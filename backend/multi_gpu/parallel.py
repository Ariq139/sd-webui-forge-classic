"""Experimental primitives for independent predictions and split linear layers.

These primitives do not install model replicas or alter Forge sampling.
Callers must own independent models and their lifecycle.
"""

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F


def _run_prediction(function, arguments):
    with torch.inference_mode():
        return function(*arguments)


class ParallelPredictions:
    """Run two independent prediction callables concurrently."""

    def __init__(self, conditional, unconditional):
        self.conditional = conditional
        self.unconditional = unconditional
        self._workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="forge-cfg")

    def predict(self, conditional_args, unconditional_args):
        first = self._workers.submit(_run_prediction, self.conditional, conditional_args)
        second = self._workers.submit(_run_prediction, self.unconditional, unconditional_args)
        # Always join both branches before exposing errors or returning. This
        # prevents a failed branch from leaving another running after cleanup.
        try:
            conditional = first.result()
        except BaseException:
            try:
                second.result()
            except BaseException:
                pass
            raise
        unconditional = second.result()
        return conditional, unconditional

    def close(self):
        self._workers.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ParallelCFG:
    """Experimental sampler adapter for explicitly supplied model replicas."""

    def __init__(self, primary, secondary, devices):
        self.primary = primary
        self.secondary = secondary
        self.devices = tuple(torch.device(device) for device in devices)
        if len(self.devices) != 2 or any(device.type != "cuda" or device.index is None for device in self.devices) or self.devices[0] == self.devices[1]:
            raise ValueError("Two distinct indexed CUDA devices are required")
        if primary is secondary:
            raise ValueError("CFG branches require independent models")
        self._conditional_cache = {}
        self._unconditional_cache = {}
        self._conditional_sources = ()
        self._unconditional_sources = ()
        self._executor = ParallelPredictions(self._conditional, self._unconditional)

    @staticmethod
    def eligible(cond, uncond, model_options, x):
        from backend.args import dynamic_args
        from backend.sampling.condition import Condition, ConditionCrossAttn

        if x.device.type != "cuda" or uncond is None or len(cond) != 1 or len(uncond) != 1:
            return False
        if (dynamic_args.ref_latents or dynamic_args.context_handler is not None or dynamic_args.is_referencing
                or dynamic_args.concat_latent is not None or any(value is not None for value in dynamic_args.lq_latent)
                or dynamic_args.kontext or dynamic_args.edit or dynamic_args.qwen21
                or dynamic_args.klein or dynamic_args.wan or dynamic_args.pid or dynamic_args.moss):
            return False
        if any(key != "transformer_options" for key in model_options):
            return False
        transformer_options = model_options.get("transformer_options", {})
        if set(transformer_options) - {"sampling_sigmas"}:
            return False
        if "sampling_sigmas" in transformer_options and not isinstance(transformer_options["sampling_sigmas"], torch.Tensor):
            return False
        for item in (cond[0], uncond[0]):
            if set(item) - {"cross_attn", "pooled_output", "model_conds", "strength", "batch_indices"} or item.get("strength", 1) != 1:
                return False
            if "batch_indices" in item and list(item["batch_indices"]) != list(range(x.shape[0])):
                return False
            if any(type(value) not in (Condition, ConditionCrossAttn) or not isinstance(value.cond, torch.Tensor) for value in item["model_conds"].values()):
                return False
        return True

    def _conditional(self, cond, x, timestep, model_options):
        from backend.sampling.sampling_function import calc_cond_uncond_batch
        with torch.cuda.device(self.devices[0]):
            cache = self._condition_cache(cond, "conditional")
            return calc_cond_uncond_batch(self.primary, cond, None, x, timestep, model_options, cache, materialize_uncond=False)[0]

    def _unconditional(self, uncond, x, timestep, model_options):
        from backend.sampling.sampling_function import calc_cond_uncond_batch
        with torch.cuda.device(self.devices[1]):
            options = model_options.get("transformer_options", {})
            options = {key: value.to(self.devices[1]) for key, value in options.items()}
            cache = self._condition_cache(uncond, "unconditional")
            return calc_cond_uncond_batch(self.secondary, [], uncond, x.to(self.devices[1]), timestep.to(self.devices[1]), {"transformer_options": options}, cache)[1]

    def _condition_cache(self, conditions, branch):
        cache = getattr(self, f"_{branch}_cache")
        previous = getattr(self, f"_{branch}_sources")
        sources = []
        for item in conditions:
            for value in item["model_conds"].values():
                tensor = value.cond
                try:
                    version = tensor._version
                except RuntimeError:
                    # Inference tensors do not track in-place changes.
                    cache.clear()
                    setattr(self, f"_{branch}_sources", ())
                    return None
                sources.append((tensor, version))
        if len(previous) != len(sources) or any(old is not new or old_version != new_version for (old, old_version), (new, new_version) in zip(previous, sources)):
            cache.clear()
            setattr(self, f"_{branch}_sources", tuple(sources))
        return cache

    def predict(self, cond, uncond, x, timestep, model_options):
        if x.device != self.devices[0]:
            raise ValueError("Latents must be on the primary replica device")
        # Full-batch indices are equivalent to no selection and avoiding the
        # selection context keeps both workers from mutating shared state.
        cond = [{key: value for key, value in item.items() if key != "batch_indices"} for item in cond]
        uncond = [{key: value for key, value in item.items() if key != "batch_indices"} for item in uncond]
        conditional, unconditional = self._executor.predict((cond, x, timestep, model_options), (uncond, x, timestep, model_options))
        return conditional, unconditional.to(self.devices[0])

    def close(self):
        self._executor.close()
        self._conditional_cache.clear()
        self._unconditional_cache.clear()
        self._conditional_sources = ()
        self._unconditional_sources = ()


class SplitLinear(torch.nn.Module):
    """Reference tensor-parallel linear operation with explicit gathering.

    This prototype handles ordinary floating-point weights and inference only.
    It does not replace quantized, patched, or online-LoRA layers.
    """

    def __init__(self, weight, bias, devices, partition="column"):
        super().__init__()
        if partition not in ("column", "row"):
            raise ValueError("partition must be column or row")
        if weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError("weight must be a floating-point matrix")
        self.devices = tuple(torch.device(device) for device in devices)
        if len(self.devices) != 2 or self.devices[0] == self.devices[1]:
            raise ValueError("two distinct devices are required")
        dimension = 0 if partition == "column" else 1
        if weight.shape[dimension] < 2:
            raise ValueError("partitioned dimension must contain at least two elements")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("bias must match the output dimension")
        self.partition = partition
        parts = torch.tensor_split(weight.detach(), 2, dim=dimension)
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(part.to(device).contiguous(), requires_grad=False)
            for part, device in zip(parts, self.devices)
        ])
        self.bias = torch.nn.Parameter(bias.detach().to(self.devices[0]), requires_grad=False) if bias is not None else None

    def forward(self, x):
        if torch.is_grad_enabled():
            raise RuntimeError("SplitLinear supports inference only")
        if self.partition == "column":
            outputs = [F.linear(x.to(device), weight) for device, weight in zip(self.devices, self.weights)]
            result = torch.cat([output.to(self.devices[0]) for output in outputs], dim=-1)
        else:
            inputs = torch.split(x, [weight.shape[1] for weight in self.weights], dim=-1)
            outputs = [F.linear(part.to(device), weight) for part, device, weight in zip(inputs, self.devices, self.weights)]
            result = outputs[0] + outputs[1].to(self.devices[0])
        return result if self.bias is None else result + self.bias
