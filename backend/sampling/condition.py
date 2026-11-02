import math

import torch


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor


class Condition:
    def __init__(self, cond, cache_key=None):
        self.cond = cond
        self.cache_key = cache_key

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, cache=None, **kwargs):
        key = None
        if cache is not None and self.cache_key is not None:
            key = (self.cache_key, batch_size, device.type, device.index)
            cached = cache.get(key)
            if cached is not None:
                return self._copy_with(cached)

        cond = repeat_to_batch_size(self.cond, batch_size).to(device)
        if key is not None:
            cache[key] = cond
        return self._copy_with(cond)

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class ConditionNoiseShape(Condition):
    def process_cond(self, batch_size, device, area, cache=None, **kwargs):
        key = None
        if cache is not None and self.cache_key is not None:
            key = (self.cache_key, batch_size, device.type, device.index, tuple(area))
            cached = cache.get(key)
            if cached is not None:
                return self._copy_with(cached)

        data = self.cond[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
        cond = repeat_to_batch_size(data, batch_size).to(device)
        if key is not None:
            cache[key] = cond
        return self._copy_with(cond)


class ConditionCrossAttn(Condition):
    pass


class ConditionConstant(Condition):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond


def compile_conditions(cond, cache_namespace=None):
    if cond is None:
        return None
    if isinstance(cond, list):
        from modules.prompt_parser import stack_conditioning

        groups = {}
        for i, row in enumerate(cond):
            groups.setdefault(conditioning_shape(row), []).append(i)
        result = []
        for indices in groups.values():
            group_cache_namespace = (cache_namespace, tuple(indices))
            compiled = compile_conditions(stack_conditioning([cond[i] for i in indices]), cache_namespace=group_cache_namespace)[0]
            compiled["batch_indices"] = indices
            result.append(compiled)
        return result

    if isinstance(cond, torch.Tensor):
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cond, cache_key=(cache_namespace, "crossattn")),
            ),
        )
        return [result]

    cross_attn = cond["crossattn"]
    pooled_output = cond["vector"]

    result = dict(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        model_conds=dict(
            c_crossattn=ConditionCrossAttn(cross_attn, cache_key=(cache_namespace, "crossattn")),
            y=Condition(pooled_output, cache_key=(cache_namespace, "vector")),
        ),
    )

    if "guidance" in cond:
        result["model_conds"]["guidance"] = Condition(cond["guidance"], cache_key=(cache_namespace, "guidance"))

    return [result]


def compile_weighted_conditions(cond, weights, cache_namespace=None):
    if isinstance(cond, list):
        from modules.prompt_parser import stack_conditioning

        groups = {}
        for image, composition in enumerate(weights):
            for term, (index, weight) in enumerate(composition):
                key = (term, conditioning_shape(cond[index]), weight)
                groups.setdefault(key, []).append((image, index))

        results = []
        for (term, _, weight), rows in groups.items():
            image_indices = [image for image, _ in rows]
            condition_indices = [index for _, index in rows]
            group_cache_namespace = (cache_namespace, term, tuple(image_indices), tuple(condition_indices))
            compiled = compile_conditions(
                stack_conditioning([cond[index] for index in condition_indices]),
                cache_namespace=group_cache_namespace,
            )[0]
            compiled.update(batch_indices=image_indices, strength=weight)
            results.append(compiled)
        return results

    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, "advanced_indexing"):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = cond[current_indices]

        h = compile_conditions(feed, cache_namespace=(cache_namespace, tuple(current_indices)))
        h[0]["strength"] = current_weight
        results += h

    return results
