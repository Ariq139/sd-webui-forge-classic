"""Optional MMGP residency for native Diffusers pipelines."""

import gc
import logging
import os
import re
import weakref

import torch

from backend import memory_management, mmgp_loader


logger = logging.getLogger("mmgp_native")
_active: "NativeMMGP | None" = None
_warned_unavailable = False

_PROFILE_NUMBERS = {
    "HighRAM_HighVRAM": 1,
    "HighRAM_LowVRAM": 2,
    "LowRAM_HighVRAM": 3,
    "LowRAM_LowVRAM": 4,
    "VerylowRAM_LowVRAM": 5,
}

_LORA_TOKEN = re.compile(r"<lora:([^:>]+)(?::([^>]+))?>", re.IGNORECASE)


def enabled() -> bool:
    return memory_management.mmgp_runtime_enabled()


def _component_kind(name: str, model=None) -> str:
    component = getattr(model, "memory_component", "auto") if model is not None else "auto"
    if component and component != "auto":
        return component
    return mmgp_loader.component_kind(name, model)


def _main_component_name(components) -> str | None:
    for name in ("transformer", "unet", "diffusion_model", "denoiser", "model"):
        if isinstance(components.get(name), torch.nn.Module):
            return name
    for name, model in components.items():
        if isinstance(model, torch.nn.Module) and _component_kind(name, model) == "unet":
            return name
    return None


def _manager_components(components) -> tuple[dict, str | None]:
    modules = {name: model for name, model in components.items() if isinstance(model, torch.nn.Module)}
    main_name = _main_component_name(modules)
    if main_name and main_name != "transformer" and "transformer" not in modules:
        modules["transformer"] = modules.pop(main_name)
    return modules, main_name


def _find_lora_components(pipeline):
    components = []
    seen = set()
    for component in getattr(pipeline, "components", {}).values():
        if not isinstance(component, torch.nn.Module) or id(component) in seen:
            continue
        if hasattr(component, "_loras_model_data") or any(hasattr(child, "_loras_model_data") for child in component.modules()):
            seen.add(id(component))
            components.append(component)
    return tuple(components)


def _resolve_lora(name: str) -> str | None:
    """Resolve a Forge LoRA alias without scanning the model directories."""
    try:
        import networks

        entry = networks.available_network_aliases.get(name)
        if entry is not None and os.path.isfile(entry.filename):
            return entry.filename
    except Exception:
        logger.debug("Could not resolve native LoRA alias %s", name, exc_info=True)

    path = os.path.abspath(os.path.expanduser(name))
    return path if os.path.isfile(path) else None


def _prompt_loras(prompt: str) -> tuple[str, list[str], list[float], list[str]]:
    paths = []
    weights = []
    unresolved = []

    def remove_resolved(match):
        path = _resolve_lora(match.group(1).strip())
        if path is None:
            unresolved.append(match.group(1).strip())
            return match.group(0)
        try:
            weight = float(match.group(2)) if match.group(2) else 1.0
        except ValueError:
            unresolved.append(match.group(1).strip())
            return match.group(0)
        paths.append(path)
        weights.append(weight)
        return ""

    clean_prompt = _LORA_TOKEN.sub(remove_resolved, prompt or "")
    return clean_prompt.strip(), paths, weights, unresolved


class NativeMMGP:
    def __init__(self, pipeline, compile_enabled: bool | None = None):
        self.pipeline_ref = weakref.ref(pipeline)
        self.manager = None
        self.component_names = ()
        self.manager_components = None
        self.compile_enabled = compile_enabled
        self.lora_signature = None
        self.lora_components = None

    def _pipeline(self):
        return self.pipeline_ref()

    def _settings(self, pipeline):
        from modules.shared import opts
        from modules_forge.mmgp_profiles import extra_text_encoder_quantization_enabled, get_mmgp_quantization

        components = getattr(pipeline, "components", {})
        manager_components, _main_name = _manager_components(components)
        names = tuple(manager_components)
        component_kinds = {name: _component_kind(name, manager_components[name]) for name in names}
        use_budgets = memory_management.feature_enabled("budgets")
        use_pinned_memory = memory_management.feature_enabled("pinned_memory")
        use_residency_hints = memory_management.feature_enabled("residency_hints")
        budgets = {}
        pinned = []
        preferred = set(memory_management.MEMORY_RESIDENCY_COMPONENTS) if use_residency_hints else set()

        for name in names:
            kind = component_kinds[name]
            if use_budgets:
                budget = memory_management.MEMORY_BUDGETS_BYTES.get(kind, 0)
                if budget > 0:
                    budgets[name] = budget / (1024 * 1024)
            if use_pinned_memory and kind in memory_management.MEMORY_PINNED_COMPONENTS:
                pinned.append(name)

        cotenants = None
        preferred_names = [name for name in names if component_kinds[name] in preferred]
        if preferred_names:
            cotenants = {name: [other for other in preferred_names if other != name] for name in names}

        profile = getattr(opts, "forge_memory_profile", "Custom")
        quantize, _quantization_type = get_mmgp_quantization(opts)
        convert_dtype = memory_management.mmgp_compute_dtype(manager_components.get("transformer"), fallback=getattr(pipeline, "dtype", None))
        extra_models_to_quantize = [
            name for name in names
            if name != "transformer" and extra_text_encoder_quantization_enabled(opts)
            and mmgp_loader.is_large_text_encoder(manager_components[name], name)
        ]
        return {
            "profile": profile,
            "pinnedMemory": pinned or False,
            "pinnedPEFTLora": bool(pinned and getattr(opts, "forge_memory_pinned_memory_enabled", False)),
            "budgets": budgets or None,
            "workingVRAM": memory_management.MEMORY_WORKING_VRAM_BYTES / (1024 * 1024) or None,
            "asyncTransfers": memory_management.mmgp_async_transfers_enabled(),
            "quantizeTransformer": quantize and "transformer" in names,
            "extraModelsToQuantize": extra_models_to_quantize,
            "quantizationType": mmgp_loader.quantization_type(_quantization_type),
            "partialPinning": bool(getattr(opts, "forge_memory_partial_pinning_enabled", False)),
            "perc_reserved_mem_max": memory_management.MEMORY_PINNED_MEMORY_PERCENT / 100.0 if use_pinned_memory else 0,
            "compile": bool(getattr(opts, "forge_memory_compile_enabled", False) if self.compile_enabled is None else self.compile_enabled),
            "convertWeightsFloatTo": convert_dtype,
            "coTenantsMap": cotenants,
            "vram_safety_coefficient": memory_management.MEMORY_VRAM_SAFETY_PERCENT / 100.0,
            "verboseLevel": 0,
            "_manager_components": manager_components,
        }

    def release(self):
        manager = self.manager
        pipeline = self._pipeline()
        lora_components = self.lora_components
        if lora_components is None and pipeline is not None:
            lora_components = _find_lora_components(pipeline)
        self.manager = None
        self.component_names = ()
        self.manager_components = None
        self.lora_signature = None
        self.lora_components = None
        try:
            from mmgp import offload

            if pipeline is not None:
                for component in lora_components or ():
                    offload.unload_loras_from_model(component)
        except Exception:
            logger.debug("MMGP native LoRA cleanup failed", exc_info=True)
        if manager is not None:
            try:
                manager.release()
            except Exception:
                logger.debug("MMGP native manager release failed", exc_info=True)
        try:
            from mmgp import offload

            clear_caches = getattr(offload, "clear_caches", None)
            if clear_caches is not None:
                clear_caches()
            else:
                offload.shared_state.pop("_cache", None)
            offload.flush_torch_caches()
        except Exception:
            logger.debug("MMGP native cache cleanup failed", exc_info=True)
        gc.collect()

    def configure_prompt_loras(self, prompt: str) -> tuple[str, str | None]:
        """Load prompt LoRAs through MMGP when the pipeline exposes adapters.

        Returns the prompt with successfully resolved tags removed and an
        optional warning for tags that could not be applied. Non-MMGP or
        unsupported pipelines are left unchanged.
        """
        pipeline = self._pipeline()
        if pipeline is None or self.manager is None:
            return prompt, None

        clean_prompt, paths, weights, unresolved = _prompt_loras(prompt)
        signature = (tuple(paths), tuple(weights))
        if signature != self.lora_signature:
            try:
                from mmgp import offload
                from modules.shared import opts

                lora_components = self.lora_components
                if lora_components is None:
                    lora_components = _find_lora_components(pipeline)
                    self.lora_components = lora_components
                if paths and not lora_components:
                    return prompt, "Native pipeline has no MMGP LoRA adapters"

                applied_components = 0
                pinned_lora = bool(
                    memory_management.feature_enabled("pinned_memory")
                    and getattr(opts, "forge_memory_pinned_memory_enabled", False)
                )
                for component in lora_components:
                    offload.unload_loras_from_model(component)
                    if paths:
                        loaded = offload.load_loras_into_model(
                            component,
                            paths,
                            lora_multi=weights,
                            activate_all_loras=True,
                            pinnedLora=pinned_lora,
                            verboseLevel=0,
                        )
                        if loaded:
                            applied_components += 1
                if paths and not applied_components:
                    raise RuntimeError("MMGP did not find compatible native LoRA weights")
                self.lora_signature = signature
                logger.info("MMGP native LoRAs active: %s", len(paths))
            except Exception:
                for component in locals().get("lora_components", ()):
                    try:
                        offload.unload_loras_from_model(component)
                    except Exception:
                        logger.debug("Could not roll back native LoRA state", exc_info=True)
                self.lora_signature = None
                logger.exception("MMGP native LoRA loading failed; leaving the prompt unchanged")
                return prompt, None

        warning = f"Unresolved LoRA: {', '.join(unresolved)}" if unresolved else None
        return clean_prompt if paths else prompt, warning

    def build(self):
        global _warned_unavailable

        pipeline = self._pipeline()
        if pipeline is None or not torch.cuda.is_available():
            return False
        try:
            from mmgp import offload
        except Exception:
            if not _warned_unavailable:
                logger.warning("MMGP is unavailable; native pipelines will use their configured offload mode")
                _warned_unavailable = True
            return False

        settings = self._settings(pipeline)
        profile = settings.pop("profile")
        manager_components = settings.pop("_manager_components")
        self.release()
        self.manager_components = manager_components
        try:
            if profile in _PROFILE_NUMBERS:
                profile_type = getattr(offload, "profile_type", None)
                profile_value = getattr(profile_type, profile, _PROFILE_NUMBERS[profile])
                self.manager = offload.profile(manager_components, profile_value, **settings)
            else:
                self.manager = offload.all(manager_components, **settings)
        except Exception:
            self.release()
            raise

        self.component_names = tuple(manager_components)
        self.lora_components = _find_lora_components(pipeline)
        logger.info("Reference MMGP residency manager active for native pipeline (%s)", ", ".join(self.component_names))
        return True


def attach(pipeline, compile_enabled: bool | None = None) -> bool:
    global _active

    if not enabled():
        # Disabling the optional path must release whichever native pipeline
        # was previously managed, even when the newly selected pipeline differs.
        release_all()
        return False
    if _active is None or _active._pipeline() is not pipeline:
        release_all()
        _active = NativeMMGP(pipeline, compile_enabled=compile_enabled)
    if _active.manager is None:
        try:
            return _active.build()
        except Exception:
            logger.exception("Reference MMGP could not manage native pipeline; using native Forge offload")
            _active.release()
            return False
    return True


def configure_prompt_loras(pipeline, prompt: str) -> tuple[str, str | None]:
    """Configure native prompt LoRAs, or preserve the prompt when inactive."""
    if not enabled() or _active is None or _active._pipeline() is not pipeline:
        return prompt, None

    try:
        from modules.shared import opts

        if not getattr(opts, "forge_memory_dynamic_lora_enabled", True):
            # Clear adapters that were loaded before the setting was disabled.
            _active.configure_prompt_loras("")
            return prompt, None
    except Exception:
        logger.debug("Could not read the native dynamic LoRA setting", exc_info=True)

    return _active.configure_prompt_loras(prompt)


def release(pipeline=None):
    global _active
    if _active is None:
        return
    if pipeline is None or _active._pipeline() is pipeline:
        _active.release()
        _active = None


def release_all():
    release()


def active_pipeline():
    """Return the currently managed native pipeline for generic cache/export code."""
    return _active._pipeline() if _active is not None else None
