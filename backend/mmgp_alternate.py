"""Opt-in adapter for the reference MMGP residency manager.

Forge still owns patch application. MMGP owns only the CPU/GPU residency of
the three base components after their Forge patches have been prepared.
"""

import gc
import logging
import weakref

import torch

from backend import memory_management, mmgp_loader


logger = logging.getLogger("mmgp_alternate")

_active: "AlternateMMGP | None" = None
_warned_unavailable = False

_PROFILE_NUMBERS = {
    "HighRAM_HighVRAM": 1,
    "HighRAM_LowVRAM": 2,
    "LowRAM_HighVRAM": 3,
    "LowRAM_LowVRAM": 4,
    "VerylowRAM_LowVRAM": 5,
}


def _restore_hooks(modules):
    for model in modules.values():
        for module in model.modules():
            for name in ("forward", *getattr(module, "_offload_hooks", ())):
                original = getattr(module, f"_mm_{name}", None)
                if original is None:
                    continue
                setattr(module, name, original)
                delattr(module, f"_mm_{name}")
            for name in ("_mm_manager", "_mm_model_id", "_mm_blocks_name", "_mm_Id", "_mm_id"):
                if hasattr(module, name):
                    delattr(module, name)


class AlternateMMGP:
    def __init__(self, engine):
        self.engine_ref = weakref.ref(engine)
        self.manager = None
        self.modules = {}
        self.patchers = {}
        self.signatures = None
        self.failed_signature = None

    def _engine(self):
        return self.engine_ref()

    def _get_modules_and_patchers(self):
        engine = self._engine()
        if engine is None or getattr(engine, "forge_objects", None) is None:
            return {}, {}

        objects = engine.forge_objects
        unet_model = getattr(objects.unet, "model", None)
        transformer = getattr(unet_model, "diffusion_model", unet_model)
        modules = {
            "transformer": transformer,
            "text_encoder": getattr(objects.clip, "cond_stage_model", None),
            "vae": getattr(objects.vae, "first_stage_model", None),
        }
        patchers = {
            "transformer": objects.unet,
            "text_encoder": getattr(objects.clip, "patcher", None),
            "vae": getattr(objects.vae, "patcher", None),
        }
        extras = list(getattr(objects.unet, "extra_model_patchers_during_sampling", ()))
        controlnet = getattr(objects.unet, "controlnet_linked_list", None)
        while controlnet is not None:
            extras.append(getattr(controlnet, "control_model_wrapped", None))
            controlnet = getattr(controlnet, "previous_controlnet", None)
        extra_index = 0
        seen_extra_models = set()
        for patcher in extras:
            model = getattr(patcher, "model", None) if patcher is not None else None
            if model is None or id(model) in seen_extra_models:
                continue
            seen_extra_models.add(id(model))
            key = f"extra_{extra_index}"
            patchers[key] = patcher
            modules[key] = model
            extra_index += 1
        modules = {key: value for key, value in modules.items() if isinstance(value, torch.nn.Module)}
        patchers = {key: value for key, value in patchers.items() if value is not None}
        if not isinstance(modules.get("transformer"), torch.nn.Module):
            return {}, {}
        return modules, patchers

    @staticmethod
    def _signature(patchers):
        return tuple(sorted((key, id(patcher.model), str(patcher.patches_uuid)) for key, patcher in patchers.items()))

    def handles(self, models):
        if not models:
            return False
        managed_ids = {id(patcher.model) for patcher in self.patchers.values()}
        return all(id(getattr(model, "model", None)) in managed_ids for model in models)

    def _release(self):
        manager = self.manager
        modules = self.modules
        self.manager = None
        self.modules = {}
        self.patchers = {}
        self.signatures = None

        try:
            from mmgp import offload

            for model in modules.values():
                if hasattr(model, "_loras_model_data") or any(hasattr(child, "_loras_model_data") for child in model.modules()):
                    offload.unload_loras_from_model(model)

            clear_caches = getattr(offload, "clear_caches", None)
            if clear_caches is not None:
                clear_caches()
            else:
                offload.shared_state.pop("_cache", None)
        except Exception:
            logger.debug("MMGP cache cleanup failed", exc_info=True)

        if manager is not None:
            try:
                manager.release()
            except Exception:
                logger.debug("MMGP manager release failed", exc_info=True)
        _restore_hooks(modules)
        try:
            from mmgp import offload

            offload.flush_torch_caches()
        except Exception:
            logger.debug("MMGP cache flush failed", exc_info=True)
        gc.collect()

    def suspend(self):
        self._release()
        self.failed_signature = None

    @staticmethod
    def _reset_forge_patchers(patchers):
        """Undo the temporary CPU patching before Forge takes ownership again."""
        cpu = torch.device("cpu")
        for patcher in patchers.values():
            try:
                # _prepare_patchers() applies Forge patches on CPU so MMGP can
                # inspect the modules.  If profiling fails, clear that state;
                # otherwise Forge can see forge_patched_weights and skip the
                # move back to CUDA during its normal fallback load.
                patcher.unpatch_model(cpu, unpatch_weights=True)
                patcher.model_patches_to(cpu)
            except Exception:
                logger.debug("Could not restore Forge patcher after MMGP setup failure", exc_info=True)

    def _prepare_patchers(self, patchers):
        cpu = torch.device("cpu")
        for patcher in patchers.values():
            try:
                patcher.unpatch_model(cpu, unpatch_weights=True)
            except Exception:
                logger.debug("Could not reset Forge patches before MMGP setup", exc_info=True)
            patcher.model_patches_to(cpu)
            patcher.patch_model(device_to=cpu, force_patch_weights=True)

    def _settings(self):
        from modules.shared import opts

        current_modules, _ = self._get_modules_and_patchers()
        budgets = {}
        if memory_management.feature_enabled("budgets"):
            for component, module_id in (("unet", "transformer"), ("text_encoder", "text_encoder"), ("vae", "vae")):
                value = memory_management.MEMORY_BUDGETS_BYTES.get(component, 0)
                if value > 0:
                    budgets[module_id] = value / (1024 * 1024)
            controlnet_budget = memory_management.MEMORY_BUDGETS_BYTES.get("controlnet", 0)
            if controlnet_budget > 0:
                budgets.update({key: controlnet_budget / (1024 * 1024) for key in current_modules if key.startswith("extra_")})

        pinned = []
        if memory_management.feature_enabled("pinned_memory"):
            for component, module_id in (("unet", "transformer"), ("text_encoder", "text_encoder"), ("vae", "vae")):
                if component in memory_management.MEMORY_PINNED_COMPONENTS:
                    pinned.append(module_id)
            if "controlnet" in memory_management.MEMORY_PINNED_COMPONENTS:
                pinned.extend(key for key in current_modules if key.startswith("extra_"))

        profile = getattr(opts, "forge_memory_profile", "Custom")
        quantize = bool(getattr(opts, "forge_memory_alternate_quantization", False))
        engine = self._engine()
        forge_unet = getattr(getattr(engine, "forge_objects", None), "unet", None)
        convert_dtype = memory_management.mmgp_compute_dtype(getattr(forge_unet, "model", None))
        extra_models_to_quantize = []
        if quantize and profile in {"LowRAM_HighVRAM", "LowRAM_LowVRAM", "VerylowRAM_LowVRAM"}:
            text_encoder = current_modules.get("text_encoder")
            module_names = {getattr(module, "__module__", "").lower() for module in text_encoder.modules()} if text_encoder else set()
            if not mmgp_loader.has_llm_adapter(text_encoder) and any(any(marker in module_name for marker in ("t5", "llama", "llm")) for module_name in module_names):
                extra_models_to_quantize.append("text_encoder")
        preferred = set(memory_management.MEMORY_RESIDENCY_COMPONENTS) if memory_management.feature_enabled("residency_hints") else set()
        component_ids = {"unet": "transformer", "text_encoder": "text_encoder", "vae": "vae"}
        preferred_ids = {component_ids[name] for name in preferred if name in component_ids}
        preferred_ids.update(key for key in current_modules if key.startswith("extra_") and "controlnet" in preferred)
        cotenants = None
        if preferred_ids:
            cotenants = {
                key: [other for other in preferred_ids if other != key]
                for key in current_modules
            }
        return {
            "profile": profile,
            "pinnedMemory": pinned or False,
            "pinnedPEFTLora": bool(pinned and getattr(opts, "forge_memory_pinned_memory_enabled", False)),
            "budgets": budgets or None,
            "workingVRAM": memory_management.MEMORY_WORKING_VRAM_BYTES / (1024 * 1024) or None,
            "asyncTransfers": memory_management.async_transfers_enabled(),
            "quantizeTransformer": quantize,
            "extraModelsToQuantize": extra_models_to_quantize,
            "quantizationType": mmgp_loader.quantization_type(),
            "partialPinning": bool(getattr(opts, "forge_memory_partial_pinning_enabled", False)),
            "perc_reserved_mem_max": memory_management.MEMORY_PINNED_MEMORY_PERCENT / 100.0 if memory_management.feature_enabled("pinned_memory") else 0,
            "compile": bool(getattr(opts, "forge_memory_compile_enabled", False)),
            "convertWeightsFloatTo": convert_dtype,
            "coTenantsMap": cotenants,
            "vram_safety_coefficient": memory_management.MEMORY_VRAM_SAFETY_PERCENT / 100.0,
            "verboseLevel": 0,
        }

    def _build(self):
        global _warned_unavailable

        try:
            from mmgp import offload
        except Exception:
            if not _warned_unavailable:
                logger.warning("--mmgp was requested but the mmgp package is unavailable; using Forge memory management")
                _warned_unavailable = True
            return False

        modules, patchers = self._get_modules_and_patchers()
        if "transformer" not in modules:
            return False

        self._release()
        self.modules = modules
        self.patchers = patchers
        try:
            self._prepare_patchers(patchers)
            settings = self._settings()
            profile = settings.pop("profile")
            if profile in _PROFILE_NUMBERS:
                profile_type = getattr(offload, "profile_type", None)
                profile_value = getattr(profile_type, profile, _PROFILE_NUMBERS[profile])
                self.manager = offload.profile(modules, profile_value, **settings)
            else:
                self.manager = offload.all(modules, **settings)
        except Exception:
            self._reset_forge_patchers(patchers)
            self._release()
            raise

        self.signatures = self._signature(patchers)
        logger.info("Reference MMGP residency manager active for %s", ", ".join(modules))
        return True

    def load(self, models):
        if not torch.cuda.is_available():
            return False

        current_modules, current_patchers = self._get_modules_and_patchers()
        current_patch_ids = {id(patcher.model) for patcher in current_patchers.values()}
        if not all(id(getattr(model, "model", None)) in current_patch_ids for model in models):
            self.suspend()
            return False
        signature = self._signature(current_patchers)
        if self.manager is None and signature == self.failed_signature:
            return False
        if self.manager is None or signature != self.signatures or set(current_modules) != set(self.modules):
            try:
                built = self._build()
            except Exception as error:
                self.failed_signature = signature
                logger.warning(
                    "Reference MMGP is incompatible with this Forge model (%s); using Forge memory management",
                    type(error).__name__,
                )
                logger.debug("Reference MMGP setup details", exc_info=True)
                built = False
            if not built:
                return False

        if not self.handles(models):
            self._release()
            return False

        return True


def get_manager_for_models(models):
    global _active

    if not memory_management.mmgp_enabled() or not memory_management.feature_enabled("enabled"):
        if _active is not None:
            _active.suspend()
            _active = None
        return None

    from modules.shared import sd_model

    if sd_model is None:
        return None
    if _active is None or _active._engine() is not sd_model:
        if _active is not None:
            _active.suspend()
        _active = AlternateMMGP(sd_model)
    return _active


def release():
    global _active
    if _active is not None:
        _active.suspend()
        _active = None
