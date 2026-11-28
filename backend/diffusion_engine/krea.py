from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.single_gpu_fastpaths import version_reference_latents
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.misc.image_resize import resize_reference
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.krea2_engine import Qwen3VL4BEngine
from modules.shared import opts


class Krea2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Krea2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen3VL4BEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        initial_latent = self.ini_latent
        reference_latents = list(dynamic_args.ref_latents)
        try:
            return self._get_learned_conditioning(prompt)
        except Exception as error:
            if not memory_management.is_oom(error):
                raise
        # Release the failed call's traceback before freeing memory and retrying.
        self.ini_latent = initial_latent
        dynamic_args.ref_latents = list(reference_latents)
        clip_patcher = self.forge_objects.clip.patcher
        devices = {clip_patcher.load_device}
        unet = self.forge_objects.unet
        if hasattr(unet, "secondary_device"):
            devices.update((unet.load_device, unet.secondary_device))
        cleanup_errors = []
        for device in devices:
            try:
                memory_management.unload_models_except([clip_patcher], device=device)
            except Exception as error:
                cleanup_errors.append(error)
                memory_management.logger.exception("Failed to release models on %s before Krea conditioning retry", device)
        try:
            memory_management.soft_empty_cache(force=True)
        except Exception as error:
            cleanup_errors.append(error)
            memory_management.logger.exception("Failed to clear the cache before Krea conditioning retry")
        if cleanup_errors:
            raise RuntimeError("Krea conditioning retry aborted because memory cleanup failed") from cleanup_errors[0]
        try:
            return self._get_learned_conditioning(prompt)
        except Exception as error:
            self.ini_latent = initial_latent
            dynamic_args.ref_latents = list(reference_latents)
            if memory_management.is_oom(error):
                # No further retry is possible. Release the encoder too so a
                # failed conditioning pass does not occupy both devices.
                for device in devices:
                    try:
                        memory_management.unload_models_except([], device=device)
                    except Exception:
                        memory_management.logger.exception("Failed to release models on %s after Krea conditioning OOM", device)
                try:
                    memory_management.soft_empty_cache(force=True)
                except Exception:
                    memory_management.logger.exception("Failed to clear the cache after Krea conditioning OOM")
            raise

    def _get_learned_conditioning(self, prompt: "SdConditioning"):
        clip_patcher = self.forge_objects.clip.patcher
        has_image_conditioning = self.ini_latent is not None or bool(self.ref_latents)
        device = clip_patcher.load_device
        headroom = clip_patcher.model_size() + max(4 * 1024**3, memory_management.minimum_inference_memory())
        if any(model is not clip_patcher for model in memory_management.loaded_models()) and (
            has_image_conditioning or memory_management.get_free_memory(device) < headroom
        ):
            memory_management.unload_models_except([clip_patcher], device=device)
        memory_management.load_model_gpu(clip_patcher)

        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if opts.krea2_do_reference and bool(_references):
                return self.get_learned_conditioning_with_image(prompt, _references)
            else:
                dynamic_args.ref_latents.clear()

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor]):
        images_vl, ref_latents = [], []
        for image in images:
            v, r = self.encode_vision(image)
            images_vl.append(v)
            ref_latents.append(r.squeeze(2))

        dynamic_args.ref_latents = version_reference_latents(ref_latents, flag="krea_reference_cache")
        return self.text_processing_engine_qwen(prompt, images=images_vl)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = image.movedim(-1, 1)  # b, c, h, w

        _vision = resize_reference(samples, 1, 768).movedim(1, -1)[:, :, :, :3]

        s = resize_reference(samples)
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3], output_device=s.device)
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        return (_vision, _latent)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if opts.krea2_do_reference:
            self.store_reference(x[:1].movedim(1, -1).mul(0.5).add(0.5).cpu())

        if not dynamic_args.is_referencing:
            return super().encode_first_stage(x)
