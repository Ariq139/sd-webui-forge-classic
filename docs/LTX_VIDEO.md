# LTX-2.3 video

Neo includes an **LTX Video** page and an `ltx2` UI preset for Diffusers-format
LTX-2.3 models. It supports text-to-video and image-to-video generation,
optional audio output, CPU offload, VAE tiling, and an optional PrunaVAED
decoder.

## Models

Use a local, complete Diffusers directory under the normal model folder, for
example:

- `models/Stable-diffusion/LTX-2.3-Diffusers/`

The pipeline requires the LTX-2.3 transformer, Gemma text encoder, text
connectors, video VAE, audio VAE, and vocoder supplied by the selected
Diffusers model.

## PrunaVAED

Place a compatible local PrunaVAED decoder in `models/VAE` and select it from
the **PrunaVAED** dropdown. It replaces only the LTX-2.3 video decoder. It does
not replace the denoising model, change the latent format, or speed up
denoising. If no compatible decoder is found, the dropdown remains empty.

PrunaVAED is distributed under the LTX-2 Community License. Review the model
card and its restrictions before using or redistributing it:

<https://huggingface.co/PrunaAI/PrunaVAED>

## Shared UI-preset workflow

1. Select **UI Preset → ltx2** in the top quicksettings row.
2. Select the LTX-2.3 model in the global **LTX-2.3 Model** dropdown.
3. Use the normal **txt2img** prompt and **Generate** button for text-to-video.
4. Use the normal **img2img** page with an input image and **Generate** for
   image-to-video.
5. Adjust LTX-specific defaults in **Settings → Presets → LTX2** or on the
   **LTX Video** page. The values are shared and saved with the preset.

The shared txt2img/img2img controls for prompt, styles, width, height, frames,
CFG, and steps are used for the LTX request. The shared route generates one
video per request; batch count and batch img2img are disabled.

The shared preset route uses the LTX runner instead of Forge's image-model
loader. Forge's regular checkpoint, VAE/text-encoder, and low-bit controls are
not used while `ltx2` is selected; LTX loads its required transformer, Gemma
encoder, connectors, video/audio VAEs, and vocoder from the selected model.

## Direct page workflow

1. Open **LTX Video**.
2. Select the model from the global checkpoint quicksetting.
3. Optionally select a compatible PrunaVAED decoder.
4. Enter a prompt. Add an image for image-to-video generation.
5. Set resolution, frame count, FPS, steps, and guidance values.
6. Enable sequential offload when VRAM is limited.
7. Click **Generate**.

The current page generates a single LTX-2.3 stage. Two-stage latent upscaling
and LTX-specific control modules remain separate work.
