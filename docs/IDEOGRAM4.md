# Ideogram 4

Neo includes an **Ideogram 4** page and an `ideogram` UI preset for the
Diffusers-format `ideogram-ai/ideogram-v4` pipeline.

Place a local copy of the Diffusers repository under
`models/Stable-diffusion/`; it will appear in the normal model dropdown.

## Shared UI-preset workflow

1. Select **UI Preset → ideogram** in the quicksettings row.
2. Select the model in the global **Ideogram 4 Model** dropdown.
3. Use the normal **txt2img** prompt and **Generate** button.
4. Set width, height, image count, steps, guidance, schedule values, and seed
using the normal controls or **Settings → Presets → Ideogram 4**.

The shared txt2img route supports up to 8 total images from **Batch Count ×
Batch Size**. Ideogram's native sampler and scheduler are managed internally.

Ideogram 4 is text-to-image only in this integration. The normal img2img,
Hires fix, sampler, scheduler, VAE/text-encoder, low-bit, and negative-prompt
controls are disabled or hidden because the native Diffusers pipeline does not
consume them. Its VAE and text encoder are loaded from the selected Ideogram
model.

## Direct page

The **Ideogram 4** page exposes the native options directly, including CPU
offload, precision, VAE tiling, flow schedule `mu` and `std`, prompt
upsampling, prompt temperature, image count, and seed.

Prompt upsampling requires the optional prompt-enhancer component bundled or
configured with the selected model. It is disabled by default.

## Model reference

See the [Diffusers Ideogram 4 documentation](https://huggingface.co/docs/diffusers/main/api/pipelines/ideogram4)
for model files, native pipeline behavior, and prompt-encoding details.
