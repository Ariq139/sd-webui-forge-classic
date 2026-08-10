# Neo Instructions

This page combines the edit/reference and video-generation workflows. It does
not replace the upstream Forge documentation:

- [Download Models](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
  lists checkpoints, text encoders, VAEs, model folders, and model-specific
  downloads.
- [Inference References](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)
  contains example outputs and recommended parameters.

## Checkpoints, text encoders, and VAEs

For normal Forge image models, use these folders:

- `models/Stable-diffusion` for checkpoints, UNets, and DiTs
- `models/text_encoder` for text encoders
- `models/VAE` for VAEs

Use the model-specific download table before combining components. A filename
match does not guarantee compatibility; a wrong VAE or text encoder can cause a
state-dict/tensor mismatch or incorrect output.

## Wan 2.2 text-to-video and image-to-video

Neo uses the normal generation pages for Wan; there is no separate Wan page.
Wan 2.2 support currently targets the 14B models, not the 5B models.

1. Put the compatible Wan model in `models/Stable-diffusion` and refresh the
   model list.
2. Select **UI Preset → wan**, then select the Wan checkpoint.
3. For text-to-video, open **txt2img**, enter a prompt, set the **Frames**
   control, and click **Generate**.
4. For image-to-video, open **img2img**, provide the source image, use high
   denoising strength, and click **Generate**.
5. For first/last-frame video, add the images through **ImageStitch
   Integrated** and use the Wan img2img workflow.
6. Use **Refiner** when the model requires Wan 2.2 high-noise/low-noise model
   switching.

FFmpeg must be installed to export the generated video. Video settings such as
frame saving, container, CRF, and output directory are under **Settings →
Saving Videos**.

## LTX-2.3 text-to-video and image-to-video

LTX supports both workflows:

1. Place a complete Diffusers-format LTX model under
   `models/Stable-diffusion`.
2. Select **UI Preset → ltx2** and the LTX model in the global model dropdown.
3. Use **txt2img** for text-to-video or **img2img** with a source image for
   image-to-video.
4. Set **Frames**, width, height, steps, CFG, and the LTX options on the
   **LTX Video** page when needed.

The LTX model must contain its compatible transformer, Gemma text encoder,
connectors, video/audio VAEs, and vocoder. Forge's generic VAE/text-encoder
selector is not applied to LTX. The optional decoder can be selected from the
global **VAE / Decoder** quicksetting when a compatible local decoder is
installed in `models/VAE`; the selector is empty when none is available.

## Ideogram 4

Ideogram 4 is text-to-image only in this integration. Select **UI Preset →
ideogram** and use **txt2img** or the **Ideogram 4** page. The **img2img** page
is hidden for this preset. Ideogram's VAE and text encoder are loaded from the
complete selected Diffusers model; normal Forge VAE/text-encoder, negative
prompt, sampler, scheduler, and Hires controls are not used.

## Edit and reference models

### Common ImageStitch workflow

ImageStitch is the Neo control for reference images and multi-image edit inputs.

1. Select the checkpoint and its matching external text encoder/VAE, if the
   model requires them.
2. Open **Generation → ImageStitch Integrated**.
3. Add a source image with **Image to Upload**, then click **Append Pasted
   Image**. Use **Replace Selected Image** or **Delete Selected Image** to
   maintain the reference list.
4. Enter an instruction, such as `Change the jacket to red while preserving
   the character, pose, and background.`
5. Generate from the model's required tab below.

The **Maximum Side Length** control can reduce VAE-encoding VRAM usage. Set it
to `0` for no resize limit.

### Flux.2-Klein edit

1. Select the **klein** preset and a matching Flux.2-Klein 4B or 9B
   checkpoint.
2. Select the matching Qwen3 text encoder and Flux.2 VAE from the
   [Download Models](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
   table.
3. Use **txt2img**, add the source image through **ImageStitch Integrated**,
   and describe the edit.
4. Follow the
   [Klein inference references](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References#flux2-klein).

Klein reference mode is enabled by default. For regular img2img, enable
**Settings → Stable Diffusion → [Klein] Disable Reference**.

### Qwen-Image-Edit

1. Select a Qwen-Image-Edit checkpoint and matching Qwen text encoder and
   `qwen_image_vae`.
2. Use **img2img** with the source image, or add images through ImageStitch.
3. Enter an editing instruction and generate.
4. Keep denoising strength at or above `0.9`.

Neo detects this mode when the checkpoint path contains both `qwen` and `edit`.

### Flux Kontext

Select a compatible Kontext checkpoint and modules, ensure `kontext` appears in
the checkpoint path, then use **txt2img** with ImageStitch references.

### Anima and Krea 2 edit modes

Install the required model-specific edit LoRA, enable the matching reference
setting under **Settings → Stable Diffusion**, then use **txt2img** with
ImageStitch and an edit instruction.

## Troubleshooting

- If the UI behaves like ordinary txt2img, verify the checkpoint detection
  name and the relevant reference setting.
- If Klein ignores a reference, ensure **Disable Reference** is off and the
  image was added to ImageStitch.
- If Qwen edit is weak, use denoising strength `0.9` or higher and confirm the
  checkpoint path contains `qwen` and `edit`.
- For a tensor mismatch or missing VAE state dict, remove unrelated modules and
  select the model-specific VAE/text encoder.
- If a module is not listed, place it in the correct folder and refresh the
  model list.

## Scope and attribution

The combined workflow notes and Neo-specific behavior were written by Codex for
this fork. They are not upstream Forge documentation and may change as Neo
changes. Model authors, licenses, and upstream Forge documentation remain the
authoritative sources for downloads and baseline inference guidance.
