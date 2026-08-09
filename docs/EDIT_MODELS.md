# Instructions: edit and reference models in Neo

This page documents Neo-specific UI behavior for edit/reference models. It does
not replace the upstream Forge documentation:

- [Download Models](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
  lists checkpoints, text encoders, VAEs, model folders, and model-specific
  downloads.
- [Inference References](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)
  contains example outputs and recommended parameters for the supported model
  families.

## Checkpoints, text encoders, and VAEs

Forge may load these as separate components. The top **VAE / Text Encoder**
selector keeps multiple selections, so a model can use both a VAE and one or
more text encoders.

- Put checkpoints, UNets, or DiTs in `models/Stable-diffusion`.
- Put text encoders in `models/text_encoder`.
- Put VAEs in `models/VAE`.
- Use the [Forge model download table](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
  to choose compatible files instead of mixing components from unrelated
  architectures.
- For background information, see the
  [Transformers model documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3)
  for Qwen text encoders and the
  [Diffusers VAE documentation](https://huggingface.co/docs/diffusers/main/en/api/models/autoencoderkl).

A text encoder converts the prompt into conditioning used by the diffusion
model. A VAE converts images to and from the model's latent representation. A
wrong text encoder or VAE can cause a state-dict/tensor mismatch or produce
incorrect output; selecting a file merely because its filename looks similar
does not make it compatible.

## Common ImageStitch workflow

ImageStitch is the Neo control used for reference images and multi-image edit
inputs.

1. Select the checkpoint and its matching external text encoder/VAE, if the
   model requires them.
2. Open **Generation → ImageStitch Integrated**.
3. Add a source image with **Image to Upload**, then click **Append Pasted
   Image**. Use **Replace Selected Image** or **Delete Selected Image** to
   maintain the reference list.
4. Enter an instruction in the prompt, such as `Change the jacket to red while
   preserving the character, pose, and background.`
5. Generate from the tab required by the model below.

The **Maximum Side Length** control can reduce VAE-encoding VRAM usage. Set it
to `0` for no resize limit.

If the reference image has no effect, first check the model-name detection
rules below and confirm that ImageStitch contains at least one image.

## Flux.2-Klein edit

For the normal Klein reference/edit workflow:

1. Select the **klein** UI preset and a matching Flux.2-Klein 4B or 9B
   checkpoint.
2. Select the Qwen3 text encoder and Flux.2 VAE listed for that checkpoint in
   the [Download Models](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
   table. The 4B and 9B variants do not use the same text-encoder setup.
3. Use **txt2img**, add the source image through **ImageStitch Integrated**,
   and describe the requested edit in the prompt.
4. Start with the parameters shown in the
   [Klein inference references](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References#flux2-klein).

Klein reference mode is enabled by default. To use regular img2img instead,
enable **Settings → Stable Diffusion → [Klein] Disable Reference** and then
provide the source image in the normal img2img input. That option deliberately
switches Klein away from the reference path.

## Qwen-Image-Edit

1. Select a Qwen-Image-Edit checkpoint and the matching Qwen text encoder and
   `qwen_image_vae` (or the VAE listed for that checkpoint).
2. Use **img2img** and provide the source image in its main image input, or
   add one or more images through **ImageStitch Integrated**.
3. Enter a direct editing instruction and generate.
4. Keep denoising strength at or above `0.9` for edit models. Lower values may
   be warned about and can make the edit path ineffective.

Neo detects Qwen edit mode from the checkpoint path. The path must contain
both `qwen` and `edit` (case-insensitive), for example in the filename or a
parent folder. If the checkpoint is not detected as an edit model, rename the
containing folder/file descriptively or use the model's normal non-edit
workflow.

## Flux Kontext

1. Select a Flux Kontext checkpoint and its compatible modules.
2. Ensure `kontext` appears in the checkpoint filename or one of its parent
   folders so Neo can select the Kontext path.
3. Use **txt2img** with **ImageStitch Integrated** for reference editing and
   add the source image(s).
4. Follow the parameters in the
   [Kontext inference references](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References#flux1-kontext-dev).

## Anima and Krea 2 edit modes

These workflows require the model-specific edit LoRA linked in the README:

- [Anima Edit LoRA](https://civitai.com/models/2650553/anima-edit)
- [Krea 2 Identity Edit LoRA](https://civitai.com/models/2761113/krea-2-identity-edit)

After installing the LoRA:

1. Select the matching Anima or Krea 2 checkpoint and add the required LoRA.
2. Enable **Settings → Stable Diffusion → [Anima] Enable Reference** or
   **[Krea2] Enable Reference**.
3. Use **txt2img**, add the source image through **ImageStitch Integrated**,
   and generate with an edit instruction.

These settings enable the reference/edit path and disable the corresponding
regular img2img path. Pin them to Quicksettings if switching modes frequently.

## Troubleshooting

- **The UI behaves like ordinary txt2img:** verify the checkpoint name/path
  satisfies the detection rule and that the relevant reference setting is
  enabled.
- **Klein ignores the reference:** make sure **[Klein] Disable Reference** is
  off and that the image was added to ImageStitch, not only selected in a file
  picker.
- **Qwen edit produces a warning or weak edit:** use denoising strength `0.9`
  or higher and confirm the checkpoint path contains `qwen` and `edit`.
- **Tensor mismatch or a missing VAE state dict:** remove unrelated external
  modules and select the model-specific VAE/text encoder from the download
  table. A derivative checkpoint does not automatically make its VAE
  interchangeable with the base model's VAE.
- **A module is not listed:** put it in the correct folder, refresh the model
  list, and check the matching architecture row in the download table.

## Scope and attribution

The edit/reference workflow notes and the Neo-specific behavior described here
were written by Codex for this fork during the current development work. They
are not upstream Forge documentation and may change as Neo changes. The model
files, model authors, and upstream Forge wiki remain the authoritative sources
for model licensing, downloads, and baseline inference guidance.
