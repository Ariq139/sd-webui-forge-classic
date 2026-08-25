import torch

from modules_forge.lora_detection import detect_lora_state_dict


def test_detects_diffusers_lora_targets_and_rank():
    report = detect_lora_state_dict(
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": torch.empty(4, 16),
            "transformer_blocks.0.attn.to_q.lora_B.weight": torch.empty(16, 4),
            "text_encoder.layers.0.mlp.lora_A.weight": torch.empty(4, 8),
        }
    )

    assert report.is_adapter
    assert report.families["LoRA"] == 3
    assert report.components["diffusion"] == 2
    assert report.components["text"] == 1
    assert report.ranks == {4}
    assert report.primary_architecture == "Flux/transformer"


def test_detects_lycoris_families():
    report = detect_lora_state_dict(
        {
            "lora_unet_blocks_0_hada_w1_a": torch.empty(4, 8),
            "lora_unet_blocks_0_lokr_w1": torch.empty(4, 8),
            "lora_unet_blocks_0_dora_scale": torch.empty(8),
        }
    )

    assert report.families["LoHa"] == 1
    assert report.families["LoKr"] == 1
    assert report.families["DoRA"] == 1


def test_rejects_non_adapter_state_dict():
    report = detect_lora_state_dict({"model.diffusion_model.input_blocks.0.weight": torch.empty(1)})
    assert not report.is_adapter
    assert "no recognized adapter tensors" in report.issues
