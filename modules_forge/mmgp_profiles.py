"""Named MMGP-style presets for Forge's optional memory controls."""


MEMORY_SETTING_KEYS = (
    "forge_memory_management_enabled",
    "forge_memory_budgets_enabled",
    "forge_memory_pinned_memory_enabled",
    "forge_memory_async_transfers_enabled",
    "forge_memory_residency_hints_enabled",
    "forge_memory_working_vram_mb",
    "forge_memory_unet_budget_mb",
    "forge_memory_text_encoder_budget_mb",
    "forge_memory_vae_budget_mb",
    "forge_memory_controlnet_budget_mb",
    "forge_memory_pinned_memory_percent",
    "forge_memory_vram_safety_percent",
    "forge_memory_async_streams",
    "forge_memory_pinned_components",
    "forge_memory_keep_unet_loaded",
    "forge_memory_keep_text_encoder_loaded",
    "forge_memory_keep_vae_loaded",
    "forge_memory_keep_controlnet_loaded",
)

# MMGP's minimum hardware targets; these are labels, not hard caps.
# Windows generally needs about 16 GB more system RAM than Linux.
MEMORY_PROFILE_REQUIREMENTS = {
    "HighRAM_HighVRAM": (48, 24),
    "HighRAM_LowVRAM": (48, 12),
    "LowRAM_HighVRAM": (32, 24),
    "LowRAM_LowVRAM": (32, 12),
    "VerylowRAM_LowVRAM": (24, 10),
}


def memory_profile_label(name):
    requirements = MEMORY_PROFILE_REQUIREMENTS.get(name)
    if requirements is None:
        return name

    ram_gb, vram_gb = requirements
    return f"{name} ({ram_gb} GB RAM / {vram_gb} GB VRAM)"


# Display hardware requirements while storing the stable profile key.
MEMORY_PROFILE_CHOICES = (
    "Custom",
    *((memory_profile_label(name), name) for name in MEMORY_PROFILE_REQUIREMENTS),
)

# Forge-compatible translations of MMGP's five profiles.
MEMORY_PROFILES = {
    "HighRAM_HighVRAM": {
        "forge_memory_management_enabled": True,
        "forge_memory_budgets_enabled": False,
        "forge_memory_pinned_memory_enabled": True,
        "forge_memory_async_transfers_enabled": True,
        "forge_memory_residency_hints_enabled": True,
        "forge_memory_working_vram_mb": 0,
        "forge_memory_unet_budget_mb": 0,
        "forge_memory_text_encoder_budget_mb": 0,
        "forge_memory_vae_budget_mb": 0,
        "forge_memory_controlnet_budget_mb": 0,
        "forge_memory_pinned_memory_percent": 45,
        "forge_memory_vram_safety_percent": 80,
        "forge_memory_async_streams": 2,
        "forge_memory_pinned_components": ["unet", "text_encoder", "vae", "controlnet"],
        "forge_memory_keep_unet_loaded": True,
        "forge_memory_keep_text_encoder_loaded": True,
        "forge_memory_keep_vae_loaded": True,
        "forge_memory_keep_controlnet_loaded": True,
    },
    "HighRAM_LowVRAM": {
        "forge_memory_management_enabled": True,
        "forge_memory_budgets_enabled": True,
        "forge_memory_pinned_memory_enabled": True,
        "forge_memory_async_transfers_enabled": True,
        "forge_memory_residency_hints_enabled": True,
        "forge_memory_working_vram_mb": 1024,
        "forge_memory_unet_budget_mb": 6144,
        "forge_memory_text_encoder_budget_mb": 4096,
        "forge_memory_vae_budget_mb": 2048,
        "forge_memory_controlnet_budget_mb": 2048,
        "forge_memory_pinned_memory_percent": 45,
        "forge_memory_vram_safety_percent": 80,
        "forge_memory_async_streams": 2,
        "forge_memory_pinned_components": ["unet", "text_encoder", "vae", "controlnet"],
        "forge_memory_keep_unet_loaded": False,
        "forge_memory_keep_text_encoder_loaded": True,
        "forge_memory_keep_vae_loaded": True,
        "forge_memory_keep_controlnet_loaded": True,
    },
    "LowRAM_HighVRAM": {
        "forge_memory_management_enabled": True,
        "forge_memory_budgets_enabled": False,
        "forge_memory_pinned_memory_enabled": False,
        "forge_memory_async_transfers_enabled": False,
        "forge_memory_residency_hints_enabled": True,
        "forge_memory_working_vram_mb": 0,
        "forge_memory_unet_budget_mb": 0,
        "forge_memory_text_encoder_budget_mb": 0,
        "forge_memory_vae_budget_mb": 0,
        "forge_memory_controlnet_budget_mb": 0,
        "forge_memory_pinned_memory_percent": 45,
        "forge_memory_vram_safety_percent": 80,
        "forge_memory_async_streams": 2,
        "forge_memory_pinned_components": ["unet"],
        "forge_memory_keep_unet_loaded": True,
        "forge_memory_keep_text_encoder_loaded": True,
        "forge_memory_keep_vae_loaded": True,
        "forge_memory_keep_controlnet_loaded": True,
    },
    "LowRAM_LowVRAM": {
        "forge_memory_management_enabled": True,
        "forge_memory_budgets_enabled": True,
        "forge_memory_pinned_memory_enabled": False,
        "forge_memory_async_transfers_enabled": False,
        "forge_memory_residency_hints_enabled": False,
        "forge_memory_working_vram_mb": 2048,
        "forge_memory_unet_budget_mb": 4096,
        "forge_memory_text_encoder_budget_mb": 2048,
        "forge_memory_vae_budget_mb": 1024,
        "forge_memory_controlnet_budget_mb": 2048,
        "forge_memory_pinned_memory_percent": 45,
        "forge_memory_vram_safety_percent": 80,
        "forge_memory_async_streams": 2,
        "forge_memory_pinned_components": ["unet"],
        "forge_memory_keep_unet_loaded": False,
        "forge_memory_keep_text_encoder_loaded": False,
        "forge_memory_keep_vae_loaded": False,
        "forge_memory_keep_controlnet_loaded": False,
    },
    "VerylowRAM_LowVRAM": {
        "forge_memory_management_enabled": True,
        "forge_memory_budgets_enabled": True,
        "forge_memory_pinned_memory_enabled": False,
        "forge_memory_async_transfers_enabled": False,
        "forge_memory_residency_hints_enabled": False,
        "forge_memory_working_vram_mb": 3072,
        "forge_memory_unet_budget_mb": 3072,
        "forge_memory_text_encoder_budget_mb": 1024,
        "forge_memory_vae_budget_mb": 512,
        "forge_memory_controlnet_budget_mb": 1024,
        "forge_memory_pinned_memory_percent": 30,
        "forge_memory_vram_safety_percent": 80,
        "forge_memory_async_streams": 1,
        "forge_memory_pinned_components": [],
        "forge_memory_keep_unet_loaded": False,
        "forge_memory_keep_text_encoder_loaded": False,
        "forge_memory_keep_vae_loaded": False,
        "forge_memory_keep_controlnet_loaded": False,
    },
}


def get_memory_profile(name):
    """Return a copy so UI edits cannot mutate the shared preset."""

    values = MEMORY_PROFILES.get(name)
    return dict(values) if values is not None else None
