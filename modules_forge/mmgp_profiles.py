"""Named MMGP-style presets for Forge's optional memory controls."""


MEMORY_SETTING_KEYS = (
    "forge_memory_management_enabled",
    "forge_memory_attention_backend",
    "forge_memory_vae_attention_backend",
    "forge_memory_alternate_quantization",
    "forge_memory_quantization_type",
    "forge_memory_quantization",
    "forge_memory_compile_enabled",
    "forge_memory_partial_pinning_enabled",
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
    "forge_memory_residency_components",
)

MMGP_QUANTIZATION_MODES = ("disabled", "qint8", "qint4", "qfloat8")
MMGP_COMPONENT_CHOICES = ("unet", "text_encoder", "vae", "controlnet")
MMGP_LOW_RAM_PROFILES = {"LowRAM_HighVRAM", "LowRAM_LowVRAM", "VerylowRAM_LowVRAM"}
_LEGACY_RESIDENCY_OPTIONS = (
    ("unet", "forge_memory_keep_unet_loaded"),
    ("text_encoder", "forge_memory_keep_text_encoder_loaded"),
    ("vae", "forge_memory_keep_vae_loaded"),
    ("controlnet", "forge_memory_keep_controlnet_loaded"),
)

MMGP_ATTENTION_BACKEND_CHOICES = (
    "automatic",
    "sdpa",
    "ck",
    "sage",
    "sage2",
    "sage3",
    "flash",
    "flash3",
    "radial",
    "xformers",
)

MMGP_VAE_ATTENTION_BACKEND_CHOICES = (
    "automatic",
    "sdpa",
    "xformers",
    "slice",
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
        "forge_memory_attention_backend": "automatic",
        "forge_memory_vae_attention_backend": "automatic",
        "forge_memory_alternate_quantization": True,
        "forge_memory_quantization_type": "qint8",
        "forge_memory_compile_enabled": False,
        "forge_memory_partial_pinning_enabled": False,
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
        "forge_memory_attention_backend": "automatic",
        "forge_memory_vae_attention_backend": "automatic",
        "forge_memory_alternate_quantization": True,
        "forge_memory_quantization_type": "qint8",
        "forge_memory_compile_enabled": False,
        "forge_memory_partial_pinning_enabled": False,
        "forge_memory_budgets_enabled": True,
        "forge_memory_pinned_memory_enabled": True,
        "forge_memory_async_transfers_enabled": True,
        "forge_memory_residency_hints_enabled": True,
        "forge_memory_working_vram_mb": 0,
        "forge_memory_unet_budget_mb": 1200,
        "forge_memory_text_encoder_budget_mb": 3000,
        "forge_memory_vae_budget_mb": 3000,
        "forge_memory_controlnet_budget_mb": 3000,
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
        "forge_memory_attention_backend": "automatic",
        "forge_memory_vae_attention_backend": "automatic",
        "forge_memory_alternate_quantization": True,
        "forge_memory_quantization_type": "qint8",
        "forge_memory_compile_enabled": False,
        "forge_memory_partial_pinning_enabled": False,
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
        "forge_memory_pinned_components": ["unet"],
        "forge_memory_keep_unet_loaded": True,
        "forge_memory_keep_text_encoder_loaded": True,
        "forge_memory_keep_vae_loaded": True,
        "forge_memory_keep_controlnet_loaded": True,
    },
    "LowRAM_LowVRAM": {
        "forge_memory_management_enabled": True,
        "forge_memory_attention_backend": "automatic",
        "forge_memory_vae_attention_backend": "automatic",
        "forge_memory_alternate_quantization": True,
        "forge_memory_quantization_type": "qint8",
        "forge_memory_compile_enabled": False,
        "forge_memory_partial_pinning_enabled": False,
        "forge_memory_budgets_enabled": True,
        "forge_memory_pinned_memory_enabled": True,
        "forge_memory_async_transfers_enabled": True,
        "forge_memory_residency_hints_enabled": False,
        "forge_memory_working_vram_mb": 0,
        "forge_memory_unet_budget_mb": 1200,
        "forge_memory_text_encoder_budget_mb": 3000,
        "forge_memory_vae_budget_mb": 3000,
        "forge_memory_controlnet_budget_mb": 3000,
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
        "forge_memory_attention_backend": "automatic",
        "forge_memory_vae_attention_backend": "automatic",
        "forge_memory_alternate_quantization": True,
        "forge_memory_quantization_type": "qint8",
        "forge_memory_compile_enabled": False,
        "forge_memory_partial_pinning_enabled": False,
        "forge_memory_budgets_enabled": True,
        "forge_memory_pinned_memory_enabled": False,
        "forge_memory_async_transfers_enabled": True,
        "forge_memory_residency_hints_enabled": False,
        "forge_memory_working_vram_mb": 0,
        "forge_memory_unet_budget_mb": 400,
        "forge_memory_text_encoder_budget_mb": 3000,
        "forge_memory_vae_budget_mb": 3000,
        "forge_memory_controlnet_budget_mb": 3000,
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
    if values is None:
        return None

    values = dict(values)
    values["forge_memory_quantization"] = (
        values["forge_memory_quantization_type"]
        if values.get("forge_memory_alternate_quantization", False)
        else "disabled"
    )
    values["forge_memory_residency_components"] = [
        component for component, option in _LEGACY_RESIDENCY_OPTIONS if values.get(option, False)
    ]
    return values


def get_mmgp_quantization(options):
    """Return the merged quantization toggle and type, with legacy fallback."""

    data = getattr(options, "data", {}) or {}
    mode = data.get("forge_memory_quantization")
    if mode in MMGP_QUANTIZATION_MODES:
        return mode != "disabled", ("qint8" if mode == "disabled" else mode)

    return (
        bool(getattr(options, "forge_memory_alternate_quantization", False)),
        str(getattr(options, "forge_memory_quantization_type", "qint8")),
    )


def extra_text_encoder_quantization_enabled(options) -> bool:
    """Match MMGP's extra text-encoder quantization scope for named profiles."""
    quantize, _quantization_type = get_mmgp_quantization(options)
    if not quantize:
        return False

    profile = str(getattr(options, "forge_memory_profile", "Custom") or "Custom")
    return profile == "Custom" or profile in MMGP_LOW_RAM_PROFILES


def get_mmgp_residency_components(options):
    """Return the merged residency selection, with legacy fallback."""

    data = getattr(options, "data", {}) or {}
    if "forge_memory_residency_components" in data:
        selected = set(data.get("forge_memory_residency_components") or ())
        return [component for component in MMGP_COMPONENT_CHOICES if component in selected]

    return [component for component, option in _LEGACY_RESIDENCY_OPTIONS if getattr(options, option, False)]
