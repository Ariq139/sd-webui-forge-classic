import os.path

import network
import networks
from ui_edit_user_metadata import LoraUserMetadataEditor

from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js


LORA_CATEGORY_LABELS = {
    "sd": "SD",
    "xl": "SDXL",
    "flux": "Flux",
    "klein": "Klein",
    "qwen": "Qwen",
    "lumina": "Lumina",
    "zit": "ZIT",
    "wan": "Wan",
    "anima": "Anima",
    "ernie": "Ernie",
    "pid": "PiD",
    "krea": "Krea",
    "unknown": "Unknown",
}


def infer_lora_category(metadata, name):
    """Infer a display category from common LoRA metadata fields."""
    explicit = metadata.get("sd version")
    if explicit in network.SD_VERSION and explicit != "Unknown":
        return LORA_CATEGORY_LABELS.get(explicit, explicit.upper())

    values = [name]
    for key in ("ss_base_model_version", "modelspec.architecture", "ss_sd_model_name", "modelspec.title", "ss_network_module"):
        value = metadata.get(key)
        if value is not None:
            values.append(str(value))
    text = " ".join(values).casefold()

    # Check specific/newer model families before broad SD/Flux matches.
    patterns = (
        ("anima", "anima"),
        ("flux.2", "klein"),
        ("flux2", "klein"),
        ("klein", "klein"),
        ("qwen", "qwen"),
        ("lumina", "lumina"),
        ("z-image", "zit"),
        ("zit", "zit"),
        ("wan", "wan"),
        ("ernie", "ernie"),
        ("pid", "pid"),
        ("krea", "krea"),
        ("sdxl", "xl"),
        ("stable-diffusion-xl", "xl"),
        ("illustrious", "xl"),
        ("pony", "xl"),
        ("flux", "flux"),
        ("sd1", "sd"),
        ("sd15", "sd"),
        ("sd 1.5", "sd"),
        ("stable-diffusion-v1", "sd"),
    )
    for marker, category in patterns:
        if marker in text:
            return LORA_CATEGORY_LABELS[category]

    return LORA_CATEGORY_LABELS["unknown"]


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__("Lora")
        self.allow_negative_prompt = True
        self.supports_category_grouping = True

    def refresh(self):
        networks.list_available_networks()

    def create_item(self, name, index=None, enable_filter=True):
        lora_on_disk = networks.available_networks.get(name)
        if lora_on_disk is None:
            return

        path = os.path.splitext(lora_on_disk.filename)[0]

        alias = lora_on_disk.get_alias()

        search_terms = [self.search_terms_from_path(lora_on_disk.filename)]
        if lora_on_disk.hash:
            search_terms.append(lora_on_disk.hash)

        item: dict[str, str | dict] = {
            "name": name,
            "filename": lora_on_disk.filename,
            "shorthash": lora_on_disk.shorthash,
            "preview": self.find_preview(path) or self.find_embedded_preview(path, name, lora_on_disk.metadata),
            "description": self.find_description(path),
            "search_terms": search_terms,
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": lora_on_disk.metadata,
            "sort_keys": {"default": index, **self.get_sort_keys(lora_on_disk.filename)},
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        # Keep the global default weight configurable, but guard against an
        # incomplete client-side options object during UI startup. Without
        # this fallback the card inserts an `undefined` LoRA weight.
        default_weight = "typeof opts !== 'undefined' && opts.extra_networks_default_multiplier != null ? String(Number(opts.extra_networks_default_multiplier) === 1 ? '1.0' : opts.extra_networks_default_multiplier) : '1.0'"
        weight = str(preferred_weight) if preferred_weight else f"({default_weight})"
        item["prompt"] = quote_js(f"<lora:{alias}:") + " + " + weight + " + " + quote_js(">")

        if activation_text:
            item["prompt"] += " + " + quote_js(" " + activation_text)

        negative_prompt = item["user_metadata"].get("negative text", "")
        item["negative_prompt"] = quote_js(negative_prompt)

        sd_version: str = item["user_metadata"].get("sd version", None)
        if sd_version in network.SD_VERSION:
            item["sd_version"] = sd_version
        else:
            sd_version = "Unknown"

        item["category"] = infer_lora_category(item["user_metadata"], name)

        if enable_filter and shared.opts.lora_preset_filter and sd_version not in ("Unknown", shared.opts.forge_preset):
            return None

        return item

    def list_items(self):
        names = list(networks.available_networks)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir, *shared.cmd_opts.lora_dirs]

    def get_category_order(self, prioritize=True):
        all_categories = [LORA_CATEGORY_LABELS[key] for key in ("sd", "xl", "flux", "klein", "qwen", "lumina", "zit", "wan", "anima", "ernie", "pid", "krea", "unknown")]
        if not prioritize:
            return all_categories

        related_categories = {
            "sd": ("SD",),
            "xl": ("SDXL",),
            "flux": ("Flux", "Klein"),
            "klein": ("Klein", "Flux"),
            "qwen": ("Qwen",),
            "lumina": ("Lumina",),
            "zit": ("ZIT",),
            "wan": ("Wan",),
            "anima": ("Anima",),
            "ernie": ("Ernie",),
            "pid": ("PiD",),
            "krea": ("Krea",),
        }
        preset = str(getattr(shared.opts, "forge_preset", "")).casefold()
        preferred = list(related_categories.get(preset, ()))
        if "Unknown" in all_categories:
            preferred.append("Unknown")
        return preferred + [category for category in all_categories if category not in preferred]

    def get_category_open_categories(self):
        """Expand only model-family groups related to the active UI preset."""
        related_categories = {
            "sd": {"SD"},
            "xl": {"SDXL"},
            "flux": {"Flux", "Klein"},
            "klein": {"Flux", "Klein"},
            "qwen": {"Qwen"},
            "lumina": {"Lumina"},
            "zit": {"ZIT"},
            "wan": {"Wan"},
            "anima": {"Anima"},
            "ernie": {"Ernie"},
            "pid": {"PiD"},
            "krea": {"Krea"},
        }
        preset = str(getattr(shared.opts, "forge_preset", "")).casefold()
        return {"Unknown", *related_categories.get(preset, set())}

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)
