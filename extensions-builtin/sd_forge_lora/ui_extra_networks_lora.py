import os.path
import re

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


LORA_CATEGORY_MARKERS = {
    "anima": ("anima",),
    "klein": ("flux 2", "flux2", "klein"),
    "qwen": ("qwen",),
    "lumina": ("lumina",),
    "zit": ("z image", "zimage", "zit"),
    "wan": ("wan",),
    "ernie": ("ernie",),
    "pid": ("pid",),
    "krea": ("krea",),
    "xl": ("sdxl", "sd xl", "stable diffusion xl", "illustrious", "pony"),
    "flux": ("flux", "flux 1", "flux1"),
    "sd": ("sd 1 5", "sd1 5", "sd15", "sd v1", "sd v2", "sd1", "sd2", "stable diffusion v1", "stable diffusion v2", "stable diffusion 1"),
}

LORA_MODULE_MARKERS = {
    "anima": ("lora anima",),
    "klein": ("lora klein", "lora flux 2", "lora flux2"),
    "qwen": ("lora qwen",),
    "lumina": ("lora lumina",),
    "zit": ("lora zit", "lora z image", "lora zimage"),
    "wan": ("lora wan",),
    "ernie": ("lora ernie",),
    "pid": ("lora pid",),
    "krea": ("lora krea",),
    "xl": ("lora sdxl", "lora sd xl"),
    "flux": ("lora flux",),
}

LORA_RELATED_CATEGORIES = {
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


def _active_preset():
    preset = getattr(shared.opts, "forge_preset", "")
    return str(getattr(preset, "name", preset)).rsplit(".", 1)[-1].casefold()


def _normalized_text(values):
    text = " ".join(str(value) for value in values if value is not None).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _category_from_markers(text, markers):
    for category, category_markers in markers.items():
        if any(re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", text) for marker in category_markers):
            return LORA_CATEGORY_LABELS[category]
    return None


def infer_lora_category(metadata, name):
    """Infer a display category from metadata."""
    metadata = metadata or {}

    explicit = str(metadata.get("sd version") or "").strip().casefold()
    for version in network.SD_VERSION:
        if explicit == version.casefold() and version != "Unknown":
            return LORA_CATEGORY_LABELS.get(version, version.upper())

    module_text = _normalized_text([metadata.get("ss_network_module")])
    if category := _category_from_markers(module_text, LORA_MODULE_MARKERS):
        return category

    source_values = [
        metadata.get("ss_base_model_version"),
        metadata.get("modelspec.architecture"),
        metadata.get("ss_sd_model_name"),
        metadata.get("modelspec.title"),
    ]
    if category := _category_from_markers(_normalized_text(source_values), LORA_CATEGORY_MARKERS):
        return category

    if category := _category_from_markers(_normalized_text([name]), LORA_CATEGORY_MARKERS):
        return category

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

        category_metadata = {**(item.get("metadata") or {}), **(item.get("user_metadata") or {})}
        item["category"] = infer_lora_category(category_metadata, name)

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
        all_categories = [
            LORA_CATEGORY_LABELS[key]
            for key in ("sd", "xl", "flux", "klein", "qwen", "lumina", "zit", "wan", "anima", "ernie", "pid", "krea", "unknown")
        ]
        if not prioritize:
            return all_categories

        preset = _active_preset()
        preferred = list(LORA_RELATED_CATEGORIES.get(preset, ()))
        if "Unknown" in all_categories:
            preferred.append("Unknown")
        return preferred + [category for category in all_categories if category not in preferred]

    def get_category_open_categories(self):
        """Expand only model-family groups related to the active UI preset."""
        preset = _active_preset()
        return {"Unknown", *LORA_RELATED_CATEGORIES.get(preset, ())}

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)
