"""Bounded, aligned gallery aggregation for directory/upload img2img."""
from dataclasses import dataclass


_FIELDS = ("all_prompts", "all_negative_prompts", "all_seeds", "all_subseeds")


@dataclass
class GalleryEntry:
    image: object
    kind: str
    infotext: str | None
    values: tuple = ()


class BatchGallery:
    def __init__(self, limit):
        self.limit = int(limit)
        self.entries = []
        self.result = None

    def add(self, processed):
        if processed is None:
            return
        if self.result is None:
            self.result = processed
        remaining = None if self.limit < 0 else max(0, self.limit - len(self.entries))
        if remaining == 0:
            return
        first = max(0, min(len(processed.images), processed.index_of_first_image))
        ordinal = 0
        for index, image in enumerate(processed.images):
            if remaining is not None and remaining <= 0:
                return
            text = processed.infotexts[index] if index < len(processed.infotexts) else None
            kind = "grid" if index < first else "auxiliary" if getattr(image, "_forge_auxiliary", False) else "image"
            values = ()
            if kind == "image":
                values = tuple(
                    items[ordinal] if ordinal < len(items) else None
                    for items in (getattr(processed, field, None) or [] for field in _FIELDS)
                )
                ordinal += 1
            self.entries.append(GalleryEntry(image, kind, text, values))
            if remaining is not None:
                remaining -= 1
        for image in processed.extra_images or []:
            if remaining is not None and remaining <= 0:
                return
            self.entries.append(GalleryEntry(image, "auxiliary", None))
            if remaining is not None:
                remaining -= 1

    def finish(self):
        result = self.result
        if result is None:
            return None
        primary = [entry for entry in self.entries if entry.kind != "auxiliary"]
        auxiliary = [entry for entry in self.entries if entry.kind == "auxiliary"]
        ordered = primary + auxiliary
        generated = [entry for entry in primary if entry.kind == "image"]
        result.images = [entry.image for entry in primary]
        result.extra_images = [entry.image for entry in auxiliary]
        result.infotexts = [entry.infotext for entry in ordered]
        result.gallery_kinds = [entry.kind for entry in ordered]
        result.index_of_first_image = next((i for i, entry in enumerate(primary) if entry.kind != "grid"), len(primary))
        for index, field in enumerate(_FIELDS):
            setattr(result, field, [entry.values[index] for entry in generated])
        if generated:
            result.prompt, result.negative_prompt, result.seed, result.subseed = generated[0].values
            result.info = generated[0].infotext or ""
        else:
            result.info = ""
        return result
