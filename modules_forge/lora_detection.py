"""Cheap, model-independent LoRA format detection.

This module intentionally inspects keys and tensor shapes only.  It is used
before the Comfy-compatible mapper so a bad or differently-targeted adapter
can be diagnosed without doing a model clone or a GPU operation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping


_ADAPTER_MARKERS = (
    "lora_",
    ".lora_",
    "lokr_",
    ".lokr_",
    "loha_",
    ".loha_",
    "hada_",
    ".hada_",
    "oft_",
    ".oft_",
    "boft_",
    ".boft_",
    "dora_scale",
)


@dataclass(frozen=True)
class LoRAKeyInfo:
    key: str
    family: str
    component: str
    role: str


@dataclass
class LoRADetection:
    total_keys: int = 0
    adapter_keys: list[LoRAKeyInfo] = field(default_factory=list)
    ignored_keys: list[str] = field(default_factory=list)
    families: Counter = field(default_factory=Counter)
    components: Counter = field(default_factory=Counter)
    architectures: Counter = field(default_factory=Counter)
    ranks: set[int] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)

    @property
    def is_adapter(self) -> bool:
        return bool(self.adapter_keys)

    @property
    def primary_family(self) -> str:
        return self.families.most_common(1)[0][0] if self.families else "unknown"

    @property
    def primary_architecture(self) -> str:
        return self.architectures.most_common(1)[0][0] if self.architectures else "unknown"

    def summary(self) -> str:
        families = ", ".join(f"{name}={count}" for name, count in self.families.most_common()) or "none"
        components = ", ".join(f"{name}={count}" for name, count in self.components.most_common()) or "unknown"
        architecture = self.primary_architecture
        ranks = ",".join(str(x) for x in sorted(self.ranks)) or "?"
        return f"family={families}; component={components}; architecture={architecture}; ranks={ranks}"


def classify_lora_key(key: str) -> LoRAKeyInfo | None:
    """Classify common A1111, Diffusers, LyCORIS, and Comfy key styles."""
    lower = key.lower()
    has_diff = lower.endswith(".diff") or ".diff." in lower or ".diff_b" in lower
    if not has_diff and not any(marker in lower for marker in _ADAPTER_MARKERS):
        return None

    if any(token in lower for token in ("lokr_", ".lokr_")):
        family = "LoKr"
    elif any(token in lower for token in ("loha_", ".loha_", "hada_", ".hada_")):
        family = "LoHa"
    elif any(token in lower for token in ("oft_", ".oft_", "boft_", ".boft_")):
        family = "OFT/BOFT"
    elif "dora_scale" in lower:
        family = "DoRA"
    elif ".diff" in lower or lower.endswith(".diff") or ".diff_b" in lower:
        family = "diff"
    else:
        family = "LoRA"

    if any(token in lower for token in ("text_encoder", "text_encoders", "clip", "t5", "lora_te_", "lora_clip")):
        component = "text"
    elif any(token in lower for token in ("unet", "diffusion_model", "transformer", "lora_unet_", "transformer_blocks")):
        component = "diffusion"
    else:
        component = "unknown"

    role = "other"
    if any(token in lower for token in ("lora_a", "lora_down", "hada_w1_a", "hada_w2_a", "lokr_w1_a")):
        role = "down"
    elif any(token in lower for token in ("lora_b", "lora_up", "hada_w1_b", "hada_w2_b", "lokr_w1_b")):
        role = "up"
    elif "alpha" in lower or "dora_scale" in lower:
        role = "metadata"

    return LoRAKeyInfo(key, family, component, role)


def detect_lora_state_dict(state_dict: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> LoRADetection:
    result = LoRADetection(total_keys=len(state_dict))
    for key, value in state_dict.items():
        info = classify_lora_key(str(key))
        if info is None:
            result.ignored_keys.append(str(key))
            continue
        result.adapter_keys.append(info)
        result.families[info.family] += 1
        result.components[info.component] += 1

        shape = getattr(value, "shape", ())
        if info.role == "down" and len(shape) >= 2:
            result.ranks.add(int(shape[0] if len(shape) == 2 else shape[0]))

        lower = info.key.lower()
        if "flux" in lower or "transformer_blocks" in lower or "single_transformer_blocks" in lower:
            result.architectures["Flux/transformer"] += 1
        elif "qwen" in lower or "img_mlp" in lower or "txt_mlp" in lower:
            result.architectures["Qwen/transformer"] += 1
        elif "lora_unet_" in lower or "input_blocks" in lower or "middle_block" in lower:
            result.architectures["Stable Diffusion"] += 1
        else:
            result.architectures["generic"] += 1

    if not result.is_adapter:
        result.issues.append("no recognized adapter tensors")
    if result.components.get("unknown", 0) and not result.components.get("diffusion", 0) and not result.components.get("text", 0):
        result.issues.append("target component cannot be inferred from keys")
    if metadata:
        network_module = str(metadata.get("ss_network_module", "")).lower()
        if "lokr" in network_module and not result.families.get("LoKr"):
            result.issues.append("metadata says LoKr but no LoKr keys were found")
        if "loha" in network_module and not result.families.get("LoHa"):
            result.issues.append("metadata says LoHa but no LoHa keys were found")
    return result
