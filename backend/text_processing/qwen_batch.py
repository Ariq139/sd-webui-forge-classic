"""Exact prompt grouping for the two audited, unweighted Qwen engines."""
import numbers

import torch

from backend import memory_management
from backend.single_gpu_fastpaths import single_gpu_flag
from backend.sampling.invariant_cache import cache_dependencies_safe


def _resident_encoder(encoder):
    from backend.nn.llm.llama import Qwen3_4B, Qwen3_8B
    model = encoder.transformer
    if (type(model) not in (Qwen3_4B, Qwen3_8B) or torch.is_grad_enabled()
            or not cache_dependencies_safe(model)
            or encoder._forward_hooks or encoder._forward_pre_hooks):
        return None
    device = model.get_input_embeddings().weight.device
    if device.type != "cuda" or not all(p.device == device for p in model.parameters()):
        return None
    return device


def _single_unweighted_section(chunks):
    if len(chunks) != 1 or not chunks[0]:
        return None
    if any(not isinstance(token, numbers.Integral) or weight != 1.0 for token, weight in chunks[0]):
        return None
    return [int(token) for token, _ in chunks[0]]


def encode_prompt_pairs(encoder, tokenizer, formatted_lines, format_condition):
    """Return per-prompt outputs; None means retain the ordinary sequential path."""
    if not single_gpu_flag("qwen_batched_prompts"):
        return None
    unique = list(dict.fromkeys(formatted_lines))
    if len(unique) < 2:
        return None
    device = _resident_encoder(encoder)
    if device is None:
        return None
    chunks = {line: tokenizer.tokenize_with_weights(line, disable_weights=True) for line in unique}
    tokens = {line: _single_unweighted_section(value) for line, value in chunks.items()}
    groups = {}
    for line, row in tokens.items():
        if row is not None:
            groups.setdefault(len(row), []).append(line)
    cache = {}
    for length, group in groups.items():
        for i in range(0, len(group) - 1, 2):
            a, b = group[i:i + 2]
            # Conservative scratch allowance; never evict weights to batch.
            hidden = encoder.transformer.model.config.hidden_size
            intermediate = encoder.transformer.model.config.intermediate_size
            required = 512 * 1024**2 + 2 * length * (hidden * 12 + intermediate * 4) * 4 + 2 * length * length * 4
            if torch.cuda.mem_get_info(device)[0] < required:
                continue
            result = encoder([tokens[a], tokens[b]])[0]
            if result.shape[0] != 2:
                raise RuntimeError("Unexpected native Qwen prompt batch output")
            for line, row in zip((a, b), result):
                cache[line] = format_condition(row.unsqueeze(0).to(device=memory_management.intermediate_device()))
    for line in unique:
        if line not in cache:
            cache[line] = format_condition(encoder.encode_token_weights(chunks[line])[0])
    return [row for line in formatted_lines for row in cache[line]]
