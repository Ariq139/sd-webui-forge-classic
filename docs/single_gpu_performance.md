# Experimental single-GPU performance batch

Scope: native Anima, FLUX.1 / FLUX.2 Klein, Z-Image / Lumina, Krea. Intended for a single RTX 3060 Ti (8 GiB); no multi-GPU scheduling changes. The flags are disabled by default (except the pre-existing opt-in activation-space LoRA flag, whose supported paths are extended). Not a measured GPU speedup or a merge-ready performance guarantee.

| Launch flag | Behavior |
| --- | --- |
| `--invariant-cache-mib 32` | Sampling-scoped Anima position/cross-KV, FLUX text projection and RoPE, Z-Image text/refiner/RoPE, converted online LoRA adapters. Conservative admission and revision/group/version guards. |
| `--activation-space-lora` | Eligible online LoRA adapters preserve packed GGUF base execution; fallback computes the base once plus the activation delta. |
| `--shared-activation-quantization` | Reuse dynamic INT8 row quantization for compatible unrotated weights on the pinned comfy-kitchen 0.2.40 eager backend. Native and Triton kernels keep their existing path. |
| `--weight-prefetch-mib 64` | Stage pinned ordinary weights or original packed GGUF bytes one block ahead on an independent stream, with event readiness and a bounded two-block window. |
| `--fused-denoiser-pointwise` | Optional torch.compile for pure modulation, compatible resident LayerNorm/RMSNorm+modulation, and gated MLP expressions. Eager fallback for compiler rejection. Krea's existing fused RMSNorm is retained. |
| `--native-lumina-gqa` | Avoid explicit K/V expansion where unmasked PyTorch flash SDPA supports native GQA; otherwise retain expansion. |
| `--direct-full-area-cfg` | Avoid unit-weight full-image accumulators for one cond/uncond pair and redundant standard CFG1 condition copies on audited native model paths. |
| `--vae-direct-pixels` | Full native VAE float32 pixels without the latent-dtype round trip. Slight rounding differences are intentional. |
| `--fast-image-postprocess` | Convert CHW float pixels directly to contiguous HWC uint8, avoiding an image-sized floating scratch buffer. |

### Validation and limitations

Run from the full Forge checkout in its normal Python environment:

```bash
python -m pytest -q tests/test_invariant_cache.py tests/test_cache_batch_review.py tests/test_single_gpu_helpers.py tests/test_single_gpu_integration.py tests/test_shared_int8_rows.py tests/test_quant_execution_review.py tests/test_image_output_review.py tests/test_cfg_allocations.py tests/test_performance_optimizations.py tests/test_gguf_fastpath.py tests/test_krea_cache.py tests/test_sampling_condition_cache.py tests/test_vae_allocation_paths.py tests/test_vae_device_routing.py
```

The development environment was CPU-only and did not have a complete Forge checkout. The final isolated CPU review reran the invariant cache, INT8 rowwise arithmetic, compiler fallback and image-output tests: 26 passed and 2 skipped. Exact helper-file blob hashes were verified. Earlier batches separately tested prefetch event accounting with mocked CUDA streams. The full application integration tests, real dependency equivalence, CUDA tests, and RTX 3060 Ti benchmarks remain unexecuted. The preview latent remains a detached clone because interruption recovery and asynchronous preview consumers require an immutable snapshot; it is now separately profiled. Mixed-precision tensor-subclass offload weights that are not pinned ordinary parameters or original packed GGUF bytes retain the existing transfer path. Unknown wrappers, masks, patched model paths, unsupported layouts and compiler failures are not evidence of fast-path coverage. Generation-scoped prompt caches exclude online weight wrappers; converted standard adapters can still use the separate bounded cast cache. Nunchaku-specific kernels and non-native VAE paths are not claimed to be accelerated.

Start by enabling one flag at a time on one visible GPU (`--gpu-device-id 0` for the correct physical device). Compare identical prompt, seed, sampler, resolution, steps, batch and precision, warm up the compiler separately, then record at least five warm generations. Record denoise time, end-to-end time, peak allocated/reserved VRAM, H2D transfer waits, per-step output differences, OOM/interrupt recovery, and whether the intended profiling regions actually execute. Reject a flag that does not help the measured workload or materially changes output beyond the intended numerical tolerance. Keep the draft unmerged until the RTX 3060 Ti validation is completed.

Final audit notes: native CFG condition reuse checks the diffusion module (not the KModel wrapper's training flag) and rejects custom fields and hooks. Direct pixels require the original native VAE and decode methods and exclude interrupted/skipped output. New invariant caching requires an exact supported native model class; inherited Nunchaku Z-Image models do not enter native block prefetch. All new optimizations remain experimental. The original sampling path remains authoritative for unsupported layouts, regions, masks, custom conditions, decoder overrides and extension hooks. Do not interpret these guards as a claim of complete runtime coverage or an observed speedup. Separate profiling of the retained preview snapshot does not accelerate it. Packed GGUF prefetch requires actually pinned original bytes, and ordinary mixed-precision tensor-subclass storage remains excluded.
