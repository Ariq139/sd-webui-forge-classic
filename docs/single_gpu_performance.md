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

## Z-Image / FLUX.1 / FLUX.2 Klein follow-up

The new opt-in flags below are experiments for one RTX 3060 Ti 8 GiB, not measured speedups. Existing flags and fallback behavior are retained.

| Flag | Added path |
| --- | --- |
| `--native-token-assembly` | Z-Image no-op pad elimination, broadcast pad/position assembly, batched uniform unpatchify. |
| `--static-sigma-control` | Versioned schedule snapshots for DPM++ 2M, LCM, Euler churn and rectified-flow Euler ancestral control flow; GPU arithmetic is retained. |
| `--qwen-intermediate-only` | Native Qwen3 4B/8B stops after required hidden states; no final output is consumed by these conditioning engines. |
| `--qwen-batched-prompts` | Two equal-length, single-section unweighted prompts, only for resident native encoder with conservative VRAM headroom. |
| `--flux-fused-qk-rope` | Eligible native CUDA/Triton RMSNorm+rotary in FLUX double/single streams. |
| `--native-int8-mlp-activation` | Pinned Comfy-Kitchen 0.2.40 CUDA ConvRot activation producer; packed GELU/SiLU in native FLUX and BF16 paired activation in ZIT/YakMLP. |
| `--fused-flux-residual` | In-place gated residual multiply/add for matching dtype and nonsegmented modulation. |
| `--flux-reference-cache` | Requires invariant cache budget; versioned reference transfer/patchification and eligible resident dense image projection, plus reference-layout RoPE. |
| `--target-only-output-head` | Dense, tokenwise native FLUX reference and ZIT caption/padding head trimming. |
| `--assembled-condition-cache` | Requires invariant cache and direct full-area CFG flags; reuse eligible assembled native conditioning. |
| `--pin-quantized-weights` | Requires existing shared-memory pinning; register original full-storage packed GGUF bytes, retain registration owner, unregister before model migration. |

The CFG schedule allowlist now permits ordinary `sampling_sigmas` metadata. Uncached Nunchaku FLUX no longer reads a timestep scalar unnecessarily. Both are source-level fixes. Existing prefetch now allows a large eligible allocation to occupy the remaining byte budget; in-flight allocations still count. CPU helper tests cover layer selection conventions, schedule metadata, uniform unpatchify, snapshot mutation and reference identity/version. Full Forge integration and GPU tests remain required. Run `python -m pytest -q tests/test_additional_static_paths.py tests/test_qwen_shortcut.py tests/test_single_gpu_integration.py` in the real installation.

Deferred hardware work: load-time packed INT8 gate/up weight packing with zero duplicate permanent storage; general quantized tensor-subclass pin/stage reconstruction; single-stream segmented activation quantization and producer QKV destinations; cross-step Nunchaku packed rotary caching; full CUDA Graph replay; full assembled conditioning reconstruction and predictor coefficient precomputation. These require real layout/kernel/lifetime validation and are not represented as completed. Native ZIT already has RMS+RoPE and Nunchaku already packs gate/up; no redundant implementation. Existing outer WebUI prompt conditioning cache remains authoritative; no additional cross-job text cache. Reference projection and head trimming, fused RMS/rotary and native INT8 producer may change numerical rounding. Custom extension mutation via untracked tensor storage is not supported by snapshot reuse. Test each flag independently and record actual kernel dispatch, output tolerance, VRAM, host pinning and timing before merging.


## Follow-up review of PR #1

- `--nunchaku-rope-cache` uses the existing sampling invariant budget for native Nunchaku Z-Image packed rotary views; per-forward packing now retains source views and distinguishes layout/version instead of only pointer and shape. Exact native type, geometry and hook guards remain required.
- `--conditioning-reconstruction-cache` retains at most one native reconstructed conditioning entry, up to 32 MiB with conservative CUDA headroom; version/storage-aware schedule signatures invalidate changed sources. Inference-mode entries still depend on the native read-only schedule contract. Custom conditioning and extension paths are excluded.
- `--static-sigma-control` also reuses fixed log/logit transforms for eligible DPM++ solvers. Intermediate solver sigmas and original GPU arithmetic remain authoritative.
- Prefetch snapshots retain actual source storage, reject pointer/layout/quantized-metadata changes, and hold partial-copy owners until cleanup. Earlier INT8 data and scale registration/staging is restricted to the pinned layout. Native FLUX passes optimization-only arguments only to its unchanged implementation.
- Remaining: resident CUDA Graph replay, zero-duplicate load-time INT8 gate/up packing, segmented quantization, direct QKV output-buffer producers, additional quantized layouts beyond supported GGUF and weight-only INT8, and broader dynamic predictor coefficient reuse. They require full runtime/kernel/lifetime validation.

Focused local CPU regression suite: 43 passed, 3 skipped (two CUDA cases and one pinned dependency case). This is not the full Forge test suite. Helper/test blob hashes were checked against the tested files. CUDA model generation, actual kernel dispatch, memory transfers, performance and numerical equivalence remain unverified; keep the PR draft until those checks are complete.


## Expanded packed-weight staging

The existing opt-in `--pin-quantized-weights` and `--weight-prefetch-mib` now admit compatible Comfy-Kitchen 0.2.40 FP8 E4M3/E5M2, AWQ W4A16, ConvRot W4A4, and canonical (non-MMA-interleaved) W4A8/W6A8 layouts. Packed data, scales, zeros, and optional correction/codebook tensors move together. Exact parameter classes, shape/dtype, full independent CPU storage, version counters and metadata are checked; unknown layouts and transposed or partial-storage weights fall back. Pinning owns detached storage aliases, and all prefetched fields remain charged to the existing byte budget. The checkpoint format and normal linear dispatch are unchanged. This is a transfer optimization, not native FP8 compute on Ampere.

Focused CPU helper suite: 52 passed, 9 skipped. Nine additional storage-contract tests use simulated parameter wrappers; six actual pinned-dependency cases are skipped because Comfy-Kitchen is unavailable locally. Existing CUDA tests remain skipped. Helper/test bytes were checked against Git blobs. Full Forge integration, pinned DMA and RTX 3060 Ti timings remain unverified. Load-time INT8 gate/up coalescing, segmented activation kernels, direct shared QKV output producers, and resident CUDA Graph capture remain pending: the independent module load/unload and LoRA ownership contracts cannot safely be bypassed by retaining duplicate packed weights or using untested CUDA buffer views. MMA-interleaved grouped weights, MXFP8 and NVFP4 remain on their original transfer path.


## Packed staging review and native CUDA row reuse

The preceding packed-storage change was rechecked against Comfy-Kitchen 0.2.40 and the isolated helper suite. Exact wrapper identity now precedes attribute access; INT8 reconstruction uses captured metadata rather than rereading mutable source parameters. The supported packed layouts and original transfer fallbacks are unchanged.

The experimental `--native-shared-int8-rows` flag requires `--shared-activation-quantization`. In an audited read-only mixed-precision projection group, it reuses unrotated or eligible fused ConvRot-256 INT8 activation rows and scales with the existing pinned CUDA CUTLASS epilogue. SM86 only; 32 MiB total per group; stream, rotation mode, source storage and version separate cache entries. Backend choice, layout, dtype, alignment and native kernel eligibility are checked; small GEMV, unsupported and declined CUTLASS paths fall back. A declined kernel drops its cached rows. No new permanent weight allocation or change to named parameters/checkpoints; no custom compiled CUDA kernel. This is not load-time gate/up packing or direct shared QKV output production.

Local isolated helper suite: 57 passed, 11 skipped. New tests cover cache policy, wrapper rejection, and mocked native producer/epilogue dispatch and decline handling. Two new actual SM86 parity tests are unexecuted; six earlier pinned dependency cases and three earlier CUDA/dependency cases remain skipped. Actual CUDA kernel execution, full model integration, DMA and performance remain unverified. The new route calls version-pinned private dependency functions and must remain opt-in until hardware verification. Remaining implementation work: zero-duplicate load-time gate/up packing, segmented activation quantization, direct shared QKV output producers, resident CUDA Graph replay, and other unadmitted storage layouts.
