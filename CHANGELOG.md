# Changelog

All notable changes to Dendrite are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — v0.1.0

### Added — Milestone 8: Launch Prep (2026-03-25)

- **`PrefixCache`** (`cache/prefix_cache.rs`) — cascade attention block reuse via `RadixTree` + `BlockPool`
  - `commit()` pins completed-sequence blocks; `lookup()` returns `PrefixHit` with matched token count
  - `evict_lru(n)` / `evict_older_than(duration)` — LRU and TTL eviction back to pool
  - `PrefixCacheStats` — hit_rate, tokens_saved, eviction counts
  - Impact: 4096-token shared system prompt across 64 requests → 262K fewer KV ops per step
- **`CompressedKvCache`** (`model/kv_cache.rs`) — drop-in replacement for `KvCache` with transparent compression
  - `CompressedLayerCache` / `CompressedKvCache` — same `append(k, v) → (k, v)` API
  - Pluggable via `KvCompressor` trait (TurboQuant, PolarQuant, Identity)
- **`TurboQuant` KV compression** (`cache/compress.rs`)
  - `PolarQuantCompressor` — L2-normalize + fixed angular grid (2/4/8-bit), no per-block scale constants
  - `QjlCompressor` — deterministic Rademacher projection (seeded), store sign bits, zero storage overhead
  - `TurboQuantCompressor` — two-stage: PolarQuant primary + QJL residual
  - Measured: **3.05x compression vs FP16** at head_dim=128 (4-bit + 64-dim QJL)
  - `KvCompressor` trait + `IdentityCompressor` baseline
- Launch documentation (`docs/launch/`)
  - `blog-post.md` — technical deep-dive for the Dendrite launch
  - `hn-post.md` — Hacker News "Show HN" draft
  - `twitter-thread.md` — 9-tweet thread

### Added — Milestone 7: Continuous Batching (2026-03-25)

- **`ContinuousBatcher`** (`scheduler/`) — Orca algorithm implementation
  - Mixed decode + chunked prefill steps per iteration
  - Per-request state machine: `Waiting → Running → Done`
- **Criterion benchmark suite** (`benches/scheduler.rs`)
  - `decode_throughput`, `prefill_latency`, `mixed_step_overhead`, `add_request` groups
- **`BlockPool` batch allocation** (`cache/pool.rs`)
  - `allocate_batch(n)` / `free_batch(ids)` — atomic batch ops (3.2x faster than N×allocate)
  - `PoolStats` with `watermark_used`, `watermark_free`
  - `with_headroom()` constructor — reserves 2% for CoW emergency
- **Example gallery** (`examples/`)
  - `continuous_batching.rs` — invariant verification, ~7µs/step
  - `memory_pool.rs` — 3.2x batch speedup demo, CoW headroom

### Added — Milestone 6: FP8 Quantization (2026-03-25)

- **`Fp8Attention`** and **`Fp8SwiGluMlp`** (`model/fp8_layer.rs`)
  - MXFP8-quantized attention and FFN layers
  - Pure-Rust dequant-on-forward; <5% MAE vs FP16 baseline
- **TE FFI stub** (`dendrite-ffi/src/te_ffi.rs`)
  - `is_te_available()`, `te_fp8_gemm()` interface defined; wires to real NVIDIA Transformer Engine when CUDA available
- **`quantize_weights()`** — batch FP8 quantization for weight matrices

### Added — Milestone 4: Grammar-Constrained Decoding (2026-03-23)

- **`ConstrainedDecoder`** (`grammar/`) — structured output via llguidance
  - Token-level mask application during beam search
  - JSON / regex / CFG constraint support
- **`TokenMask`** — GPU-compatible token validity mask
  - `from_llg_bytes()` constructor for llguidance integration
- **Tokenizer bridge** (`grammar/tokenizer_bridge.rs`)
  - `apply_llg_mask()` — applies llguidance token mask to logit slice
  - Supports HuggingFace tokenizer format

### Added — Milestone 5: Tree Search + Paged KV (2025-12-26)

- **`RadixTree`** (`cache/radix.rs`) — O(L) prefix lookup for KV block sharing
  - `find_prefix()`, `find_exact()`, `remove_by_value()` (eviction support)
  - `RadixTreeStats` — utilization metrics
- **`PagedKvCache`** (`cache/paged.rs`) — vLLM-style paged attention
  - `PagePool` with `PageTable` per sequence
  - `slice_set` for token-level KV writes
- **Beam/tree search** (`search/`, `tree/`)
  - `BeamSearch` with configurable width and length penalty
  - `TreeState` — fork/merge token trees with O(1) CoW
  - Golden harness for determinism testing

### Added — Milestone 1-3: Core Infrastructure (2025-12-25)

- **Transformer model** (`model/`)
  - `TransformerLayer`, `TransformerModel` with RoPE, RMSNorm, SwiGLU
  - `KvCache` / `LayerCache` — grow-on-append per-layer cache
  - FP8 linear layer (`quantization/fp8_linear.rs`)
- **Reference attention backend** (`attention/backend.rs`)
  - Scaled dot-product attention with causal masking
  - Flash attention FFI interface (GPU path)
- **`BlockPool`** (`cache/pool.rs`) — pre-allocated block manager with CoW
  - `Block` + `BlockId`, reference-counted
  - `BlockTable` — per-sequence logical→physical mapping
- **`Scheduler`** (`scheduler/`) — request lifecycle management
  - Waiting/running/done state machine
  - Token budget enforcement
- **Project infrastructure**
  - Cargo workspace: `dendrite`, `dendrite-core`, `dendrite-ffi`
  - CI, benchmarks, property tests (proptest)
  - `ROADMAP.md` with 14-milestone plan

---

## Stats (as of 2026-03-25)

| Metric | Value |
|--------|-------|
| Tests | **359** passing |
| Crates | 3 (`dendrite`, `dendrite-core`, `dendrite-ffi`) |
| Milestones complete | M1–M7 ✅ |
| Active milestone | M8 (Launch) |
| Compression | 3x vs FP16 (TurboQuant, CPU) |
| Benchmark | ~7µs/step continuous batching (CPU) |
