# Dendrite Roadmap

## Overview

14-week implementation plan for production-ready agent-native inference engine.

---

## Development Methodology

**TDD + Property-Based Testing** — The gold standard for infrastructure:

```
1. Write failing test (defines expected behavior)
2. Implement minimum code to pass
3. Refactor while keeping tests green
4. Add property tests for invariants
5. CI blocks merge on any failure
```

**Why this builds trust:**
- Contributors can safely refactor with green test suite
- Property tests find edge cases humans miss
- Benchmarks prove performance claims empirically
- Miri catches undefined behavior

**CI Pipeline (GitHub Actions):**
- `cargo check` — Compilation
- `cargo fmt --check` — Formatting
- `cargo clippy --all-targets -D warnings` — Linting (strict)
- `cargo test` — Unit + integration tests
- `cargo doc` — Documentation builds
- `cargo-deny` — Dependency audit
- MSRV check (Rust 1.75)

---

## Milestone 1: Foundation (Weeks 1-2) ✅ COMPLETE
**Goal:** Correct reference implementation for testing

- [x] Workspace scaffold (3 crates)
- [x] KV cache with copy-on-write blocks
- [x] Tree state management with O(1) fork
- [x] Scheduler with prefill/decode separation
- [x] Attention backend trait
- [x] CI pipeline (GitHub Actions)
- [x] Code quality configs (clippy.toml, rustfmt.toml, deny.toml)
- [x] Issue/PR templates, CONTRIBUTING.md
- [x] Unit tests: Block, BlockId, NodeId, TreeNode (10 tests)
- [x] Unit tests: BlockPool (12 tests)
- [x] Unit tests: BlockTable (17 tests)
- [x] Unit tests: TreeState (22 tests)
- [x] Unit tests: KvCache (8 tests)
- [x] Unit tests: Scheduler (14 tests)
- [x] Unit tests: Request/Batch (23 tests)
- [x] Unit tests: Attention (13 tests)
- [x] Property-based tests for CoW invariants (5 proptest invariants)
- [x] Fork benchmark harness (O(blocks) proven, all <50μs)
- [x] Reference attention implementation (CPU)
- [x] API documentation with examples

**Test Coverage:** 256 unit tests (includes transformer, search, radix, golden harness modules)
**Exit Criteria:** ✅ All invariant tests pass, fork is demonstrably O(1)

---

## Milestone 2: FlashInfer Integration (Weeks 3-4)
**Goal:** GPU-accelerated attention kernels

- [ ] FlashInfer FFI bindings via bindgen
- [ ] BatchDecodeWithPagedKVCacheWrapper integration
- [ ] BatchPrefillWithPagedKVCacheWrapper integration
- [ ] Cascade attention for shared prefixes
- [ ] CUDA stream management
- [ ] Kernel launch benchmarks

**Exit Criteria:** FlashInfer kernels callable from Rust, <100μs decode latency

---

## Milestone 3: Model Loading (Weeks 5-6) 🟡 IN PROGRESS
**Goal:** Load and run Llama-3-8B

- [x] SafeTensors weight loading
- [x] RoPE position embeddings
- [x] RMSNorm implementation
- [x] SwiGLU MLP
- [x] Transformer architecture with Candle
- [x] End-to-end inference tests (random weights)
- [ ] GQA attention with FlashInfer (requires GPU)
- [ ] Full Llama-3-8B weight loading

**Exit Criteria:** Generate coherent text from Llama-3-8B

---

## Milestone 4: Grammar Constraints (Weeks 7-8) 🟡 IN PROGRESS
**Goal:** Structured output via llguidance

- [x] llguidance Rust integration
- [x] GrammarConstraint with JSON/regex/CFG support
- [x] TokenMask generation from parser
- [ ] TokenMask GPU transfer (requires GPU)
- [ ] Mask computation benchmarks (<50μs target)

**Exit Criteria:** Generate valid JSON from schema, mask compute <50μs

---

## Milestone 5: Tree Search (Weeks 9-10) ✅ COMPLETE
**Goal:** First-class ToT and MCTS support

- [x] Tree expansion API (TokenExpander, UniformExpander)
- [x] Branch scoring interface (UCT, Greedy, PUCT scorers)
- [x] MCTS implementation with UCT selection
- [x] Beam search with length normalization
- [x] MCTS example with simulated environment
- [x] Beam search example with mock language model
- [ ] Parallel branch generation (future enhancement)
- [ ] Pruning and garbage collection (future enhancement)

**Exit Criteria:** ✅ MCTS and Beam Search examples running

---

## Milestone 6: FP8/MXFP8 Quantization (Weeks 11-12)
**Goal:** Memory-efficient quantized inference

- [ ] FP8 E4M3 tensor support
- [ ] MXFP8 block scaling (Blackwell-native)
- [ ] Transformer Engine FFI bindings
- [ ] FP8 forward pass implementation
- [ ] FP8 perplexity validation (within 1% of FP16)
- [ ] FP16 mask application for numerical stability
- [ ] Memory profiling benchmarks

**Exit Criteria:** FP8 inference with <1% accuracy loss, reduced memory footprint

---

## Milestone 7: Performance & Polish (Weeks 13-14)
**Goal:** Production-ready performance

- [ ] Continuous batching optimization
- [ ] Memory pool tuning
- [ ] Benchmark suite (fork, decode, prefill)
- [ ] Profiling and hotspot elimination
- [ ] Documentation polish
- [ ] Example gallery

**Exit Criteria:** Meet all performance targets from PRD

---

## Milestone 8: Launch (Weeks 15-16)
**Goal:** Public release and community building

- [ ] Blog post: "Why We Built Dendrite"
- [ ] Hacker News launch
- [ ] Twitter/X thread
- [ ] Discord community setup
- [ ] Issue triage and community response
- [ ] First external contributor PR

**Exit Criteria:** 1,000 GitHub stars

---

## Test Coverage Targets

| Module | Unit | Property | Integration | Status |
|--------|------|----------|-------------|--------|
| cache/block.rs | ✓ | - | - | Done |
| cache/block_table.rs | ✓ | - | - | Done |
| cache/pool.rs | ✓ | ✓ | - | Done |
| tree/node.rs | ✓ | - | - | Done |
| tree/state.rs | ✓ | ✓ | - | Done |
| scheduler/* | ✓ | - | - | Done |
| attention/* | ✓ | - | ✓ | Done |
| grammar/* | ✓ | - | - | Done |
| model/transformer.rs | ✓ | - | ✓ | Done |
| search/mcts.rs | ✓ | - | - | Done |
| search/beam.rs | ✓ | - | - | Done |
| search/scorer.rs | ✓ | - | - | Done |
| search/expander.rs | ✓ | - | - | Done |
| search/integrated.rs | ✓ | - | - | Done |
| cache/radix.rs | ✓ | - | - | Done |
| model/golden.rs | ✓ | - | - | Done |

**Key Invariants to Test:**
1. Refcount sum equals active references
2. No cycles in block graph
3. Free list contains only refcount=0 blocks
4. All sequences have valid block mappings

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Fork latency | < 50 μs | Pending |
| Grammar mask | < 50 μs | Pending |
| Decode latency | < 100 μs | Pending |
| Memory overhead per fork | < 5% | Pending |
| Cache utilization | > 80% | Pending |

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Rust | 1.75+ | Language |
| Candle | latest | ML framework |
| FlashInfer | 0.2.x | Attention kernels |
| CUDA | 12.x | GPU acceleration |
| llguidance | TBD | Grammar constraints |
| proptest | 1.5+ | Property-based testing |

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| FlashInfer API changes | Pin to specific commit |
| llguidance Rust bindings unavailable | C FFI fallback |
| GB10 hardware access delays | Develop on consumer GPU first |
| Candle CUDA compatibility issues | Upstream contributions |

---

*Last Updated: 2025-12-25 (M1+M5 complete, M3-4 in progress, added M6 FP8/MXFP8)*
