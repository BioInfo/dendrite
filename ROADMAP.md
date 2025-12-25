# Dendrite Roadmap

## Overview

14-week implementation plan for production-ready agent-native inference engine.

---

## Milestone 1: Foundation (Weeks 1-2)
**Goal:** Correct reference implementation for testing

- [x] Workspace scaffold (3 crates)
- [x] KV cache with copy-on-write blocks
- [x] Tree state management with O(1) fork
- [x] Scheduler with prefill/decode separation
- [x] Attention backend trait
- [ ] Reference attention implementation (CPU)
- [ ] Property-based tests for invariants
- [ ] Fork correctness test suite

**Exit Criteria:** All invariant tests pass, fork is demonstrably O(1)

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

## Milestone 3: Model Loading (Weeks 5-6)
**Goal:** Load and run Llama-3-8B

- [ ] SafeTensors weight loading
- [ ] RoPE position embeddings
- [ ] RMSNorm implementation
- [ ] GQA attention with FlashInfer
- [ ] SwiGLU MLP
- [ ] End-to-end generation loop

**Exit Criteria:** Generate coherent text from Llama-3-8B

---

## Milestone 4: Grammar Constraints (Weeks 7-8)
**Goal:** Structured output via llguidance

- [ ] llguidance Rust bindings or FFI
- [ ] TokenMask GPU transfer
- [ ] JSON schema constraint
- [ ] Regex constraint
- [ ] CFG constraint
- [ ] Mask computation benchmarks (<50μs target)

**Exit Criteria:** Generate valid JSON from schema, mask compute <50μs

---

## Milestone 5: Tree Search (Weeks 9-10)
**Goal:** First-class ToT and MCTS support

- [ ] Tree expansion API
- [ ] Branch scoring interface
- [ ] Parallel branch generation
- [ ] Pruning and garbage collection
- [ ] MCTS example with UCT
- [ ] Beam search example

**Exit Criteria:** MCTS example solving Game of 24

---

## Milestone 6: Performance & Polish (Weeks 11-12)
**Goal:** Production-ready performance

- [ ] Continuous batching optimization
- [ ] Memory pool tuning
- [ ] Benchmark suite (fork, decode, prefill)
- [ ] Profiling and hotspot elimination
- [ ] Documentation polish
- [ ] Example gallery

**Exit Criteria:** Meet all performance targets from PRD

---

## Milestone 7: Launch (Weeks 13-14)
**Goal:** Public release and community building

- [ ] Blog post: "Why We Built Dendrite"
- [ ] Hacker News launch
- [ ] Twitter/X thread
- [ ] Discord community setup
- [ ] Issue triage and community response
- [ ] First external contributor PR

**Exit Criteria:** 1,000 GitHub stars

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

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| FlashInfer API changes | Pin to specific commit |
| llguidance Rust bindings unavailable | C FFI fallback |
| GB10 hardware access delays | Develop on consumer GPU first |
| Candle CUDA compatibility issues | Upstream contributions |

---

*Last Updated: 2025-12-25*
