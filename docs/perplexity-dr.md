# Dendrite PRD Refinement: Agent-Native Inference Engine

**Author**: Perplexity  
**Date**: 2025-12-25  
**Status**: Research Complete

## 1. Problem & Positioning

### 1.1 Problem Dendrite Solves

Most serving engines (vLLM, TGI, TensorRT-LLM) are optimized for **multi-tenant throughput**: many independent requests, continuous batching, and high GPU utilization.

Agentic workloads (Tree-of-Thought, MCTS, self-refinement) have a fundamentally different profile:

- A **single agent** or small number of agents
- Highly **branching** reasoning with many partial trajectories sharing long prefixes
- Repeated **re-prefill** of the same context for each branch when using standard engines
- Tight **latency budgets** for interactive use; throughput is secondary

**PRD Core Statement**:

> "Existing LLM engines optimize for multi-tenant throughput; Dendrite optimizes for single-agent branching latency by representing KV cache as a tree of paged blocks with copy-on-write branching and explicit tree-aware scheduling."

For agentic workflows, prefilling dominates cost; eliminating redundant prefills via prefix sharing and O(1) branching yields large end-to-end gains even if raw tokens/sec is similar.

## 2. Core Design Principles

### 2.1 Correctness-First

Dendrite must be **trustworthy infrastructure** first, optimization engine second:

- **Golden-token determinism** vs. a pinned reference implementation (HF transformers, exact model revision, tokenizer)
- **Numerical tolerances** that acknowledge mixed precision and FP8 behavior
- **Memory-safety invariants** on KV paging and copy-on-write

**PRD Implication**: Treat **correctness harness** as a first-class deliverable, not a later test pass. No performance claims without defined benchmark suite backing them.

### 2.2 Agent-Optimized State Model

The core conceptual model:

- **KV cache = tree of blocks** (pages) with fixed-size blocks (8–32 tokens) allocated from a block allocator
- **Branches = paths** in that tree where forks share prefix blocks by reference
- **Copy-on-Write**: Divergence triggers CoW at block granularity
- **Scheduler = tree walker**: Knows branch relationships, never recomputes shared prefixes

**PRD Implication**: Define **KV tree** and **branch** as explicit domain objects. The scheduler API should expose branch identities and relationships, not just anonymous "requests".

## 3. Tree-Aware KV Cache: Requirements

### 3.1 Paged KV Structure

Borrowing from vLLM's PagedAttention and SGLang's RadixAttention:

- KV stored in **fixed-size blocks** (pages) rather than per-sequence contiguous slabs
- **Block table** maps `(sequence_id, block_index)` to block descriptor (pointer, dtype, length)
- Blocks can be **shared across sequences** via reference counting

**PRD Requirements**:

- Block size configurable (8/16/32 tokens) with default tuned for latency vs. fragmentation
- Block allocator supports O(1) allocate/free and high utilization (>80%)
- Block-table operations: O(1) lookup and efficient iteration

### 3.2 Fork & Copy-on-Write Semantics

For a branch fork:

- **Fork operation**: Creates new logical sequence whose block table initially shares all blocks with parent (increases refcounts), cost O(number of block table entries), **no block copies**
- **Copy-on-write**: When branch diverges and writes into shared block (refcount > 1), block is copied and refcount decremented
- **Invariants**: No write to block with refcount > 1 without first copying; no refcount-0 blocks remain in allocator

**PRD Requirements**:

- Fork API is O(1) in memory and negligible CPU time relative to forward pass
- CoW happens **only when needed** and at block granularity
- Refcount semantics testable: Fork → increment, Drop → decrement + potential free
- Tree depth/branching factor constraints explicit (tested up to depth 5, factor 8 for V0.1)

### 3.3 Layout Correctness Guarantees

Because paged KV and CoW are subtle sources of bugs (off-by-one, stale pointers):

- **Per-block canaries** (sentinel values) at block boundaries to detect overwrites
- Optional **checksums** for integrity checks in debug mode
- Clear **failure behavior**: In debug builds panic with diagnostic dump; in release attempt recovery

**PRD Requirements**:

- "KV integrity mode" toggle (e.g., `DENDRITE_KV_DEBUG`) enabling canaries + periodic verification
- Logging for every allocate/fork/CoW/free when debug mode is enabled

## 4. Scheduling Model for Branching Workloads

### 4.1 Differences from Continuous Batching

vLLM's continuous batching operates over largely **independent sequences** trying to maximize GPU utilization. For Dendrite:

- **Dominant sharing** within single agent's tree
- Branches share large prefixes but diverge at different depths
- Scheduler must ensure **no redundant prefill** and balance exploration/exploitation

**PRD Requirements**:

- Represent active state as a **tree of sequences**, not flat set of requests
- Support configurable search strategies: BFS across levels or priority-based (frontier with heuristic scores)
- Fit into simple mental model: agents pass tree-shaped structure (or frontier list with parent references), Dendrite schedules respecting those relations

### 4.2 First-Release Scheduler Scope

**In-scope**:
- Single-agent or small fixed number of agents (<8) sharing one model instance
- BFS or "frontier queue" style scheduling
- No preemption beyond iteration-level; minimal fairness concerns

**Out-of-scope**:
- Multi-tenant, unbounded request admission
- Sophisticated QoS or SLAs
- Complex integration with external cluster schedulers

## 5. Grammar & Schema-Constrained Decoding

### 5.1 Why Inside the Sampler

Post-processing output to check JSON/schema validity is both **wasteful** and **unsafe**:

- Violating tokens already consumed GPU compute
- Structured tasks often need **guaranteed valid outputs** (tool calls, database queries)

State-of-the-art uses **token masking**: invalid tokens at each step get probability 0 (logit = −∞).

**PRD Requirements**:

- Constrained decoding is part of **core sampler**, not optional layer
- Support at least: JSON-schema constraints and EBNF/regex patterns
- API to: compile constraints into FSM, maintain FSM state per branch, mask logits before sampling

### 5.2 Performance Expectations

FSM-based constrained decoding with compressed automata can add **<10–20% latency** vs. unconstrained while dramatically reducing invalid outputs.

**PRD Requirements**:

- Set clear performance budget: ≤15% overhead for JSON outputs of typical length (256 tokens)
- Provide explicit "constraints-on vs. constraints-off" benchmark

## 6. Hardware: Grace–Blackwell (DGX Spark) Reality

### 6.1 What GB10 Actually Gives

- **Unified memory**: ~128GB LPDDR5x shared between Grace CPU and Blackwell GPU
- **Bandwidth**: ~273 GB/s (lower than H100 HBM but higher capacity)
- **Compute**: Blackwell with FP8/FP4 support and sparse acceleration
- **CPU**: 20 ARM Neoverse cores (cut-down Grace)

Implications:

- Dendrite's value: **huge model + context trees** in unified memory
- Main bottleneck: **memory bandwidth**, reinforcing need for paged KV and FP8 KV compression

### 6.2 "Do Now" vs. "Do Later"

**Do Now (V0.1)**:
- Paged KV on GPU; block allocator aware of unified memory
- CPU orchestrator for agent logic + scheduler with minimal overhead
- FP16/BF16 for weights, FP16/FP32 accum, optional FP8 KV
- GB10 as primary test target; also H100/A100 for validation on discrete GPUs

**Do Later (V0.2+)**:
- FP4 paths (need more stability data)
- Multi-node/multi-device scaling

**PRD Refinement**:
- Explicitly list **GB10 unified-memory assumption** for first iteration
- Mark FP8 KV support as **optional but supported**, FP4 as **experimental/future**
- State **multi-GPU and distributed** are out-of-scope for V0.1

## 7. Correctness Harness: PRD-Level Commitments

### 7.1 Golden Token Suite

- Use HF transformers on CPU (float32) as reference
- Pin model IDs, revisions (commit hashes), tokenizer versions
- For each: deterministic greedy decoding on fixed prompts
- Compare first N tokens (e.g., 128) exactly

Dendrite's KV tree system, grammar masking, FP8 KV, and schedulers must all pass in "correctness-first" config.

### 7.2 Numerical Tolerance Policy

Define **logit-based tolerance** suite:
- Dense vs. paged KV (same dtype)
- BF16 vs. FP8 KV for same model

Acceptable error norms:
- L2(logits) < 1e−3 for BF16
- L2(logits) < 5e−2 for FP8 KV

### 7.3 Structural Invariants

Treat these as explicit, testable invariants:

- No KV block written while shared (refcount > 1) without CoW
- Every forked branch either holds reference to block or explicitly released it (no dangling references)
- Tree structure acyclic with single root per agent session

Include "Debug Invariants" mode where every KV-modifying operation runs additional checks; violations trigger immediate abort with detailed dump.

## 8. Benchmark Suite: PRD-Level Metrics

Benchmarks are **part of the PRD**, with clearly defined goals.

### 8.1 Core Benchmarks

| Benchmark | Purpose | Acceptance Criteria |
|-----------|---------|-------------------|
| Decode throughput/latency | Show overhead relative to existing systems acceptable | <5% slower than vLLM baseline |
| Fork overhead microbench | Time to fork at different depths | <10µs per fork (O(1) validation) |
| Tree vs. linear ToT | Fixed problem set (GSM8K subset) baseline vs Dendrite | ≥2× speedup on representative ToT |
| Constraint overhead | Unconstrained vs JSON-schema-constrained | ≤15% latency overhead |

**Release criteria**: "We consider V0.1 shippable only if Tree-of-Thought benchmark shows at least 2× speedup over vLLM baseline on representative ToT workload and constraint overhead is ≤15%."

### 8.2 Reporting Format

For each benchmark:
- Hardware (GB10 vs H100/A100), model, sequence length, tree parameters
- Confidence intervals (multiple runs)
- Release notes including benchmark tables

## 9. Feature Scope: In / Out for V0.1

### 9.1 In-Scope (V0.1)

- Rust-native library embedded in agent code (no network server required)
- Single-GPU, single-host; optimized for unified memory (GB10), works on H100/A100
- Paged KV cache with CoW fork and tree representation
- Tree-aware iteration-level scheduler for one or few agents
- Grammar-constrained decoding via FSM token masking
- Optional FP8 KV cache with correctness gates
- Correctness suite and benchmarks integrated into CI

### 9.2 Deferred (V0.2 or Later)

- Speculative decoding (draft/verify)
- FP4 quantization and sparse acceleration
- Multi-GPU or multi-node distribution
- CPU-side draft models for structured generation
- Advanced constraint languages

### 9.3 Explicitly Out of Scope

- Serving as generic, high-throughput multi-tenant API server
- Model training or fine-tuning
- On-device mobile inference

## 10. API & Developer Experience

### 10.1 Minimal Core API

```rust
Engine::new(ModelConfig, EngineConfig)
Session::new(&Engine)
Session::prefill(prompt)
Session::fork(node_id) -> node_id
Session::decode_step(node_id, constraints?) -> token
Session::run_tree(strategy, constraints?)
```

### 10.2 Observability Hooks

API for grabbing:
- KV tree statistics (blocks, refcounts, pages per branch)
- Per-branch latency & token counts
- Constraint state (FSM states branches in)

Debug flags to log:
- Fork operations and CoW events
- Branch creation and pruning
- Scheduler decisions

## 11. Recommended PRD Structure

1. **Overview & Goals**
   - Problem statement (agentic latency vs. throughput serving)
   - Target users (agent frameworks, research labs)

2. **Non-Goals**
   - Multi-tenant API serving, training, etc.

3. **Architecture Overview**
   - KV paged tree model
   - Scheduler design
   - Grammar enforcement
   - Hardware assumptions

4. **Core Requirements**
   - Correctness & determinism
   - KV data structures & invariants
   - Tree scheduling behavior
   - Constrained decoding guarantees

5. **Performance Targets**
   - Fork cost
   - Tree-vs-linear speedup
   - Constraint overhead

6. **Hardware Targets**
   - Primary: GB10; secondary: H100/A100

7. **Milestones**
   - V0.1: single-GPU core + correctness + benchmarks
   - V0.2+: speculative decoding, FP4, Python bindings, multi-GPU

8. **Risks & Unknowns**
   - FFI overhead, FP8 stability, real-world tree shapes

9. **Key References**
   - vLLM, SGLang, FlashInfer papers and systems
   - Benchmark definitions

---

This refinement captures the essence of academic literature and system landscape while keeping scope realistic and tightly aligned with the 1000-star goal. It transforms vague aspirations into concrete, measurable requirements.
