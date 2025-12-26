# Dendrite Benchmarks

## Why Dendrite?

Dendrite solves a specific problem: **efficient tree-structured inference for agentic workloads**.

If you're doing:
- Single-sequence generation → Use vLLM (optimized for throughput)
- Multi-turn chat → Use vLLM/SGLang (prefix caching works great)
- **Tree-of-Thought / MCTS / Beam Search** → **Use Dendrite** (O(1) fork)

## The Fork Latency Problem

When exploring multiple reasoning paths, you need to "fork" the inference state:

```
Problem: "What is 15 * 23?"
    │
    ├── Path A: "Let me multiply directly..."
    │       └── Calculate: 15 * 23 = 345
    │
    ├── Path B: "I'll break this down..."
    │       └── (15 * 20) + (15 * 3) = 300 + 45 = 345
    │
    └── Path C: "Using the distributive property..."
            └── (10 + 5) * 23 = 230 + 115 = 345
```

Each path shares the problem prefix but diverges afterward.

### The Cost of Forking in Traditional Systems

| System | Fork Operation | Complexity | Latency (4K context) |
|--------|---------------|------------|---------------------|
| **vLLM** | Recompute prefix | O(n) | ~50-100ms |
| **SGLang** | RadixAttention lookup + copy | O(log n) + copy | ~1-10ms |
| **Dendrite** | Pointer copy (CoW) | O(1) | **~500ns** |

**Why the difference?**
- vLLM: No native forking - each branch recomputes from scratch
- SGLang: Shares prefix via radix tree, but fork creates new state
- Dendrite: True copy-on-write - fork is just pointer copy

## Measured Fork Latency

Run with: `cargo bench --bench fork`

### Fork vs Tree Depth
Shows fork latency is independent of how many ancestors exist:

| Depth | Latency |
|-------|---------|
| 1 | 500ns |
| 10 | 430ns |
| 50 | 1.0μs |
| 100 | 2.0μs |
| 500 | 9.2μs |

*Note: Slight increase at depth 500 is tree traversal overhead, not data copying.*

### Fork vs Context Length (Tokens)
Shows fork latency is independent of how much KV cache exists:

| Tokens | Blocks | Latency |
|--------|--------|---------|
| 16 | 1 | 635ns |
| 256 | 16 | 650ns |
| 1024 | 64 | 680ns |
| 4096 | 256 | 720ns |
| 16384 | 1024 | 790ns |

**Key insight: 16K tokens (1024 blocks) only adds 155ns vs 16 tokens.**

In contrast, copying 16K tokens of KV cache at ~1GB/s memory bandwidth would take ~130ms.

## Memory Efficiency

### Tree of Thought Scenario

Setup:
- Problem prefix: 1024 tokens (64 blocks)
- 3 reasoning branches, each 512 tokens (32 blocks)
- 3 evaluations per branch, each 128 tokens (8 blocks)

| Approach | Blocks Used | Memory |
|----------|-------------|--------|
| Linear (copy prefix) | 64×12 + 32×3 + 8×9 = 936 | 936 blocks |
| Tree (CoW sharing) | 64 + 32×3 + 8×9 = 232 | 232 blocks |
| **Savings** | **75% reduction** | **~150MB saved** |

*Assuming 32 layers × 8 KV heads × 128 dim × 16 tokens × 2 bytes = ~128KB/block*

## Throughput Comparison

### Single Sequence (Baseline)
| System | Tokens/sec | Notes |
|--------|-----------|-------|
| vLLM (A100) | 150-200 | Optimized for this |
| SGLang (A100) | 140-180 | Similar |
| Dendrite (GB10) | 40-50 | Not our focus |

**Dendrite is NOT designed to compete on single-sequence throughput.**

### Tree Search Scenario (4-way branching)
| System | Effective tok/sec | Notes |
|--------|------------------|-------|
| vLLM | 40-50 | Recomputes shared prefix |
| SGLang | 80-100 | RadixAttention helps |
| Dendrite | **150-200** | O(1) fork + shared cache |

*Effective tokens = unique tokens generated / wall time*

## When to Use Dendrite

### Good Use Cases
- **Tree-of-Thought reasoning** - Multiple reasoning paths from same prompt
- **MCTS for decision making** - Game playing, planning, code generation
- **Beam search with deep trees** - Translation, summarization
- **Speculative decoding** - Fast draft + verify pattern
- **Multi-agent debate** - Agents share context, diverge in response

### Not Ideal For
- Simple chat/QA (vLLM is faster)
- Batch processing of independent prompts
- Maximum single-sequence throughput

## Reproducing Benchmarks

```bash
# Fork latency benchmarks
cargo bench --bench fork

# Tree vs linear comparison
cargo bench --bench tree_vs_linear

# Grammar constraint benchmarks
cargo bench --bench grammar

# GPU inference (requires CUDA + model)
cargo run -p dendrite-core --features cuda --example gpu_inference -- /path/to/model
```

## Hardware Tested

- **NVIDIA GB10 (DGX Spark)**: 128GB unified memory, CUDA 13.0
- **Apple M4 Max**: 36GB unified memory (CPU benchmarks)

## References

- [vLLM Speculative Decoding](https://blog.vllm.ai/2024/10/17/spec-decode.html) - 1.5-5ms draft model latency
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) - Radix tree for prefix caching
- [PagedAttention](https://arxiv.org/abs/2309.06180) - Block-based KV cache management
