# Dendrite

**Agent-native inference engine with O(1) fork latency for tree-structured reasoning.**

Dendrite is a specialized LLM inference engine designed for agentic workloads that require exploring multiple reasoning paths simultaneously. Unlike traditional inference engines that optimize for single-sequence throughput, Dendrite provides constant-time forking of inference state, enabling efficient tree-of-thought, MCTS, and beam search algorithms.

## Key Features

- **O(1) Fork Latency**: Create reasoning branches without copying the KV cache using copy-on-write semantics
- **Tree-Structured KV Cache**: Share memory across branches with reference counting
- **PagedAttention**: Memory-efficient KV cache management with 16-token blocks
- **FlashInfer Integration**: High-performance attention kernels with cascade support
- **Grammar Constraints**: Structured output via llguidance integration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         dendrite                                │
│                    (High-level API)                             │
├─────────────────────────────────────────────────────────────────┤
│                      dendrite-core                              │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌─────────────────┐ │
│  │  Cache  │  │   Tree   │  │ Scheduler │  │     Model       │ │
│  │  (CoW)  │  │  State   │  │ (Batch)   │  │  (Transformer)  │ │
│  └─────────┘  └──────────┘  └───────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      dendrite-ffi                               │
│              (FlashInfer + CUDA bindings)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use dendrite::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create engine
    let engine = Engine::builder()
        .model_path("model.safetensors")
        .max_seq_len(8192)
        .build()
        .await?;

    // Generate with tree search
    let result = engine
        .generate("Let's solve this step by step:")
        .max_tokens(256)
        .with_tree_search(TreeSearchConfig {
            num_branches: 3,
            max_depth: 5,
            scoring: ScoringMethod::LogProb,
        })
        .execute()
        .await?;

    println!("{}", result.text);
    Ok(())
}
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `dendrite` | High-level API and engine |
| `dendrite-core` | Core engine: KV cache, scheduler, tree management |
| `dendrite-ffi` | FFI bindings for FlashInfer and CUDA |

## Requirements

- Rust 1.83+ (edition 2024)
- CUDA 12.x (optional, for GPU acceleration)
- FlashInfer 0.2.x (optional, for optimized kernels)

## Building

```bash
# CPU-only build
cargo build --release

# With CUDA support
cargo build --release --features cuda

# Full build with all features
cargo build --release --features full
```

## Running Examples

```bash
# Tree of Thought example
cargo run --example tree_of_thought

# JSON output with grammar constraints
cargo run --example json_output
```

## Benchmarks

```bash
# Fork latency benchmark
cargo bench --bench fork

# Tree search benchmark
cargo bench --bench tree_search

# Memory usage benchmark
cargo bench --bench memory
```

## Target Hardware

Dendrite is optimized for:
- **NVIDIA GB10/DGX Spark**: 128GB unified memory, NVLink-C2C
- **Grace Hopper**: Unified memory architecture
- Works on any CUDA-capable GPU with reduced memory

## Performance Targets

| Metric | Target |
|--------|--------|
| Fork latency | < 50 μs |
| Grammar mask | < 50 μs |
| Decode throughput | > 100 tok/s (single) |
| Memory overhead | < 5% per fork |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for attention kernels
- [vLLM](https://github.com/vllm-project/vllm) for PagedAttention inspiration
- [llguidance](https://github.com/microsoft/llguidance) for grammar constraints
- [Candle](https://github.com/huggingface/candle) for Rust ML framework
