# Dendrite

**Agent-native inference engine with O(1) fork latency for tree-structured reasoning.**

Dendrite is a specialized LLM inference engine designed for agentic workloads that require exploring multiple reasoning paths simultaneously. Unlike traditional inference engines that optimize for single-sequence throughput, Dendrite provides constant-time forking of inference state, enabling efficient tree-of-thought, MCTS, and beam search algorithms.

## Why Dendrite? (vs vLLM/SGLang)

**TL;DR:** If your agent explores multiple reasoning paths, Dendrite is 1000-10000x faster at branching.

| Scenario | vLLM | SGLang | Dendrite |
|----------|------|--------|----------|
| Fork 4K context | 50-100ms | 5-10ms | **3μs** |
| 6-branch exploration | 300-600ms | 30-60ms | **18μs** |
| MCTS (500 forks) | 25-50s | 2.5-5s | **1.5ms** |
| Memory (6 branches, 4K prefix) | 6GB | ~2GB | **1.1GB** |

**Use Dendrite for:** Tree-of-Thought, MCTS, Beam Search, Speculative Decoding, Multi-Agent
**Use vLLM for:** Single-sequence generation, simple chat (higher throughput)

See [BENCHMARKS.md](BENCHMARKS.md) and [PRACTICAL_IMPACT.md](PRACTICAL_IMPACT.md) for details.

## Key Features

- **O(1) Fork Latency**: Create reasoning branches without copying the KV cache using copy-on-write semantics
- **Tree-Structured KV Cache**: Share memory across branches with reference counting
- **PagedAttention**: Memory-efficient KV cache management with 16-token blocks
- **MCTS & Beam Search**: Built-in tree search algorithms with UCT scoring
- **FlashInfer Integration**: High-performance attention kernels with cascade support
- **Grammar Constraints**: Structured output via llguidance integration

## How It Works

```
Traditional (vLLM/SGLang):               Dendrite (Copy-on-Write):

Fork = Copy entire KV cache              Fork = Copy block table pointers
       O(context_length)                        O(num_blocks) ≈ O(1)

┌─────────────────────┐                  ┌─────────────────────┐
│ Parent: 4K tokens   │                  │ Block Table (Parent)│
│ [================]  │                  │ [0][1][2][3]        │
└─────────────────────┘                  └──│──│──│──│────────┘
         │ fork                             │  │  │  │
         ▼                                  ▼  ▼  ▼  ▼
┌─────────────────────┐                  ┌──────────────────────┐
│ Child: Copy 4K      │                  │ Physical Blocks      │
│ [================]  │  ← 50-100ms      │ [B0][B1][B2][B3]     │
└─────────────────────┘                  │ ref=2 (shared!)      │
                                         └──────────────────────┘
                                                    ▲
                                         ┌──│──│──│──│────────┐
                                         │ [0][1][2][3]        │
                                         │ Block Table (Child) │ ← 500ns
                                         └─────────────────────┘
```

Memory is duplicated **only when branches diverge** and **only at block granularity** (16 tokens).

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| KV Cache (CoW) | ✅ Complete | O(1) fork with reference counting |
| Paged KV Cache | ✅ Complete | Memory-efficient page pool |
| Tree State | ✅ Complete | Radix tree for prefix sharing |
| Scheduler | ✅ Complete | Fair batch scheduling |
| Attention Backend | ✅ Complete | Reference + Flash Attention |
| Transformer | ✅ Complete | Full LLaMA architecture |
| Tokenizer | ✅ Complete | HuggingFace tokenizers |
| MCTS Search | ✅ Complete | UCT scoring |
| Beam Search | ✅ Complete | Top-k beams |
| Grammar Constraints | ✅ Complete | llguidance integration |
| GPU Inference | ✅ Complete | candle-flash-attn |
| FP8 Quantization | 🔄 In Progress | E4M3/E5M2/MXFP8 |
| FlashInfer FFI | 🔄 In Progress | Paged attention kernels |

**272 tests passing** | **40.8 tok/s** on NVIDIA GB10 (DGX Spark) with TinyLlama-1.1B

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
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │        Search           │  │        Grammar              │  │
│  │  (MCTS, Beam, UCT)      │  │  (llguidance constraints)   │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      dendrite-ffi                               │
│              (FlashInfer + CUDA bindings)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Text Generation

```rust
use dendrite_core::model::{ModelConfig, Tokenizer, Transformer};
use dendrite_core::attention::ReferenceBackend;
use candle_core::Device;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let model_path = std::path::Path::new("/path/to/tinyllama");
    let config = ModelConfig::from_file(&model_path.join("config.json"))?;
    let tokenizer = Tokenizer::from_dir(model_path)?;

    let backend = Arc::new(ReferenceBackend::new());
    let mut transformer = Transformer::new(config, backend, Device::Cpu)?;
    transformer.load_weights(model_path)?;

    // Generate text
    let text = transformer
        .generate_text(&tokenizer, "Hello, my name is", 50, 0.0)
        .await?;

    println!("{}", text);
    Ok(())
}
```

### GPU Inference with KV Cache

```rust
use dendrite_core::model::{ModelConfig, Transformer, KvCache};
use dendrite_core::attention::FlashAttnBackend;  // requires "cuda" feature
use candle_core::{Device, Tensor};

// Load model on GPU
let device = Device::new_cuda(0)?;
let backend = Arc::new(FlashAttnBackend::new(0)?);
let mut transformer = Transformer::new(config, backend, device.clone())?;
transformer.load_weights(model_path)?;

// Create KV cache for efficient autoregressive generation
let mut cache = transformer.create_cache();

// Prefill prompt
let prompt_tokens = vec![1u32, 15043, 29892];  // "<s>Hello,"
let input = Tensor::from_slice(&prompt_tokens, (1, 3), &device)?;
let logits = transformer.forward_with_cache(&input, &mut cache).await?;

// Decode with cached KV (O(1) per token)
let next_token = transformer.sample(&logits, 0.0)?;
println!("Generated token: {}", next_token);
```

### O(1) Fork for Tree Search

```rust
use dendrite_core::cache::{PagedKvCache, DEFAULT_PAGE_SIZE};
use candle_core::DType;

// Create paged cache with O(1) fork support
let cache = PagedKvCache::new(
    32,                    // num_layers
    1000,                  // initial pages
    DEFAULT_PAGE_SIZE,     // 16 tokens/page
    8,                     // num_kv_heads
    128,                   // head_dim
    DType::F16,
    device,
)?;

// Allocate parent sequence
let parent = cache.allocate_sequence();

// O(1) fork - shares pages via copy-on-write
let child1 = cache.fork_sequence(parent)?;
let child2 = cache.fork_sequence(parent)?;

// Each child can diverge independently
// Pages are copied only when modified
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
# GPU inference with TinyLlama (requires model weights)
cargo run -p dendrite-core --features cuda --example gpu_inference -- /path/to/tinyllama

# Text generation with tokenizer integration
cargo run -p dendrite-core --example text_inference -- /path/to/tinyllama

# Golden token validation (verify vs HuggingFace)
python scripts/generate_golden.py --model /path/to/tinyllama --output golden_cases.json
cargo run -p dendrite-core --example golden_validation -- /path/to/tinyllama golden_cases.json

# MCTS search with simulated environment
cargo run -p dendrite-core --example mcts_search

# Beam search with mock language model
cargo run -p dendrite-core --example beam_search
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

## Performance (Measured)

| Metric | Target | Actual |
|--------|--------|--------|
| Fork latency | < 50 μs | **~500ns** (100x better) |
| Grammar mask | < 50 μs | **~1.6μs** (30x better) |
| Decode latency | < 10 ms | **10ms** on GB10 |
| Memory overhead | < 5% per fork | **~0.1%** (CoW) |

See [docs/architecture.md](docs/architecture.md) for detailed design.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for attention kernels
- [vLLM](https://github.com/vllm-project/vllm) for PagedAttention inspiration
- [llguidance](https://github.com/microsoft/llguidance) for grammar constraints
- [Candle](https://github.com/huggingface/candle) for Rust ML framework
