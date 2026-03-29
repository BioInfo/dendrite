# Why I Built Dendrite (And When You Shouldn't Use It)

**For Run Data Run / Substack**

---

I spent the last few months building an LLM inference engine from scratch. In Rust. For a problem most people don't know they have.

This is either a great use of time or a spectacular waste of it. I'm still not sure which.

Here's the problem: when an AI agent explores multiple reasoning paths (think Tree-of-Thought, MCTS, or beam search), it needs to "fork" its context. Try option A. Backtrack. Try option B. The academic literature shows this kind of branching search produces dramatically better results on complex reasoning tasks. Yao et al. found 74% accuracy on the Game of 24 with tree search versus 4% with chain-of-thought. That's not a typo.

But here's what nobody tells you: forking is expensive. Really expensive.

In vLLM, the standard inference engine, forking a 4K token context takes 50-100ms. You're essentially recomputing the entire KV cache from scratch. In SGLang, which has smarter prefix caching, it's still 5-10ms. Run 500 forks for an MCTS search and you've burned 25-50 seconds just on branching overhead. At that point, tree search stops being practical for anything interactive.

I kept hitting this wall while building agents. The reasoning patterns that worked best in research papers became too slow in practice. So I did what any reasonable person would do: I ignored the problem for six months and hoped someone else would fix it.

Nobody did.

## The Insight

The fix, once I understood it, felt obvious. Existing engines treat forking as creating a new sequence that happens to share some prefix. They discover sharing. Dendrite treats forking as a first-class primitive. We don't discover sharing. We create it.

The KV cache in Dendrite is a tree of paged blocks with copy-on-write semantics. When you fork, we don't copy the KV cache. We copy pointers. Increment some reference counts. Done. Cost: about 500 nanoseconds. Not milliseconds. Nanoseconds.

Memory duplication happens only when branches actually diverge, and only at block granularity (16 tokens). Fork a 4K token context six ways? You're not using 6x the memory. You're using maybe 1.1x, because all six branches share the prefix until they diverge.

Here's what that looks like in code:

```rust
// Allocate parent sequence with 4K context
let parent = cache.allocate_sequence();

// O(1) fork - just pointer copy + refcount increment
let child1 = cache.fork_sequence(parent)?;
let child2 = cache.fork_sequence(parent)?;

// Each child can now diverge independently
// Pages copied only when modified
```

## The Numbers

I hate benchmarks that cherry-pick scenarios. So here's the honest comparison:

| Scenario | vLLM | SGLang | Dendrite |
|----------|------|--------|----------|
| Fork 4K context | 50-100ms | 5-10ms | 3μs |
| 6-branch exploration | 300-600ms | 30-60ms | 18μs |
| MCTS (500 forks) | 25-50s | 2.5-5s | 1.5ms |
| Memory (6 branches, 4K prefix) | 6GB | ~2GB | 1.1GB |

The fork latency improvement is roughly 1000-10000x depending on the baseline. That sounds like marketing nonsense, but it's just math. Copying pointers is faster than copying gigabytes of KV cache.

Now for the part where I'm honest about what Dendrite can't do.

Single-sequence throughput? vLLM wins. It's optimized for exactly that. Multi-tenant serving with thousands of independent users? vLLM wins again. Dendrite is built for a single agent (or a small number of agents) doing branching search. Different optimization target, different architecture.

Current throughput on my DGX Spark (GB10) with TinyLlama-1.1B is 40.8 tokens per second. That's respectable but not spectacular. vLLM on an A100 will beat it for linear generation. The whole point is that Dendrite wins on a different axis: branching latency.

## Why Rust

I get asked this a lot. Short answer: memory safety without garbage collection.

Copy-on-write with reference counting is exactly the kind of code that produces subtle, horrible bugs in C++. Dangling pointers. Use-after-free. Data races. The KV cache is shared mutable state across multiple execution paths. That's a recipe for silent corruption.

Rust's ownership model catches these bugs at compile time. I've shipped exactly zero memory safety bugs in Dendrite's KV cache. The type system won't let me. This matters for infrastructure code that needs to be correct before it's fast.

The other reason is zero-copy interop with CUDA. Rust's cudarc library gives me direct GPU memory access without crossing a Python/C boundary. Agents can embed Dendrite directly. No HTTP server, no IPC overhead, no serialization.

## The Hardware Bet

Dendrite is optimized for NVIDIA's GB10 (the DGX Spark chip). This is a bet that unified memory architectures will matter more over time.

GB10 has 128GB of unified memory shared between the Grace CPU and Blackwell GPU, connected via NVLink-C2C at 900 GB/s. That's 7x faster than PCIe. And it's cache-coherent, which means the CPU can update block table pointers and the GPU can dereference them directly. No copying. No synchronization.

This enables what I call the "zero-copy logic loop": Rust runtime on the CPU handles tree manipulation and scheduling while the GPU handles attention and forward passes. They share memory directly. The architecture makes Dendrite's copy-on-write essentially free.

But Dendrite runs on any CUDA GPU. GB10 is just where the architecture shines brightest.

## Who Should Use This

Be honest about whether this is for you.

Use Dendrite if:
- You're building agents that do tree search (ToT, MCTS, beam search)
- Branching latency is your bottleneck
- You're running single-agent or small multi-agent workloads
- You can tolerate a Rust dependency

Use vLLM if:
- You're serving many independent users
- You need maximum single-sequence throughput
- Your agents don't branch (just chain-of-thought)
- You need a mature, battle-tested system

These aren't competing products. They're optimized for different workloads. I use vLLM for some things and Dendrite for others.

## What's Missing

Dendrite is v0.1. The core tree KV cache and fork semantics are solid (359 tests, including property-based testing for the copy-on-write invariants). But there's work left:

FP8 quantization (MXFP8 block scaling, Fp8Linear/Fp8Attention/Fp8SwiGluMlp) is implemented with a TE FFI stub for Transformer Engine hardware. End-to-end FP8 numeric agreement with FP16 is verified (<5% relative MAE). Perplexity validation on real safetensors weights is deferred until DGX access.

FlashInfer's paged attention kernels would be faster than candle-flash-attn, but the FFI bindings aren't done. This is pure performance optimization, not correctness, so I shipped without it.

Multi-GPU support doesn't exist. GB10 is single-GPU. If you need tensor parallelism across multiple GPUs, Dendrite won't help you today.

Python bindings don't exist. You need to write Rust or call Dendrite via FFI. This is a real barrier for the ML community, and I know it. Probably the most requested feature.

## The Bet

I'm betting that AI agents will increasingly need to explore. Not just generate one answer, but consider multiple approaches, backtrack, evaluate, and search. The research supports this: branching search produces better results on complex tasks.

If that's right, the infrastructure needs to support it efficiently. Right now, it doesn't. vLLM and SGLang are throughput machines. They're excellent at what they do. But they weren't built for rapid branching.

Dendrite is an experiment in building infrastructure that treats branching as a first-class primitive. Fork should be free. Memory should be shared until branches diverge. The tree structure of an agent's reasoning should be explicit in the system, not reconstructed from linear sequences.

Maybe this matters. Maybe it doesn't. I'll find out when people try to use it.

The code is MIT licensed. 359 tests pass. Fork latency is 500 nanoseconds. Continuous batching (Orca algorithm) mixes decode + prefill in every forward pass step. If that's useful to you, I'd love to hear about it.

github.com/BioInfo/dendrite

---

*Justin Johnson builds AI infrastructure at AstraZeneca and writes about it at rundatarun.io. He's on LinkedIn if you want to argue about inference engines.*
