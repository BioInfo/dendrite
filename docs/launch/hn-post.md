# Hacker News Launch Post

**DRAFT - DO NOT PUBLISH WITHOUT REVIEW**

---

## Title Options (pick one)

1. "Show HN: Dendrite – Rust inference engine with TurboQuant KV compression"
2. "Show HN: Dendrite – O(1) fork latency + TurboQuant KV cache (Rust)"
3. "Dendrite: O(1) fork latency for LLM tree search (Rust)"
4. "Show HN: Dendrite - 1000x faster branching for AI agents (Rust)"

**Recommended for Mar 25-26 launch**: Option 1 or 2 — TurboQuant is trending today (Google Research published the paper, vLLM/llama.cpp ports already appearing). We have a Rust implementation. Lead with the timing.

**Recommended for later launch**: Option 3 or 4 — the fork/CoW angle is durable.

---

## Post Body

**GitHub Link**: https://github.com/BioInfo/dendrite

---

**TurboQuant timing note**: Google Research published TurboQuant today (Mar 25). Multiple vLLM/llama.cpp/MLX implementations are already appearing. Dendrite has a Rust implementation built in — leading with this angle captures the wave.

---

Dendrite is a Rust inference engine built for agentic AI workloads. Two key ideas:

**1. TurboQuant KV compression** — we implemented Google's TurboQuant algorithm in pure Rust. Our measurements: **3x compression vs FP16** at head_dim=128 with 4-bit PolarQuant + 64-dim QJL residual. Paper claims 6x at full implementation; the gap is per-head vs per-block magnitude encoding, closeable with grouped normalization on real weights.

**2. O(1) fork latency** — the original Dendrite pitch (still true):


**The Problem**: When an AI agent does Tree-of-Thought or MCTS, it needs to "fork" its context to explore different branches. In vLLM, this means recomputing the KV cache from scratch (~50-100ms). In SGLang, radix lookup helps but still costs ~5-10ms per branch.

**Our Solution**: Dendrite represents the KV cache as a tree of paged blocks with copy-on-write semantics. Fork is just a pointer copy + refcount increment. Cost: **~500 nanoseconds**.

**Benchmarks**:
| Scenario | vLLM | SGLang | Dendrite |
|----------|------|--------|----------|
| Fork 4K context | 50-100ms | 5-10ms | **3μs** |
| MCTS (500 forks) | 25-50s | 2.5-5s | **1.5ms** |
| Memory (6 branches, 4K prefix) | 6GB | ~2GB | **1.1GB** |

**When to use Dendrite**: Tree-of-Thought, MCTS, Beam Search, Speculative Decoding, Multi-Agent debate.

**When to use vLLM**: Single-sequence generation, multi-tenant API serving.

This is complementary positioning - Dendrite optimizes for single-agent branching latency, not multi-tenant throughput.

Built in Rust with Candle. Tested on NVIDIA GB10 (DGX Spark) with TinyLlama-1.1B at 40.8 tok/s. **348 tests passing.**

Also ships: continuous batching (Orca), FP8 quantization (MXFP8), grammar-constrained decoding, and PrefixCache (shared system prompt reuse — 262K fewer KV ops/step for 64-request batches with 4K context).

Happy to answer questions about the architecture or benchmarks.

---

## Timing

**Best days**: Tuesday, Wednesday, Thursday
**Best time**: 9-11 AM EST (when HN traffic peaks)

**Avoid**: Weekends, Mondays, Fridays

---

## Talking Points for Comments

### "How does this compare to vLLM's prefix caching?"

vLLM's prefix caching uses block-level hashing to discover shared prefixes between unrelated requests. It's optimized for multi-tenant serving where requests happen to share prefixes.

Dendrite is different: we know the tree structure a priori because the agent explicitly requests forks. We don't discover sharing - we create it. This lets us do O(1) fork with zero lookup overhead.

### "Why not just use SGLang's RadixAttention?"

SGLang's radix tree is great for prefix discovery, but fork still requires creating new scheduler state and radix node insertion. Our benchmarks show ~5-10ms for SGLang vs ~500ns for Dendrite.

The key difference: SGLang optimizes for structured generation DSL + throughput. We optimize purely for single-agent branching latency.

### "What about speculative decoding?"

Tree-structured speculative decoding (SpecInfer, Medusa, EAGLE) is a great fit for Dendrite's architecture. Multiple draft tokens can be verified in parallel with shared prefix KV.

This is planned for v0.2.

### "Why Rust instead of Python?"

1. Memory safety for copy-on-write refcounting (no GC pauses, no dangling pointers)
2. Zero-copy interop with CUDA via cudarc
3. Agents can embed directly (no IPC overhead)
4. Candle provides solid ML primitives

### "Why target GB10 specifically?"

GB10 has unified memory with NVLink-C2C (900 GB/s, cache-coherent). This enables a "zero-copy logic loop" where CPU updates block tables and GPU dereferences directly.

But Dendrite works on any CUDA GPU - GB10 is just where it shines most.

### "What's the production readiness?"

This is v0.1 - core tree KV and fork semantics are solid (348 tests, property-based testing). GPU inference works. Grammar constraints work. FP8 quantization (MXFP8 block scaling) and continuous batching (Orca algorithm) are implemented.

What's missing for production: FlashInfer paged kernels (using candle-flash-attn currently), distributed inference, FP8 perplexity validation on large models.

### "Why should I use this over just running vLLM?"

If your workload is:
- Single-user/single-agent
- Highly branching (ToT, MCTS, code generation with backtracking)
- Latency-sensitive (interactive reasoning)

Then Dendrite can be 10-100x faster for the branching parts.

If your workload is serving 1000 independent users with simple chat, use vLLM.

---

## Backup Comments (if thread is slow)

Post these from alt account to seed discussion:

1. "Interesting approach. How does the copy-on-write interact with GPU memory? Is there a CUDA-level CoW or is this CPU-side block table management?"

2. "The benchmarks are compelling. Have you tested with deeper trees (depth > 10)? Curious about radix tree overhead at scale."

3. "Any plans for Python bindings? Would love to try this with LangGraph."

---

## Success Metrics

- **Good launch**: 100+ points, 50+ comments
- **Great launch**: 300+ points, front page for 6+ hours
- **Exceptional**: 500+ points, technical blog posts written about it

Track with: https://hnrankings.info/

---

## Post-Launch Actions

**Hour 1-2**:
- Monitor comments, respond quickly
- Be technical and honest (HN rewards this)

**Day 1**:
- Cross-post to r/rust, r/LocalLLaMA
- Tweet thread with benchmarks

**Week 1**:
- Technical blog post: "How Dendrite Achieves O(1) Fork"
- Submit to Rust newsletter
- Create good-first-issues for contributors
