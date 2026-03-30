# Dendrite Twitter/X Launch Thread

**DRAFT — DO NOT POST WITHOUT JUSTIN'S REVIEW**

---

## Thread (copy-paste ready)

---

**1/**
Introducing Dendrite — an LLM inference engine where forking a 4K context costs 500 nanoseconds.

Not milliseconds. Nanoseconds.

Here's why that matters for AI agents. 🧵

---

**2/**
When an AI agent does Tree-of-Thought or MCTS, it "forks" its context to explore branches.

In vLLM:    ~50-100ms per fork  
In SGLang:  ~5-10ms per fork  
In Dendrite: ~500ns per fork  

That's 1000-10000x faster. Here's how.

---

**3/**
Existing engines treat forking as "creating a new sequence that happens to share a prefix."

They *discover* sharing.

Dendrite treats fork as a first-class primitive. We *create* sharing — the KV cache is a tree of paged blocks with copy-on-write semantics from day one.

---

**4/**
Fork = shallow copy of the block table + ref count increments.

No KV data copied until branches diverge. When they do, only diverged blocks are copied.

```rust
let child1 = cache.fork_sequence(parent)?; // O(1)
let child2 = cache.fork_sequence(parent)?; // O(1)
```

---

**5/**
Memory impact for 6-branch MCTS with 4K shared prefix:

vLLM:     ~6 GB (6 full copies)  
SGLang:   ~2 GB (radix dedup)  
Dendrite: ~1.1 GB (CoW pages, copy only on diverge)  

---

**6/**
Real-world: MCTS search with 500 forks.

vLLM:     25-50 seconds  
SGLang:   2.5-5 seconds  
Dendrite: 1.5 milliseconds  

This makes tree search practical for interactive reasoning, not just research experiments.

---

**7/**
M7 ships continuous batching (Orca algorithm):

Every forward pass step mixes ALL decoding sequences + chunked prefill of new arrivals.

Scheduling overhead: ~7µs/step.  
Memory pool: batch alloc/free, CoW headroom reservation.

---

**8/**
When to use Dendrite:
✅ Tree-of-Thought reasoning  
✅ MCTS for planning / code gen  
✅ Beam search with deep trees  
✅ Speculative decoding  
✅ Multi-agent debate  

When NOT to use Dendrite:
❌ Multi-tenant API serving → use vLLM  
❌ Simple chat → use vLLM  

---

**9/**
Built in Rust with Candle. MIT licensed.

359 tests. FP8 quantization (MXFP8 block scaling). Continuous batching. Criterion benchmark suite.

github.com/BioInfo/dendrite

Happy to answer architecture questions. 👇

---

## Alt shorter version (if thread feels long)

> Introduced Dendrite today: an LLM inference engine where forking a 4K token context costs 500 nanoseconds (vs 50-100ms in vLLM).
>
> Built for AI agents doing Tree-of-Thought / MCTS. KV cache is a tree of paged CoW blocks — fork is just a pointer copy.
>
> 359 tests, MIT licensed, continuous batching, FP8 quant.
> github.com/BioInfo/dendrite

---

## Timing

Same as HN: Mon Mar 30, ~9:30 AM EST. Post HN first.

Post HN first (use hn-dendrite-draft.md), then tweet with HN link in reply to thread for social proof.
