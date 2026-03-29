# Dendrite Launch Timeline

**PRIVATE - Launch Planning Document**

---

## Pre-Launch Checklist

### Technical (Must Complete)

- [x] Core tree KV cache with O(1) fork
- [x] GPU inference working (40.8 tok/s on GB10)
- [x] 328 tests passing (M6 FP8 quantization + M7 continuous batching + benchmarks)
- [x] Fork benchmark: ~500ns
- [x] Grammar benchmark: ~1.6μs
- [x] BENCHMARKS.md with competitive comparison
- [x] PRACTICAL_IMPACT.md with real-world scenarios
- [x] README "Why Dendrite?" section
- [x] Architecture diagram (docs/architecture.md)
- [x] Demo GIF recorded and placed in /assets/ (2.5MB, 1100×700, Dracula theme)

### Documentation

- [x] README.md polished
- [x] CONTRIBUTING.md
- [x] ROADMAP.md updated
- [x] Issue templates
- [x] PR template
- [x] CI pipeline working

### Launch Materials

- [x] HN post drafted (docs/launch/hn-post.md)
- [x] Demo GIF created (assets/demo.gif, 2.5MB)
- [x] Twitter thread drafted (docs/launch/twitter-thread.md)
- [x] Reddit posts drafted (in timeline.md social templates)
- [x] Blog post outlined and drafted (docs/launch/blog-post.md)

---

## Launch Week Schedule

### T-7 Days: Final Polish

- [ ] Record demo GIF using VHS or asciinema
- [ ] Final README review
- [ ] Test all examples still work
- [ ] Verify benchmarks on fresh clone
- [ ] Create 3-5 good-first-issues

### T-3 Days: Soft Launch

- [ ] Share with 10-20 trusted contacts
- [ ] Ask for honest feedback
- [ ] Fix any issues found
- [ ] Collect 20-50 initial stars (network seeding)

### T-1 Day: Final Prep

- [ ] Draft all social posts
- [ ] Set up HN monitoring (hnrankings.info)
- [ ] Prepare talking points for comments
- [ ] Clear calendar for launch day
- [ ] Tag v0.1.0 release

### Launch Day (Tuesday-Thursday, 9-11 AM EST)

**Hour 0**:
- [ ] Post to Hacker News (use title option 1 or 2)
- [ ] Monitor for first comments

**Hour 1-2**:
- [ ] Respond to every comment thoughtfully
- [ ] Be technical, honest, humble
- [ ] Don't over-promise

**Hour 3-6**:
- [ ] Cross-post to r/rust
- [ ] Cross-post to r/LocalLLaMA
- [ ] Tweet thread with key benchmarks

**End of Day 1**:
- [ ] Summarize feedback
- [ ] Note feature requests
- [ ] Thank commenters

### T+1 Week: Follow-up

- [ ] Technical blog post: "How Dendrite Achieves O(1) Fork"
- [ ] Submit to This Week in Rust newsletter
- [ ] Respond to any GitHub issues
- [ ] Start working on most-requested features

---

## Social Media Templates

### Twitter Thread

```
1/ Introducing Dendrite: O(1) fork latency for LLM tree search

When AI agents explore multiple reasoning paths (Tree-of-Thought, MCTS), they need to "fork" their context.

In vLLM: ~50-100ms per fork
In SGLang: ~5-10ms per fork
In Dendrite: ~500 nanoseconds

Thread 🧵

2/ How? We represent the KV cache as a tree of paged blocks with copy-on-write.

Fork = shallow copy of block table + increment refcounts
Memory duplication only when branches actually diverge

[diagram image]

3/ Real-world impact:

MCTS with 500 forks:
- vLLM: 25-50 seconds
- SGLang: 2.5-5 seconds
- Dendrite: 1.5 milliseconds

This makes tree search practical for real-time applications.

4/ When to use Dendrite:
- Tree-of-Thought reasoning
- MCTS for planning/code generation
- Beam search with deep trees
- Speculative decoding
- Multi-agent debate

5/ When NOT to use Dendrite:
- Multi-tenant API serving (use vLLM)
- Simple chat/QA (use vLLM)
- Maximum single-sequence throughput (use vLLM)

Complementary positioning, not replacement.

6/ Built in Rust with:
- Candle for ML primitives
- 328 tests passing
- Property-based testing for CoW invariants
- Tested on NVIDIA GB10 (DGX Spark)

MIT licensed, contributions welcome.

github.com/BioInfo/dendrite
```

### Reddit r/rust

```
Title: Dendrite: Rust inference engine with O(1) fork for LLM tree search

Hey r/rust!

I built Dendrite, a Rust inference engine optimized for AI agents that explore multiple reasoning paths (Tree-of-Thought, MCTS, beam search).

The key insight: existing engines like vLLM optimize for multi-tenant throughput. When an agent forks to explore branches, they recompute the KV cache from scratch (~50-100ms).

Dendrite uses a tree-structured KV cache with copy-on-write blocks. Fork is just a pointer copy. Cost: ~500 nanoseconds.

Why Rust?
- Memory safety for refcounting (no GC pauses)
- Zero-copy CUDA interop via cudarc
- Agents embed directly (no IPC overhead)
- Candle provides solid ML primitives

Current state:
- 328 tests passing
- TinyLlama-1.1B at 40.8 tok/s on GB10
- Property-based tests with proptest
- MIT licensed

Looking for feedback on the architecture and contributions welcome!

GitHub: github.com/BioInfo/dendrite
```

### Reddit r/LocalLLaMA

```
Title: Dendrite: 1000x faster branching for Tree-of-Thought and MCTS

Built an inference engine specifically for AI agents that need to explore multiple reasoning paths.

The problem: When you do Tree-of-Thought or MCTS, you need to "fork" the context to try different branches. In vLLM, each fork recomputes the entire KV cache (~50-100ms). That adds up fast.

Dendrite solution: Tree-structured KV cache with copy-on-write. Fork costs ~500 nanoseconds because we just copy pointers and increment refcounts. Memory is shared until branches actually diverge.

Benchmarks:
| Scenario | vLLM | Dendrite |
|----------|------|----------|
| Fork 4K context | 50-100ms | 3μs |
| MCTS 500 forks | 25-50s | 1.5ms |
| Memory 6 branches | 6GB | 1.1GB |

Currently running TinyLlama-1.1B at 40.8 tok/s on DGX Spark (GB10).

This is NOT a vLLM replacement for serving users. It's complementary - use vLLM for throughput, Dendrite for single-agent reasoning.

MIT licensed: github.com/BioInfo/dendrite
```

---

## Success Metrics

### Launch Day

| Metric | Minimum | Good | Great |
|--------|---------|------|-------|
| HN points | 50 | 150 | 400+ |
| GitHub stars | 50 | 200 | 500+ |
| GitHub forks | 5 | 20 | 50+ |

### Week 1

| Metric | Minimum | Good | Great |
|--------|---------|------|-------|
| Total stars | 200 | 500 | 1000+ |
| Contributors | 1 | 3 | 10+ |
| Issues opened | 10 | 30 | 100+ |

### Month 1

| Metric | Target |
|--------|--------|
| Stars | 1000+ |
| External blog posts | 2+ |
| Integration PRs | 1+ (LangGraph, etc.) |

---

## Risk Mitigation

### "Benchmarks are fake/misleading"

Mitigation:
- All benchmarks reproducible with cargo bench
- Document exact hardware, model, settings
- Be honest about limitations (single-sequence throughput)

### "vLLM already does this"

Mitigation:
- Acknowledge vLLM's prefix caching
- Explain the difference: discovery vs. explicit fork
- Show benchmark comparison with prefix caching enabled

### "Why not just contribute to vLLM?"

Mitigation:
- Different optimization target (latency vs. throughput)
- Rust vs. Python architecture
- Clean-slate design for tree-native semantics

### "This won't scale"

Mitigation:
- Be honest: v0.1 is single-GPU
- Multi-GPU is roadmap item
- Target use case is single-agent, not cluster serving

---

## Post-Launch Improvements (Based on Expected Feedback)

1. **Python bindings** - Most requested, plan for v0.2
2. **Llama-3 support** - Larger models
3. **FlashInfer paged kernels** - Better performance
4. **Speculative decoding** - Natural fit for tree structure
5. **LangGraph integration** - Example/cookbook
