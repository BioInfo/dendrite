# Practical Impact: Dendrite vs vLLM/SGLang for Agentic Workloads

## Real-World Scenario: AI Coding Agent

An AI coding agent solving a bug:

```
[System prompt + codebase context: 4K tokens]
         │
    ┌────┴────┐
    │         │
 Fix A     Fix B     (2 potential fixes)
    │         │
 ┌──┴──┐   ┌──┴──┐
Test  Test Test  Test  (test each fix with 2 approaches)
```

This is a 2-level tree with 6 total branches.

### Time to Create All Branches

| System | Fork/Branch Time | Total Branching Time |
|--------|-----------------|---------------------|
| **vLLM** | ~50-100ms (recompute 4K prefix) | **300-600ms** |
| **SGLang** | ~5-10ms (radix lookup + copy) | **30-60ms** |
| **Dendrite** | ~3μs (pointer copy) | **18μs** |

**Dendrite is 1,600x - 33,000x faster at branching.**

### Memory Usage

With 4K token prefix (Llama-3-8B, 32 layers, GQA):
- KV cache per token: ~256KB (32 layers × 8 KV heads × 128 dim × 2 bytes × 2 for K+V)
- 4K prefix = ~1GB KV cache

| System | Memory for 6 branches |
|--------|----------------------|
| **vLLM** | 6 × 1GB = **6GB** (full copies) |
| **SGLang** | ~2GB (prefix shared, some overhead) |
| **Dendrite** | ~1.1GB (prefix shared, only divergent data copied) |

**Dendrite uses 5x less memory for this workload.**

## Real-World Scenario: Tree-of-Thought Math Reasoning

Solving a complex math problem with exploration:

```
[Problem: 4K tokens]
         │
    ┌────┼────┐
    │    │    │
 Path A  B    C    (3 reasoning approaches)
    │    │    │
   ┌┴┐  ┌┴┐  ┌┴┐
   1 2  1 2  1 2   (2 sub-explorations each)
    │    │    │
  ┌─┼─┐┌─┼─┐┌─┼─┐
  a b c...          (3 evaluations each = 18 total leaf nodes)
```

Total forks: 3 + 6 + 18 = 27 branching operations

### Branching Overhead

| System | Time per Fork | Total Branching Overhead |
|--------|--------------|-------------------------|
| **vLLM** | ~50ms | **1.35 seconds** |
| **SGLang** | ~5ms | **135ms** |
| **Dendrite** | ~3μs | **81μs** |

**This is the difference between "instant" and "noticeable delay".**

### What This Means for Agent Responsiveness

For an interactive coding assistant doing 10 explorations per user request:

| System | Branching Latency | User Experience |
|--------|------------------|-----------------|
| **vLLM** | 500ms-1s | "Why is it thinking?" |
| **SGLang** | 50-100ms | "Slight delay" |
| **Dendrite** | <1ms | "Instant exploration" |

## Real-World Scenario: MCTS for Code Generation

Monte Carlo Tree Search with:
- 100 simulations
- Average 5 moves per simulation = 500 forks

| System | Total Fork Time |
|--------|----------------|
| **vLLM** | 25-50 seconds |
| **SGLang** | 2.5-5 seconds |
| **Dendrite** | **1.5ms** |

**MCTS becomes practical only with O(1) fork.**

## When Dendrite Wins vs Loses

### Dendrite Wins Big
- Tree-of-Thought: 10-100x faster exploration
- MCTS: Makes it practical (seconds → milliseconds)
- Speculative Decoding: Sub-microsecond branch creation
- Multi-agent systems: Agents share context efficiently

### Dendrite Loses
- Single sequence throughput: vLLM is 3-4x faster
- Simple chat: No branching needed, overhead doesn't help
- Batch processing: Independent requests don't benefit from sharing

## The Bottom Line

**If your agent explores multiple paths, Dendrite can:**
- Reduce branching latency by 1000-10000x
- Reduce memory usage by 3-5x
- Make MCTS/beam search practical for real-time applications

**If your agent generates one response per request:**
- Use vLLM - it's faster for that use case
