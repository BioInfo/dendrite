# Benchmarks

**Withdrawn 2026-10-24.** This page compared Dendrite's fork latency and memory against vLLM and SGLang on the assumption that both recompute a shared prefix on fork. They do not: vLLM's automatic prefix caching (default in V1) and SGLang's RadixAttention both reuse prefix KV blocks. The Dendrite-only numbers measured block-table bookkeeping, not model inference.

A measured comparison on the same model and hardware will replace this page. Until then, see the README for what Dendrite does.
