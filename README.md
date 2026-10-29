# Dendrite

Faster serving for [kev](https://github.com/jaredpalmer/kev) decision models on hybrid Qwen3.5 backbones.

A kev request is one document (the state) and a set of yes/no, multiple-choice or rating questions about it. Kev runs the state once, then one branch per question on top of the cached state. Dendrite changes two things about how that works:

- **Question branches share one copy of the state.** Kev copies the state's attention cache once per question. `dendrite.fork` keeps a single copy and runs every branch against it with cascade attention, so branch time stops growing with the length of the document.
- **A state that grows is not re-run.** Kev's server reuses a state only when it repeats exactly. A case history, an agent trajectory or any document that gets appended to misses every time and pays for the whole state again. `dendrite.extend` continues a cached state with only the new tokens, and the result is bit-identical to running the whole state from scratch. `dendrite.serve` wires that into kev's server.

Kev is not modified. Dendrite patches it at runtime and every answer is checked against kev's own path.

**Status: experimental.** Everything below was measured on one machine (an NVIDIA DGX Spark, GB10) with one model (Kev-9B). Nothing has been tested on another GPU yet.

## Results

Kev-9B (`jaredpalmer/kev-9b@db029f0`), bf16, kev's fused kernels, on a GB10 that was also serving other models during the runs. Raw rows are in [`results/`](results/).

**Question branches** (`dendrite.fork`, 64 questions on one cached state, median of 5):

| State tokens | kev | Dendrite | Speedup |
|---|---|---|---|
| 1,024 | 795 ms | 565 ms | 1.4x |
| 4,096 | 2,375 ms | 577 ms | 4.1x |
| 8,192 | 5,814 ms | 589 ms | 9.9x |
| 16,384 | 7,116 ms | 615 ms | 11.6x |

Fork is exact up to floating-point reassociation, not bit-exact. Its gate requires every answer's argmax to agree with kev and every probability to stay within twice kev's own bf16-versus-fp32 difference. It passes with little room: max |dp| 0.0086 against a tolerance of 0.0096, 0 flips in 48 questions. The gate's selftest breaks the attention merge on purpose and the gate catches it (max |dp| 0.28, 2 flips). Fork holds more memory than kev's default path, 3.6 GiB against 0.8 GiB at 16k x 64, because kev runs its branches in small chunks. Kev with the chunking lifted runs out of memory at 8k x 64.

**Growing states** (`dendrite.extend`, appends to a 16k-token state; a full state pass takes 5,701 ms):

| Tokens appended | Time | Speedup |
|---|---|---|
| 5 | 125 ms | 45x |
| 212 | 264 ms | 22x |
| 1,008 | 464 ms | 12x |
| 4,421 | 1,569 ms | 3.6x |

Every cache tensor and every answer probability is bit-identical to a full state pass, in 10 of 10 cases: paragraph appends, a chain of four appends, and token-level splits at every offset class. The gate's three planted faults (DeltaNet state not advanced, no query padding, no GEMM padding) all fail it. On 218 real paragraph appends, 100% continued a stored checkpoint.

**Serving** (`dendrite.serve`, kev's server defaults: bf16, fused kernels, CUDA graphs): a session of 15 requests on one growing document plus one batch of six, each answer compared with kev answering the same request cold. All answers are bit-identical. Requests that grew a 6k-16k-token state took a median 1.44 s against kev's 3.58 s with 16 questions each. Most of the remaining time is kev's own question branches, because the server does not use `dendrite.fork` yet. Over HTTP, a 35k-token document took 13.1 s cold (first request, warmup included) and 0.54 s after one paragraph was appended.

## How the exact reuse works

A kev state is a token sequence, so a new state that starts with a cached state's tokens can continue from it: attention layers append the new keys and values, and the Gated DeltaNet layers carry their convolution window and recurrent state forward. Getting the same bits as a full pass took three more things, each found by comparing layer by layer:

1. **64-token alignment.** The chunked delta-rule kernel works in blocks of 64, so a continuation has to start on a multiple of 64. Dendrite keeps a checkpoint at the last 64-aligned point at least 8 tokens before the state's end, which also absorbs the tokenizer re-merging tokens at the join.
2. **Query padding.** With few query rows, flash attention switches to a split-KV schedule that sums in a different order. Extension passes pad their queries with zero rows to at least 512.
3. **GEMM row padding.** cuBLAS picks a different algorithm for some row counts. Which counts depends only on the weight shape, so `extend.install()` measures the exact row counts per projection shape when the model loads (about 30 s on Kev-9B) and pads inputs to the next one. The calibration belongs to the GPU and library versions it ran on.

## Limits

- One GPU (GB10) and one model size (Kev-9B) measured. Kev-27B in bf16 does not fit beside the other services on that machine and has not been run through these gates.
- The fork gate's margin is thin. A new GPU or model needs its own tolerance, measured the same way.
- `dendrite.serve` handles states longer than kev's graphed state pass (1,024 tokens). Shorter states and exact repeats go to kev unchanged.
- Checkpoints cost memory. Each one keeps its state's attention cache alive (about 0.5 GiB for 16k tokens on Kev-9B). The store holds 8 checkpoints and 131,072 tokens by default.
- A state edited before its last checkpoint is a miss and runs in full.
- FP8 weights were measured and are not included: 1.46-1.72x faster state passes, but 17 of 1,440 test answers changed.

## Install

Linux, Python 3.12, a CUDA GPU, and a source checkout of kev at the commit Dendrite was tested against. Kev pins `torch<2.9`; the GB10 needs torch 2.9 with CUDA 13, so kev goes in without its dependencies.

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
git clone https://github.com/jaredpalmer/kev && git -C kev checkout 2074cc4
pip install --no-deps -e kev
pip install accelerate datasets peft scikit-learn pydantic
pip install -e .
```

Tested with torch 2.9.0+cu130, transformers 5.20.1, flash-linear-attention 0.5.2 and triton 3.5.0. On a GB10, export `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, because Triton's bundled ptxas rejects sm_121a.

## Use

Serve. The server has kev's API (`POST /v1/systemone`, TypeSafe System One compatible), plus `GET /v1/dendrite` for checkpoint stats:

```bash
python -m dendrite.serve --run jaredpalmer/kev-9b --port 8008
```

`DENDRITE_CHECKPOINTS` and `DENDRITE_CHECKPOINT_TOKENS` size the checkpoint store. Kev's own `KEV_*` variables work as before.

As a library:

```python
import torch
from kev.checkpoint import LoadOptions, load
from kev.model import admit
from dendrite import extend, fork

tok, model = load("jaredpalmer/kev-9b", "cuda", LoadOptions(dtype=torch.bfloat16, fused=True))
extend.install(model.lm)                       # calibrates the GEMM row counts once

enc = admit(model, tok, record)                # a kev record: {"state": ..., "questions": [...]}
ck = extend.checkpoint(model, enc)             # keep this between requests

grown = admit(model, tok, grown_record)        # the same document with more text appended
prefix, ck = extend.advance(model, ck, grown)  # (None, None) when it does not continue the checkpoint
probs = fork.fork_probs(model, grown, prefix)  # every question, on one shared copy of the state
```

## Gates

Each gate needs a CUDA GPU and downloads the model on first run. Run them under the memory guard on a unified-memory GPU, where an allocation overrun can take down the whole machine instead of raising:

```bash
export DENDRITE_GPU_CAP_GIB=40                 # torch allocator cap
python -m dendrite.memguard --floor 8 -- python gates/fork_gate.py --run jaredpalmer/kev-9b --fused
python -m dendrite.memguard --floor 8 -- python gates/fork_gate.py --run jaredpalmer/kev-9b --fused --tol 0.01 --selftest
python -m dendrite.memguard --floor 8 -- python gates/extend_gate.py --run jaredpalmer/kev-9b
python -m dendrite.memguard --floor 8 -- python gates/serve_gate.py --run jaredpalmer/kev-9b
python gates/bench.py --run jaredpalmer/kev-9b --fused --arms kev,fork
```

`memguard` kills the command's process tree when free host memory falls below `--floor` GiB. `python -m dendrite.memguard --selftest` checks both directions without a GPU. The benchmark text is kev's own documentation, read from the kev checkout (`KEV_DOCS` overrides the path).

## The Rust engine

Until October 2026 this repository held a Rust inference engine with a copy-on-write paged KV cache and tree search. It is kept as the [`rust-engine`](../../tree/rust-engine) branch and the `rust-engine-final` tag. Its README compared fork latency against vLLM and SGLang recomputing a shared prefix. That comparison was wrong, because both engines reuse prefix KV blocks, and it was withdrawn on 2026-10-24.

## License

MIT. Kev is Apache-2.0, by Jared Palmer.
