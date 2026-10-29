# Contributing to Dendrite

Issues and pull requests are welcome. Dendrite is small and experimental, so open an issue before a large change.

## Setup

Follow the Install section of the README: torch 2.9 for your CUDA version, a kev checkout at the tested commit installed with `--no-deps`, then `pip install -e .`.

## Before a pull request

- `uvx ruff check --select F dendrite gates` is clean.
- `python -m dendrite.memguard --selftest` passes.
- A change to `dendrite.fork`, `dendrite.extend` or `dendrite.serve` comes with its gate run on a GPU: `gates/fork_gate.py` (and `--selftest`), `gates/extend_gate.py` or `gates/serve_gate.py`. Paste the final verdict line, the GPU, and the kev checkpoint in the pull request.
- A new speed claim comes with the benchmark rows behind it and the hardware it ran on.

## Gates must be able to fail

Every gate here carries planted-fault controls that must fail it. A new gate, or a change to one, keeps that property: say which line you broke to make it fail and paste the failing output.
