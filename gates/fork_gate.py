"""Correctness gate: fork_probs vs kev's own eager row path on the same cached state.

PASS = every question's argmax agrees and max |dp| stays within the bf16 noise floor, measured as kev bf16 vs kev fp32
on the same records (the reference path disagreeing with itself across dtypes). --selftest breaks the merge on purpose
(drops the branch half of the cascade) and must FAIL.
"""
import argparse, json, sys, time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import dendrite.gpucap  # noqa: F401  (must precede any CUDA allocation)
from dendrite import fork as forkattn
from records import make_record
from kev.checkpoint import LoadOptions, load
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

RUN = "jaredpalmer/kev-4b@v1.0"


def run_gate(model, tok, states, questions, tol):
    worst, flips, n = 0.0, 0, 0
    for S in states:
        rec = make_record(tok, S, questions)
        enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        prefix = model.prefix(enc)
        ref = model._branch_rows_from_prefix(enc, prefix[1])
        got = forkattn.fork_probs(model, enc, prefix)
        d = max((a - b).abs().max().item() for a, b in zip(ref, got))
        f = sum(int(a.argmax() != b.argmax()) for a, b in zip(ref, got))
        print(json.dumps({"state_tokens": enc["state_tokens"], "questions": len(ref), "max_dp": round(d, 5), "flips": f}), flush=True)
        worst, flips, n = max(worst, d), flips + f, n + len(ref)
    ok = worst <= tol and flips == 0
    print(json.dumps({"verdict": "PASS" if ok else "FAIL", "max_dp": round(worst, 5), "tol": tol, "flips": flips, "questions": n}), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="512,2048,8192")
    ap.add_argument("--questions", type=int, default=16)
    ap.add_argument("--tol", type=float, default=None, help="max |dp|; default = measured kev bf16-vs-fp32 floor x2")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fused", action="store_true", help="load kev's fused Triton layers for the bf16 model; the reference is then kev's fused row path")
    ap.add_argument("--run", default=RUN, help="kev checkpoint; 27B has no room for the fp32 floor, so pass --tol")
    a = ap.parse_args()
    states = [int(s) for s in a.states.split(",")]
    tol = a.tol
    if tol is None:
        # the noise floor: kev's reference path, fp32 vs bf16, on the shortest state
        tok, m32 = load(a.run, "cuda", LoadOptions(dtype=torch.float32))
        rec = make_record(tok, states[0], a.questions)
        enc = m32.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        p32 = m32._branch_rows_from_prefix(enc, m32.prefix(enc)[1])
        del m32; torch.cuda.empty_cache()
    tok, model = load(a.run, "cuda", LoadOptions(dtype=torch.bfloat16, fused=a.fused))
    if tol is None:
        p16 = model._branch_rows_from_prefix(enc, model.prefix(enc)[1])
        floor = max((x - y).abs().max().item() for x, y in zip(p32, p16))
        tol = max(2 * floor, 1e-3)
        print(json.dumps({"noise_floor_bf16_vs_fp32": round(floor, 5), "tol": round(tol, 5)}), flush=True)
    if a.selftest:
        real = forkattn._flash
        def broken(q, k, v, causal, scale):
            o, lse = real(q, k, v, causal, scale)
            return (o, lse) if not causal else (torch.zeros_like(o), torch.full_like(lse, -1e4))
        forkattn._flash = broken
        ok = run_gate(model, tok, states[:1], a.questions, tol)
        print(json.dumps({"selftest": "PASS (broken merge was caught)" if not ok else "FAIL (broken merge passed the gate)"}))
        sys.exit(0 if not ok else 1)
    sys.exit(0 if run_gate(model, tok, states, a.questions, tol) else 1)


if __name__ == "__main__":
    t = time.time(); main()
