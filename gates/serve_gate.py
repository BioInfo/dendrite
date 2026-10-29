"""Gate for dendrite.serve: does a kev server with Dendrite checkpoints answer exactly what kev answers, and what does a
growing state save?

One loaded model under kev.serve's serving defaults (bf16, fused, CUDA graphs), one kev.serve.Server. Every request runs
twice through Server._run: with Dendrite (its checkpoints and kev's prefix cache, kept across the session as a live
server keeps them) and as reference (kev's own probs_batch, a prefix cache that never hits: what kev serves when the
state changed since it was cached). Answers must be bit-identical (torch.equal per question).

Session: a state that grows by paragraphs from ~6k to ~16k tokens (appends of a few to a few thousand tokens), a branch
(two different continuations of one state), an exact repeat (kev's own cache hit), a short state (left to kev's graphed
state pass), an edit in the middle (no checkpoint continues it), and one batch of six requests (more eligible states than
one built group). Path counts are checked against what each request should take. After the session every stored
checkpoint must still equal model.prefix of its tokens (nothing wrote into the shared keys/values).
Controls that must FAIL: DeltaNet states not advanced, and no GEMM padding, each on a growth step.
Inference only.
"""
import argparse, json, statistics, sys, time
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import dendrite.gpucap  # noqa: F401
from dendrite import extend as ex
from dendrite import serve as dserve
import kev.model as km
import kev.serve as ks
from kev.checkpoint import Checkpoint, LoadOptions, fused_available
from extend_gate import same_cache, rec
from records import corpus


def progress(msg):
    with open(Path("progress.md"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} serve_gate: {msg}\n")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", type=int, default=16)
    ap.add_argument("--out", default="results/serve.jsonl")
    a = ap.parse_args()
    out = open(a.out, "a")
    emit = lambda row: (print(json.dumps(row), flush=True), out.write(json.dumps(row) + "\n"), out.flush())

    opts = replace(LoadOptions.from_env(), dtype=torch.bfloat16, cuda_graphs=True, fused=fused_available())   # kev.serve's CUDA defaults
    ck = Checkpoint(a.run)
    tok, model = ck.load("cuda", opts)
    server = ks.Server(ck, tok, model, "cuda")
    t = time.time(); dserve.install(server); t_install = time.time() - t
    dcache, rcache = server.prefix_cache, ks.PrefixCache(0, 0)
    dstore = server.checkpoints
    emit({"run": a.run, "fused": opts.fused, "cuda_graphs": model.graphs is not None, "install_s": round(t_install, 1),
          "graph_state": __import__("kev.cuda_graphs", fromlist=["x"]).GRAPH_STATE, "kev_prefix_cache_size": dcache.size})
    wrapped, kev_batch = model.probs_batch, model._kev_probs_batch

    def run(encs, dendrite):
        server.prefix_cache = dcache if dendrite else rcache
        model.probs_batch = wrapped if dendrite else kev_batch
        with server.lock:
            torch.cuda.synchronize(); t = time.perf_counter()
            res = server._run(encs)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t) * 1000
        model.probs_batch = wrapped; server.prefix_cache = dcache
        return [[torch.tensor(q) for q in r[0]] for r in res], dt

    E = lambda text: km.admit(model, tok, rec(text, a.questions))
    text = corpus()
    while len(km.user_tokens(tok, text)) < 18000: text = text + "\n\n" + corpus()
    cuts = [i for i in range(len(text)) if text.startswith("\n\n", i)]
    ntok = lambda c: len(km.user_tokens(tok, text[:c]))
    def cut_near(n):
        return min(cuts, key=lambda c: abs(ntok(c) - n)) if len(cuts) < 400 else _bisect(n)
    def _bisect(n):
        lo, hi, best = 0, len(cuts) - 1, cuts[0]
        while lo <= hi:
            mid = (lo + hi) // 2
            if ntok(cuts[mid]) <= n: best, lo = cuts[mid], mid + 1
            else: hi = mid - 1
        return best

    # (name, text, expected Dendrite path: extended / cold / exact / skipped)
    grow = [cut_near(n) for n in (6000, 6300, 6340, 7900, 8000, 11000, 12000, 15000, 16000)]
    grow = [c if k == 0 or c > grow[k - 1] else min(x for x in cuts if x > grow[k - 1]) for k, c in enumerate(grow)]   # distinct states: a repeat is kev's exact hit, tested below
    for k in range(1, len(grow)): grow[k] = max(grow[k], min(x for x in cuts if x > grow[k - 1]))
    steps = [("grow_0", text[:grow[0]], "cold")] + [(f"grow_{k}", text[:c], "extended") for k, c in enumerate(grow[1:], 1)]
    steps.append(("append_5_chars", text[:grow[-1]] + " Then", "extended"))
    base = text[:grow[4]]
    steps.append(("branch_a", base + "\n\nThe patient later reported new symptoms.", "extended"))
    steps.append(("branch_b", base + "\n\nThe customer then asked for a refund.", "extended"))
    steps.append(("repeat_branch_b", base + "\n\nThe customer then asked for a refund.", "exact"))
    steps.append(("short_state", text[:cut_near(500)], "skipped"))
    mid = len(text[:grow[3]]) // 2
    steps.append(("edit_middle", text[:mid] + " (corrected)" + text[mid:grow[3]], "cold"))

    paths = lambda: (dstore.extended, dstore.cold, dstore.skipped, dcache.hits)
    rows, ok_all = [], True
    for name, t_, want in steps:
        enc = E(t_)
        before = paths()
        (pd,), td = run([enc], True)
        after = paths()
        took = ["extended", "cold", "skipped", "exact"][[x != y for x, y in zip(before, after)].index(True)] if before != after else "none"
        (pr,), tr = run([enc], False)
        same = all(torch.equal(x, y) for x, y in zip(pd, pr))
        dp = max((x - y).abs().max().item() for x, y in zip(pd, pr))
        ok = same and took == want
        ok_all &= ok
        row = {"step": name, "state_tokens": enc["state_tokens"], "path": took, "expected": want, "bit_identical": same, "max_dp": dp,
               "dendrite_ms": round(td, 1), "kev_ms": round(tr, 1), "verdict": "PASS" if ok else "FAIL"}
        rows.append(row); emit(row); progress(f"{name} {row['verdict']} {took} {td:.0f} vs {tr:.0f} ms")

    # one batch of six: five growth variants of a fresh document (more than EAGER_STATES to build) + a short state
    t2 = "\n\n".join(reversed(corpus().split("\n\n")))   # the same paragraphs in reverse order: no checkpoint continues it
    c2 = [i for i in range(len(t2)) if t2.startswith("\n\n", i)]
    seed_enc = E(t2[: c2[len(c2) // 2]])
    run([seed_enc], True)   # its checkpoint
    batch_encs = [E(t2[: c2[len(c2) // 2 + k]]) for k in range(1, 6)] + [E(text[:cut_near(400)])]
    before = paths()
    pd, td = run(batch_encs, True)
    after = paths()
    pr = [run([e], False)[0][0] for e in batch_encs]
    same = all(torch.equal(x, y) for p1, p2 in zip(pd, pr) for x, y in zip(p1, p2))
    counts = {k: y - x for k, x, y in zip(("extended", "cold", "skipped", "exact"), before, after)}
    ok = same and counts["extended"] == 5 and counts["skipped"] == 1
    ok_all &= ok
    emit({"step": "batch_6", "bit_identical": same, "paths": counts, "dendrite_ms": round(td, 1), "verdict": "PASS" if ok else "FAIL"})

    # stored checkpoints are intact
    intact = all(same_cache(c["prefix"][1], model.prefix(c["enc"])[1])[0] for c in dstore.entries.values())
    ok_all &= intact
    emit({"checkpoints_intact": intact, "checked": len(dstore.entries), "store": dstore.stats()})

    # controls: a growth step with a fault planted must NOT match kev
    def control(flag, name):
        server.checkpoints = dserve.Checkpoints()
        e0, e1 = E(text[:grow[5]]), E(text[:grow[6]])
        run([e0], True)
        flag[0] = False
        try: (pd,), _ = run([e1], True)
        finally: flag[0] = True
        server.checkpoints = dstore
        (pr,), _ = run([e1], False)
        caught = not all(torch.equal(x, y) for x, y in zip(pd, pr))
        emit({"control": name, "caught": caught, "max_dp": max((x - y).abs().max().item() for x, y in zip(pd, pr))})
        return caught
    dcache.clear()
    c1 = control(ex._ADVANCE, "no_advance")
    dcache.clear()
    c2_ = control(ex._PAD, "no_gemm_padding")

    gate = "PASS" if ok_all and c1 and c2_ else "FAIL"
    growth = [r for r in rows if r["path"] == "extended"]
    emit({"gate": gate, "steps": len(rows) + 1, "controls_caught": [c1, c2_],
          "extended_median_ms": statistics.median(r["dendrite_ms"] for r in growth), "kev_median_ms_same_requests": statistics.median(r["kev_ms"] for r in growth),
          "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
    progress(f"END gate={gate}")


if __name__ == "__main__":
    main()
