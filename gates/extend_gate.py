"""State reuse (dendrite.extend checkpoint/advance): bit-exact against a full state pass? How often does real appended text
hit? What does it save?

1. Prefix rate (tokenizer only): kev's docs split at paragraph boundaries; for each pair of consecutive boundaries,
   are the tokens of the shorter text, minus the checkpoint slack, a prefix of the tokens of the longer one?
2. Gate: a state cut at a paragraph boundary, checkpointed, then advanced by the rest, against model.prefix of the whole
   state: every cache tensor (attention keys/values, DeltaNet conv and recurrent states) torch.equal, and the question
   probabilities (fork and kev rows) identical. Single appends of several sizes, a chain of appends, and token-level
   splits at every offset class. Controls that must FAIL the bitwise gate: DeltaNet states not advanced (kev's own
   continuation), and no query padding (flash's split-KV schedule).
3. Timing: full state pass of the new state vs advance (both its extensions, cache copies included).
Inference only.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import dendrite.gpucap  # noqa: F401
from dendrite import extend as ex
from dendrite import fork as forkattn
import kev.model as km
from kev.checkpoint import LoadOptions, load
from records import POOL, corpus


def progress(msg):
    with open(Path("progress.md"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} extend_gate: {msg}\n")


def rec(text, Q):
    return {"state": text, "questions": [{"instr": POOL[i % len(POOL)][0], "options": POOL[i % len(POOL)][1], "label": 0} for i in range(Q)]}


def diff(ref, got):
    return max((a - b).abs().max().item() for a, b in zip(ref, got)), sum(int(a.argmax() != b.argmax()) for a, b in zip(ref, got))


def same_cache(c1, c2):
    """(all tensors bit-identical, number of tensors compared, worst relative difference)."""
    n, worst, eq = 0, 0.0, True
    for x, y in zip(c1.layers, c2.layers):
        pairs = ([(x.conv_states[0], y.conv_states[0]), (x.recurrent_states[0], y.recurrent_states[0])] if isinstance(x, LinearAttentionCacheLayerMixin)
                 else [(x.keys, y.keys), (x.values, y.values)])
        for a, b in pairs:
            n += 1
            if a.shape != b.shape: return False, n, float("inf")
            if not torch.equal(a, b):
                eq = False; worst = max(worst, ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item())
    return eq, n, worst


def ms(fn, reps):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--total", type=int, default=16384, help="state tokens after the append")
    ap.add_argument("--appends", default="64,256,1024,4096")
    ap.add_argument("--questions", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/extend.jsonl")
    a = ap.parse_args()
    out = open(a.out, "a")
    emit = lambda row: (print(json.dumps(row), flush=True), out.write(json.dumps(row) + "\n"), out.flush())
    tok, model = load(a.run, "cuda", LoadOptions(dtype=torch.bfloat16, fused=True))
    emit({"run": a.run, "deltanet_mixers_patched": ex.install(model.lm), "gemm_exact_rows": {f"{k[0]}x{k[1]}": [len(v), v[:6]] for k, v in ex._GOOD.items()}, "align": ex.ALIGN, "slack": ex.SLACK, "query_rows": ex.EXT_QUERY_ROWS})
    E = lambda text: model.encode(tok, rec(text, a.questions), max_state=km.SERVE_MAX_STATE, max_branch=km.SERVE_MAX_BRANCH)

    text = corpus()
    while len(km.user_tokens(tok, text)) < a.total: text = text + "\n\n" + corpus()
    cuts = [i for i in range(len(text)) if text.startswith("\n\n", i)]

    # 1. hit rate: raw token prefix, and prefix up to the checkpoint (what advance needs)
    sample = cuts[:: max(1, len(cuts) // 200)]
    raw = slack = 0
    for c0, c1 in zip(sample, sample[1:]):
        t0, t1 = km.user_tokens(tok, text[:c0]), km.user_tokens(tok, text[:c1])
        raw += t1[: len(t0)] == t0
        k = ex._point(len(t0) + 1) - 1   # checkpoint point in user tokens (the <state> token is position 0)
        slack += t1[: max(k, 0)] == t0[: max(k, 0)]
    m = len(sample) - 1
    emit({"appends_checked": m, "raw_prefix_rate": round(raw / m, 3), "checkpoint_hit_rate": round(slack / m, 3)})

    def cut_near(n_tokens):
        best, lo, hi = None, 0, len(cuts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if len(km.user_tokens(tok, text[: cuts[mid]])) + 1 <= n_tokens: best, lo = mid, mid + 1
            else: hi = mid - 1
        return cuts[best]

    end = cut_near(a.total)
    enc_new = E(text[:end])
    p_full = model.prefix(enc_new)
    ref_kev = model._branch_rows_from_prefix(enc_new, p_full[1])
    ref_fork = forkattn.fork_probs(model, enc_new, p_full)
    t_full = ms(lambda: model.prefix(enc_new), a.reps)
    emit({"new_state_tokens": enc_new["state_tokens"], "full_prefix_ms": round(t_full, 1)})

    def check(name, encs, expect_exact=True):
        """Checkpoint encs[0], advance through the rest (the last = enc_new), compare with the full pass."""
        ck = ex.checkpoint(model, encs[0])
        for e in encs[1:]:
            p, ck = ex.advance(model, ck, e)
            if p is None:
                emit({"case": name, "miss": True}); progress(f"{name} MISS"); return None
        eq, n, worst = same_cache(p[1], p_full[1])
        d_k, f_k = diff(ref_kev, model._branch_rows_from_prefix(enc_new, p[1]))
        d_f, f_f = diff(ref_fork, forkattn.fork_probs(model, enc_new, p))
        ok = eq and d_k == 0 and d_f == 0
        row = {"case": name, "steps": len(encs) - 1, "old_state_tokens": encs[0]["seg"].count(0), "cache_bit_identical": eq, "tensors": n,
               "worst_rel": round(worst, 7), "kev_rows": [round(d_k, 6), f_k], "fork": [round(d_f, 6), f_f], "verdict": "PASS" if ok else "FAIL"}
        emit(row); progress(f"{name} {row['verdict']} bit_identical={eq} dp={max(d_k, d_f):.2e}")
        return ok

    verdicts = []
    for n in map(int, a.appends.split(",")):
        enc_old = E(text[: cut_near(a.total - n)])
        verdicts.append(check(f"append_{n}", [enc_old, enc_new]))
        ck = ex.checkpoint(model, enc_old)
        if ex.advance(model, ck, enc_new)[0] is not None:
            t_adv = ms(lambda: ex.advance(model, ck, enc_new), a.reps)
            emit({"appended_tokens": enc_new["state_tokens"] - enc_old["state_tokens"], "recomputed_tokens": enc_new["state_tokens"] - ck["enc"]["seg"].count(0),
                  "full_prefix_ms": round(t_full, 1), "advance_ms": round(t_adv, 1), "speedup": round(t_full / t_adv, 1)})
            progress(f"append {enc_new['state_tokens'] - enc_old['state_tokens']} tok: advance {t_adv:.0f} ms vs full {t_full:.0f} ms")
        del ck; torch.cuda.empty_cache()

    chain = [E(text[: cut_near(a.total - k * 1024)]) for k in (4, 3, 2, 1)] + [enc_new]
    verdicts.append(check("chain_4x1k", chain))

    # token-level splits (no tokenizer effects): every residue class of the old length mod 64
    L = enc_new["seg"].count(0)
    for A in (L - 5, L - 63, L - 64, L - 600, L - 2049):
        old = {"ids": enc_new["ids"][:A], "pos": enc_new["pos"][:A], "seg": [0] * A}
        verdicts.append(check(f"token_split_{A}", [old, enc_new]))

    # controls: each must FAIL the bitwise gate
    old = {"ids": enc_new["ids"][:L - 600], "pos": enc_new["pos"][:L - 600], "seg": [0] * (L - 600)}
    ex._ADVANCE[0] = False
    try: c1 = check("control_no_advance", [old, enc_new]) is False
    finally: ex._ADVANCE[0] = True
    rows0 = ex.EXT_QUERY_ROWS; ex.EXT_QUERY_ROWS = 0
    try: c2 = check("control_no_query_padding", [{"ids": enc_new["ids"][:L - 5], "pos": enc_new["pos"][:L - 5], "seg": [0] * (L - 5)}, enc_new]) is False
    finally: ex.EXT_QUERY_ROWS = rows0
    ex._PAD[0] = False
    try: c3 = check("control_no_gemm_padding", [{"ids": enc_new["ids"][:L - 600], "pos": enc_new["pos"][:L - 600], "seg": [0] * (L - 600)}, enc_new]) is False
    finally: ex._PAD[0] = True
    gate = "PASS" if all(v is True for v in verdicts) else "FAIL"
    emit({"gate": gate, "cases": len(verdicts), "misses": sum(v is None for v in verdicts), "control_no_advance_caught": c1, "control_no_padding_caught": c2, "control_no_gemm_padding_caught": c3})
    progress(f"END gate={gate} controls={c1},{c2},{c3}")


if __name__ == "__main__":
    main()
