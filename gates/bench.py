"""Branch-phase benchmark on one cached state: kev default vs kev one-pass vs fork, over a (state x questions) grid.

Arms (all start from the same model.prefix(enc); the state pass is identical across arms and reported once):
  kev       DecisionModel._branch_rows_from_prefix as shipped (ROW_PASS_TOKENS budget: replicated state, rows in chunks)
  kev_1pass the same with the pass budget lifted so every row runs in one pass (still one state copy per row)
  fork      forkattn.fork_probs (one shared state copy, all rows in one pass)
  kev_graphs DecisionModel.probs_batch with kev's CUDA graphs (its CUDA serving path; states <= 4,096 only), graphs
            captured before timing. Needs --fused (kev.serve runs graphs on the fused layers).
--fused loads kev's fused Triton layers (kev.serve's CUDA default) for every arm.
Metrics: median ms over --reps after one warmup, peak extra memory over the resident model + prefix (GiB).
Writes one JSON line per (arm, state, questions) to --out and appends a progress line per cell.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import dendrite.gpucap  # noqa: F401  (must precede any CUDA allocation)
from dendrite import fork as forkattn
import kev.model as km
from records import make_record
from kev.checkpoint import LoadOptions, load

RUN = "jaredpalmer/kev-4b@v1.0"
DEFAULT_BUDGET = km.rows_per_pass.__defaults__


def timed(fn, reps):
    fn(); torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000, (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="1024,2048,4096,8192,16384")
    ap.add_argument("--questions", default="1,8,16,32,64")
    ap.add_argument("--arms", default="kev,kev_1pass,fork")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--out", default="results/bench.jsonl")
    ap.add_argument("--run", default=RUN)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(exist_ok=True)
    tok, model = load(a.run, "cuda", LoadOptions(dtype=torch.bfloat16, fused=a.fused, cuda_graphs=a.fused))
    log = open(Path("progress.md"), "a")
    for S in map(int, a.states.split(",")):
        for Q in map(int, a.questions.split(",")):
            rec = make_record(tok, S, Q)
            enc = model.encode(tok, rec, max_state=km.SERVE_MAX_STATE, max_branch=km.SERVE_MAX_BRANCH)
            torch.cuda.synchronize(); t = time.perf_counter(); prefix = model.prefix(enc); torch.cuda.synchronize()
            state_ms = (time.perf_counter() - t) * 1000
            for arm in a.arms.split(","):
                if arm == "kev":
                    km.rows_per_pass.__defaults__ = DEFAULT_BUDGET
                    fn = lambda: model._branch_rows_from_prefix(enc, prefix[1])
                elif arm == "kev_1pass":
                    km.rows_per_pass.__defaults__ = (10**9,)
                    fn = lambda: model._branch_rows_from_prefix(enc, prefix[1])
                elif arm == "kev_graphs":
                    rows = km.rows_of(enc)[2]
                    if model.graphs is None or not model.graphs.admits(len(km.rows_of(enc)[0]), [len(r["ids"]) for r in rows], True):
                        continue
                    fn = lambda: model.probs_batch([enc], [prefix], [False])[0][0]
                    fn(); model.graphs.capture_pending()
                else:
                    fn = lambda: forkattn.fork_probs(model, enc, prefix)
                try:
                    ms, gib = timed(fn, a.reps); err = None
                except torch.OutOfMemoryError:
                    ms, gib, err = None, None, "OOM"
                finally:
                    km.rows_per_pass.__defaults__ = DEFAULT_BUDGET
                    torch.cuda.empty_cache()
                row = {"arm": arm + ("+fused" if a.fused else ""), "state": enc["state_tokens"], "questions": Q, "branch_ms": ms and round(ms, 1),
                       "q_per_s": ms and round(Q / ms * 1000, 1), "peak_extra_gib": gib and round(gib, 2),
                       "state_pass_ms": round(state_ms, 1), "error": err}
                print(json.dumps(row), flush=True)
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
                log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} bench {arm} S={S} Q={Q} ms={row['branch_ms']} gib={row['peak_extra_gib']}\n"); log.flush()
            del prefix; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
