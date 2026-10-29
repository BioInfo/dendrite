"""Dendrite serving: kev.serve with exact state reuse across requests.

kev's PrefixCache reuses a state only when a request's state tokens equal a cached state's. A state that grows between
requests (a case history, an agent trajectory, a document) misses every time and runs in full. This module keeps, beside
kev's cache, a checkpoint of each long state (extend.checkpoint: the cache at a 64-aligned point just before the state's
end) and, when a request's state starts with a checkpoint's tokens, builds its state prefix with extend.advance instead
of a full state pass. The prefix is bit-identical to model.prefix of the whole state (gates/extend_gate.py), so the answers are
the ones kev itself gives; the request then runs kev's own branch passes on it.

Only states kev runs through the eager state pass are handled (longer than the graphed state pass, GRAPH_STATE tokens):
there kev's cold path is model.prefix, the reference gates/extend_gate.py checks against. Shorter states, and every exact hit, go
to kev unchanged.

A checkpoint's attention keys/values are views into the state prefix built with it (the extension appends by
concatenation, so they are the same values), so a checkpoint costs its DeltaNet states (~50 MB on Kev-9B), not a second
copy of the state's KV, as long as that prefix is held anyway.

  python -m dendrite.serve --run jaredpalmer/kev-9b --port 8008      # kev.serve's API plus GET /v1/dendrite
"""
import os
from dataclasses import dataclass, field

from transformers.cache_utils import LinearAttentionCacheLayerMixin

from . import extend as ex
import kev.serve as ks
from kev.device import out_of_memory
from kev.model import EAGER_STATES, rows_of

CHECKPOINTS = int(os.environ.get("DENDRITE_CHECKPOINTS", "8"))                    # checkpoints kept, least recently used evicted
CHECKPOINT_TOKENS = int(os.environ.get("DENDRITE_CHECKPOINT_TOKENS", "131072"))  # state tokens they cover in all


def _share_kv(ck_prefix, prefix):
    """Point a checkpoint's attention keys/values at the leading positions of the state prefix extended from it."""
    n = ck_prefix[0]
    for a, b in zip(ck_prefix[1].layers, prefix[1].layers):
        if not isinstance(a, LinearAttentionCacheLayerMixin):
            a.keys, a.values = b.keys[..., :n, :], b.values[..., :n, :]


@dataclass
class Checkpoints:
    """State checkpoints, least recently used first: tuple(checkpoint token ids) -> extend checkpoint."""
    size: int = CHECKPOINTS
    max_tokens: int = CHECKPOINT_TOKENS
    entries: dict = field(default_factory=dict)
    extended: int = 0     # requests whose state prefix was advanced from a checkpoint
    cold: int = 0         # eligible requests with no checkpoint to continue (checkpoint + tail)
    skipped: int = 0      # misses left to kev: states its graphed state pass takes
    saved_tokens: int = 0  # state tokens not re-run thanks to a checkpoint
    oom_clears: int = 0

    def find(self, ids):
        """The longest checkpoint whose tokens start `ids`."""
        best = None
        for key, ck in self.entries.items():
            if len(key) < len(ids) and (best is None or len(key) > len(best[0])) and ids[: len(key)] == list(key):
                best = key, ck
        if best is None: return None
        self.entries[best[0]] = self.entries.pop(best[0])   # most recent
        return best[1]

    def add(self, ck):
        key = tuple(ck["enc"]["ids"])
        self.entries.pop(key, None); self.entries[key] = ck
        while len(self.entries) > self.size or sum(map(len, self.entries)) > self.max_tokens:
            self.entries.pop(next(iter(self.entries)))

    def stats(self):
        return {"checkpoints": len(self.entries), "checkpoint_tokens": sum(map(len, self.entries)), "size": self.size, "max_tokens": self.max_tokens,
                "extended": self.extended, "cold": self.cold, "skipped": self.skipped, "saved_tokens": self.saved_tokens, "oom_clears": self.oom_clears}


def eligible(model, enc):
    """Whether kev would run this request's state through the eager state pass (model.prefix)."""
    if not model.hybrid: return False
    graphs = getattr(model, "graphs", None)
    if graphs is None: return True
    S, _, rows = rows_of(enc)
    return not graphs.admits(len(S), [len(r["ids"]) for r in rows], False)


def state_prefix(model, store, enc):
    """-> enc's state prefix (Ls, cache, None) from a checkpoint when one continues it, else checkpoint + tail; the new
    checkpoint is stored either way."""
    ids = ex.state_ids(enc)[0]
    ck = store.find(ids)
    if ck is not None:
        prefix, nck = ex.advance(model, ck, enc)
        store.extended += 1; store.saved_tokens += len(ck["enc"]["ids"])
    else:
        nck = ex.checkpoint(model, enc)
        if nck is None: return model.prefix(enc)
        prefix = ex.extend(model, nck["prefix"], nck["enc"], enc) if len(ids) > len(nck["enc"]["ids"]) else nck["prefix"]
        store.cold += 1
    if nck is not ck:
        if prefix is not nck["prefix"]: _share_kv(nck["prefix"], prefix)
        store.add(nck)
    return prefix


def install(server, store=None):
    """Route server.model.probs_batch through the checkpoints. -> the Checkpoints store. Answers are kev's: a request
    gets the state prefix kev's own state pass would build, bit for bit, and kev's branch passes on it."""
    model = server.model
    ex.install(model.lm)
    server.checkpoints = store or Checkpoints()
    batch = model.probs_batch

    def probs_batch(encs, prefixes, keep):
        store = server.checkpoints   # read per batch, so a test can swap it
        prefixes = list(prefixes)
        todo = [i for i, p in enumerate(prefixes) if p is None and eligible(model, encs[i])]
        store.skipped += sum(p is None for p in prefixes) - len(todo)
        ps, out = [None] * len(encs), [None] * len(encs)
        # at most EAGER_STATES built prefixes held at once, as kev's own batched runs do
        groups = [todo[j:j + EAGER_STATES] for j in range(0, len(todo), EAGER_STATES)] or [[]]
        rest = [i for i in range(len(encs)) if i not in todo]
        try:
            for g, idx in enumerate(groups):
                for i in idx: prefixes[i] = state_prefix(model, store, encs[i])
                run = idx + (rest if g == 0 else [])
                if not run: continue
                p, o = batch([encs[i] for i in run], [prefixes[i] for i in run], [keep[i] for i in run])
                for i, pi, oi in zip(run, p, o):
                    ps[i], out[i] = pi, (prefixes[i] if keep[i] else None) if i in idx else oi
                for i in idx:
                    if not keep[i]: prefixes[i] = None   # release it before the next group's prefixes are built
        except Exception as e:
            if out_of_memory(e) and store.entries:
                store.entries.clear(); store.oom_clears += 1
            raise
        return ps, out

    model.probs_batch = probs_batch
    model._kev_probs_batch = batch
    return server.checkpoints


@ks.app.get("/v1/dendrite")
def dendrite_stats():
    s = ks.server()
    return {"checkpoints": s.checkpoints.stats(), "prefix_cache": {"hits": s.prefix_cache.hits, "misses": s.prefix_cache.misses}}


def main():
    Server = ks.Server
    class DendriteServer(Server):
        def __post_init__(self):
            super().__post_init__(); install(self)
    ks.Server = DendriteServer
    ks.main()


if __name__ == "__main__":
    main()
