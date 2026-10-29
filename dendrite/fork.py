"""Copy-free question forks for kev's hybrid (Qwen3.5) backbones.

kev serves a record as its state pass, then one causal row per question continuing the cached state. Its row pass
replicates the state cache once per row (reorder_cache(zeros(N)) in DecisionModel._rows_hidden), so N questions hold N
copies of the state's attention keys/values. Here the attention layers keep ONE copy:

- a row's queries attend to the shared state (every state token visible, no mask) and to their own branch (causal),
  as two flash-attention calls, merged exactly by their log-sum-exp (cascade attention);
- the state call folds every row and every query head of a KV group into one sequence, so it reads the state's keys
  once per KV head, not once per row;
- only the DeltaNet conv/recurrent states (fixed size, independent of the state length) are copied per row.

Exact up to floating-point reassociation. kev itself is not modified.
"""
import copy

import torch
from transformers import AttentionInterface
from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin

NAME = "kev_fork"
_PREFIX = {}   # layer_idx -> (keys, values) [1, kv_heads, S, d] of the shared state, set by fork_rows for one pass


def _flash(q, k, v, causal, scale):
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, causal, False, scale=scale)[:2]
    return out, lse


def fork_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query [R, H, Lb, d]; key/value [R, kvh, Lb, d] = the rows' own branch keys (ForkLayer.update returns only those).
    -> ([R, Lb, H, d], None), the layout transformers' attention layers expect."""
    pk, pv = _PREFIX[module.layer_idx]
    R, H, Lb, d = query.shape
    kvh = pk.shape[1]; g = H // kvh
    # state part: fold rows and the g query heads of each KV head into the sequence axis; no mask, so this is exact
    qs = query.view(R, kvh, g, Lb, d).permute(1, 0, 2, 3, 4).reshape(1, kvh, R * g * Lb, d)
    o_s, lse_s = _flash(qs, pk, pv, False, scaling)
    o_s = o_s.view(kvh, R, g, Lb, d).permute(1, 0, 2, 3, 4).reshape(R, H, Lb, d)
    lse_s = lse_s.view(kvh, R, g, Lb).permute(1, 0, 2, 3).reshape(R, H, Lb)
    # branch part: each row's own keys, causal (rows are right-padded, so pads never reach a real query)
    o_b, lse_b = _flash(query, key.repeat_interleave(g, 1), value.repeat_interleave(g, 1), True, scaling)
    m = torch.maximum(lse_s, lse_b)
    w_s, w_b = torch.exp(lse_s - m).unsqueeze(-1), torch.exp(lse_b - m).unsqueeze(-1)
    out = (o_s.float() * w_s + o_b.float() * w_b) / (w_s + w_b)
    return out.to(query.dtype).transpose(1, 2).contiguous(), None


AttentionInterface.register(NAME, fork_attention)


class ForkLayer(DynamicLayer):
    """An attention layer of the fork cache: the state's keys/values live in _PREFIX (one copy); update() keeps and
    returns only the branch keys/values of this pass."""

    def __init__(self, state_len):
        super().__init__()
        self.state_len = state_len

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized: self.lazy_initialization(key_states, value_states)
        self.keys, self.values = key_states, value_states
        return key_states, value_states

    def get_seq_length(self):
        return self.state_len


def fork_cache(cache, rows):
    """A cache for `rows` question rows continuing kev's state cache `cache` (batch 1). Attention layers: the state's
    keys/values registered once in _PREFIX. DeltaNet layers: their conv/recurrent states expanded to `rows` (a copy;
    fixed size). The caller's cache is left untouched."""
    out = copy.copy(cache)
    out.layers = []
    _PREFIX.clear()
    for i, src in enumerate(cache.layers):
        if isinstance(src, LinearAttentionCacheLayerMixin):
            dst = copy.copy(src)
            for name in ("conv_states", "recurrent_states"):
                states = getattr(src, name)
                setattr(dst, name, {j: None if t is None else t.expand(rows, *t.shape[1:]).contiguous() for j, t in states.items()})
            for name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
                setattr(dst, name, getattr(src, name).copy())
        else:
            _PREFIX[i] = (src.keys, src.values)
            dst = ForkLayer(src.keys.shape[-2])
        out.layers.append(dst)
    return out


@torch.no_grad()
def fork_rows_hidden(model, rows, cache, prefix_len, row_tokens=None):
    """DecisionModel._rows_hidden(rows, cache, prefix_len) with the state shared instead of replicated: one pass holds as
    many rows as fit `row_tokens` branch tokens (None = all of them). -> one [L_i, d] fp32 tensor per row."""
    lm = model.lm
    chunk = len(rows) if row_tokens is None else max(1, row_tokens // max(len(ids) for ids, _ in rows))
    saved = lm.config._attn_implementation
    out = []
    try:
        lm.config._attn_implementation = NAME
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, _ = model._pad_rows(part)
            past = fork_cache(cache, len(part))
            h = lm(input_ids=ids, position_ids=pos, attention_mask=None, past_key_values=past, use_cache=True).last_hidden_state.float()
            out += [h[i, : len(row_ids)] for i, (row_ids, _) in enumerate(part)]
    finally:
        lm.config._attn_implementation = saved
        _PREFIX.clear()
    return out


@torch.no_grad()
def fork_probs(model, enc, prefix, row_tokens=None):
    """DecisionModel._branch_rows_from_prefix with fork_rows_hidden: per-question probabilities from a cached state."""
    import torch.nn.functional as F
    from kev.model import rows_of
    Ls, cache, _ = prefix
    S, _, rows = rows_of(enc)
    hs = fork_rows_hidden(model, [(r["ids"], r["pos"]) for r in rows], cache, len(S), row_tokens)
    ps = [F.softmax(model.head(h[r["decide"]], h[torch.tensor(r["opts"], device=model.device)]), -1) for h, r in zip(hs, rows)]
    return list(torch.cat(ps).cpu().split([len(p) for p in ps]))
