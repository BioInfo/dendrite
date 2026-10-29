"""State reuse for kev's hybrid backbones: continue a cached state prefix with only the new tokens, instead of running the
whole state again.

A kev state is [<state>] + the text's tokens at positions 0..Ls-1, so when a new state's tokens start with a cached
state's tokens (an append: a case history, an agent trajectory, a document that grows), every layer's state at the end of
the old tokens is exactly what a full pass would have left there:
- attention layers: the cached keys/values, to which the new tokens' keys/values are appended;
- Gated DeltaNet layers: the conv window and the recurrent state, from which the new tokens continue.
So extend(prefix, new tokens) costs the new tokens only (their attention also reads the cached keys).

A hybrid cache cannot be cropped back (the recurrent state has no per-token history), so reuse needs the cached tokens
to be an exact prefix of the new state's tokens; anything else is a miss and runs in full.

kev's fused DeltaNet forward deliberately leaves a cached state as it was when a pass continues it (question rows never
continue from each other); an extension must advance it, so this module binds an advancing forward to every DeltaNet
mixer, active only inside extend().

Exactness (bit for bit against model.prefix of the whole new state) needs two things, both handled by checkpoint/advance:
- the DeltaNet chunked delta rule works in 64-token chunks, so an extension starts at a multiple of 64 (a checkpoint);
- a pass of a few new tokens makes flash attention pick a split-KV schedule that sums in another order, so extension
  passes pad the attention queries with zero rows to at least EXT_QUERY_ROWS, which keeps the full pass's schedule;
- cuBLAS picks a different GEMM algorithm (split-K) for some row counts, which rounds differently; which counts depends
  only on the weight's shape, so install() calibrates, per projection shape, the row counts whose rows are bit-identical
  to a long pass's, and extension passes pad every projection's input with zero rows to the next such count.

  ck = checkpoint(model, enc)                 # cache at the last 64-aligned point >= SLACK tokens before the state's end
  prefix, ck = advance(model, ck, enc_new)    # enc_new's state prefix + its checkpoint; (None, None) on a miss
  p_new = extend(model, p_old, enc_old, enc_new)   # the raw step (exact only from a 64-aligned start)
"""
import copy, types

import torch
from transformers import AttentionInterface
from transformers.cache_utils import LinearAttentionCacheLayerMixin

import kev.fused_qwen35 as fq
from fla.modules.conv import causal_conv1d
from fla.modules.fused_norm_gate import rms_norm_gated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

ALIGN, SLACK, EXT_QUERY_ROWS, CAL_MAX = 64, 8, 512, 2048
_GOOD = {}          # (N, K) -> sorted row counts <= CAL_MAX whose GEMM rows equal a long pass's
_EXTENDING = [False]
_PAD = [True]       # False disables the GEMM padding (control)
NAME = "kev_extend"
_ADVANCE = [True]   # extend() advances the DeltaNet states; False reproduces kev's non-advancing continuation (selftest)


def _deltanet_extend(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """kev.fused_qwen35.deltanet_forward for a pass that continues a cached state AND advances it (conv window and
    recurrent state written back)."""
    if not getattr(self, "_extending", False):
        return self._kev_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    hidden_states = fq.apply_mask_to_padding_states(hidden_states, attention_mask)
    B, T, _ = hidden_states.shape
    mixed, z, b, a = fq.F.linear(hidden_states, self.in_proj).split(self.splits, -1)
    layer = cache_params.layers[self.layer_idx]
    adv = _ADVANCE[0]
    mixed, conv_state = causal_conv1d(mixed, self.conv_weight, None, initial_state=layer.conv_states[0], output_final_state=adv, activation="silu")
    q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], -1)
    out, recurrent = chunk_gated_delta_rule(
        q.reshape(B, T, -1, self.head_k_dim), k.reshape(B, T, -1, self.head_k_dim), v.reshape(B, T, -1, self.head_v_dim),
        g=a, beta=b, initial_state=layer.recurrent_states[0], output_final_state=adv,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias, use_beta_sigmoid_in_kernel=True)
    if adv:
        layer.conv_states[0] = conv_state.to(layer.conv_states[0].dtype)
        layer.recurrent_states[0] = recurrent.to(layer.recurrent_states[0].dtype)
    out = rms_norm_gated(out, z.reshape(B, T, -1, self.head_v_dim), self.norm.weight, None, activation="swish", eps=self.norm.variance_epsilon)
    return self.out_proj(out.reshape(B, T, -1))


def extend_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query [1, H, n, d] for the n new tokens; key/value [1, kvh, L, d], the whole state after the cache update.
    Causal, aligned to the end (new token i sees keys up to L - n + i). The queries are left-padded with zero rows to
    EXT_QUERY_ROWS so flash attention runs the schedule of the full pass (sdpa with enable_gqa = flash on repeated keys,
    bit for bit). -> ([1, n, H, d], None)."""
    B, H, n, d = query.shape
    L = key.shape[-2]
    g = H // key.shape[1]
    rows = min(max(n, EXT_QUERY_ROWS), L)
    q = query if rows == n else torch.cat([query.new_zeros(B, H, rows - n, d), query], 2)
    out = torch.ops.aten._scaled_dot_product_flash_attention(q, key.repeat_interleave(g, 1), value.repeat_interleave(g, 1), 0.0, True, False,
                                                             scale=scaling)[0]
    return out[:, :, -n:].transpose(1, 2).contiguous(), None


AttentionInterface.register(NAME, extend_attention)


def _exact_linear(fn):
    """fn(x, w) with x's rows zero-padded, during an extension, to the next calibrated row count for w's shape."""
    import bisect

    def linear(x, w, *args, **kw):
        if not (_EXTENDING[0] and _PAD[0]) or not isinstance(w, torch.Tensor):
            return fn(x, w, *args, **kw)
        flat = x.reshape(-1, x.shape[-1]); n = flat.shape[0]
        good = _GOOD.get(tuple(w.shape))
        if good is None or n > CAL_MAX:
            return fn(x, w, *args, **kw)
        m = good[bisect.bisect_left(good, n)] if n <= good[-1] else n
        if m != n:
            flat = torch.cat([flat, flat.new_zeros(m - n, flat.shape[1])])
        return fn(flat, w, *args, **kw)[:n].reshape(*x.shape[:-1], -1)
    return linear


@torch.no_grad()
def calibrate(shapes, device, dtype=torch.bfloat16, ref_rows=4096):
    for N, K in shapes:
        if (N, K) in _GOOD: continue
        w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
        x = torch.randn(ref_rows, K, device=device, dtype=dtype)
        full = torch.nn.functional.linear(x, w)
        _GOOD[(N, K)] = [m for m in range(1, CAL_MAX + 1) if torch.equal(torch.nn.functional.linear(x[-m:], w), full[-m:])]
        assert _GOOD[(N, K)] and _GOOD[(N, K)][-1] == CAL_MAX, f"no exact GEMM row count up to {CAL_MAX} for {(N, K)}"


def install(lm):
    """Bind the advancing forward to every DeltaNet mixer, route every projection GEMM through the exact padding, and
    calibrate the padding (all no-ops outside extend())."""
    shapes = set()
    fq.F = types.SimpleNamespace(**{k: getattr(fq.F, k) for k in dir(fq.F) if not k.startswith("__")}) if not hasattr(fq.F, "_ext") else fq.F
    if not hasattr(fq.F, "_ext"):
        fq.F.linear = _exact_linear(fq.F.linear); fq.F._ext = True
    for mod in lm.modules():
        for name in ("in_proj", "qkv", "gate_up"):
            t = getattr(mod, name, None)
            if isinstance(t, torch.Tensor) and t.dim() == 2: shapes.add(tuple(t.shape))
        if isinstance(mod, torch.nn.Linear) and not hasattr(mod, "_ext"):
            shapes.add(tuple(mod.weight.shape))
            mod._ext = True
            mod.forward = (lambda m: (lambda x: _exact_linear(torch.nn.functional.linear)(x, m.weight, m.bias)))(mod)
    calibrate(shapes, next(lm.parameters()).device)
    n = 0
    for mod in lm.modules():
        if hasattr(mod, "in_proj") and hasattr(mod, "conv_weight") and not hasattr(mod, "_kev_forward"):
            mod._kev_forward = mod.forward
            mod.forward = types.MethodType(_deltanet_extend, mod)
            n += 1
    return n


def clone_cache(cache):
    """A copy of a state cache that an extension can advance without touching the original. Attention layers append by
    concatenation (a new tensor), so their keys/values are shared, not copied; the DeltaNet conv/recurrent states (fixed
    size) are copied."""
    out = copy.copy(cache)
    out.layers = []
    for src in cache.layers:
        dst = copy.copy(src)
        if isinstance(src, LinearAttentionCacheLayerMixin):
            for name in ("conv_states", "recurrent_states"):
                setattr(dst, name, {j: None if t is None else t.clone() for j, t in getattr(src, name).items()})
            for name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state"):
                setattr(dst, name, getattr(src, name).copy())
        out.layers.append(dst)
    return out


def state_ids(enc):
    Ls = enc["seg"].count(0)
    return enc["ids"][:Ls], enc["pos"][:Ls]


@torch.no_grad()
def extend(model, prefix, enc_old, enc_new, inplace=False):
    """-> the state prefix of enc_new (as model.prefix returns it), computed from enc_old's prefix plus the new tokens;
    None when enc_old's state tokens are not a prefix of enc_new's (a miss: run model.prefix(enc_new))."""
    old, _ = state_ids(enc_old)
    new, pos = state_ids(enc_new)
    if len(new) <= len(old) or new[: len(old)] != old:
        return None
    Ls_old, cache, h = prefix
    if h is not None:
        raise NotImplementedError("extend covers hybrid backbones (no state hidden states kept)")
    cache = cache if inplace else clone_cache(cache)
    mixers = [m for m in model.lm.modules() if hasattr(m, "_kev_forward")]
    if not mixers:
        raise RuntimeError("extend.install(model.lm) first")
    ids = torch.tensor([new[len(old):]], device=model.device); p = torch.tensor([pos[len(old):]], device=model.device)
    saved = model.lm.config._attn_implementation
    for m in mixers: m._extending = True
    _EXTENDING[0] = True
    try:
        model.lm.config._attn_implementation = NAME
        model.lm(input_ids=ids, position_ids=p, past_key_values=cache, use_cache=True)
    finally:
        model.lm.config._attn_implementation = saved
        _EXTENDING[0] = False
        for m in mixers: m._extending = False
    return len(new), cache, None


def _cut(enc, n):
    ids, pos = state_ids(enc)
    return {"ids": ids[:n], "pos": pos[:n], "seg": [0] * n}


def _point(n_state):
    return max(0, (n_state - SLACK) // ALIGN * ALIGN)


@torch.no_grad()
def checkpoint(model, enc):
    """-> a checkpoint of enc's state: {"enc": its tokens up to the point, "prefix": the cache there}. The point is the
    last multiple of ALIGN at least SLACK tokens before the state's end (None when the state is shorter than that)."""
    c = _point(len(state_ids(enc)[0]))
    if c == 0:
        return None
    e = _cut(enc, c)
    return {"enc": e, "prefix": model.prefix(e)}


@torch.no_grad()
def advance(model, ck, enc_new):
    """-> (enc_new's state prefix, enc_new's checkpoint), both from ck without re-running ck's tokens: one extension from
    ck's point to enc_new's point (kept as the new checkpoint) and one from there to enc_new's end. (None, None) when ck's
    tokens are not a prefix of enc_new's (a miss)."""
    old = ck["enc"]["ids"]; new = state_ids(enc_new)[0]
    if new[: len(old)] != old:
        return None, None
    c2 = _point(len(new))
    nck = ck
    if c2 > len(old):
        e2 = _cut(enc_new, c2)
        nck = {"enc": e2, "prefix": extend(model, ck["prefix"], ck["enc"], e2)}
    tail = extend(model, nck["prefix"], nck["enc"], enc_new) if len(new) > len(nck["enc"]["ids"]) else nck["prefix"]
    return tail, nck
