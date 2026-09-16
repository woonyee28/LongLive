"""Arm D bisection hooks: a BF16 shadow of the FP4 KV cache, and a switch that
swaps `fp4_attention.attend()` for a reference attention over that shadow.

Why: Arm D's output collapses (VBench imaging_quality 23.6 vs Arm C's 65.5;
latent std 0.54 vs 0.93, |max| pinned at ~2.5) and every prior validation of
the kernel used cosine similarity, which cannot separate "correct" from
"half-contaminated by mean(V)" (fully broken scored ~0.83, "fixed" ~0.98).
These hooks answer the only question that matters first: is the damage done
inside `attend()` at all?

    LLV2_FP4_ATTN_MODE=kernel   (default) production path, no shadow kept
    LLV2_FP4_ATTN_MODE=ref      BF16 attention over the shadow cache replaces
                                attend(); everything else identical. Healthy
                                output here => bug is inside attend().
    LLV2_FP4_ATTN_MODE=ref_rt   as ref, but Q/K/V round-tripped through NVFP4
                                (same scale_mode as production) first. Healthy
                                here but not in `kernel` => the kernel's own
                                math (P quantization, accumulation, addressing)
                                is at fault, not the input quantization.
    LLV2_FP4_ATTN_MODE=split    FP4 kernel over the PAST blocks only, BF16
                                FlashAttention over the current block (which
                                carries ~68% of the softmax mass and which
                                Arm C also attends in BF16), merged with the
                                two log-sum-exps. Emulated accuracy: attention
                                rel_l2 0.19 -> 0.05 vs exact.
    LLV2_FP4_ATTN_CHECK=1       run both kernel and ref every call and append
                                norm ratio / rel_l2 / cosine per call to
                                LLV2_FP4_ATTN_CHECK_LOG (or stderr).

The shadow holds exactly what the FP4 cache holds -- already RoPE'd,
already k_smooth'd K, and V -- and is rolled/inserted with the same token
ranges, so `ref` differs from production in nothing but the attention call.
"""

from __future__ import annotations

import os
import sys

import torch

MODE = os.environ.get("LLV2_FP4_ATTN_MODE", "kernel")
# LLV2_SINK_BF16=1: keep a BF16 copy of the sink block (the first block, pinned for the
# whole video) and attend it in BF16 through cuDNN instead of reading its NVFP4 pages;
# the FP4 kernel is pointed past the sink with a page-table view. Works in `split`
# (three-way LSE merge: sink | past | current) and in `kernel` (two-way: sink | rest).
SINK_BF16 = os.environ.get("LLV2_SINK_BF16", "0") == "1"
CHECK = os.environ.get("LLV2_FP4_ATTN_CHECK", "0") == "1"
CHECK_LOG = os.environ.get("LLV2_FP4_ATTN_CHECK_LOG", "")
if MODE not in ("kernel", "ref", "ref_rt", "split"):
    raise ValueError(f"LLV2_FP4_ATTN_MODE must be kernel|ref|ref_rt|split, got {MODE!r}")

_NEED_SHADOW = MODE in ("ref", "ref_rt") or CHECK
_call_index = 0
_log_fh = None
_dumped = False


def _log(line: str) -> None:
    global _log_fh
    if CHECK_LOG:
        if _log_fh is None:
            _log_fh = open(CHECK_LOG, "a")
        _log_fh.write(line + "\n")
        _log_fh.flush()
    else:
        print(line, file=sys.stderr, flush=True)


def on_insert(kv_cache: dict, start: int, end: int, k_sm: torch.Tensor, v: torch.Tensor) -> None:
    """Mirror `fp4_attention.insert_block`. k_sm, v: [S, H, D], S == end - start."""
    if MODE == "split":
        kv_cache["cur_k"], kv_cache["cur_v"], kv_cache["cur_len"] = k_sm, v, end - start
    if SINK_BF16 and start == 0:
        # Only the first block ever lands at token 0 (the roll pins it), and its last
        # insert is the clean t=0 forward -- exactly what the FP4 pages hold afterwards.
        kv_cache["sink_k"], kv_cache["sink_v"], kv_cache["sink_len"] = k_sm, v, end - start
    if not _NEED_SHADOW:
        return
    if "shadow_k" not in kv_cache:
        n = int(kv_cache["max_blocks"]) * int(kv_cache["block_token_size"])
        kv_cache["shadow_k"] = torch.zeros(n, *k_sm.shape[1:], dtype=k_sm.dtype, device=k_sm.device)
        kv_cache["shadow_v"] = torch.zeros(n, *v.shape[1:], dtype=v.dtype, device=v.device)
    assert k_sm.shape[0] == end - start, (k_sm.shape, start, end)
    kv_cache["shadow_k"][start:end].copy_(k_sm)
    kv_cache["shadow_v"][start:end].copy_(v)


def on_roll(kv_cache: dict, sink_tok: int, evict_tok: int, roll_tok: int) -> None:
    """Mirror `fp4_attention.roll_blocks`, in tokens. Ranges may overlap, so clone src first."""
    if not _NEED_SHADOW or "shadow_k" not in kv_cache:
        return
    for name in ("shadow_k", "shadow_v"):
        s = kv_cache[name]
        src = s[sink_tok + evict_tok : sink_tok + evict_tok + roll_tok].clone()
        s[sink_tok : sink_tok + roll_tok].copy_(src)


def _round_trip(x: torch.Tensor) -> torch.Tensor:
    from nvfp4_decode_kernel.reference import nvfp4_round_trip
    from utils.fp4_attention import _scale_mode

    d = x.shape[-1]
    y = nvfp4_round_trip(x.float().reshape(-1, d), scale_mode=_scale_mode())
    return y.reshape(x.shape).to(x.dtype)


def attend_or_ref(
    kv_cache: dict,
    roped_query: torch.Tensor,   # [1, S, H, D]
    seqused_fp4: torch.Tensor,   # [1] int32, valid tokens
    softmax_scale: float,
    heads: int,
    attend_fn,
    past_tokens: int | None = None,
    total_tokens: int | None = None,
    sink_tokens: int | None = None,
) -> torch.Tensor:               # [1, S, H, D]
    """Production `attend()` unless a debug mode/check is on.

    `past_tokens` / `total_tokens` are the caller's own Python ints (start and
    end of the current block in the window). Split mode uses them so it never
    has to read `seqused_fp4` back from the device: a `.item()` there is a
    stream sync per attention call, 5,760 per video, and each one drains the
    launch queue -- measured as ~+4 ms per call on top of the kernels.
    """
    global _call_index
    from wan_5b.modules.attention import attention

    def kernel_out() -> torch.Tensor:
        """The production output for the active mode (kernel / kernel+sink / split[+sink])."""
        if MODE == "split":
            return attend_split(kv_cache, roped_query, seqused_fp4, softmax_scale, heads, attend_fn,
                                past_tokens=past_tokens, total_tokens=total_tokens, sink_tokens=sink_tokens)
        if SINK_BF16:
            return attend_sink_kernel(kv_cache, roped_query, softmax_scale, heads, attend_fn,
                                      total_tokens=total_tokens, sink_tokens=sink_tokens)
        return attend_fn(
            roped_query.squeeze(0), kv_cache["fp4_cache"], seqused_fp4, softmax_scale, heads
        ).unsqueeze(0)

    if MODE in ("kernel", "split") and not CHECK:
        return kernel_out()

    n = int(seqused_fp4.item())
    shadow_k = kv_cache["shadow_k"][:n].unsqueeze(0)
    shadow_v = kv_cache["shadow_v"][:n].unsqueeze(0)

    if MODE == "ref_rt":
        ref_out = attention(_round_trip(roped_query), _round_trip(shadow_k), _round_trip(shadow_v))
    else:
        ref_out = attention(roped_query, shadow_k, shadow_v)

    if CHECK:
        k_out = kernel_out()
        kf, rf = k_out.float(), ref_out.float()
        rn = rf.norm()
        norm_ratio = (kf.norm() / rn).item() if rn > 0 else float("nan")
        rel_l2 = ((kf - rf).norm() / rn).item() if rn > 0 else float("nan")
        cos = torch.nn.functional.cosine_similarity(kf.flatten(), rf.flatten(), dim=0).item()
        # How much of the kernel's output is explained by mean(V) over the window
        mean_v = shadow_v.float().mean(dim=1, keepdim=True)          # [1,1,H,D]
        r_dist = (rf - mean_v).norm()
        k_dist = (kf - mean_v).norm()
        _log(
            f"call={_call_index} n_tok={n} norm_ratio={norm_ratio:.4f} rel_l2={rel_l2:.4f} "
            f"cos={cos:.4f} dist_from_meanV_ratio={(k_dist / r_dist).item():.4f}"
        )
        # One real full-window call, for offline kernel-boundary experiments.
        global _dumped
        n_max = int(kv_cache["max_blocks"]) * int(kv_cache["block_token_size"])
        if not _dumped and n == n_max and CHECK_LOG:
            dump_path = os.path.join(os.path.dirname(CHECK_LOG), "dump_full_window.pt")
            torch.save(
                {
                    "q": roped_query.squeeze(0).detach().cpu(),
                    "k": kv_cache["shadow_k"][:n].detach().cpu(),
                    "v": kv_cache["shadow_v"][:n].detach().cpu(),
                    "kernel_out": k_out.squeeze(0).detach().cpu(),
                    "ref_out": ref_out.squeeze(0).detach().cpu(),
                    "softmax_scale": softmax_scale,
                    "call": _call_index,
                },
                dump_path,
            )
            _log(f"dumped full-window call {_call_index} -> {dump_path}")
            _dumped = True
        _call_index += 1
        if MODE in ("kernel", "split"):
            return k_out
    return ref_out.to(roped_query.dtype)


_seqused_cache: dict[tuple[int, int], torch.Tensor] = {}


def _seqused_for(n: int, device: torch.device) -> torch.Tensor:
    key = (device.index, n)
    t = _seqused_cache.get(key)
    if t is None:
        t = torch.tensor([n], dtype=torch.int32, device=device)
        _seqused_cache[key] = t
    return t


def attend_split(kv_cache, roped_query, seqused_fp4, softmax_scale, heads, attend_fn,
                 past_tokens=None, total_tokens=None, sink_tokens=None):
    """FP4 kernel over the past blocks + BF16 attention over the current block, LSE-merged.
    The current block is the last `cur_len` tokens of the valid window and was inserted into the
    FP4 cache just before this call; the kernel is simply asked for one block less. Fully
    asynchronous: no device reads, no per-call allocations beyond the outputs.
    With LLV2_SINK_BF16=1 the sink block is also taken out of the kernel's range (page-table
    view starting after it) and attended in BF16 from the copy kept by `on_insert`, giving a
    three-way merge sink | past | current. All three are exact partitions of the same softmax.
    """
    if past_tokens is None:                                               # only when called outside the model
        past_tokens = int(seqused_fp4.item()) - int(kv_cache["cur_len"])
    q = roped_query.squeeze(0)                                            # [S,H,D]
    o2, l2 = _bf16_attention_with_lse(roped_query, kv_cache["cur_k"].unsqueeze(0),
                                      kv_cache["cur_v"].unsqueeze(0), softmax_scale)   # o2 [S,H,D], l2 [S,H,1]
    if past_tokens <= 0:
        return o2.unsqueeze(0)
    if not (SINK_BF16 and "sink_k" in kv_cache):
        o1, l1 = attend_fn(q, kv_cache["fp4_cache"], _seqused_for(past_tokens, q.device),
                           softmax_scale, heads, return_lse=True)         # o1 [S,H,D], l1 [H,S]
        # merge weights are tiny ([S,H,1]); the two big tensors are touched once each
        w1 = torch.sigmoid(l1.t().unsqueeze(-1) - l2)                     # = e^l1 / (e^l1 + e^l2), stable
        out = torch.lerp(o2, o1, w1.to(o1.dtype))                         # o2 + w1*(o1 - o2), one fused pass
        return out.unsqueeze(0)
    parts = _sink_parts(kv_cache, roped_query, softmax_scale, heads, attend_fn,
                        past_tokens, sink_tokens)
    parts.append((o2, l2))
    return _merge_lse(parts).unsqueeze(0)


def attend_sink_kernel(kv_cache, roped_query, softmax_scale, heads, attend_fn,
                       total_tokens=None, sink_tokens=None):
    """Pure-FP4 path (LLV2_FP4_ATTN_MODE=kernel) with only the sink in BF16: two-way merge."""
    q = roped_query.squeeze(0)
    if "sink_k" not in kv_cache or total_tokens is None:
        return attend_fn(q, kv_cache["fp4_cache"], _seqused_for(total_tokens, q.device),
                         softmax_scale, heads).unsqueeze(0)
    parts = _sink_parts(kv_cache, roped_query, softmax_scale, heads, attend_fn,
                        total_tokens, sink_tokens)
    return _merge_lse(parts).unsqueeze(0)


def _sink_parts(kv_cache, roped_query, softmax_scale, heads, attend_fn, upto_tokens, sink_tokens):
    """[(o, lse)] for the BF16 sink and, if any, the FP4 pages between the sink and `upto_tokens`."""
    from utils.fp4_attention import PAGE_SIZE
    q = roped_query.squeeze(0)
    s = int(kv_cache["sink_len"])
    if sink_tokens is not None:
        assert s == sink_tokens, f"BF16 sink copy holds {s} tokens but the model pins {sink_tokens}"
    n_sink = min(s, upto_tokens)
    o_s, l_s = _bf16_attention_with_lse(roped_query, kv_cache["sink_k"][:n_sink].unsqueeze(0),
                                        kv_cache["sink_v"][:n_sink].unsqueeze(0), softmax_scale)
    parts = [(o_s, l_s)]
    mid = upto_tokens - s
    if mid > 0:
        cache = kv_cache["fp4_cache"]
        o1, l1 = attend_fn(q, cache, _seqused_for(mid, q.device), softmax_scale, heads,
                           return_lse=True, page_table=_page_table_from(cache, s // PAGE_SIZE))
        parts.append((o1, l1.t().unsqueeze(-1)))
    return parts


def _merge_lse(parts):
    """Exact softmax merge of attention outputs over disjoint key sets: (o_i [S,H,D], lse_i [S,H,1]).
    Chained pairwise lerps: each step is one fused pass over the big BF16 tensors, and the
    running log-sum-exp is a tiny [S,H,1] logaddexp. Three parts cost two lerps (~0.1 ms).
    """
    o, l = parts[0]
    for o_i, l_i in parts[1:]:
        w = torch.sigmoid(l - l_i)                                        # e^l / (e^l + e^l_i), stable
        o = torch.lerp(o_i, o, w.to(o.dtype))                             # o_i + w*(o - o_i)
        l = torch.logaddexp(l, l_i)
    return o


_page_table_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _page_table_from(cache, first_page: int) -> torch.Tensor:
    """Identity page table shifted to start at `first_page` (entries clamped to the last page)."""
    key = (cache.page_table.device.index, id(cache), first_page)
    t = _page_table_cache.get(key)
    if t is None:
        n = cache.total_pages
        t = torch.clamp(torch.arange(first_page, first_page + n, device=cache.page_table.device),
                        max=n - 1).to(torch.int32).reshape(1, n)
        _page_table_cache[key] = t
    return t


BF16_BACKEND = os.environ.get("LLV2_SPLIT_BF16_BACKEND", "cudnn")   # cudnn | fa2


def _bf16_attention_with_lse(q, k, v, softmax_scale):
    """BF16 attention over the current block, returning (out [S,H,D], lse [S,H,1] natural log).

    cuDNN's SDPA is Blackwell-native and measured 3.3x faster than FA2 at this
    shape (0.43 vs 1.42 ms for 7040x7040x24x128); FA2 is kept as a fallback.
    """
    if BF16_BACKEND == "cudnn":
        # aten op wants (B,H,S,D); returns out (B,H,S,D) and lse (B,H,S,1) in natural log
        r = torch.ops.aten._scaled_dot_product_cudnn_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, True, 0.0, False, False,
            scale=softmax_scale,
        )
        return r[0][0].transpose(0, 1), r[1][0].permute(1, 0, 2)          # [S,H,D], [S,H,1]
    from flash_attn import flash_attn_func

    o, l, _ = flash_attn_func(q, k, v, softmax_scale=softmax_scale, return_attn_probs=True)
    return o[0], l[0].t().unsqueeze(-1)
