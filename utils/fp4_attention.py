"""Arm D: NVFP4 tensor-core self-attention for LongLive, via a hand-built
`FP4DecodeKernel(is_varlen_q=True, seqlen_q_static_one=False, pack_gqa=False)`.

Design (matches LongLive's causal_model.py KV-cache layout exactly):
  - LongLive's self-attention is MHA (heads_q == heads_kv == 24, head_dim=128)
    with absolute RoPE (K is cached *after* RoPE, never re-roped -- see
    `use_relative_rope` in causal_model.py, default False and unset by every
    arm config so far). This lets K/V be quantized once, at insertion, rather
    than every attention call.
  - Every ring-buffer "block" LongLive ever uses (checked against every arm
    config) is `num_frame_per_block * frame_seq_length` tokens, and every
    config's `local_attn_size * frame_seq_length` (the KV cache capacity) is
    an exact multiple of BitDecoding's fixed PAGE_SIZE=128. armB/C/D's actual
    numbers: 7040-token blocks = 55 pages/block, 28160-token window = 220
    pages across 4 blocks. Pages within a block are physically contiguous and
    in temporal order, so "roll" is a page-range `.copy_()` between block
    slots, and "insert" is a fresh `quantize_key_pages`/`quantize_value_pages`
    call on just the arriving block (55 pages), copied into that block's
    slot -- never a full-cache dequant/requant.
"""

from __future__ import annotations

import atexit
import os
import sys
from dataclasses import dataclass

_BITDECODING_SRC = os.environ.get(
    "BITDECODING_FP4_SRC", "/workspace/Blackwell-NVFP4/BitDecoding-FP4/src"
)
if _BITDECODING_SRC not in sys.path:
    sys.path.insert(0, _BITDECODING_SRC)

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack, make_ptr

from nvfp4_decode_kernel import _quantize
from nvfp4_decode_kernel._decode import _page_scales_for_kernel, _page_stride_bytes
from nvfp4_decode_kernel.fp4_decode_kernel import FP4DecodeKernel
from nvfp4_decode_kernel.quantize_kv_kernel import _allocate_scales
from nvfp4_vllm.rotation import hadamard

PAGE_SIZE = 128
HEAD_DIM = 128
_prefill_compile_cache: dict[tuple[int, int, int, int, bool, bool], object] = {}
_rotation_cache: dict[torch.device, torch.Tensor] = {}
# K centering (LLV2_KEY_SHIFT): loaded shift per (device, k_smooth), [layers, heads, head_dim] bf16
_key_shift_cache: dict[tuple[torch.device, bool], torch.Tensor] = {}
# K centering calibration (LLV2_KEY_SHIFT_CALIBRATE): float64 sum of raw post-RoPE K per layer, token count
_key_shift_sums: dict[int, torch.Tensor] = {}
_key_shift_tokens: dict[int, int] = {}
# LLV2_HESSIAN_FEEDBACK factors per device: [layers, heads, 2, 64, 64] fp32
_feedback_factor_cache: dict[torch.device, torch.Tensor] = {}
# LLV2_HESSIAN_CALIBRATE state: sum of q q^T per layer, token counts, sampled K ring
_query_moment_sums: dict[int, torch.Tensor] = {}
_query_moment_tokens: dict[int, int] = {}
_feedback_key_ring: dict[int, torch.Tensor] = {}
_feedback_key_seen: dict[int, int] = {}
_FEEDBACK_KEY_RING = 512
_FEEDBACK_KEY_PER_CALL = 8     # evenly spaced: no RNG, so the noise draws do not move


def _query_terms() -> int:
    """How many NVFP4 terms Q is being quantised.
    """
    value = int(os.environ.get("LLV2_Q_TERMS", "2"))
    if value not in (1, 2):
        raise ValueError(f"LLV2_Q_TERMS must be 1 or 2, got {value}")
    return value


def _scale_mode() -> str:
    """Which NVFP4 block-scale candidate set Arm D's Q/K/V quantizers use.

    In our code, we offer `amax6`` - the single fixed ``amax/6`` rule, and
    ``four_six_sixhalf`` also tries 4 and 6.5 per block and keeps whichever
    costs least in mse.
    """
    return os.environ.get("LLV2_SCALE_MODE", "four_six_sixhalf")


def _p_scale_search() -> bool:
    """Per-group {6, 2} candidate search for P's (post-softmax) block scale.
    Off by default due to performance issue, havent really optimise it yet.
    """
    return os.environ.get("LLV2_P_SCALE_SEARCH", "0") == "1"


def _p_scale_mode() -> str:
    """P group scale rule: ``amax6`` (default) or ``code10`` (amax/2 when amax <= 21/1024)."""
    mode = os.environ.get("LLV2_P_SCALE_MODE", "amax6")
    if mode not in ("amax6", "code10"):
        raise ValueError(f"LLV2_P_SCALE_MODE must be amax6 or code10, got {mode!r}")
    if mode == "code10" and _p_scale_search():
        raise ValueError("LLV2_P_SCALE_MODE=code10 and LLV2_P_SCALE_SEARCH=1 are exclusive")
    return mode


def _p_scale_from_scores() -> bool:
    """Derive P's block scales from the raw score maxima (kernel `p_scale_from_scores`)."""
    return os.environ.get("LLV2_P_SCALE_FROM_SCORES", "0") == "1"


def _p_sp1() -> bool:
    """Pre-scale P by 448*6 before its per-group E4M3 block scale is formed
    (the kernel's `compute_sp1`). OFF, and do not turn it on here.
    """
    return os.environ.get("LLV2_P_SP1", "0") == "1"


def _rotation() -> str:
    """``none`` or ``hadamard``: orthogonal rotation of Q and K before NVFP4; logits are unchanged."""
    value = os.environ.get("LLV2_ROTATION", "none")
    if value not in ("none", "hadamard"):
        raise ValueError(f"LLV2_ROTATION must be none or hadamard, got {value!r}")
    return value


def rotate_qk(x: torch.Tensor) -> torch.Tensor:
    """Apply LLV2_ROTATION over the head dim; Q and every K the attention reads must go through it."""
    if _rotation() == "none":
        return x
    rot = _rotation_cache.get(x.device)
    if rot is None:
        rot = _rotation_cache[x.device] = hadamard(x.shape[-1], x.device)
    return (x.float() @ rot).to(x.dtype)   # H is symmetric, so x @ H == x @ H^T


def _k_smooth_enabled() -> bool:
    """LongLive's per-token K smoothing, on by default; unlike K centering it changes the softmax."""
    value = os.environ.get("LLV2_K_SMOOTH", "1")
    if value not in ("0", "1"):
        raise ValueError(f"LLV2_K_SMOOTH must be 0 or 1, got {value!r}")
    return value == "1"


def _key_shift_paths() -> tuple[str, str]:
    """(`LLV2_KEY_SHIFT`, `LLV2_KEY_SHIFT_CALIBRATE`); at most one may be set."""
    apply_path = os.environ.get("LLV2_KEY_SHIFT", "")
    calibrate_path = os.environ.get("LLV2_KEY_SHIFT_CALIBRATE", "")
    if apply_path and calibrate_path:
        raise ValueError("set LLV2_KEY_SHIFT or LLV2_KEY_SHIFT_CALIBRATE, not both")
    return apply_path, calibrate_path


def _observe_keys(layer: int, key: torch.Tensor) -> None:
    """Accumulate raw post-RoPE K per layer in float64 for the calibration mean."""
    total = _key_shift_sums.get(layer)
    if total is None:
        if not _key_shift_sums:
            atexit.register(_write_key_shift)
        total = _key_shift_sums[layer] = torch.zeros(
            key.shape[1:], dtype=torch.float64, device=key.device)
    total += key.double().sum(dim=0)
    _key_shift_tokens[layer] = _key_shift_tokens.get(layer, 0) + key.shape[0]


def _write_key_shift() -> None:
    """At exit: mean = sum / tokens per layer, saved as a `key_shift` safetensors sidecar."""
    from safetensors.torch import save_file

    _, path = _key_shift_paths()
    layers = sorted(_key_shift_sums)
    if not path or not layers:
        return
    if layers != list(range(len(layers))):
        raise RuntimeError(f"key shift calibration saw layers {layers}, expected 0..{len(layers) - 1}")
    mean = torch.stack([_key_shift_sums[l] / _key_shift_tokens[l] for l in layers])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file({"key_shift": mean.float().cpu().contiguous()}, path, metadata={
        # the model's raw post-RoPE basis: before k_smooth and before any rotation
        "basis": "raw_post_rope",
        "tokens_per_layer": str(_key_shift_tokens[layers[0]]),
        "prompts": os.environ.get("LLV2_KEY_SHIFT_PROMPTS", ""),
    })
    print(f"[LLV2_KEY_SHIFT] wrote {path}: key_shift {tuple(mean.shape)}, "
          f"{_key_shift_tokens[layers[0]]} tokens per layer", flush=True)


def _key_shift_for(layer: int, device: torch.device, smooth: bool) -> torch.Tensor | None:
    """The LLV2_KEY_SHIFT constant for one layer, as P(mu) = mu - mean(mu) when k_smooth is on."""
    path, _ = _key_shift_paths()
    if not path:
        return None
    shift = _key_shift_cache.get((device, smooth))
    if shift is None:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            basis = (f.metadata() or {}).get("basis")
            if basis != "raw_post_rope":
                raise ValueError(f"{path}: key_shift basis is {basis!r}, expected 'raw_post_rope'")
            mu = f.get_tensor("key_shift").to(device=device, dtype=torch.float32)
        if mu.dim() != 3 or mu.shape[-1] != HEAD_DIM:
            raise ValueError(f"{path}: key_shift has shape {tuple(mu.shape)}, expected [layers, heads, {HEAD_DIM}]")
        if smooth:
            mu = mu - mu.mean(dim=-1, keepdim=True)
        shift = _key_shift_cache[(device, smooth)] = mu.to(torch.bfloat16)
    if layer >= shift.shape[0]:
        raise ValueError(f"key_shift has {shift.shape[0]} layers, asked for layer {layer}")
    return shift[layer]


def _hessian_paths() -> tuple[str, str]:
    """(`LLV2_HESSIAN_FEEDBACK`, `LLV2_HESSIAN_CALIBRATE`); at most one may be set."""
    apply_path = os.environ.get("LLV2_HESSIAN_FEEDBACK", "")
    calibrate_path = os.environ.get("LLV2_HESSIAN_CALIBRATE", "")
    if apply_path and calibrate_path:
        raise ValueError("set LLV2_HESSIAN_FEEDBACK or LLV2_HESSIAN_CALIBRATE, not both")
    return apply_path, calibrate_path


def _observe_queries(layer: int, query: torch.Tensor, heads_kv: int) -> None:
    """Accumulate sum_h q_h q_h^T per KV head over the query heads that share it."""
    total = _query_moment_sums.get(layer)
    if total is None:
        if not _query_moment_sums:
            atexit.register(_write_query_moment)
        total = _query_moment_sums[layer] = torch.zeros(
            heads_kv, HEAD_DIM, HEAD_DIM, dtype=torch.float64, device=query.device)
    grouped = query.reshape(query.shape[0], heads_kv, -1, HEAD_DIM)
    for start in range(0, grouped.shape[0], 2048):     # bounds the float64 copy
        block = grouped[start:start + 2048].double()
        total += torch.einsum("thri,thrj->hij", block, block)
    _query_moment_tokens[layer] = _query_moment_tokens.get(layer, 0) + query.shape[0]


def _observe_feedback_keys(layer: int, key: torch.Tensor) -> None:
    """Keep evenly spaced K tokens (exactly what gets quantized) in a per-layer ring."""
    ring = _feedback_key_ring.get(layer)
    if ring is None:
        ring = _feedback_key_ring[layer] = torch.zeros(
            _FEEDBACK_KEY_RING, *key.shape[1:], dtype=torch.float32, device=key.device)
    index = torch.linspace(0, key.shape[0] - 1, _FEEDBACK_KEY_PER_CALL, device=key.device).long()
    seen = _feedback_key_seen.get(layer, 0)
    slots = (torch.arange(_FEEDBACK_KEY_PER_CALL, device=key.device) + seen) % _FEEDBACK_KEY_RING
    ring[slots] = key[index].float()
    _feedback_key_seen[layer] = seen + _FEEDBACK_KEY_PER_CALL


def _write_query_moment() -> None:
    """At exit: save E[q q^T] per layer and KV head and report the feedback / RTN error ratio."""
    from safetensors.torch import save_file
    from nvfp4_vllm.hessian import MAX_FEEDBACK_RATIO, factors_from_moment, measure_feedback_ratio

    _, path = _hessian_paths()
    layers = sorted(_query_moment_sums)
    if not path or not layers:
        return
    if layers != list(range(len(layers))):
        raise RuntimeError(f"query moment calibration saw layers {layers}, expected 0..{len(layers) - 1}")
    moment = torch.stack([_query_moment_sums[l] / _query_moment_tokens[l] for l in layers]).float()
    metadata = {
        # the basis K is quantized in: post-RoPE, after LLV2_ROTATION
        "basis": _rotation(),
        "tokens_per_layer": str(_query_moment_tokens[layers[0]]),
        "prompts": os.environ.get("LLV2_HESSIAN_PROMPTS", ""),
    }
    if sorted(_feedback_key_ring) == layers:
        kept = min(_FEEDBACK_KEY_RING, min(_feedback_key_seen.values()))
        keys = torch.stack([_feedback_key_ring[l][:kept] for l in layers])   # [L, T, H, D]
        ratio = measure_feedback_ratio(keys, moment, factors_from_moment(moment), scale_mode=_scale_mode())
        metadata["feedback_ratio"] = f"{ratio:.4f}"
        verdict = "ok" if ratio <= MAX_FEEDBACK_RATIO else f"ABOVE the {MAX_FEEDBACK_RATIO} gate"
        print(f"[LLV2_HESSIAN] feedback / RTN M-weighted K error on {kept} sampled tokens per layer: "
              f"{ratio:.4f} ({verdict})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file({"query_moment": moment.cpu().contiguous()}, path, metadata=metadata)
    print(f"[LLV2_HESSIAN] wrote {path}: query_moment {tuple(moment.shape)}, "
          f"{_query_moment_tokens[layers[0]]} query tokens per layer", flush=True)


def _feedback_factors_for(layer: int, device: torch.device) -> torch.Tensor | None:
    """`LLV2_HESSIAN_FEEDBACK` factors for one layer, [1, heads, 2, 64, 64] fp32, or None when off."""
    path, _ = _hessian_paths()
    if not path:
        return None
    factors = _feedback_factor_cache.get(device)
    if factors is None:
        from safetensors import safe_open
        from nvfp4_vllm.hessian import factors_from_moment

        with safe_open(path, framework="pt") as f:
            basis = (f.metadata() or {}).get("basis")
            if basis != _rotation():
                raise ValueError(f"{path}: query_moment measured with LLV2_ROTATION={basis!r}, "
                                 f"this run uses {_rotation()!r}")
            moment = f.get_tensor("query_moment").to(device=device, dtype=torch.float32)
        if moment.dim() != 4 or moment.shape[-2:] != (HEAD_DIM, HEAD_DIM):
            raise ValueError(f"{path}: query_moment has shape {tuple(moment.shape)}, "
                             f"expected [layers, heads, {HEAD_DIM}, {HEAD_DIM}]")
        factors = _feedback_factor_cache[device] = factors_from_moment(moment).contiguous()
    if layer >= factors.shape[0]:
        raise ValueError(f"query_moment has {factors.shape[0]} layers, asked for layer {layer}")
    return factors[layer:layer + 1]


def transform_query_for_attend(layer: int, query: torch.Tensor, heads_kv: int) -> torch.Tensor:
    """Q as the FP4 attention reads it; observed under LLV2_HESSIAN_CALIBRATE."""
    query = rotate_qk(query)
    _, calibrate_path = _hessian_paths()
    if calibrate_path:
        _observe_queries(layer, query[0], heads_kv)
    return query


def transform_key_for_cache(layer: int, key: torch.Tensor) -> torch.Tensor:
    """Everything done to K between RoPE and the NVFP4 cache.

    Order: (calibrate) -> k_smooth -> subtract LLV2_KEY_SHIFT -> LLV2_ROTATION. Every K a query
    attends to must go through here, or the shift no longer cancels in the softmax.
    """
    from utils.quant import k_smooth

    _, calibrate_path = _key_shift_paths()
    if calibrate_path:
        _observe_keys(layer, key)
    smooth = _k_smooth_enabled()
    if smooth:
        key = k_smooth(key)
    shift = _key_shift_for(layer, key.device, smooth)
    if shift is not None:
        key = key - shift    # one [heads, head_dim] constant, broadcast over all tokens
    key = rotate_qk(key)
    if _hessian_paths()[1]:
        _observe_feedback_keys(layer, key)
    return key


def _to_cute_tensor(tensor: torch.Tensor, *, assumed_align: int, leading_dim: int):
    result = from_dlpack(tensor.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _compile_prefill_decode(device_index: int, heads_q: int, heads_kv: int, with_lse: bool = False):
    """Compile (once per device/head-shape) the genuine varlen-Q kernel."""
    query_terms = _query_terms()
    p_scale_search = _p_scale_search()
    p_scale_code10 = _p_scale_mode() == "code10"
    p_scale_from_scores = _p_scale_from_scores()
    p_sp1 = _p_sp1()
    cache_key = (device_index, heads_q, heads_kv, query_terms, p_scale_search, p_scale_code10,
                 p_scale_from_scores, p_sp1, with_lse)
    compiled = _prefill_compile_cache.get(cache_key)
    if compiled is not None:
        return compiled

    operation = FP4DecodeKernel(
        HEAD_DIM,
        HEAD_DIM,
        qhead_per_kvhead=heads_q // heads_kv,
        is_causal=False,
        is_local=False,
        is_split_kv=False,
        pack_gqa=(heads_q != heads_kv),
        m_block_size=PAGE_SIZE,
        n_block_size=PAGE_SIZE,
        is_persistent=True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors=False,
        paged_kv_non_tma=False,
        is_varlen_q=True,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        fused_residual_first_block=False,
        fused_sink_block=False,
        residual_source="paged_bf16",
        use_out_indices=False,
        seqlen_q_static_one=False,
        transpose_s=True,  # overridden to False internally for this path
        query_terms=query_terms,
        p_scale_search=p_scale_search,
        p_scale_code10_threshold=p_scale_code10,
        p_scale_from_scores=p_scale_from_scores,
    )
    fake_stream = cute.runtime.make_fake_stream()
    q_pointer = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    k_pointer = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    v_pointer = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    device = torch.device(f"cuda:{device_index}")
    fake_output = torch.empty(1, heads_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    fake_page_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    fake_seqused = torch.zeros(1, dtype=torch.int32, device=device)
    fake_cu_seqlens_q = torch.zeros(2, dtype=torch.int32, device=device)
    fake_query = torch.zeros(1, heads_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if query_terms == 2:
        (
            _,
            fake_query_scales,
            _,
            fake_query_scales_second,
        ) = _quantize.quantize_query_varlen(
            fake_query, query_terms=2, scale_mode=_scale_mode()
        )
    else:
        _, fake_query_scales = _quantize.quantize_query_varlen(
            fake_query, scale_mode=_scale_mode()
        )
        fake_query_scales_second = fake_query_scales
    _, fake_kv_scales_raw = _allocate_scales(1, heads_kv, 1, HEAD_DIM // 64, device)
    fake_kv_scales_raw = fake_kv_scales_raw.view(torch.float8_e4m3fn)
    fake_kv_scales = _page_scales_for_kernel(fake_kv_scales_raw, "fake_kv_scales")

    output_tensor = _to_cute_tensor(fake_output, assumed_align=16, leading_dim=2)
    # LSE for the varlen path is (heads_q, total_q) fp32, natural log; the kernel
    # re-selects modes [1, 0] internally (fp4_decode_kernel.py:864). Only the
    # split-attention path asks for it.
    fake_lse = torch.empty(heads_q, 1, dtype=torch.float32, device=device)
    lse_tensor = _to_cute_tensor(fake_lse, assumed_align=4, leading_dim=1) if with_lse else None
    page_table_tensor = _to_cute_tensor(fake_page_table, assumed_align=4, leading_dim=1)
    seqused_tensor = _to_cute_tensor(fake_seqused, assumed_align=4, leading_dim=0)
    cu_seqlens_q_tensor = _to_cute_tensor(fake_cu_seqlens_q, assumed_align=4, leading_dim=0)
    query_scales_tensor = _to_cute_tensor(fake_query_scales, assumed_align=16, leading_dim=3)
    query_scales_second_tensor = _to_cute_tensor(
        fake_query_scales_second, assumed_align=16, leading_dim=3
    )
    kv_scales_tensor = _to_cute_tensor(fake_kv_scales, assumed_align=16, leading_dim=3)

    symbolic_q_shape = tuple(Int32(0) for _ in range(3))
    symbolic_k_shape = tuple(Int32(0) for _ in range(4))
    symbolic_v_shape = tuple(Int32(0) for _ in range(4))

    compile_args = (
        operation,
        q_pointer,
        k_pointer,
        v_pointer,
        output_tensor,
        lse_tensor,
        Float32(1.0),
        fake_stream,
        cu_seqlens_q_tensor,
        None,
        None,
        seqused_tensor,
        page_table_tensor,
        None,
        None,
        None,
        None,
        None,
        query_scales_tensor,
        kv_scales_tensor,
        kv_scales_tensor,
        symbolic_q_shape,
        symbolic_k_shape,
        symbolic_v_shape,
    )
    compile_kwargs = dict(
        k_page_stride=Int32(0),
        v_page_stride=Int32(0),
        k_sf_page_stride=Int32(0),
        v_sf_page_stride=Int32(0),
        mQSecond=q_pointer,
        mSFQSecond=query_scales_second_tensor,
        compute_sp1=p_sp1,
    )
    compiled = cute.compile(*compile_args, **compile_kwargs, options="--enable-tvm-ffi")
    _prefill_compile_cache[cache_key] = compiled
    return compiled


@dataclass
class Fp4KVCache:
    """Persistent BitDecoding-native paged K/V storage for one attention layer."""

    key_pages_fp4: torch.Tensor      # [total_pages, PAGE_SIZE, heads_kv, HEAD_DIM // 2] uint8
    value_pages_fp4: torch.Tensor    # [total_pages, heads_kv, HEAD_DIM, PAGE_SIZE // 2] uint8
    key_scales_raw: torch.Tensor     # _allocate_scales layout, page axis FIRST -- page-sliceable
    value_scales_raw: torch.Tensor
    page_table: torch.Tensor         # [1, total_pages] int32, static identity
    total_pages: int
    pages_per_block: int
    heads_kv: int


def init_fp4_kv_cache(
    max_blocks: int,
    pages_per_block: int,
    heads_kv: int,
    device: torch.device,
) -> Fp4KVCache:
    total_pages = max_blocks * pages_per_block
    key_pages_fp4 = torch.zeros(
        total_pages, PAGE_SIZE, heads_kv, HEAD_DIM // 2, dtype=torch.uint8, device=device
    )
    value_pages_fp4 = torch.zeros(
        total_pages, heads_kv, HEAD_DIM, PAGE_SIZE // 2, dtype=torch.uint8, device=device
    )
    # K and V share the same (rest_m=1, rest_k=HEAD_DIM//64=2) scale-storage shape 
    rest_k = HEAD_DIM // 64
    _, key_scales_raw = _allocate_scales(total_pages, heads_kv, 1, rest_k, device)
    _, value_scales_raw = _allocate_scales(total_pages, heads_kv, 1, rest_k, device)
    key_scales_raw = key_scales_raw.view(torch.float8_e4m3fn)
    value_scales_raw = value_scales_raw.view(torch.float8_e4m3fn)
    key_scales_raw.zero_()
    value_scales_raw.zero_()

    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(1, total_pages)

    return Fp4KVCache(
        key_pages_fp4=key_pages_fp4,
        value_pages_fp4=value_pages_fp4,
        key_scales_raw=key_scales_raw,
        value_scales_raw=value_scales_raw,
        page_table=page_table,
        total_pages=total_pages,
        pages_per_block=pages_per_block,
        heads_kv=heads_kv,
    )


def roll_blocks(cache: Fp4KVCache, sink_blks: int, evict_blks: int, roll_blks: int) -> None:
    """Shift `roll_blks` block-slots left by `evict_blks`, starting at `sink_blks`."""
    ppb = cache.pages_per_block
    for i in range(roll_blks):
        src = sink_blks + evict_blks + i
        dst = sink_blks + i
        src_pages = slice(src * ppb, (src + 1) * ppb)
        dst_pages = slice(dst * ppb, (dst + 1) * ppb)
        cache.key_pages_fp4[dst_pages].copy_(cache.key_pages_fp4[src_pages])
        cache.value_pages_fp4[dst_pages].copy_(cache.value_pages_fp4[src_pages])
        cache.key_scales_raw[dst_pages].copy_(cache.key_scales_raw[src_pages])
        cache.value_scales_raw[dst_pages].copy_(cache.value_scales_raw[src_pages])


def insert_block(
    cache: Fp4KVCache, start_blk: int, n_blks: int, k_bf16: torch.Tensor, v_bf16: torch.Tensor,
    layer: int | None = None,
) -> None:
    """Quantize already-RoPE'd, k_smooth'd BF16 K/V into block-slots [start_blk, start_blk+n_blks).

    k_bf16/v_bf16: [n_blks * block_token_size, heads_kv, HEAD_DIM] BF16, contiguous.
    layer selects the LLV2_HESSIAN_FEEDBACK factors.
    """
    ppb = cache.pages_per_block
    n_pages = n_blks * ppb
    heads_kv = cache.heads_kv
    k_pages_in = k_bf16.reshape(n_pages, PAGE_SIZE, heads_kv, HEAD_DIM).contiguous()
    v_pages_in = v_bf16.reshape(n_pages, PAGE_SIZE, heads_kv, HEAD_DIM).contiguous()
    factors = None
    if _hessian_paths()[0]:
        if layer is None:
            raise ValueError("LLV2_HESSIAN_FEEDBACK needs insert_block(..., layer=)")
        factors = _feedback_factors_for(layer, k_bf16.device)
    new_key_fp4, new_key_scales = _quantize.quantize_key_pages(
        k_pages_in, scale_mode=_scale_mode(), factors=factors
    )
    new_value_fp4, new_value_scales = _quantize.quantize_value_pages(
        v_pages_in, scale_mode=_scale_mode()
    )

    dst_pages = slice(start_blk * ppb, (start_blk + n_blks) * ppb)
    cache.key_pages_fp4[dst_pages].copy_(new_key_fp4)
    cache.value_pages_fp4[dst_pages].copy_(new_value_fp4)
    cache.key_scales_raw[dst_pages].copy_(new_key_scales)
    cache.value_scales_raw[dst_pages].copy_(new_value_scales)

_cu_seqlens_q_cache: dict[tuple[int, int], torch.Tensor] = {}


def _cu_seqlens_q_for(total_q: int, device: torch.device) -> torch.Tensor:
    key = (device.index, total_q)
    cu_seqlens_q = _cu_seqlens_q_cache.get(key)
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=device)
        _cu_seqlens_q_cache[key] = cu_seqlens_q
    return cu_seqlens_q


def attend(
    q_bf16: torch.Tensor,
    cache: Fp4KVCache,
    seqused_fp4: torch.Tensor,
    softmax_scale: float,
    heads_kv: int,
    return_lse: bool = False,
    page_table: torch.Tensor | None = None,
):
    """NVFP4 tensor-core self-attention: decode `q_bf16`'s rows against `cache`.

    q_bf16: [total_q, heads_q, HEAD_DIM] BF16, flat (no batch axis, every
    row of this one LongLive block attends to the same cache window).
    seqused_fp4: [1] int32, how many tokens of the page sequence are valid.
    page_table: optional [1, total_pages] int32 view replacing the cache's identity
        table, so a call can start at an arbitrary page (used to skip the BF16 sink).
    Returns: [total_q, heads_q, HEAD_DIM] BF16.
    """
    total_q, heads_q, head_dim = q_bf16.shape
    assert head_dim == HEAD_DIM
    assert heads_kv == cache.heads_kv
    device = q_bf16.device

    query_terms = _query_terms()
    if query_terms == 2:
        (
            query_fp4,
            query_scales,
            query_fp4_second,
            query_scales_second,
        ) = _quantize.quantize_query_varlen(
            q_bf16, query_terms=2, scale_mode=_scale_mode()
        )
    else:
        query_fp4, query_scales = _quantize.quantize_query_varlen(
            q_bf16, scale_mode=_scale_mode()
        )
        query_fp4_second, query_scales_second = query_fp4, query_scales
    key_scales = _page_scales_for_kernel(cache.key_scales_raw, "key_scales")
    value_scales = _page_scales_for_kernel(cache.value_scales_raw, "value_scales")
    cu_seqlens_q = _cu_seqlens_q_for(total_q, device)
    output = torch.empty(total_q, heads_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    lse = torch.empty(heads_q, total_q, dtype=torch.float32, device=device) if return_lse else None

    with torch.cuda.device(device):
        compiled = _compile_prefill_decode(device.index, heads_q, heads_kv, with_lse=return_lse)
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        q_pointer = make_ptr(cutlass.Float4E2M1FN, query_fp4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        q_second_pointer = make_ptr(
            cutlass.Float4E2M1FN,
            query_fp4_second.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        k_pointer = make_ptr(cutlass.Float4E2M1FN, cache.key_pages_fp4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        v_pointer = make_ptr(cutlass.Float4E2M1FN, cache.value_pages_fp4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

        run_args = (
            q_pointer,
            k_pointer,
            v_pointer,
            output,
            lse,
            softmax_scale,
            stream,
            cu_seqlens_q,
            None,
            None,
            seqused_fp4,
            cache.page_table if page_table is None else page_table,
            None,
            None,
            None,
            None,
            None,
            query_scales,
            key_scales,
            value_scales,
            (total_q, heads_q, HEAD_DIM),
            (cache.total_pages, PAGE_SIZE, heads_kv, HEAD_DIM),
            (cache.total_pages, PAGE_SIZE, heads_kv, HEAD_DIM),
        )
        run_kwargs = dict(
            k_page_stride=_page_stride_bytes(cache.key_pages_fp4, "key_pages_fp4") * 2,
            v_page_stride=_page_stride_bytes(cache.value_pages_fp4, "value_pages_fp4") * 2,
            k_sf_page_stride=key_scales.stride()[-1],
            v_sf_page_stride=value_scales.stride()[-1],
            mQSecond=q_second_pointer,
            mSFQSecond=query_scales_second,
        )
        compiled(*run_args, **run_kwargs)
    if return_lse:
        return output, lse
    return output
