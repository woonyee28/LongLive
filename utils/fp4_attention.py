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

PAGE_SIZE = 128
HEAD_DIM = 128
_prefill_compile_cache: dict[tuple[int, int, int, int, bool, bool], object] = {}


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


def _p_sp1() -> bool:
    """Pre-scale P by 448*6 before its per-group E4M3 block scale is formed
    (the kernel's `compute_sp1`). OFF, and do not turn it on here.
    """
    return os.environ.get("LLV2_P_SP1", "0") == "1"


def _to_cute_tensor(tensor: torch.Tensor, *, assumed_align: int, leading_dim: int):
    result = from_dlpack(tensor.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _compile_prefill_decode(device_index: int, heads_q: int, heads_kv: int, with_lse: bool = False):
    """Compile (once per device/head-shape) the genuine varlen-Q kernel."""
    query_terms = _query_terms()
    p_scale_search = _p_scale_search()
    p_sp1 = _p_sp1()
    cache_key = (device_index, heads_q, heads_kv, query_terms, p_scale_search, p_sp1, with_lse)
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


def insert_block(cache: Fp4KVCache, start_blk: int, n_blks: int, k_bf16: torch.Tensor, v_bf16: torch.Tensor) -> None:
    """Quantize `n_blks` worth of freshly-computed, already-RoPE'd, already
    k_smooth'd BF16 K/V and write them into block-slots [start_blk, start_blk+n_blks).

    k_bf16/v_bf16: [n_blks * block_token_size, heads_kv, HEAD_DIM] BF16
    (block_token_size == pages_per_block * PAGE_SIZE), contiguous.
    """
    ppb = cache.pages_per_block
    n_pages = n_blks * ppb
    heads_kv = cache.heads_kv
    k_pages_in = k_bf16.reshape(n_pages, PAGE_SIZE, heads_kv, HEAD_DIM).contiguous()
    v_pages_in = v_bf16.reshape(n_pages, PAGE_SIZE, heads_kv, HEAD_DIM).contiguous()
    new_key_fp4, new_key_scales = _quantize.quantize_key_pages(
        k_pages_in, scale_mode=_scale_mode()
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
):
    """NVFP4 tensor-core self-attention: decode `q_bf16`'s rows against `cache`.

    q_bf16: [total_q, heads_q, HEAD_DIM] BF16, flat (no batch axis, every
    row of this one LongLive block attends to the same cache window).
    seqused_fp4: [1] int32, how many of cache's leading pages are valid.
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
            cache.page_table,
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
