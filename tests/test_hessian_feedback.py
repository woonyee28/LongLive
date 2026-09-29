"""Hessian feedback rounding (LLV2_HESSIAN_FEEDBACK / LLV2_HESSIAN_CALIBRATE) in utils.fp4_attention.

python tests/test_hessian_feedback.py   (run from LongLive/; the last test needs an SM100 GPU)
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.fp4_attention as fa

torch.manual_seed(0)
H, D = 4, 128
FLAGS = ("LLV2_HESSIAN_FEEDBACK", "LLV2_HESSIAN_CALIBRATE", "LLV2_KEY_SHIFT",
         "LLV2_KEY_SHIFT_CALIBRATE", "LLV2_ROTATION", "LLV2_K_SMOOTH")


def reset(**env):
    for k in FLAGS:
        os.environ.pop(k, None)
    os.environ.update(env)
    for cache in (fa._feedback_factor_cache, fa._query_moment_sums, fa._query_moment_tokens,
                  fa._feedback_key_ring, fa._feedback_key_seen):
        cache.clear()


def correlated_queries(tokens, heads, device="cpu"):
    """Queries whose channels are coupled, so E[q q^T] has real off-diagonal structure."""
    mix = torch.eye(D, device=device) + 0.3 * torch.randn(heads, D, D, device=device) / D ** 0.5
    mix[:, torch.arange(1, D, 2), torch.arange(0, D, 2)] += 0.8     # Wan's RoPE pairs (2j, 2j+1)
    return torch.einsum("hij,thj->thi", mix, torch.randn(tokens, heads, D, device=device))


def save_moment(moment, basis="none"):
    from safetensors.torch import save_file
    path = os.path.join(tempfile.mkdtemp(), "qm.safetensors")
    save_file({"query_moment": moment.float().contiguous()}, path, metadata={"basis": basis})
    return path


def test_query_path_unchanged_without_flags():
    reset()
    q = torch.randn(1, 64, H, D).bfloat16()
    assert torch.equal(fa.transform_query_for_attend(0, q, H), fa.rotate_qk(q))


def test_both_flags_rejected():
    reset(LLV2_HESSIAN_FEEDBACK="a", LLV2_HESSIAN_CALIBRATE="b")
    try:
        fa.transform_query_for_attend(0, torch.randn(1, 8, H, D), H)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_moment_is_per_head_second_moment():
    """MHA: E[q_h q_h^T]; GQA: summed over the query heads that share a KV head."""
    reset(LLV2_HESSIAN_CALIBRATE=os.path.join(tempfile.mkdtemp(), "unused.safetensors"))
    chunks = [torch.randn(1, 300, H, D) for _ in range(3)]
    for q in chunks:
        fa.transform_query_for_attend(0, q, H)
    fa.transform_query_for_attend(1, chunks[0], H // 2)           # GQA: 2 query heads per KV head
    q = torch.cat(chunks, dim=1)[0].double()
    want = torch.einsum("thi,thj->hij", q, q) / q.shape[0]
    got = fa._query_moment_sums[0] / fa._query_moment_tokens[0]
    assert torch.allclose(got, want, rtol=1e-10, atol=1e-10)
    g = chunks[0][0].double().reshape(300, H // 2, 2, D)
    want_gqa = torch.einsum("thri,thrj->hij", g, g) / 300
    assert torch.allclose(fa._query_moment_sums[1] / 300, want_gqa, rtol=1e-10, atol=1e-10)
    reset()


def test_calibration_writes_moment_and_ratio():
    out = os.path.join(tempfile.mkdtemp(), "cal.safetensors")
    reset(LLV2_HESSIAN_CALIBRATE=out, LLV2_K_SMOOTH="0")
    for layer in range(2):
        for _ in range(4):
            fa.transform_query_for_attend(layer, correlated_queries(256, H)[None], H)
            fa.transform_key_for_cache(layer, torch.randn(256, H, D).bfloat16())
    fa._write_query_moment()
    from safetensors import safe_open
    with safe_open(out, framework="pt") as f:
        meta, moment = f.metadata(), f.get_tensor("query_moment")
    assert moment.shape == (2, H, D, D) and meta["basis"] == "none"
    assert float(meta["feedback_ratio"]) < 0.95, meta      # coupled queries: feedback must pay off
    reset()


def test_basis_mismatch_rejected():
    reset(LLV2_HESSIAN_FEEDBACK=save_moment(torch.eye(D).expand(1, H, D, D), basis="hadamard"))
    try:
        fa._feedback_factors_for(0, torch.device("cpu"))
    except ValueError:
        reset()
        return
    raise AssertionError("expected ValueError")


def test_gpu_feedback_lowers_attention_error():
    """End to end on the real cache: RTN vs feedback K, same Q/V, NVFP4 attention vs exact."""
    if not torch.cuda.is_available():
        print("    (no GPU, skipped)")
        return
    dev = torch.device("cuda:0")
    heads, blocks, ppb = 8, 2, 2
    tokens = blocks * ppb * fa.PAGE_SIZE
    q = correlated_queries(1024, heads, dev)
    k = (torch.randn(tokens, heads, D, device=dev) * 2).bfloat16()
    v = torch.randn(tokens, heads, D, device=dev).bfloat16()
    moment = torch.einsum("thi,thj->hij", q.double(), q.double()).float() / q.shape[0]
    path = save_moment(moment[None].cpu())
    q = q.bfloat16()
    seq = torch.tensor([tokens], dtype=torch.int32, device=dev)
    exact = torch.softmax(torch.einsum("qhd,nhd->hqn", q.double(), k.double()) * D ** -0.5, -1)
    exact = torch.einsum("hqn,nhd->qhd", exact, v.double())

    outs = {}
    for name, env in (("rtn", {}), ("off_with_layer", {}), ("feedback", {"LLV2_HESSIAN_FEEDBACK": path})):
        reset(**env)
        cache = fa.init_fp4_kv_cache(blocks, ppb, heads, dev)
        fa.insert_block(cache, 0, blocks, k, v, layer=None if name == "rtn" else 0)
        outs[name] = (fa.attend(q, cache, seq, D ** -0.5, heads), cache.key_pages_fp4.clone(),
                      cache.key_scales_raw.view(torch.uint8).clone())
    reset()
    assert all(torch.equal(a, b) for a, b in zip(outs["rtn"], outs["off_with_layer"]))   # flag off: bit-identical
    assert torch.equal(outs["rtn"][2], outs["feedback"][2])                               # scales unchanged
    changed = (outs["rtn"][1] != outs["feedback"][1]).float().mean().item()
    err = {n: ((o[0].double() - exact).norm() / exact.norm()).item() for n, o in outs.items()}
    print(f"    K bytes changed {changed:.1%}; attention rel err vs exact: "
          f"RTN {err['rtn']:.5f} -> feedback {err['feedback']:.5f} ({100 * (err['feedback'] / err['rtn'] - 1):+.1f}%)")
    assert err["feedback"] < err["rtn"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
