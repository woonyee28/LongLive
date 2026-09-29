"""K centering (LLV2_KEY_SHIFT) and the k_smooth switch in utils.fp4_attention.

CPU only, no model: python tests/test_key_shift.py   (run from LongLive/)
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.fp4_attention as fa
from utils.quant import k_smooth
from nvfp4_decode_kernel.reference import nvfp4_quantize

torch.manual_seed(0)
L, H, D, S = 3, 4, 128, 512
FLAGS = ("LLV2_KEY_SHIFT", "LLV2_KEY_SHIFT_CALIBRATE", "LLV2_K_SMOOTH", "LLV2_ROTATION")


def reset(**env):
    for k in FLAGS:
        os.environ.pop(k, None)
    os.environ.update(env)
    fa._key_shift_cache.clear()
    fa._key_shift_sums.clear()
    fa._key_shift_tokens.clear()


def keys(offset=True):
    """Post-RoPE-like K: per-channel offsets that are the same for every token, plus noise."""
    k = torch.randn(S, H, D)
    if offset:
        k = k + 6.0 * torch.randn(1, H, D)
    return k.bfloat16()


def save_shift(mu):
    from safetensors.torch import save_file
    path = os.path.join(tempfile.mkdtemp(), "ks.safetensors")
    save_file({"key_shift": mu.float().contiguous()}, path, metadata={"basis": "raw_post_rope"})
    return path


def attn(q, k):
    return torch.softmax(q.double() @ k.double().transpose(-1, -2) / D ** 0.5, dim=-1)


def test_default_path_unchanged():
    reset()
    k = keys()
    assert torch.equal(fa.transform_key_for_cache(0, k), fa.rotate_qk(k_smooth(k)))


def test_k_smooth_off_is_identity():
    reset(LLV2_K_SMOOTH="0")
    k = keys()
    assert torch.equal(fa.transform_key_for_cache(0, k), k)


def test_shift_leaves_attention_unchanged():
    """q.(k - mu) = q.k - q.mu: the same constant for every key, so softmax ignores it."""
    mu = torch.randn(L, H, D) * 5
    path = save_shift(mu)
    q = torch.randn(H, 64, D)
    for smooth in ("1", "0"):
        k = keys()
        reset(LLV2_K_SMOOTH=smooth)
        base = fa.transform_key_for_cache(1, k).float()
        reset(LLV2_K_SMOOTH=smooth, LLV2_KEY_SHIFT=path)
        shifted = fa.transform_key_for_cache(1, k).float()
        assert not torch.allclose(base, shifted)              # K really changed
        a0, a1 = attn(q, base.transpose(0, 1)), attn(q, shifted.transpose(0, 1))
        err = (a1 - a0).abs().max().item()
        assert err < 2e-2, (smooth, err)                      # only bf16 rounding of k - mu remains


def test_k_smooth_is_not_invariant():
    """Contrast: k_smooth changes the softmax (it subtracts a different amount per key)."""
    reset(LLV2_K_SMOOTH="0")
    q = torch.randn(H, 64, D) + 1.0                           # sum(q) != 0
    k = keys()
    raw = fa.transform_key_for_cache(0, k).float()
    reset(LLV2_K_SMOOTH="1")
    sm = fa.transform_key_for_cache(0, k).float()
    assert (attn(q, sm.transpose(0, 1)) - attn(q, raw.transpose(0, 1))).abs().max() > 1e-2


def test_projected_shift_equals_measuring_after_smoothing():
    """With k_smooth on the file's raw mu is used as P(mu): P(k) - P(mu) == P(k - mu)."""
    mu = torch.randn(L, H, D) * 5
    reset(LLV2_K_SMOOTH="1", LLV2_KEY_SHIFT=save_shift(mu))
    k = keys()
    got = fa.transform_key_for_cache(2, k).float()
    want = (k.float() - mu[2]) - (k.float() - mu[2]).mean(dim=-1, keepdim=True)
    # equal up to bf16 rounding (~1 step, 0.125 at |k|~27, in the worst element)
    assert ((got - want).norm() / want.norm()).item() < 5e-3


def test_calibration_mean():
    out = os.path.join(tempfile.mkdtemp(), "cal.safetensors")
    reset(LLV2_KEY_SHIFT_CALIBRATE=out)
    chunks = {layer: [keys() for _ in range(3)] for layer in range(L)}
    for layer, ks in chunks.items():
        for k in ks:
            fa.transform_key_for_cache(layer, k)
    fa._write_key_shift()
    from safetensors import safe_open
    with safe_open(out, framework="pt") as f:
        mu = f.get_tensor("key_shift")
        assert f.metadata()["basis"] == "raw_post_rope"
    want = torch.stack([torch.cat(chunks[l]).double().mean(0) for l in range(L)]).float()
    assert mu.shape == (L, H, D) and torch.allclose(mu, want, atol=1e-5)


def test_both_flags_rejected():
    reset(LLV2_KEY_SHIFT="a", LLV2_KEY_SHIFT_CALIBRATE="b")
    try:
        fa.transform_key_for_cache(0, keys())
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_shift_reduces_nvfp4_error():
    """Removing per-channel offsets shrinks group maxima, so NVFP4 error drops."""
    k = keys()
    mu = k.double().mean(0, keepdim=True).float()[0]             # measured mean, [H, D]

    def rel_err(x):
        flat = x.float().reshape(-1, 16)
        rt, _, _ = nvfp4_quantize(flat, group=16, scale_mode="four_six_sixhalf")
        return ((rt - flat).norm() / flat.norm()).item(), (rt - flat).norm().item()

    reset(LLV2_K_SMOOTH="1")
    base = fa.transform_key_for_cache(0, k)
    reset(LLV2_K_SMOOTH="1", LLV2_KEY_SHIFT=save_shift(mu.unsqueeze(0).repeat(L, 1, 1)))
    cent = fa.transform_key_for_cache(0, k)
    (_, e0), (_, e1) = rel_err(base), rel_err(cent)
    print(f"    absolute NVFP4 error on K: k_smooth only {e0:.1f} -> +centering {e1:.1f} ({100 * (e1 / e0 - 1):+.0f}%)")
    assert e1 < e0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    reset()
