"""Parity checks for the optional Ascend fused ops.

Run on the NPU box from the repo root:

    python tools/ascend_parity/check_fused_ops.py

Compares each torch_npu op against the PyTorch reference used in the vendored
transformers code. Exits non-zero on failure. No test framework by design.
"""

import os
import sys

import torch
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:
    torch_npu = None

TOL_BF16 = 1e-2
TOL_FP32 = 1e-3

failures = []


def check(name, out, ref, tol, dtype_note=""):
    diff = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().clamp_min(1e-6).item()
    rel = diff / scale
    ok = rel <= tol
    print(f"{'PASS' if ok else 'FAIL'} {name}{dtype_note}: maxdiff={diff:.3e} rel={rel:.3e} tol={tol}")
    if not ok:
        failures.append(name)


def rotate_half(t):
    x1, x2 = t.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def reference_rms_norm(x, w, eps):
    xf = x.float()
    return (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)))


def main():
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        print("SKIP: torch_npu / NPU not available")
        return 0

    dev = "npu"
    torch.manual_seed(0)
    print(f"torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', '?')}")

    for n in ("npu_rms_norm", "npu_swiglu", "npu_rotary_mul", "npu_gelu", "npu_fusion_attention",
              "npu_add_rms_norm", "npu_group_norm_silu", "npu_silu"):
        print("available", n, hasattr(torch_npu, n))

    # --- RMSNorm ---
    x = torch.randn(24, 421, 3584, device=dev, dtype=torch.bfloat16)
    w = torch.randn(3584, device=dev, dtype=torch.bfloat16)
    ref = reference_rms_norm(x, w, 1e-6)
    out = torch_npu.npu_rms_norm(x, w, epsilon=1e-6)[0]
    check("rms_norm bf16", out, ref, TOL_BF16)

    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    torch_npu.npu_rms_norm(x2, w2, epsilon=1e-6)[0].float().pow(2).mean().backward()
    finite = bool(torch.isfinite(x2.grad).all()) and bool(torch.isfinite(w2.grad).all())
    print(f"{'PASS' if finite else 'FAIL'} rms_norm backward finite: "
          f"dx_max={float(x2.grad.abs().max()):.3e} dw_max={float(w2.grad.abs().max()):.3e}")
    if not finite:
        failures.append("rms_norm backward")

    # --- SwiGLU (tensor, dim=-1 convention) ---
    gate = torch.randn(24, 421, 18944, device=dev, dtype=torch.bfloat16)
    up = torch.randn(24, 421, 18944, device=dev, dtype=torch.bfloat16)
    ref = F.silu(gate.float()) * up.float()
    out = torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)
    check("swiglu bf16", out, ref, TOL_BF16)

    # --- mRoPE / rotate_half, full-width cos/sin ---
    B, H, L, D = 24, 28, 421, 128
    q = torch.randn(B, H, L, D, device=dev, dtype=torch.bfloat16)
    cos = torch.randn(B, 1, L, D, device=dev, dtype=torch.bfloat16)
    sin = torch.randn(B, 1, L, D, device=dev, dtype=torch.bfloat16)
    ref = q.float() * cos.float() + rotate_half(q.float()) * sin.float()
    out = torch_npu.npu_rotary_mul(q, cos, sin)
    check("rotary_mul bf16", out, ref, TOL_BF16)

    # --- vision RoPE shapes ---
    S, HV, DV = 1024, 16, 64
    qv = torch.randn(S, HV, DV, device=dev, dtype=torch.bfloat16)
    cosv = torch.randn(S, 1, DV, device=dev, dtype=torch.bfloat16)
    sinv = torch.randn(S, 1, DV, device=dev, dtype=torch.bfloat16)
    refv = qv.float() * cosv.float() + rotate_half(qv.float()) * sinv.float()
    outv = torch_npu.npu_rotary_mul(qv, cosv, sinv)
    check("rotary_mul vision bf16", outv, refv, TOL_BF16)

    # --- exact GELU ---
    xg = torch.randn(24, 421, 18944, device=dev, dtype=torch.bfloat16)
    refg = F.gelu(xg.float(), approximate="none")
    outg = torch_npu.npu_gelu(xg, approximate="none")
    check("gelu exact bf16", outg, refg, TOL_BF16)

    # --- residual add + RMSNorm ---
    if hasattr(torch_npu, "npu_add_rms_norm"):
        try:
            xa = torch.randn(24, 421, 3584, device=dev, dtype=torch.bfloat16)
            xb = torch.randn_like(xa)
            wa = torch.randn(3584, device=dev, dtype=torch.bfloat16)
            y, rstd, summed = torch_npu.npu_add_rms_norm(xa, xb, wa, 1e-6)
            check("add_rms_norm residual", summed, xa.float() + xb.float(), TOL_BF16)
            check("add_rms_norm y", y, reference_rms_norm(xa + xb, wa, 1e-6), TOL_BF16)
            xb2, wa2 = xb.clone().requires_grad_(True), wa.clone().requires_grad_(True)
            torch_npu.npu_add_rms_norm(xa, xb2, wa2, 1e-6)[0].float().pow(2).mean().backward()
            finite = bool(torch.isfinite(xb2.grad).all()) and bool(torch.isfinite(wa2.grad).all())
            print(f"{'PASS' if finite else 'FAIL'} add_rms_norm backward finite")
            if not finite:
                failures.append("add_rms_norm backward")
        except Exception as e:
            print("SKIP add_rms_norm:", type(e).__name__, str(e).splitlines()[0][:160])
            failures.append("add_rms_norm")
    else:
        print("unavailable npu_add_rms_norm")

    # --- residual add + RMSNorm, v2_functional + explicit backward ---
    if hasattr(torch_npu, "npu_add_rms_norm_v2_functional"):
        try:
            import importlib.util
            _p = os.path.join(os.path.dirname(__file__), "..", "..",
                              "reference/transformers/src/transformers/models/qwen2_5_vl/npu_fused_ops.py")
            _spec = importlib.util.spec_from_file_location("_npu_fused_ops_v2", _p)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)

            xa = torch.randn(24, 421, 3584, device=dev, dtype=torch.bfloat16)
            xb = torch.randn_like(xa)
            wa = torch.randn(3584, device=dev, dtype=torch.bfloat16)
            gout, gsum = torch.randn_like(xa), torch.randn_like(xa)

            xa1 = xa.clone().requires_grad_(True)
            xb1 = xb.clone().requires_grad_(True)
            wa1 = wa.clone().requires_grad_(True)
            out, summed = _mod._FusedAddRMSNormV2.apply(xa1, xb1, wa1, 1e-6)
            check("add_rms_v2 y", out, reference_rms_norm(xa + xb, wa, 1e-6), TOL_BF16)
            check("add_rms_v2 summed", summed, xa.float() + xb.float(), TOL_BF16)
            ((out.float() * gout.float()).sum() + (summed.float() * gsum.float()).sum()).backward()

            xa2 = xa.clone().requires_grad_(True)
            xb2 = xb.clone().requires_grad_(True)
            wa2 = wa.clone().requires_grad_(True)
            s2 = xa2 + xb2
            ref_out = reference_rms_norm(s2, wa2, 1e-6)
            ((ref_out.float() * gout.float()).sum() + (s2.float() * gsum.float()).sum()).backward()

            for name, got, ref in (("dx1", xa1.grad, xa2.grad),
                                   ("dx2", xb1.grad, xb2.grad),
                                   ("dw", wa1.grad, wa2.grad)):
                rel = ((got.float() - ref.float()).abs().max()
                       / ref.float().abs().max().clamp_min(1e-6)).item()
                ok = rel <= TOL_BF16
                print(f"{'PASS' if ok else 'FAIL'} add_rms_v2 backward {name}: "
                      f"rel={rel:.3e} tol={TOL_BF16}")
                if not ok:
                    failures.append(f"add_rms_v2 backward {name}")
        except Exception as e:
            print("SKIP add_rms_v2:", type(e).__name__, str(e).splitlines()[0][:160])
            failures.append("add_rms_v2")
    else:
        print("unavailable npu_add_rms_norm_v2_functional")

    # --- GroupNorm + SiLU ---
    if hasattr(torch_npu, "npu_group_norm_silu"):
        try:
            xg2 = torch.randn(2, 32, 16, 16, device=dev, dtype=torch.bfloat16)
            gw = torch.randn(32, device=dev, dtype=torch.bfloat16)
            gb = torch.randn(32, device=dev, dtype=torch.bfloat16)
            out = torch_npu.npu_group_norm_silu(xg2, gw, gb, 8, 1e-5)
            out = out[0] if isinstance(out, (tuple, list)) else out
            ref = F.silu(F.group_norm(xg2.float(), 8, gw.float(), gb.float(), 1e-5))
            check("group_norm_silu bf16", out, ref, TOL_BF16)
            xg3 = xg2.clone().requires_grad_(True)
            y3 = torch_npu.npu_group_norm_silu(xg3, gw, gb, 8, 1e-5)
            y3 = y3[0] if isinstance(y3, (tuple, list)) else y3
            y3.float().pow(2).mean().backward()
            finite = xg3.grad is not None and bool(torch.isfinite(xg3.grad).all())
            print(f"{'PASS' if finite else 'FAIL'} group_norm_silu backward finite")
            if not finite:
                failures.append("group_norm_silu backward")
        except Exception as e:
            print("SKIP group_norm_silu:", type(e).__name__, str(e).splitlines()[0][:160])
            failures.append("group_norm_silu")
    else:
        print("unavailable npu_group_norm_silu")

    # --- SiLU ---
    if hasattr(torch_npu, "npu_silu"):
        try:
            xs = torch.randn(24, 421, 18944, device=dev, dtype=torch.bfloat16)
            check("silu bf16", torch_npu.npu_silu(xs), F.silu(xs.float()), TOL_BF16)
        except Exception as e:
            print("SKIP silu:", type(e).__name__, str(e).splitlines()[0][:160])
            failures.append("silu")
    else:
        print("unavailable npu_silu")

    # --- FA2 vs SDPA (causal, GQA 28/4, BSND) ---
    # NOTE: this CANN/Ascend 950 build requires a mask of at least 2048x2048
    # for sparse_mode 2/3; a seq-sized mask fails kernel tiling.
    Bq, Sq, Hq, Hkv, Dq = 2, 512, 28, 4, 128
    try:
        qa = torch.randn(Bq, Sq, Hq, Dq, device=dev, dtype=torch.bfloat16)
        ka = torch.randn(Bq, Sq, Hkv, Dq, device=dev, dtype=torch.bfloat16)
        va = torch.randn(Bq, Sq, Hkv, Dq, device=dev, dtype=torch.bfloat16)
        mask = torch.triu(torch.ones(2048, 2048, device=dev, dtype=torch.bool), diagonal=1)
        fa = torch_npu.npu_fusion_attention(
            qa, ka, va, Hq, "BSND", keep_prob=1.0, scale=Dq ** -0.5,
            atten_mask=mask, sparse_mode=3,
        )[0]
        rep = Hq // Hkv
        k_r = ka.transpose(1, 2).repeat_interleave(rep, dim=1)
        v_r = va.transpose(1, 2).repeat_interleave(rep, dim=1)
        sdpa = F.scaled_dot_product_attention(
            qa.transpose(1, 2).float(), k_r.float(), v_r.float(), is_causal=True
        ).transpose(1, 2)
        check("fa2 causal vs sdpa bf16", fa, sdpa, TOL_BF16)
    except Exception as e:
        print("FAIL fa2 causal vs sdpa:", type(e).__name__, str(e).splitlines()[0][:160])
        failures.append("fa2")

    print()
    if failures:
        print("FAILURES:", ", ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
