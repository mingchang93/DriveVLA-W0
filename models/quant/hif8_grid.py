# -*- coding: utf-8 -*-
"""HiF8 的精确量化器：直接查**实测网格**，不用任何档位模型。

来历：先前按「一个 binade 内的码字数减 1」推尾数位数，偏移定不准（硬件把
2^-15 与 1.5*2^-15 归进同一码字，而计数法给出 2 个点），154 个码字组里 60 个
校验不过。改为由 hif8_table.py 实测每个码字的舍入单元、取中点得到网格值，
存在 hif8_grid.json；这里用 searchsorted 查最近网格点，按构造即精确。

用法：q = quant_dequant(x)          # x 已除过 scale
     e = quant_dequant(x/s)*s - x  # 真实量化误差
"""
import json
import math
import os

import torch

_JSON = os.environ.get(
    "HIF8_GRID_JSON", "/mnt/sfs_turbo/zyc/drivevla-w0-zyc/hif8_grid.json")
_g = {"vals": None, "mids": None}


def _load(device, dtype=torch.float32):
    if _g["vals"] is not None and _g["vals"].device == torch.device(device):
        return _g["vals"], _g["mids"]
    d = json.load(open(_JSON, encoding="utf-8"))
    vals = sorted(float(v) for v in d["grid"].values() if float(v) > 0)
    # 规整到真正的二进制网格：实测中点带采样误差，吸附到最近的 m/2^j 形式
    snapped = []
    for v in vals:
        k = math.floor(math.log2(v))
        for bits in (3, 2, 1, 0):
            step = 2.0 ** (k - bits) if bits else 2.0 ** k
            r = round(v / step) * step
            if abs(r - v) / v < 0.02:
                snapped.append(r)
                break
        else:
            snapped.append(v)
    vals = sorted(set(snapped))
    t = torch.tensor(vals, dtype=dtype, device=device)
    mids = (t[1:] + t[:-1]) / 2          # round-to-nearest 的判别阈值
    _g["vals"], _g["mids"] = t, mids
    return t, mids


def quant_dequant(x):
    """按 HiF8 实测网格做 round-to-nearest 并还原。scale 由调用方先除掉。"""
    a = x.detach().abs().to(torch.float32)
    sign = torch.where(x.detach() < 0, -1.0, 1.0).to(torch.float32)
    vals, mids = _load(a.device)
    idx = torch.searchsorted(mids, a.reshape(-1).contiguous())
    q = vals[idx].reshape(a.shape)
    q = torch.where(a < vals[0] / 2, torch.zeros_like(q), q)   # 下溢归零
    return (sign * q).to(x.dtype)


def selfcheck(dev="npu:0"):
    """硬件一致性：码字相同的输入，查表还原值必须相同。"""
    import torch_npu
    HIF8 = getattr(torch_npu, "hifloat8", None)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    bad = tot = 0
    worst = 0.0
    for kk in range(-14, 15):
        v = torch.pow(2.0, kk + torch.arange(2048, dtype=torch.float64) / 2048.0)
        vf = v.to(torch.float32)
        code = torch_npu.npu_quantize(vf.to(dev), one, None, HIF8, -1, True) \
                        .to(torch.int32).cpu()
        qa = quant_dequant(vf)
        rel = ((qa.to(torch.float64) - v) / v).abs()
        worst = max(worst, float(rel.max()))
        for c in sorted(set(int(z) for z in code.tolist())):
            sel = (code == c)
            u = set(round(float(z), 9) for z in qa[sel].tolist())
            tot += 1
            if len(u) != 1:
                bad += 1
                if bad <= 5:
                    print("    不一致 k=%d code=%d -> %s" % (kk, c, sorted(u)[:3]))
    print("  一致性：%d/%d 个码字组内还原值恒定（不一致 %d）" % (tot - bad, tot, bad))
    print("  最大相对误差 %.4f%%（3-bit 档理论上限 6.25%%，1-bit 档 25%%）"
          % (100 * worst))
    return bad == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if selfcheck() else 1)
