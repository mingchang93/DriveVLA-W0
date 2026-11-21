# -*- coding: utf-8 -*-
"""量化误差的**相干性**测量 —— 判「确定性偏置」还是「零均值噪声」。

动机：grad_norm 看不见坐标级的偏置（各坐标正负交错、net 幅度为零），所以
「量化误差在幅度上无偏」并不排除「它在每个坐标上每步都往同一方向推」。
后者会随步数线性累积，正是「偏差随训练/重放单调放大」的形状。

定义（对固定形状的张量，逐元素累积）：
    S_T = sum_{t<=T} e_t          e_t = Q(x_t/s_t)*s_t - x_t
    Q_T = sum_{t<=T} ||e_t||^2
    R(T) = ||S_T|| / sqrt(Q_T)
零均值噪声 -> R ~ 1；完全相干的偏置 -> R ~ sqrt(T)。T=150 时两者差 12 倍。
报告 R 与 R/sqrt(T)（后者 ~1 为完全相干、~1/sqrt(T) 为纯噪声）。

只跟 role=w：权重形状固定，可逐元素累积（x/g 的 token 数每步变，没法对齐）。
而且权重量化恰是「确定性扭曲」那条通道：CTS 的 s_w = amax(w)/15 只是 w 的函数。

环境变量：
    HIF8_COH=/path/out.csv    写到哪（不设则关闭）
    HIF8_COH_EVERY=10         每多少步写一行（累积每步都做）
    HIF8_COH_ROLE=w           跟哪个角色（默认 w）
"""
import io
import math
import os

import torch

try:
    from models.quant.hif8_grid import quant_dequant
except Exception:
    try:
        from hif8_grid import quant_dequant
    except Exception:
        quant_dequant = None

PATH = os.environ.get("HIF8_COH", "")
EVERY = int(os.environ.get("HIF8_COH_EVERY", "10") or 10)
ROLE = os.environ.get("HIF8_COH_ROLE", "w")
ENABLED = bool(PATH) and quant_dequant is not None

_st = {}            # label -> dict(S=累积误差, q=平方和, t=步数, absum=|e| 的和)
_fh = {"f": None}
_target = {"label": None}


def _out():
    if _fh["f"] is None:
        f = io.open(PATH, "a", encoding="utf-8", newline="\n")
        if f.tell() == 0:
            f.write("t,role,label,numel,R,sqrtT,R_over_sqrtT,"
                    "mean_abs_e,rms_e,rel_rms\n")
        _fh["f"] = f
    return _fh["f"]


def record(role, label, x, scale):
    """x 是量化前的张量，scale 是实际用于除法的那个值。"""
    if not ENABLED or role != ROLE:
        return
    if _target["label"] is None:
        _target["label"] = label
    if label != _target["label"]:
        return
    try:
        with torch.no_grad():
            s = scale.reshape(-1)[0] if torch.is_tensor(scale) else torch.tensor(float(scale))
            s = s.to(torch.float32)
            if float(s) <= 0:
                return
            xf = x.detach().to(torch.float32)
            e = quant_dequant(xf / s) * s - xf
            st = _st.get(label)
            if st is None or st["S"].shape != e.shape:
                st = {"S": torch.zeros_like(e), "q": 0.0, "t": 0,
                      "absum": 0.0, "xn": 0.0}
                _st[label] = st
            st["S"] += e
            st["q"] += float((e * e).sum())
            st["absum"] += float(e.abs().sum())
            st["xn"] += float((xf * xf).sum())
            st["t"] += 1
            t = st["t"]
            if t % EVERY != 0:
                return
            Sn = float(torch.linalg.vector_norm(st["S"]))
            qn = math.sqrt(st["q"]) if st["q"] > 0 else 0.0
            if qn <= 0:
                return
            R = Sn / qn
            n = e.numel()
            _out().write("%d,%s,%s,%d,%.6g,%.4f,%.6g,%.6g,%.6g,%.6g\n"
                         % (t, role, label, n, R, math.sqrt(t), R / math.sqrt(t),
                            st["absum"] / (t * n),
                            math.sqrt(st["q"] / (t * n)),
                            math.sqrt(st["q"] / max(st["xn"], 1e-30))))
            _out().flush()
    except Exception:
        return
