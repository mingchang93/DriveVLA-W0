# -*- coding: utf-8 -*-
"""HiF8 量化健康探针：把量化点的分布统计写成 CSV。

环境变量开启，默认完全不生效：
    HIF8_PROBE=/path/out.csv     写到哪（不设则探针关闭）
    HIF8_PROBE_EVERY=25          同一个量化点每多少次调用记一次（默认 25）
    HIF8_PROBE_MODULES=6         最多跟踪几个量化点（按首次出现顺序，默认 6）
    HIF8_PROBE_RANK=0            只有这个 rank 写（默认 0；-1 = 所有 rank）

计数按「每个量化点自己的调用次数」，不需要往 Trainer 插钩子：GA=1 时一个点的
x/w 每步各调一次、g 每步调一次，所以 call 数就等于步数。

为什么不只看上溢/下溢：amax 被映射到 dst_type_max（前向 15、反向 224），而 HiF8
表示上限 32768、动态范围 38 个 binade，按这种映射真正的饱和与下溢几乎不可能。
对渐变精度格式真正决定误差的是**分布主体落在哪个尾数档**（3-bit 覆盖 7 个阶码值，
之外降到 2-bit、1-bit），所以核心量是「主体离上界有多远」。

列：
    call,role,label,numel    调用序号、角色(x/w/g)、量化点、元素数
    amax,scale               当前 amax、实际用于除法的 scale
    p50,p99,r50,r99          |x| 中位数/99 分位，以及 amax/p50、amax/p99
    bin3,bin2,bin1           落在 3-bit / 2-bit / 1-bit 尾数档的占比（近似）
    uf,ovr,sat               下溢占比、越过 dst_max 占比、真饱和(>32768)占比
"""

import io
import os

import torch

try:
    from models.quant import hif8_coh as _coh
except Exception:
    _coh = None

PATH = os.environ.get("HIF8_PROBE", "")
EVERY = int(os.environ.get("HIF8_PROBE_EVERY", "25") or 25)
NMOD = int(os.environ.get("HIF8_PROBE_MODULES", "6") or 6)
ONLY_RANK = int(os.environ.get("HIF8_PROBE_RANK", "0") or 0)
ENABLED = bool(PATH)

# 档位边界（arXiv:2409.16626：3-bit 尾数覆盖 7 个阶码值，2-bit 覆盖 8 个，1-bit 覆盖 16 个）
_B3, _B2 = 7.0, 15.0
_UF_LOG2 = -29.0
_SAT = 32768.0

_cnt = {}
_dw_cnt = {}
# 跟踪名额按角色各算。早先写成跨角色共用是错的：前向先于反向执行，
# x/w 会把名额占满，role=g 永远排不上 —— 而梯度恰恰是最需要看的那档。
_order = {}
_fh = {"f": None}


def _rank():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


def _out():
    if _fh["f"] is None:
        f = io.open(PATH, "a", encoding="utf-8", newline="\n")
        if f.tell() == 0:
            f.write("call,role,label,numel,zfrac,amax,scale,p50,p99,r50,r99,"
                    "bin3,bin2,bin1,uf,ovr,sat,e3,e2,e1,top1\n")
        _fh["f"] = f
    return _fh["f"]


# ---- 逐步 ‖Δw‖ 记录（HIF8_DW=<csv> 开启，默认完全不生效）----
DW_PATH = os.environ.get("HIF8_DW", "")
_dw_prev = {}
_dw_fh = {"f": None}


def _dw_out():
    if _dw_fh["f"] is None:
        f = io.open(DW_PATH, "a", encoding="utf-8", newline="\n")
        if f.tell() == 0:
            f.write("step,label,dw,wnorm,dw_rel\n")
        _dw_fh["f"] = f
    return _dw_fh["f"]


def _dw_record(label, w, step):
    """w 是本步前向用的权重。留一份副本，下一步算 ‖w_t - w_{t-1}‖。"""
    try:
        with torch.no_grad():
            cur = w.detach().to(torch.float32)
            prev = _dw_prev.get(label)
            _dw_prev[label] = cur.clone()
            if prev is None or prev.shape != cur.shape:
                return
            d = float(torch.linalg.vector_norm(cur - prev))
            n = float(torch.linalg.vector_norm(cur))
            _dw_out().write("%d,%s,%.6e,%.6e,%.6e\n"
                            % (step, label, d, n, d / n if n > 0 else 0.0))
            _dw_out().flush()
    except Exception:
        return


# ---- 逐元素量化误差测量（HIF8_ELEM=<csv> 开启，默认完全不生效）----
# 在同一个进程里对同一个真实梯度算两种配方的误差，量化器行为被完全隔离。
# 核心量是误差相对梯度的平行/垂直分解：平行只是增益变化，垂直才旋转更新方向。
ELEM_PATH = os.environ.get("HIF8_ELEM", "")
ELEM_EVERY = int(os.environ.get("HIF8_ELEM_EVERY", "10") or 10)
ELEM_ROLES = os.environ.get("HIF8_ELEM_ROLES", "g").strip().lower()
ELEM_NMOD = int(os.environ.get("HIF8_ELEM_MODULES", "4") or 4)
# scale 陈旧倍数列表：s = amax(g)/dst_max * mult。空则回退到 live + 首次种下的 frozen。
_mstr = os.environ.get("HIF8_ELEM_MULTS", "").strip()
ELEM_MULTS = [float(t) for t in _mstr.split(",") if t.strip()]
_elem_order = {}
_elem_cnt = {}
_elem_froz = {}
_elem_fh = {"f": None}
_elem_prev = {}
_elem_first = {}
_ELEM_HEAD = "call,role,label,recipe,numel,nzfrac,scale,ytop,gnorm,enorm,epar,eperp,cos,relperp,zerofrac,mean_rel,std_rel,ac1,ac0,rbar"
_ELEM_FMT = "%d,%s,%s,%s,%d,%.6f,%.6e,%.4f,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6e,%.6e,%.6f,%.6f,%.6e"


def _elem_out():
    if _elem_fh["f"] is None:
        f = io.open(ELEM_PATH, "a", encoding="utf-8", newline=chr(10))
        if f.tell() == 0:
            f.write(_ELEM_HEAD + chr(10))
        _elem_fh["f"] = f
    return _elem_fh["f"]


def _elem_one(call, role, label, g, s_val, dst_max, recipe):
    """对给定 scale 用硬件算一遍精确量化误差，写一行。"""
    import torch_npu
    s = torch.full((1,), float(s_val), dtype=torch.float32, device=g.device)
    codes = torch_npu.npu_quantize(g, s, None, torch_npu.hifloat8, -1, True)
    ghat = torch_npu.npu_anti_quant(codes, s).float()
    gf = g.detach().float()
    e = ghat - gf
    gn = float(torch.linalg.vector_norm(gf))
    en = float(torch.linalg.vector_norm(e))
    dot = float((e * gf).sum())
    epar = dot / gn if gn > 0 else 0.0
    eperp = max(0.0, en * en - epar * epar) ** 0.5
    # 沿 token 轴归约成每特征向量：特征维有持久身份，可以算跨步方向相关。
    # 这也是误差进入 dW = sum_t x_t (x) g_t 的那条路径的代理。
    F = gf.shape[-1]
    ebar = e.reshape(-1, F).sum(0)
    gbar = gf.reshape(-1, F).sum(0)
    ebn = float(torch.linalg.vector_norm(ebar))
    gbn = float(torch.linalg.vector_norm(gbar))
    kk = "%s|%s|%s" % (role, label, recipe)
    ac1 = ac0 = 0.0
    pv = _elem_prev.get(kk)
    if pv is not None and pv.shape == ebar.shape and ebn > 0:
        pn = float(torch.linalg.vector_norm(pv))
        if pn > 0:
            ac1 = float((ebar * pv).sum()) / (ebn * pn)
    fv = _elem_first.get(kk)
    if fv is None:
        _elem_first[kk] = ebar.clone()
        ac0 = 1.0
    elif fv.shape == ebar.shape and ebn > 0:
        fn = float(torch.linalg.vector_norm(fv))
        if fn > 0:
            ac0 = float((ebar * fv).sum()) / (ebn * fn)
    _elem_prev[kk] = ebar.clone()
    rbar = ebn / gbn if gbn > 0 else 0.0
    nz = gf != 0
    nzn = int(nz.sum())
    if nzn:
        zf = float((ghat[nz] == 0).float().mean())
        r = e[nz] / gf[nz]
        mr, sr = float(r.mean()), float(r.std())
    else:
        zf = mr = sr = 0.0
    _elem_out().write((_ELEM_FMT + chr(10)) % (
        call, role, label, recipe, gf.numel(), nzn / max(1, gf.numel()),
        s_val, (float(gf.abs().max()) / s_val) if s_val > 0 else 0.0,
        gn, en, epar, eperp,
        dot / (en * gn) if en > 0 and gn > 0 else 0.0,
        eperp / gn if gn > 0 else 0.0, zf, mr, sr, ac1, ac0, rbar))
    _elem_out().flush()


def _elem_record(role, label, x, dst_max):
    if role not in [t.strip() for t in ELEM_ROLES.split(",")]:
        return
    key = "%s|%s" % (role, label)
    if key not in _elem_cnt:
        # 名额按角色各算，与探针 NMOD 一致。反向先到 layer 27，正是要看的那几个点。
        seen = _elem_order.setdefault(role, [])
        if len(seen) >= ELEM_NMOD:
            return
        seen.append(key)
    _elem_cnt[key] = _elem_cnt.get(key, 0) + 1
    call = _elem_cnt[key]
    try:
        with torch.no_grad():
            g = x.detach()
            amax = float(g.abs().max())
            if amax <= 0:
                return
            if key not in _elem_froz:
                _elem_froz[key] = amax
            if call % ELEM_EVERY != 0:
                return
            if ELEM_MULTS:
                # 同一梯度上扫多个陈旧倍数，recipe 列记为 m<倍数>
                for mu in ELEM_MULTS:
                    _elem_one(call, role, label, g, amax / dst_max * mu,
                              dst_max, "m%g" % mu)
            else:
                _elem_one(call, role, label, g, amax / dst_max, dst_max, "live")
                _elem_one(call, role, label, g, _elem_froz[key] / dst_max,
                          dst_max, "frozen")
    except Exception:
        return


# ---- 梯度结构分解（HIF8_STRUCT=<csv> 开启，默认完全不生效）----
STRUCT_PATH = os.environ.get("HIF8_STRUCT", "")
STRUCT_EVERY = int(os.environ.get("HIF8_STRUCT_EVERY", "50") or 50)
STRUCT_NMOD = int(os.environ.get("HIF8_STRUCT_MODULES", "8") or 8)
STRUCT_ROLES = os.environ.get("HIF8_STRUCT_ROLES", "g").strip().lower()
STRUCT_TOPK = int(os.environ.get("HIF8_STRUCT_TOPK", "10") or 10)
_st_cnt = {}
_st_order = {}
_st_fh = {"f": None}
_ST_COLS = ("call,role,label,nrow,ncol,amax,"
            "rp99,rp50,rsecond,rows_half,rtop_frac,"
            "cp99,cp50,csecond,cols_half,ctop_frac,"
            "rtop_idx,rtop_vals,ctop_idx")


def _st_out():
    if _st_fh["f"] is None:
        f = io.open(STRUCT_PATH, "a", encoding="utf-8", newline=chr(10))
        if f.tell() == 0:
            f.write(_ST_COLS + chr(10))
        _st_fh["f"] = f
    return _st_fh["f"]


def _st_stats(v, n):
    """返回 (max, p99, p50, second, 超过 max/2 的个数)。v 已是非负一维。"""
    vs = torch.sort(v, descending=True).values
    mx = float(vs[0])
    sec = float(vs[1]) if n > 1 else 0.0
    p99 = float(vs[max(0, int(n * 0.01))])
    p50 = float(vs[n // 2])
    half = int((v > mx * 0.5).sum())
    return mx, p99, p50, sec, half


def _st_record(role, label, x, call):
    try:
        with torch.no_grad():
            t = x.detach()
            if t.dim() < 2:
                return
            a = t.abs().reshape(-1, t.shape[-1]).float()
            nrow, ncol = int(a.shape[0]), int(a.shape[1])
            if nrow < 4 or ncol < 4:
                return
            amax = float(a.max())
            if amax <= 0:
                return
            rinf = a.amax(dim=1)
            cinf = a.amax(dim=0)
            _, rp99, rp50, rsec, rhalf = _st_stats(rinf, nrow)
            _, cp99, cp50, csec, chalf = _st_stats(cinf, ncol)
            thr = amax / 8.0
            ri = int(torch.argmax(rinf))
            ci = int(torch.argmax(cinf))
            rtop_frac = float((a[ri] > thr).float().mean())
            ctop_frac = float((a[:, ci] > thr).float().mean())
            k = min(STRUCT_TOPK, nrow)
            tv, ti = torch.topk(rinf, k)
            ridx = " ".join(str(int(i)) for i in ti.tolist())
            rvals = " ".join("%.3e" % float(v) for v in tv.tolist())
            k2 = min(STRUCT_TOPK, ncol)
            _, cidx_t = torch.topk(cinf, k2)
            cidx = " ".join(str(int(i)) for i in cidx_t.tolist())
            row = ("%d,%s,%s,%d,%d,%.6e,"
                   "%.6e,%.6e,%.6e,%d,%.6f,"
                   "%.6e,%.6e,%.6e,%d,%.6f,"
                   "%s,%s,%s"
                   % (call, role, label, nrow, ncol, amax,
                      rp99, rp50, rsec, rhalf, rtop_frac,
                      cp99, cp50, csec, chalf, ctop_frac,
                      ridx, rvals, cidx))
            _st_out().write(row + chr(10))
            _st_out().flush()
    except Exception:
        return


def record(role, label, x, scale, dst_max):
    if STRUCT_PATH and _rank() == 0 and role in STRUCT_ROLES:
        _k = "%s|%s" % (role, label)
        if _k not in _st_cnt:
            _seen = _st_order.setdefault(role, [])
            if len(_seen) < STRUCT_NMOD:
                _seen.append(_k)
                _st_cnt[_k] = 0
        if _k in _st_cnt:
            _st_cnt[_k] += 1
            if _st_cnt[_k] % STRUCT_EVERY == 0:
                _st_record(role, label, x, _st_cnt[_k])
    if ELEM_PATH and _rank() == 0:
        _elem_record(role, label, x, dst_max)
    if DW_PATH and role == "w" and _rank() == 0:
        _dw_cnt[label] = _dw_cnt.get(label, 0) + 1
        if len(_dw_prev) < 4 or label in _dw_prev:
            _dw_record(label, x, _dw_cnt[label])
    if not ENABLED:
        return
    if ONLY_RANK >= 0 and _rank() != ONLY_RANK:
        return
    # 相干性测量要逐步累积，必须在下面的 EVERY 抽样门之前
    if _coh is not None and _coh.ENABLED:
        _coh.record(role, label, x, scale)
    key = "%s|%s" % (role, label)
    if key not in _cnt:
        seen = _order.setdefault(role, [])
        if len(seen) >= NMOD:
            return
        seen.append(key)
        _cnt[key] = 0
    _cnt[key] += 1
    c = _cnt[key]
    if c % EVERY != 0:
        return

    try:
        with torch.no_grad():
            a_all = x.detach().abs().reshape(-1).float()
            n_all = a_all.numel()
            amax = float(a_all.max())
            s = float(scale.reshape(-1)[0]) if torch.is_tensor(scale) else float(scale)
            if amax <= 0 or s <= 0 or n_all < 2:
                return
            # 精确零单独统计：padding / 被 mask 的位置梯度本来就是 0，
            # 把它们算进「下溢」或 1-bit 档会把指标彻底污染（实测会让 r50 虚高到 1e26）。
            nz = a_all > 0
            zfrac = 1.0 - float(nz.float().mean())
            a = a_all[nz]
            n = a.numel()
            if n < 2:
                return
            p50 = float(a.median())
            k = min(n - 1, max(0, int(n * 0.99)))
            p99 = float(a.kthvalue(k + 1).values)
            y = a / s
            ly = torch.log2(torch.clamp(y, min=1e-45))
            # 档位按 HiF8 自己的**绝对**阶码空间判（实测，见 hif8_bands.py），
            # 不再按「低于 dst_max 多少个 binade」—— 后者在反向 224 上是错的。
            kk = torch.floor(ly)
            m3 = (kk >= -3) & (kk <= 3)
            m2 = ((kk >= -7) & (kk <= -4)) | ((kk >= 4) & (kk <= 7))
            m1 = ((kk >= -15) & (kk <= -8)) | ((kk >= 8) & (kk <= 14))
            b3 = float(m3.float().mean())
            b2 = float(m2.float().mean())
            b1 = float(m1.float().mean())
            uf = float((kk < -15).float().mean())
            ovr = float((y > float(dst_max)).float().mean())
            sat = float((y > _SAT).float().mean())
            # 按能量（a^2）加权的档位占比。bin* 是按「元素个数」算的，而 1-bit 档里
            # 装的恰好是最小的元素、对梯度范数几乎无贡献 —— 实测 grad_norm 在 CTS
            # loss 偏差 +2.5% 时仍与 bf16 差 1.3% 以内，说明承载能量的大元素一直在
            # 3-bit 档里。所以要判「量化是否真的伤了梯度」，必须看能量加权这一组。
            w = a * a
            tot = float(w.sum())
            if tot > 0:
                e3 = float(w[m3].sum()) / tot
                e2 = float(w[m2].sum()) / tot
                e1 = float(w[m1].sum()) / tot
                top1 = float(w.max()) / tot          # 最大单元素占总能量比，衡量离群主导程度
            else:
                e3 = e2 = e1 = top1 = 0.0
    except Exception:
        return

    _out().write("%d,%s,%s,%d,%.5f,%.6g,%.6g,%.6g,%.6g,%.4g,%.4g,"
                 "%.5f,%.5f,%.5f,%.3e,%.3e,%.3e,%.5f,%.5f,%.5f,%.5f\n"
                 % (c, role, label, n_all, zfrac, amax, s, p50, p99,
                    amax / max(p50, 1e-30), amax / max(p99, 1e-30),
                    b3, b2, b1, uf, ovr, sat, e3, e2, e1, top1))
    _out().flush()
