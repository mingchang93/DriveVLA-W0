# -*- coding: utf-8 -*-
"""HiF8 延迟张量缩放（DTS），按我们自己上个项目的实现移植。

参考：/mnt/sfs_turbo/zyc/hif8/FSDPTurbo/fsdp_turbo/quantization/hif8_formats/hif8_dts.py
（364 行。本文件是它在 HF Trainer + DeepSpeed 这条线上的移植：去掉 FSDPTurbo 的
config 对象依赖，量化点由 HiF8DTSLinear 直接创建，其余语义逐条照搬。）

CTS 每次调用都从当前张量重算 scale。DTS 为每个 (module, role) 保持一个常驻 scale，
复用 interval 步，然后从 amax 历史窗口刷新一次 —— 并在数据并行 rank 间做 MAX 规约，
使所有 rank 对同一个 scale 量化。

这个 recipe 存在的理由是吞吐，所以 amax 一个窗口只采样一次：

* 未采样的那些步是纯静态量化：npu_quantize 只扫张量一遍。CTS 的 npu_dynamic_quant
  要扫两遍 —— 一遍归约 amax、一遍除以刚算出的 scale。省下的这一遍就是 DTS 的
  全部性能收益。
* 采样的那一步用 npu_quant_max，一遍同时返回码点和 amax，所以采样也不额外多扫一遍。

amax_sample_last 默认 1：只采样刷新前的那一步，也是该次刷新能用到的最新样本。

对 TE 参考的两处有意偏离（沿用参考实现的选择，理由见它的 docstring）：
* 保留 per-role target（inputs/weight 15、grads 224），不用 TE 硬编码的 32768。
  实测前向 GEMM 相对误差：32768 是 1.45e-1，15 只有 3.80e-2。
* 历史窗口默认更短：history_len=1 + most_recent，scale 就是窗口采样步的 amax，
  是 DTS 能做到的最紧。更长的 max 窗口只买到抗截断，而这里的余量
  （32768/15 = 2185 倍、32768/224 = 146 倍）本来就不会截断。

重算激活（gradient_checkpointing）不需要特殊保护：scale 只在 optimizer 的
post-step hook 里变，也就是只在步与步之间变，所以重算的前向读到的就是原前向用的
那个 scale；而窗口把 amax 折成 running MAX，是幂等的，同一张量观测两次不会偏。

scale 要钳住下界：CTS 能容忍全零张量，因为 npu_quant_matmul 是乘 scale，零 scale
给出零结果（是对的）；npu_quantize 是除，同样的零会产出 inf。

我第一版写错的地方（已全部修掉），留在这里以免再犯：
  1. 采样步用了 npu_dynamic_quant（两遍），把 DTS 的性能理由弄丢了
  2. 刷新步当场算 amax 并立刻用 -> 实际是「第 0 步 CTS + 后 9 步静态」，不是延迟
  3. 没有跨 rank MAX，各 rank 用各自的 scale
  4. 没有 scale 下界钳位
  5. 时钟靠 Trainer 回调注入，而不是 optimizer post-step hook
"""

import logging
import os
import weakref
from typing import Optional

import torch
import torch.nn as nn

try:
    from models.quant import hif8_probe as _probe
except Exception:  # 探针缺失不影响训练
    _probe = None

logger = logging.getLogger(__name__)

# 时钟探测：只在 rank0 打到 stdout。logger.info 实测进不了 train.log，
# 所以这里用 print —— 由 HIF8_DTS_CLOCK=1 开，默认完全静默。
_CLOCK_DEBUG = os.environ.get("HIF8_DTS_CLOCK", "") == "1"


def _say(msg):
    """rank0 打到 stdout。实测 logger.info 进不了 train.log，所以关键诊断用 print。"""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        print("[HiF8-DTS] %s" % msg, flush=True)
    except Exception:
        pass


def _clk(msg):
    if not _CLOCK_DEBUG:
        return
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        print("[HiF8-DTS-CLOCK] %s" % msg, flush=True)
    except Exception:
        pass

try:
    import torch_npu
    _HIF8 = getattr(torch_npu, "hifloat8", None)
except Exception:
    torch_npu = None
    _HIF8 = None

HIF8_MAX = 32768.0
_SCALE_FLOOR = torch.finfo(torch.float32).tiny
AMAX_COMPUTE_ALGOS = ("most_recent", "max")

# 与 CTS 同源，照搬 MindSpeed 的两档动态范围
FWD_DST_MAX = 15.0    # FormatEnum.HIF8_15
BWD_DST_MAX = 224.0   # FormatEnum.HIF8_224

DEFAULT_APPLY = ("model.language_model.layers.",)
DEFAULT_IGNORE = ("lm_head", "o_proj")
SMALL_LINEAR_DIM = 64

# npu_quant_max 会校验 round_mode 与目标类型是否匹配并直接报错，不是忽略：
# round（向最近整数）是 hifloat8 的模式，rint（五成双）是 e5m2/e4m3fn 的。
_ROUND_MODES = {}
if hasattr(torch, "float8_e5m2"):
    _ROUND_MODES[torch.float8_e5m2] = "rint"
if hasattr(torch, "float8_e4m3fn"):
    _ROUND_MODES[torch.float8_e4m3fn] = "rint"


def hif8_available() -> bool:
    return (
        torch_npu is not None
        and _HIF8 is not None
        and hasattr(torch_npu, "npu_quantize")
        and hasattr(torch_npu, "npu_quant_max")
        and hasattr(torch_npu, "npu_quant_matmul")
    )


def _probe_rec(point, x):
    """只读记录。role 从 label 后缀取，dst_max 用 point.target，与 CTS 侧同口径。"""
    if _probe is None or not _probe.ENABLED:
        return
    try:
        _probe.record(point.label.rsplit(".", 1)[-1], point.label,
                      x, point.scale, point.target)
    except Exception:
        pass


# ---- scale 用什么统计量：默认 amax，可换成分位数（HIF8_DTS_STAT）----
# amax 是千万个坐标里的单点极值（实测 amax/p99 = 4.1~34.5），用它定 scale 等于
# 把整张量的栅格绑在一个离群点上。HIF8_DTS_STAT=p99 换成稳健统计量。
# 分位数只在**非零**元素上算：role=g 有 96.53% 的元素是精确零（padding/mask），
# 全张量的 0.99 分位只相当于非零部分的 0.71 分位，不是一回事。
_STAT = os.environ.get("HIF8_DTS_STAT", "amax").strip().lower()
_STAT_N = int(os.environ.get("HIF8_DTS_STAT_N", "1048576") or 1048576)
_STAT_ROLES = os.environ.get("HIF8_DTS_STAT_ROLES", "g").strip().lower()
if _STAT in ("", "amax", "max"):
    _STAT_Q = 0.0
elif _STAT.startswith("p"):
    _STAT_Q = float(_STAT[1:]) / 100.0
else:
    _STAT_Q = float(_STAT)
if not 0.0 <= _STAT_Q < 1.0:
    raise ValueError("HIF8_DTS_STAT 只接受 amax / p<xx> / 0<q<1，收到 %r" % _STAT)


def _stat_applies(point):
    if _STAT_Q <= 0.0:
        return False
    role = point.label.rsplit(".", 1)[-1] if point.label else ""
    return role in [t.strip() for t in _STAT_ROLES.split(",")]


def _scale_stat(point, x):
    """返回用来定 scale 的标量，shape (1,) float32。"""
    a = x.detach().abs().reshape(-1)
    if not _stat_applies(point):
        return a.max().reshape(1)
    n = a.numel()
    if n > _STAT_N:
        # 随机下标，不用定步长：梯度按 (token, feature) 展平，定步长会和 feature
        # 维混叠、永远取到同一批特征位。采样噪声无害 —— 已实测 scale 的每步抖动
        # 对精度没有影响（压 20 倍、偏差分毫不动）。
        a = a.index_select(0, torch.randint(0, n, (_STAT_N,), device=a.device))
    a = a.float()
    nz = a[a > 0]
    m = nz.numel()
    if m < 1000:
        return a.max().reshape(1)        # 非零太少，分位数不可靠，退回 amax
    k = min(m, max(1, int(m * _STAT_Q) + 1))
    return nz.kthvalue(k).values.reshape(1)


class ScalePoint:
    """一个量化点：一个常驻 scale 和观测它的窗口。"""

    __slots__ = ("scale", "inv_scale", "pending", "initialized", "target",
                 "order", "label")

    def __init__(self, target, order=0, label=""):
        # 0.0 表示「用 hifloat8 自己的最大值」，与 dst_type_max 的约定一致
        self.target = HIF8_MAX if target == 0.0 else target
        self.scale = None          # (1,) float32
        # scale 的倒数：npu_quant_max 是乘，npu_quantize 是除
        self.inv_scale = None
        # 本窗口已采样步的 amax 的 MAX；窗口之间为 None
        self.pending = None
        self.initialized = False
        self.order = order
        self.label = label


class _DelayedScaler:
    def __init__(self, interval, amax_sample_last=None, history_len=1,
                 compute_algo="most_recent", allreduce=True):
        if interval < 2:
            raise ValueError("DTS 需要 interval >= 2（1 就是 CTS），收到 %r" % interval)
        self.interval = interval
        self.sample_last = 1 if amax_sample_last is None else amax_sample_last
        if not 1 <= self.sample_last <= interval - 1:
            raise ValueError(
                "amax_sample_last 必须在 1..%d（interval=%d），收到 %r。"
                "特别地 0 不是静态缩放开关：它什么都不采样，"
                "每个 scale 会整轮停在冷启动值上"
                % (interval - 1, interval, self.sample_last))
        if history_len < 1:
            raise ValueError("history_len 必须 >= 1，收到 %r" % history_len)
        if compute_algo not in AMAX_COMPUTE_ALGOS:
            raise ValueError("compute_algo 必须是 %s 之一，收到 %r"
                             % (AMAX_COMPUTE_ALGOS, compute_algo))
        self.history_len = history_len
        self.compute_algo = compute_algo
        self.allreduce = allreduce
        # 静态缩放：首次调用种下 scale、step 1 跨 rank 对齐、此后不刷新。
        # 这是 2026-11-18 前所有「DTS」运行的实际行为，实测优于真延迟缩放 56 倍。
        self.static = os.environ.get("HIF8_DTS_STATIC", "0") not in ("0", "", "false", "False", "no")
        # 只重种一次：静态配方下，在第 R 个训练步刷新一次 scale，之后再冻住。
        # 用来分开「任何改变」与「累计漂移」—— 两者在冻结臂里是绑死的。
        self.reseed = int(os.environ.get("HIF8_DTS_RESEED", "0") or 0)
        self.step = 0
        self.points = []
        self.observations = 0
        self.refreshes = 0
        self._history = None
        self._history_filled = None
        self._lead_optimizer = None
        self._hook_handle = None

    def register(self, point):
        point.order = len(self.points)
        self.points.append(point)

    def observing_now(self):
        if self.static:
            # 静态：永不采样；唯一例外是 RESEED 指定的那一步要采一次
            return self.reseed > 0 and self.step == self.reseed - 1
        pos = self.step % self.interval
        return pos != 0 and pos >= self.interval - self.sample_last

    # ---- 量化路径 ----
    def quantize(self, point, x, dst_type):
        """按 point 存的 scale 量化 x，返回 (codes, scale)。"""
        if not point.initialized:
            # 一个点的第一次调用是 CTS，落到的 scale 就是它保留的那个。
            # 不能用 npu_quant_max：它需要一个 scale 才能量化，而这正是唯一
            # 还没有 scale 的那次调用 —— 所以这里两个 kernel，每个点一次。
            self.seed(point, _scale_stat(point, x))
            _q = torch_npu.npu_quantize(x, point.scale, None, dst_type, -1, True)
            _probe_rec(point, x)
            return (_q, point.scale)
        if not self.observing_now():
            # 扫一遍 x，任何地方都不算 amax
            _q = torch_npu.npu_quantize(x, point.scale, None, dst_type, -1, True)
            _probe_rec(point, x)
            return (_q, point.scale)
        # 采样步：码点和 amax(x) 出自同一遍。npu_quant_max 算 cast(x * scale)，
        # 所以喂的是 npu_quantize 所除的那个值的倒数。
        if _stat_applies(point):
            # 分位数口径下用不了融合的 npu_quant_max（它只产出 amax），
            # 所以码点和统计量分两遍。这笔开销只在采样步付（1/interval）。
            codes = torch_npu.npu_quantize(x, point.scale, None, dst_type, -1, True)
            self.note_amax(point, _scale_stat(point, x))
        else:
            codes, amax = torch_npu.npu_quant_max(
                x, point.inv_scale, round_mode=_ROUND_MODES.get(dst_type, "round"),
                dst_dtype=dst_type)
            self.note_amax(point, amax)
        _probe_rec(point, x)
        return codes, point.scale

    def seed(self, point, amax):
        """用一个点的第一个 amax 锁定它的第一个 scale。"""
        amax = amax.detach().to(torch.float32).reshape(1)
        self._set_scale(point, torch.clamp(amax / point.target, min=_SCALE_FLOOR))
        point.initialized = True

    def note_amax(self, point, amax):
        """把一次观测到的 amax 折进这个点正在累积的窗口。"""
        amax = amax.detach().to(torch.float32).reshape(1)
        point.pending = (amax if point.pending is None
                         else torch.maximum(point.pending, amax))
        self.observations += 1

    @staticmethod
    def _set_scale(point, scale):
        # 重新绑定，绝不原地写：前向把 scale 存给反向用，刷新不能回头改变
        # 已有码点当初是按哪个 scale 量化的。
        point.scale = scale.contiguous()
        point.inv_scale = point.scale.reciprocal()

    # ---- 时钟 ----
    def on_optimizer_step(self, optimizer):
        # 只让一个 optimizer 驱动时钟；第二个（比如单独的参数组）不能把窗口
        # 在一个训练步里推两次。用 weakref 而不是 id()：CPython 会复用被回收
        # 对象的地址，一个旧 id 可能和后来无关的对象相等。
        lead = self._lead_optimizer() if self._lead_optimizer is not None else None
        if lead is None:
            self._lead_optimizer = weakref.ref(optimizer)
            _clk("lead latched: %s" % type(optimizer).__name__)
        elif lead is not optimizer:
            _clk("rejected non-lead: %s (lead=%s)"
                 % (type(optimizer).__name__, type(lead).__name__))
            return
        self._advance()

    def _advance(self):
        """把窗口推进一个训练步。每训练步必须恰好调用一次。"""
        self.step += 1
        if self.step <= 3 or self.step % self.interval == 0:
            _clk("step=%d" % self.step)
        if self.step == 1:
            self._sync_cold_start()
            return
        if self.static:
            if self.reseed > 0 and self.step == self.reseed and self.points:
                _say("RESEED：在第 %d 步重种一次 scale，之后继续冻住" % self.step)
                self._refresh()
            return                 # 静态：除 RESEED 那一步外什么都不做
        if self.step % self.interval == 0 and self.points:
            self._refresh()

    def _group_ok(self, n, device):
        """所有 rank 对「有多少个点」是否一致？

        不一致会让批量 all_reduce 形状不匹配而挂住，用 [n, -n] 探一下，
        不一致就退回 per-rank scale。device 取自调用方自己的点，不取
        self.points[0]：已注册但还没跑过的层 scale 仍是 None。
        """
        if not (self.allreduce and torch.distributed.is_initialized()):
            return False
        if torch.distributed.get_world_size() <= 1:
            return False
        probe = torch.tensor([n, -n], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(probe, op=torch.distributed.ReduceOp.MAX)
        if int(probe[0]) != n or int(probe[1]) != -n:
            logger.warning("HiF8 DTS: rank 间点数不一致（本 rank %d，最大 %d），"
                           "退回 per-rank scale", n, int(probe[0]))
            self.allreduce = False
            return False
        return True

    def _sync_cold_start(self):
        pts = [p for p in self.points if p.initialized]
        if not pts or not self._group_ok(len(pts), pts[0].scale.device):
            return
        cold = torch.cat([p.scale for p in pts])
        torch.distributed.all_reduce(cold, op=torch.distributed.ReduceOp.MAX)
        inv = cold.reciprocal()
        for p, s, i in zip(pts, cold.split(1), inv.split(1)):
            p.scale = s.contiguous()
            p.inv_scale = i.contiguous()

    def _refresh(self):
        pts = [p for p in self.points if p.pending is not None]
        self.refreshes += 1
        _clk("refresh #%d at step=%d: %d/%d points had pending"
             % (self.refreshes, self.step, len(pts), len(self.points)))
        # 硬自检：首次刷新一个点都没采到，说明时钟或采样条件坏了，scale 会就此
        # 冻结在冷启动值、整段训练退化成静态缩放。这一条必须响，不能静默。
        # （实测踩过：optimizer hook 每步触发两次 -> self.step 恒为偶数 ->
        #   pos == interval-1 永不成立 -> 全程 0/N pending。）
        if self.refreshes == 1 and not pts:
            raise RuntimeError(
                "HiF8 DTS 时钟坏了：第 1 次刷新（step=%d, interval=%d）"
                "%d 个量化点一个都没采到 amax。scale 会冻结在冷启动值、"
                "退化成静态缩放。检查时钟源是否每训练步恰好推进一次"
                "（self.step 当前=%d），以及 observing_now() 的奇偶性。"
                % (self.step, self.interval, len(self.points), self.step))
        if self.refreshes <= 3 or self.refreshes % 200 == 0:
            _say("refresh #%d at step=%d: %d/%d points sampled"
                 % (self.refreshes, self.step, len(pts), len(self.points)))
        logger.info("HiF8 DTS 第 %d 步刷新：点 %d 个，本窗口采样到 %d 个，观测 %d 次",
                    self.step, len(self.points), len(pts), self.observations)
        self.observations = 0
        if not pts:
            return
        pending = torch.cat([p.pending for p in pts])
        if self._group_ok(len(pts), pending.device):
            torch.distributed.all_reduce(pending, op=torch.distributed.ReduceOp.MAX)
        amax = self._roll_history(pts, pending)
        # per-role target，这是相对 TE 参考的偏离处
        targets = torch.tensor([p.target for p in pts], dtype=torch.float32,
                               device=amax.device)
        new = torch.clamp(amax / targets, min=_SCALE_FLOOR)
        inv = new.reciprocal()
        # 用 split(1) 而不是每点 clone：每个点拿这两个 buffer 的一个 (1,) 视图，
        # 下次刷新另外分配新的，而不是覆写它们，所以已存下的 scale 仍然有效。
        for p, s, i in zip(pts, new.split(1), inv.split(1)):
            p.scale = s.contiguous()
            p.inv_scale = i.contiguous()
            p.pending = None

    def _roll_history(self, pts, pending):
        """把 pending 推进 amax 历史，返回用来算 scale 的 amax。

        对所有刷新中的点批量做：逐点 roll 会是 4 个 kernel x 504 个点，
        正是这个 recipe 想省掉的那笔 launch 开销。
        """
        if self.history_len == 1:
            return pending
        if self._history is None:
            self._history = torch.zeros(len(self.points), self.history_len,
                                        dtype=torch.float32, device=pending.device)
            self._history_filled = torch.zeros(len(self.points), dtype=torch.bool,
                                               device=pending.device)
        idx = torch.tensor([p.order for p in pts], dtype=torch.long,
                           device=pending.device)
        rows = self._history.index_select(0, idx)
        rolled = torch.cat([rows[:, 1:], pending.unsqueeze(1)], dim=1)
        # 第一次刷新的点，窗口要填满它自己的值，而不是 buffer 分配时的那些零
        filled = self._history_filled.index_select(0, idx).unsqueeze(1)
        rows = torch.where(filled, rolled, pending.unsqueeze(1).expand_as(rolled))
        self._history.index_copy_(0, idx, rows)
        self._history_filled.index_fill_(0, idx, True)
        if self.compute_algo == "max":
            return rows.amax(dim=1)
        return rows[:, -1]


_SCALER = None


def enable(interval, amax_sample_last=None, history_len=1,
           compute_algo="most_recent"):
    """切到延迟缩放。必须在替换 Linear 之前调用。"""
    global _SCALER
    if _SCALER is not None:
        return
    # amax 历史窗口可由环境变量覆盖；不设则沿用调用方传入的值。
    # HIF8_DTS_HISTORY=16 HIF8_DTS_ALGO=max 让 scale 跟随 amax 的真实衰减、
    # 同时滤掉单步极值噪声（实测梯度 amax 每步跳 ±28.9%，且与梯度主体无关，
    # corr=-0.03；scale 跟着跳会给整张量注入一个不会相消的共模栅格平移）。
    # 取值非法时由 _DelayedScaler.__init__ 直接抛错，不静默回退。
    _env_hist = os.environ.get("HIF8_DTS_HISTORY", "").strip()
    _env_algo = os.environ.get("HIF8_DTS_ALGO", "").strip()
    if _env_hist:
        history_len = int(_env_hist)
    if _env_algo:
        compute_algo = _env_algo
    _SCALER = _DelayedScaler(interval, amax_sample_last, history_len, compute_algo)
    if _env_hist or _env_algo:
        # 必须进 stdout：被静默忽略的开关正是时钟 bug 让整段实验跑成另一个
        # 配方的原因，module logger.info 到不了 train.log。
        _say("amax 历史窗口来自环境变量：history_len=%d compute_algo=%s"
             % (_SCALER.history_len, _SCALER.compute_algo))
    # 时钟源。默认 trainer：由训练循环每步调一次 tick()，构造上保证一步一次。
    # optimizer：原实现，挂 register_optimizer_step_post_hook —— 在 HF Trainer +
    # DeepSpeed 这条线上实测每训练步触发两次，会让偶数 interval 永不采样，
    # 只保留用于复现历史运行。
    src = os.environ.get("HIF8_DTS_CLOCK_SRC", "trainer").strip().lower()
    _SCALER.clock_src = src
    if src == "optimizer":
        from torch.optim.optimizer import register_optimizer_step_post_hook
        _SCALER._hook_handle = register_optimizer_step_post_hook(
            lambda opt, args, kwargs: _SCALER.on_optimizer_step(opt))
        _say("clock source = optimizer post-step hook（历史行为）")
    elif src == "trainer":
        _say("clock source = trainer callback，训练侧必须每步调 hif8_dts.tick()")
    if _STAT_Q > 0.0:
        _say("scale 统计量 = %s（q=%.4f, 子样 %d, 只作用于 role=%s）"
             % (_STAT, _STAT_Q, _STAT_N, _STAT_ROLES))
    if _SCALER.static:
        _say("recipe = STATIC（首次调用种下 scale + step1 跨 rank 对齐，此后不刷新）"
             "；HIF8_INTERVAL 在此模式下无意义")
        if _SCALER.reseed > 0:
            _say("recipe = STATIC + RESEED@%d（全程恒定，只在第 %d 步重种一次）"
                 % (_SCALER.reseed, _SCALER.reseed))
    logger.info("HiF8 延迟缩放已启用：interval=%d amax_sample_last=%d "
                "history_len=%d compute_algo=%s", interval, _SCALER.sample_last,
                _SCALER.history_len, _SCALER.compute_algo)


def tick():
    """把 DTS 的窗口推进一个训练步。HIF8_DTS_CLOCK_SRC=trainer 时由训练循环每步调用。

    幂等性不在这里保证 —— 调用方必须每训练步恰好调一次。这正是原来挂
    optimizer hook 出的问题：同一个 optimizer 每步触发两次。
    """
    if _SCALER is None:
        return
    if getattr(_SCALER, "clock_src", "trainer") != "trainer":
        return
    _SCALER._advance()


def enabled():
    return _SCALER is not None


def new_point(target, label=""):
    point = ScalePoint(target, 0, label)
    _SCALER.register(point)
    return point


def state():
    """scaler 当前在做什么，给运行日志用。"""
    if _SCALER is None:
        return {"enabled": False}
    return {"enabled": True, "interval": _SCALER.interval,
            "sample_last": _SCALER.sample_last,
            "history_len": _SCALER.history_len,
            "compute_algo": _SCALER.compute_algo,
            "step": _SCALER.step, "points": len(_SCALER.points),
            "refreshes": _SCALER.refreshes, "allreduce": _SCALER.allreduce,
            "clock_src": getattr(_SCALER, "clock_src", "?"),
            "static": getattr(_SCALER, "static", False)}


def _mm_kwargs():
    return {"x1_dtype": _HIF8, "x2_dtype": _HIF8}


def _as_2d(t):
    if t.dim() == 2:
        return t
    return t.reshape(-1, t.shape[-1])


class _HiF8DTSMatMul(torch.autograd.Function):
    """y = x @ w.T，三处量化都走延迟缩放的常驻 scale。"""

    @staticmethod
    def forward(ctx, x, weight, px, pw, pg):
        x2 = _as_2d(x)
        mmk = _mm_kwargs()
        x_q, x_s = _SCALER.quantize(px, x2, _HIF8)
        w_q, w_s = _SCALER.quantize(pw, weight, _HIF8)
        out = torch_npu.npu_quant_matmul(
            x_q, w_q.t(), w_s, pertoken_scale=x_s, output_dtype=x.dtype, **mmk)
        if x.dim() != 2:
            out = out.reshape(*x.shape[:-1], out.shape[-1])
        ctx.x_q, ctx.x_s, ctx.w_q, ctx.w_s = x_q, x_s, w_q, w_s
        ctx.pg = pg
        ctx.out_dtype = x.dtype
        ctx.x_shape = tuple(x.shape)
        ctx.need_dx = x.requires_grad
        ctx.need_dw = weight.requires_grad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mmk = _mm_kwargs()
        g2 = _as_2d(grad_out)
        g_q, g_s = _SCALER.quantize(ctx.pg, g2, _HIF8)
        dx = dw = None
        if ctx.need_dx:
            # 前向用的是 w_q.t()，所以这里用未转置的 w_q
            dx = torch_npu.npu_quant_matmul(
                g_q, ctx.w_q, ctx.w_s, pertoken_scale=g_s,
                output_dtype=ctx.out_dtype, **mmk)
            if len(ctx.x_shape) != 2:
                dx = dx.reshape(*ctx.x_shape[:-1], dx.shape[-1])
        if ctx.need_dw:
            dw = torch_npu.npu_quant_matmul(
                g_q.T, ctx.x_q, ctx.x_s, pertoken_scale=g_s,
                output_dtype=ctx.out_dtype, **mmk)
        return dx, dw, None, None, None


class HiF8DTSLinear(nn.Linear):
    """与 nn.Linear 等价的替换件，矩阵乘走 HiF8-DTS。

    scale 用普通属性持有、不注册为 buffer：一旦进 state_dict，存出来的 ckpt
    结构就和 bf16 那把不一样，评测加载时全是 unexpected key，对照也不干净。
    """

    @classmethod
    def from_linear(cls, lin, label=""):
        new = cls(lin.in_features, lin.out_features,
                  bias=lin.bias is not None, device="meta")
        new.weight = lin.weight
        new.bias = lin.bias
        new._px = new_point(FWD_DST_MAX, label + ".x")
        new._pw = new_point(FWD_DST_MAX, label + ".w")
        new._pg = new_point(BWD_DST_MAX, label + ".g")
        return new

    def forward(self, x):
        out = _HiF8DTSMatMul.apply(x, self.weight, self._px, self._pw, self._pg)
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_to_hif8_dts(
    model,
    apply=DEFAULT_APPLY,
    ignore=DEFAULT_IGNORE,
    min_features=SMALL_LINEAR_DIM,
    interval=10,
    amax_sample_last=None,
    history_len=1,
    compute_algo="most_recent",
):
    """把符合范围的 nn.Linear 原地换成 HiF8DTSLinear。范围口径与 CTS 完全一致。"""
    if not hif8_available():
        raise RuntimeError("当前环境缺 torch_npu.hifloat8 / npu_quantize / "
                           "npu_quant_max / npu_quant_matmul")
    enable(interval, amax_sample_last, history_len, compute_algo)

    converted, skipped = [], []

    def _walk(mod, prefix):
        for name, child in list(mod.named_children()):
            full = (prefix + "." + name) if prefix else name
            if isinstance(child, nn.Linear):
                if any(p in full for p in ignore):
                    skipped.append(full)
                elif apply and not any(p in full for p in apply):
                    skipped.append(full)
                elif min(child.in_features, child.out_features) < min_features:
                    skipped.append(full)
                else:
                    setattr(mod, name, HiF8DTSLinear.from_linear(child, full))
                    converted.append(full)
            else:
                _walk(child, full)

    _walk(model, "")
    return converted, skipped
