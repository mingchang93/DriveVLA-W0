#!/usr/bin/env python3
"""Zhao's MFU methodology, kept separate from ``utils/flops.py``.

Mirrors ``utils/monitor.py:FlopsCounter`` in
``408235117/DriveVLA-W0-950DT`` (branch ``emu3-npu-port``):

* hooks ``nn.Linear`` only (no Conv, no attention score matmul),
* counts each module at most once per step (``_seen_modules``),
* forward FLOPs are ``2 * rows * in_features * out_features``,
* training FLOPs are ``3x`` the trainable forward plus ``1x`` the frozen,
* MFU = FLOPs / (wall-clock step time * 486.354 TFLOPS).

It is intentionally a second, independent counter: ``utils/flops.py`` (the
project counter, Linear + Conv + attention) is untouched.
"""

import os

import torch
import torch.nn as nn

__all__ = ["FlopsCounterZhao", "ZHAO_PEAK_TFLOPS"]

ZHAO_PEAK_TFLOPS = float(os.getenv("DRIVEVLA_PEAK_TFLOPS", "486.354"))


class FlopsCounterZhao:
    """Linear-only forward FLOPs, 3x trainable + 1x frozen."""

    def __init__(self):
        self.forward_flops_trainable = 0.0
        self.forward_flops_frozen = 0.0
        self._seen_modules = set()
        self._trainable = {}
        self._handles = []
        self._attached = False

    @property
    def attached(self):
        return self._attached

    def attach(self, model):
        if self._attached:
            return
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self._trainable[id(module)] = any(
                    p.requires_grad for p in module.parameters(recurse=False)
                )
                self._handles.append(module.register_forward_hook(self._hook))
        self._attached = True

    def _hook(self, module, inputs, output):
        mid = id(module)
        if mid in self._seen_modules:
            return
        self._seen_modules.add(mid)
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        in_features = module.in_features
        if not in_features:
            return
        rows = inputs[0].numel() // in_features
        flops = 2.0 * rows * in_features * module.out_features
        if self._trainable.get(mid, False):
            self.forward_flops_trainable += flops
        else:
            self.forward_flops_frozen += flops

    def reset(self):
        self.forward_flops_trainable = 0.0
        self.forward_flops_frozen = 0.0
        self._seen_modules.clear()

    @property
    def model_training_flops(self):
        return 3.0 * self.forward_flops_trainable + self.forward_flops_frozen

    def snapshot(self):
        return {
            "zhao_linear_trainable": self.forward_flops_trainable,
            "zhao_linear_frozen": self.forward_flops_frozen,
            "zhao_flops": self.model_training_flops,
        }

    @staticmethod
    def compute_mfu(flops, step_time_sec, peak_tflops=None):
        """(achieved_tflops, mfu_pct) with his wall-clock convention."""
        peak = (peak_tflops or ZHAO_PEAK_TFLOPS) * 1e12
        if step_time_sec <= 0:
            return 0.0, 0.0
        return flops / (step_time_sec * 1e12), flops / (step_time_sec * peak) * 100.0


if __name__ == "__main__":
    # Self-check: one Linear(4->8) on 6 rows = 384 forward FLOPs.
    counter = FlopsCounterZhao()
    lin = nn.Linear(4, 8)
    counter.attach(nn.Sequential(lin))
    counter.reset()
    nn.Sequential(lin)(torch.zeros(2, 3, 4))
    assert counter.forward_flops_trainable == 384, counter.forward_flops_trainable
    assert counter.model_training_flops == 3.0 * 384, counter.model_training_flops
    counter.reset()
    lin.weight.requires_grad_(False)
    lin.bias.requires_grad_(False)
    counter._trainable[id(lin)] = False
    nn.Sequential(lin)(torch.zeros(2, 3, 4))
    assert counter.model_training_flops == 384, counter.model_training_flops
    print("flops_zhao self-check: OK")
