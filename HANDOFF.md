# HANDOFF

Operational log for DriveVLA-W0 work. Updated after every experiment.

## Ways of working (hard rules)

1. All box interaction goes through the tmux pane `lmc-w0-hif8` on `ott9-a56`.
   No side-channel ssh commands for repo work.
2. `HANDOFF.md` is updated after every experiment (config, log path, result, window).
3. The agent monitors the pane for task status proactively.
4. Change flow: edit on the Mac -> commit after approval -> push -> land on the box
   -> run. Box repo: `/mnt/sfs_turbo/lmc/DriveVLA-W0`.

## Environment

- Box: `ott9-a56` (`liteserver-a5-2-00002.novalocal`), tmux pane `lmc-w0-hif8:0.0`,
  shell inside the Ascend container (CANN 9.1, torch 2.11 / torch_npu 2.11).
- Local repo: `/Users/liumingchang/Documents/HwWork-temp/autonomous_driving/codespaces/DriveVLA-W0`
- Branch: `qwen-hif8` (branched from `upstream/main` @ `a55766c`), pushed to
  `origin/qwen-hif8` = `git@github.com-work:mingchang93/DriveVLA-W0.git`.
- Box repo was last used on `qwen3vl-backbone`; it must be fetched/checked out to
  `qwen-hif8` before the first run of this line.

## Current state (2026-11-09)

Ported the minimum Qwen3-VL path onto `qwen-hif8` (uncommitted at time of writing):

- `reference/transformers/src/transformers/models/qwen3_vl/` vendored from
  `qwen3vl-backbone` (config/modeling/modular/processor/video processor; ROSS head
  at branch tip). No `npu_fused_ops` dependency.
- `utils/train_qwen_vla.py`: family detection from `config.json` (`qwen3_vl`),
  Qwen3 config/model/processor selection, forced `sdpa`, `torch_npu` bootstrap,
  device-aware synchronize on save.
- `qwenvl`: optional `flash_attn` / `decord` imports (aarch64).
- `requirements_qwen.txt`, `scripts/sft/zero2.json`.

Verification so far: syntax only (no torch on the Mac). Nothing has been run on
the box yet.

## Experiment log

| # | Date | What | Config | Log | Result |
|---|------|------|--------|-----|--------|
| - | - | - | - | - | no experiments yet on `qwen-hif8` |

## Next step

Commit + push the port, land `qwen-hif8` on the box, then run the Qwen3-VL-2B
startup smoke (50 steps, `scripts/sft/zero2.json`) and record it here.
