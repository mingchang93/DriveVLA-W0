> **接手前先读这条**（2026-11-13 加）
>
> 本文件的工作方式说明**已经不适用**，照着走会走错：
> - 第 5 行说"所有交互走 `ott9-a56` 上的 tmux pane `lmc-w0-hif8`" —— **a56 已释放**，
>   现在的机器是 `ott9-a51` / `a52` / `a53` / `a54`（a51 是跳板）
> - 第 13 行说"Box repo: `/mnt/sfs_turbo/lmc/DriveVLA-W0`" —— 那是 lmc 的仓库；
>   我们的工作在 **`/mnt/sfs_turbo/zyc/DriveVLA-W0`**，工作目录 `/mnt/sfs_turbo/zyc/drivevla-w0-zyc`
> - "Current state" 停在 2026-11-09
>
> **当前的交接文档是 `HANDOFF-hif8-2026-11-13.md`**（同目录，也在
> `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/drivevla-w0-hif8-handoff.md`，另有同名 `.html`）。
> 它覆盖：数据与官方对齐、HiF8 实现与 ckpt 结构、运行方式与全部环境变量、
> 已确立的事实（带「实测 / 未验证 / 已更正」标记）、抽帧与续训的根因与闭环、
> torchtitan 与 BaiZe 两个参考实现、以及 11 条踩过的坑。
>
> 本文件下面的实验记录仍有价值，当历史看。

---

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
  shell inside the Ascend container. The container is **CANN 9.2.0** with python
  3.12.15; 8 NPUs visible.
- Training venv: `/mnt/sfs_turbo/lmc/venv-hif8` (python 3.12.15, no system site
  packages). torch 2.11.0+cpu, torch_npu 2.11.0, torchvision 0.26.0+cpu,
  torchaudio 2.7.1 (`--no-deps`; no 2.11 aarch64 wheel, and audio is unused),
  transformers 4.56.0 editable from this repo, diffusers 0.39.0, deepspeed
  0.17.6, accelerate 1.15.0, einops 0.8.1, huggingface_hub 0.36.2, scipy,
  pyyaml, ninja 1.13.2.
  Verified: `torch.npu.is_available()` true with 8 devices on CANN 9.2, and the
  vendored Qwen3-VL ROSS class imports.
- `venv-w0` is stale: built against `/usr/local/python3.12.13` in a different,
  CANN 9.1 container; its `python`/`pip` targets do not exist here. Do not use it.
- Local repo: `/Users/liumingchang/Documents/HwWork-temp/autonomous_driving/codespaces/DriveVLA-W0`
- Branch: `qwen-hif8` (branched from `upstream/main` @ `a55766c`), pushed to
  `origin/qwen-hif8` = `git@github.com-work:mingchang93/DriveVLA-W0.git`.
- Box repo: `/mnt/sfs_turbo/lmc/DriveVLA-W0`, checked out on `qwen-hif8` @ `6598667`
  (untracked `run_*.sh` files from earlier work are left in place).

### Paths on the box

- Models: `/mnt/sfs_turbo/models/`
  - `Qwen3-VL-2B-Instruct/` (target for the startup smoke), `Qwen3-VL-8B-Instruct/`,
    `Qwen3-VL-30B-A3B-Instruct/`, `Qwen2.5-VL-7B-Instruct/`, `Qwen2.5-VL-3B-Instruct/`
  - `stable-diffusion-v1-5/` — pass its `unet/` as `--sd_model_path`; the code
    derives `vae/` and `scheduler/` as siblings via `"/unet" -> "/vae"|"/scheduler"`.
  - Also present: `Emu3-Stage1/`, `Emu3-VisionTokenizer/`, `physical-intelligence-fast/`.
- Data: `/mnt/sfs_turbo/lmc/navsim_openscene/`
  - `navsim_trainval_subset.pkl`, `navsim_trainval_subset_x20.pkl`,
    `navsim_flops_L420.pkl`, `openscene-v1.1/`
  - The camera `--data_root` (directory holding `<log>/CAM_F0/<frame>.jpg`) still
    has to be pinned from inside `openscene-v1.1/` before the run.

### Dataset plan (Qwen3-VL-2B SFT + hold-out eval)

Hold-out evaluation target is **NAVSIM v1.1 `navtest`** (`data_split: test`); the
paper reports PDMS (v1) and EPDMS (v2) there. This checkout only ships the v1.1
devkit, so v2 EPDMS is out of scope for now. The in-house 70M-frame set is private.

| Purpose | Artifact | Destination |
|---|---|---|
| SFT | `openscene_metadata_trainval.tgz` | `navsim_logs/trainval` |
| SFT | 200 x `openscene_sensor_trainval_camera_*.tgz` | `sensor_blobs/trainval` |
| Hold-out eval | `openscene_metadata_test.tgz` | `navsim_logs/test` |
| Hold-out eval | 32 x `openscene_sensor_test_camera_*.tgz` | `sensor_blobs/test` |
| PDMS scoring | `nuplan-maps-v1.1.zip` | `maps/` (`NUPLAN_MAPS_ROOT`) |
| derived | navtest scene-token yaml; norm stats + FAST tokenizer | generated / already in `configs/` |

Not needed: LiDAR blobs (single front camera), Emu VQ code arrays for model input
(the Qwen/ROSS path rewrites the pickle's `.npy` path to `CAM_F0/<frame>.jpg`), and
the private in-house set. The pickle generator still needs the nuPlan processed
pickle + VQ root for its pre-1s fallback lookup.

Download source: OpenScene v1.1 on Hugging Face
(`OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/...`) and the nuPlan maps zip.
`huggingface.co` and the motional S3 host time out from the box, so camera/metadata
tarballs come from `hf-mirror.com` (302 to the xet CAS CDN) and the maps via the
`vultr-server` SOCKS relay. Downloads run in their own tmux sessions so the
`lmc-w0-hif8` training pane stays free.

Exact sizes from the mirror API (`?blobs=true`): trainval camera **200 splits,
1242.2 GB** (mean 6.21 GB); test camera **32 splits, 127.9 GB**; metadata_test
0.48 GB; metadata_trainval 7.05 GB (extracted, 1310 log pickles). Total ≈ **1.37 TB**.

Download record — **complete 2026-11-09 22:58 UTC** (`DOWNLOAD COMPLETE
trainval=1310 test=147 meta_test=147 failed=0`), ~11 h wall clock.
- Producers: `lmc-dl2` ran `parallel_dl.sh` (6 workers: trainval camera 2..199,
  test metadata, test camera 0..31, `normalize_test`, 3 retry passes);
  `lmc-dl-retry` ran `retry_failed_v2.sh`, a converging sweeper over `_failed/`.
- Failure root cause: **HTTP 429 Too Many Requests from `hf-mirror.com`**, which
  `wget` treats as fatal, so throttled workers died in ~1 s and burned through the
  queue. Individual probes succeeded (302 → 206 from the xet CDN), which is why it
  looked transient. The sweeper handles it with
  `--retry-on-http-error=429,500,502,503,504 --retry-on-host-error`,
  `--waitretry=30`, per-file `wget -o _wget_logs/<name>.log` and random backoff;
  it lost no splits.
- Throughput was an aggregate **~40 MB/s per-IP ceiling** (6 connections → 47 MB/s,
  8 → 40, 10 → 39), so worker counts only decided which pool finished last.
- Splits 0 and 1 came from the earlier sequential run and have no `_done` marker;
  they are covered by the content checks below.
- Verification (`verify_download.py`, exit 0):

```
[splits] trainval done=198 +extra=[0,1] missing=0 []
[splits] test done=32 missing=0 []
[meta]   test=ok trainval=ok
[logs:trainval] metadata=1310 camera_dirs=1310 missing=0 orphan=0
[frames:trainval] checked=1310 short_logs=0
[logs:test] metadata=147 camera_dirs=147 missing=0 orphan=0
[frames:test] checked=147 short_logs=0
RESULT: COMPLETE
```

Ops scripts live next to the data: `navsim_openscene/parallel_dl.sh`,
`retry_failed_v2.sh`, `verify_download.py`; logs `parallel_dl.log`,
`retry_v2.log`, `verify_final.log`.
- Maps: `/root/.ssh/id_vultr` was authorized on `45.77.5.43` for the transfer, then
  the key and its `authorized_keys` line were deleted. Archive 970,997,691 B, sha256
  `444860429f9a3bcf89a6459d683fa82eb9219aa259d8d9b8cdefdb37f0b56b05`, extracted
  1.4 GB to `openscene-v1.1/nuplan-maps-v1.0/` (archive deleted). Export
  `NUPLAN_MAPS_ROOT=/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1`.

## Current state (2026-11-09)

Committed `ba4ce89` "Port the minimal Qwen3-VL-2B backbone path" and landed it on
the box. Contents:

- `reference/transformers/src/transformers/models/qwen3_vl/` vendored (config,
  modeling, modular, processor, video processor; ROSS head). No `npu_fused_ops`
  dependency.
- `utils/train_qwen_vla.py`: family detection from `config.json` (`qwen3_vl`),
  Qwen3 config/model/processor selection, forced `sdpa`, `torch_npu` bootstrap,
  device-aware synchronize on save.
- `qwenvl`: optional `flash_attn` / `decord` imports (aarch64).
- `requirements_qwen.txt`, `scripts/sft/zero2.json`.

`venv-hif8` is prepared and the 20-step Qwen3-VL-2B smoke passes (experiment 1).

## Data inventory — Qwen3-VL-2B path (2026-11-11)

Scope: only what `utils/train_qwen_vla.py` -> `qwenvl/dataset/data_qwen_vla.py`
(`nuplan2va_ross`, `navsim2va_ross`) and `inference/qwen/inference_action_navsim_vava.py`
read. nuPlan lives in `/mnt/sfs_turbo/datasets/nuplan-v1.1`, NAVSIM in
`/mnt/sfs_turbo/lmc/navsim_openscene`, the reference pickles in
`/mnt/sfs_turbo/models/DriveVLA-W0/DriveVLA-W0`. SFS free: 7.9 TB.

Assets every Qwen run needs, all landed: `Qwen3-VL-2B-Instruct`,
`stable-diffusion-v1-5/unet` + `.../vae` (ROSS encoder/denoiser, and the VAE
decodes FID frames), `configs/fast` (FAST action tokenizer), and the ROSS
`negative_prompt_sd15.pt` that ships in the repo.
**Not read on this path:** VQ code trees, the Emu3 tokenizer/checkpoint, LiDAR,
the other 7 cameras, nuPlan maps at training time, and any `*_vq_*.pkl`.

### Pretrain — nuPlan v1.1, front camera (`nuplan2va_ross`)

`--data_path <pickle>` + `--data_root <images root>` with
`<log>/CAM_F0/<frame>.jpg`; `configs/normalizer_nuplan/norm_stats.json` is
applied at data-prep time, not at run time.

| Artifact | Needed | Landed | Missing |
|---|---|---|---|
| Per-log sqlite DB (poses) | 1.11 TiB of zips | 12,066 `.db`: mini 64 + boston, pittsburgh, singapore, vegas_1-4 + maps | `vegas_5` 81.9/136.5 GB, `vegas_6` 65.8/175 GB, `val` 57.2/96.9 GB (in flight on a53) |
| CAM_F0 JPEGs | 43 train + 12 val + 9 mini shards (5.9 TiB of tar, ~0.72 TiB extracted) | 9 mini + 4 train shards = 193 log dirs | 6 train shards in the subset pass, then 33 train + 12 val in backfill |
| Processed pickle (step0-4) | one per split | mini only: `_mini/output/nuplan_processed_data.pkl`, 67 MB, 1,304 segments, 258,580 frames, `verify_final` PASS | full `_full/output/nuplan_processed_data.pkl`; pipeline script ready, not run |

Pickle/dataset conventions line up: step1 samples frames at **10 Hz** (20 s
segment = 200 frames), step2 emits 8 waypoints at 0.5 s, and the dataset's
`frames_per_second=10` / `gap_frames=10` give the 1 s pre-frame gap; the Qwen
nuPlan dataset derives the pre pair from the same sequence, so no `pre_1s_*`
fields are needed. Deviation from the paper: the class hardcodes
`va_pair_num = 2` (2VA) while the paper pretrains on 6VA.

**Gap:** the generated pickle has only passed `verify_final`; no smoke has run
it through `LazySupervisedNuplan2VAROSSDataset` on NPU yet.

### SFT and hold-out — NAVSIM v1.1 (`navsim2va_ross`)

`--data_path <pickle>` + `--data_root sensor_blobs/trainval`; the dataset needs
`image`, `pre_1s_image`, `text`, `pre_1s_text`, `action`, `pre_1s_action`.

| Artifact | Landed | Missing |
|---|---|---|
| `navsim_logs/trainval` 1310 pickles, `test` 147; `sensor_blobs/trainval` 1310 dirs, `test` 147 dirs (1.37 TB) | verified COMPLETE 2026-11-09 | - |
| SFT pickle `navsim_emu_vla_256_144_trainval_pre_1s.pkl` | 103,288 samples, 12 images each, all keys above | - |
| Navtest pickle `navsim_emu_vla_256_144_test_pre_1s.pkl` | 12,146 samples | no `token` key -> scene-token list needed for PDMS |
| Maps `openscene-v1.1/nuplan-maps-v1.0` | 4 cities | - |
| PDMS scoring env | - | `nuplan-devkit` + hydra outside `venv-hif8` |

The pickles are dated 2026-10-17 and shipped with the model release, so SFT
needs no pickle generation; their `.npy` paths are rewritten to
`<data_root>/<log>/CAM_F0/<frame>.jpg`.
**Gap:** only the 467-sample subset pickle has been trained on (experiment 1);
the 103,288-sample pickle has not been through a step yet.

### FID (`tools/eval_world_model_fid.py`)

Needs the pretrain pickle + held-out images (so the 12 val shards matter for the
paper's FID split), the SD VAE, and InceptionV3 pool3 features. Inception weights
are missing: `~/.cache/torch/hub/checkpoints` is empty and
`download.pytorch.org` is unreachable from the box, so fetch the file through
the mirror/proxy and pass `--inception_weights_path`.



## Hold-out evaluation path

`inference/qwen/inference_action_navsim_vava.py` is the policy runner
(`inference_2va.sh` → one JSON trajectory per token → `run_pdm_score.py`
`agent=emu_vla_agent` → PDMS v1). It previously hardcoded Qwen2.5-VL; the port adds:

- `resolve_backbone()`: `config.json` `model_type == "qwen3_vl"` → plain
  `Qwen3VLForConditionalGeneration` + `Qwen3VLProcessor` + `sdpa`; otherwise the
  original Qwen2.5 path (FA2 only on CUDA).
- Plain class, not `...ROSS`: the ROSS `forward` asserts on training masks
  (`image_hidden.shape[1] == 2`) and would fail under `generate`;
  `denoiser.*`/`vae.*` checkpoint keys load as unexpected-key warnings.
- `hccl` / `npu:{rank}` when `torch.npu` is available; `PROJECT_ROOT` added to
  `sys.path` so `reference.transformers...` resolves under `torchrun`.

Evaluation needs only the FAST-token `lm_head` — the paper's MoE action expert is
not instantiated anywhere on this path (`train_qwen_vla.py` imports nothing from
`models/policy_head`; the Emu3 pipeline owns those heads). PDMS v1 is implemented,
EPDMS v2 is not in this checkout. Rope verified: Qwen3's `get_rope_index` and the
dataset's `get_rope_index_2` give bit-identical positions for single-image samples
(same token ids 151652/151655/151656, merge 2) — maxdiff 0.

Smoke plumbing: `/mnt/sfs_turbo/lmc/smoke_qwen3vl_2b/make_eval_smoke_pkl.py`
builds a navsim-format pickle directly from log pickles (no nuPlan fallback, no VQ
codes), and `run_eval_smoke.sh` runs the ported runner on one NPU rank. Image
sizing is inherited from the checkpoint's `preprocessor_config.json`
(`max_pixels` 150528, `min_pixels` 12544), so no CLI flags are needed.

Still missing for a real navtest run: the navtest pickle over
`navsim_logs/test` with the true scene-token list (`tools/pickle_gen/`'s script
hardcodes its author's paths and needs the nuPlan pre-1s fallback, so it needs the
same treatment the smoke builder got), and the PDMS scoring step.

## Experiment log

| # | Date | What | Config | Log | Result |
|---|------|------|--------|-----|--------|
| 1 | 2026-11-09 | Qwen3-VL-2B + ROSS startup smoke, 20 steps | NAVSIM subset (467 samples), 8 NPUs x bs 2, `zero2.json`, `max_pixels` 150528, `cur_frame_idx` 3, `future_nums` 8, lr 5e-5, grad ckpt on | `/mnt/sfs_turbo/lmc/smoke_qwen3vl_2b/train.log` (kept as `train_exp1.log`) | **PASS**: 20/20 steps, `TRAIN_EXIT=0`, ~1.26 it/s (~21 s); loss 21.14 -> 6.41, grad_norm 556 -> 38, `action_loss` 7.8 -> 6.1, `ross_loss` 0.03-0.34 |
| 2 | 2026-11-09 | Eval smoke of the ported runner, 1 NPU rank | 6 scenes from extracted trainval logs, 2 images/sample, `cur_frame_idx` 3, `future_nums` 8, bf16, sdpa, `model_max_length` 2048; passed both with `--use_previous_actions` on (pre-fix) and off (post-fix, `0d709e6`) | `/mnt/sfs_turbo/lmc/smoke_qwen3vl_2b/eval_smoke/infer.log` | **PASS**: `EVAL_EXIT=0`, 6/6 JSONs, action 8x3 finite; constant dx across the horizon (20-step checkpoint) |
| 3 | 2026-11-09 | Training smoke rerun at `24a3db3` | same config as exp 1 | `/mnt/sfs_turbo/lmc/smoke_qwen3vl_2b/train.log` | **PASS**: 20/20 steps, 27.9 s (1.39 s/it, network contention from downloads), train_loss 9.23, `TRAIN_EXIT=0` |
| 4 | 2026-11-12 | NAVSIM 2VA SFT, 4000 steps, **bf16 baseline arm** | ZYC checkout `zyc-hif8` @ `63898d1` + uncommitted HiF8/determinism edits; official trainval pickle (103,288), 8 NPUs x bs 6 x ga 1 (gb 48), `zero1.json`, lr 8e-5, warmup 50, world model on (3072.6M / 2988.9M trainable), seed 42 | `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs/night_bf16/train.log` | **PASS**: 4000/4000, `TRAIN_EXIT=0`, 06:13:10 -> 08:14:22 (2 h 01 m, ~1.82 s/step), loss 1.8074; **navtest PDMS 68.43** (12,146 scenes, 0 failed) |
| 5 | 2026-11-12 | same, **HiF8 168** (`cts`, ignore `lm_head,o_proj`) | same as exp 4 | `.../runs/night_hif8_168/train.log` | **PASS**: 4000/4000, `TRAIN_EXIT=0` 08:14:54, loss 1.8021; **navtest PDMS 68.69** |
| 6 | 2026-11-12 | same, **HiF8 196** (`cts`, ignore `lm_head`) | same as exp 4 | `.../runs/night_hif8_196/train.log` | **PASS**: 4000/4000, `TRAIN_EXIT=0` 08:15:03, loss 1.8013; **navtest PDMS 69.89** |
| 7 | 2026-11-12 | same, **HiF8 edge3** (`cts`, + layers 0-2/25-27 kept bf16) | same as exp 4 | `.../runs/night_hif8_edge3/train.log` | **PASS**: 4000/4000, `TRAIN_EXIT=0` 08:15:22, loss 1.8088; **navtest PDMS 67.55** |
| 8 | 2026-11-12 | Stage-1 nuPlan 2VA trial, 50 steps, bf16 | `_full` pickle (5,385 segments), 8 NPUs x mb 12 (gb 96), `zero1.json`, 50 steps | `.../runs/pretrain_trial50/train.log` | **PASS**: 50/50, `TRAIN_EXIT=0`, final loss 4.729, 6.4 GB checkpoint |
| 9 | 2026-11-12 | Stage-1 nuPlan 2VA trial, 50 steps, HiF8 | same as exp 8 + `HIF8=cts` | `.../runs/pretrain_trial50_hif8/train.log` | **KILLED**: SIGTERM ~2 min 45 s in, no checkpoint |
| 10 | 2026-11-12 | 2B SFT, **bf16**, second recipe (`aligned` family) | same data/bs/zero1/seed as exp 4, but **lr 1e-5, warmup 200** (read off the run's own lr series) | `.../runs/aligned_bf16/train.log` | **PASS**: 4000/4000, 120 min, loss 1.8703; **navtest PDMS 81.20** |
| 11 | 2026-11-12 | same, **HiF8 196** (`cts`, ignore `lm_head`) | as exp 10 | `.../runs/aligned_hif8_196/train.log` | **PASS**: 120 min, loss 1.8915; **PDMS 81.03** |
| 12 | 2026-11-13 | same, **HiF8 168** (`cts`, ignore `lm_head,o_proj`) | as exp 10 | `.../runs/aligned_hif8_168/train.log` | **PASS**: 120 min, loss 1.8910; **PDMS 80.75** |
| 13 | 2026-11-13 | same, **HiF8 132** (`cts`, + layers 0-2/25-27 bf16) | as exp 10 | `.../runs/aligned_hif8_132/train.log` | **PASS**: 120 min, loss 1.8816; **PDMS 80.98** |
| 14 | 2026-11-13 | same, **HiF8-DTS 132** (`dts`, interval 10, same ignore as exp 13) | as exp 10 | `.../runs/aligned_hif8_132_dts/train.log` | **PASS**: 125 min, loss 1.8927; **PDMS 80.99** |
| 15 | 2026-11-13 | same, **HiF8-DTS 196** (`dts`, interval 10, ignore `lm_head`) | as exp 10 | `.../runs/aligned_hif8_196_dts/train.log` | **PASS**: 129 min, loss 1.8844; **PDMS 81.97** (+0.95 % vs bf16, paired t 2.65) |
| 16 | 2026-11-13 | 7B SFT, **bf16** (Qwen2.5-VL-7B-Instruct) | same recipe as exp 10, 8 NPUs x bs 6 (gb 48), 4000 steps, seed 42 | `.../runs/w7b_bf16/train.log` | **PASS**: 143 min, loss 1.6984; **PDMS 84.07** |
| 17 | 2026-11-13 | same, **HiF8 132** (`cts`, + layers 0-2/25-27 bf16) | as exp 16 | `.../runs/w7b_hif8_132/train.log` | **PASS** on 2nd attempt (first `TRAIN_EXIT=1`), 139 min, loss 1.6932; **PDMS 83.97** |
| 18 | 2026-11-13 | same, **HiF8 168** (`cts`, ignore `lm_head,o_proj`) | as exp 16 | `.../runs/w7b_hif8_168/train.log` | **PASS**: 137 min, loss 1.7198; **PDMS 84.00** |
| 19 | 2026-11-13 | same, **HiF8 196** (`cts`, ignore `lm_head`) | as exp 16 | `.../runs/w7b_hif8_196/train.log` | **PASS**: 137 min, loss 1.7092; **PDMS 83.74** |

Eval-smoke findings (all outside the model code):
- The exported SFT checkpoint has no `video_preprocessor_config.json`, so
  `Qwen3VLProcessor.from_pretrained(checkpoint)` falls back to `AutoConfig` and
  fails on `qwen3_vl`. Worked around by copying the base model's file; the runner
  now loads `AutoImageProcessor` only.
- `preprocess_qwen_2_visual_vla_sources` inserts action segments unconditionally;
  with `--use_previous_actions` off all three are `None` and it raises
  `AttributeError: 'NoneType' object has no attribute 'tolist'`. Fixed in the
  runner. The passing run above used `--use_previous_actions`.
- The pickle's image path must be `<log>/<frame>.npy` (two components). A
  three-component `<log>/CAM_F0/<frame>.npy` makes the rewrite produce
  `<raw_img_root>/CAM_F0/<frame>.jpg` and silently falls through to `None`.
- Navtest itself is not downloaded yet; this smoke used trainval-derived scenes to
  exercise the same code path. The navtest pickle + token yaml + PDMS run follow
  the test split.

Blockers found and fixed on the way to the smoke (all landed on `qwen-hif8`):
- `b691a7e` vendored `transformers.data` collator shim (the 4.56 tree omits it)
- `e614e09` ROSS negative-prompt path resolved relative to the module
- `6598667` negative-prompt `.pt` loaded with `map_location="cpu"`, `weights_only=False`

## Next step

Data is complete (1.37 TB, verified). Open items, in order:
1. **Full SFT** on trainval: choose `use_previous_actions` (the smoke used off; the
   reference runner/vocab path expects it on for history tokens), step count and
   batch size, then run from `venv-hif8` in the `lmc-w0-hif8` pane.
2. **Navtest eval artifacts**: generate the navtest pickle from `navsim_logs/test`
   with the true scene-token list, and the `{tokens: [...]}` yaml.
3. **PDMS run**: `run_pdm_score.py` with `agent=emu_vla_agent` once the devkit env
   (`nuplan-devkit`, hydra) is verified in the venv.


## 2026-11-10 — pretrain FID + nuPlan 2VA data prep (written, not run)

Target path: **Qwen3-VL-2B** — `Qwen3VLForConditionalGenerationROSS`, `sdpa`,
denoiser `z_channel` = `text_config.hidden_size` (2048), dataset tag
`qwen3vl_ross` exactly as `utils/train_qwen_vla.py` sets it. The FID tool detects
the family from `config.json`; the Qwen2.5 branch is only a mirror of the eval
runner and is not used here.

Box was busy, so nothing below has been executed; only `py_compile` was run locally.

New tools:
- `tools/eval_world_model_fid.py` — 2VA world-model FID. Loads the ROSS wrapper
  (family detected from `config.json`), samples pre/cur pairs with the same
  dataset + collator as training, runs `model.denoiser.inference(z, z_a, steps)`,
  decodes with the frozen VAE, dumps `gen/`, `gt/` and optionally `recon/` PNGs
  (VAE decode-of-encode control), then scores FID with InceptionV3 pool3
  features. `--mode generate|fid|all`; smoke = `--num_samples 20
  --num_inference_steps 10`. Needs an SFT checkpoint that still carries
  `denoiser.*`/`vae.*` weights (missing keys are reported).
- `tools/pickle_gen/nuplan_pickle_gen/step0_export_nuplan.py` — dependency-free
  sqlite export of each `<log>.db` (`image ⋈ camera(CAM_F0) ⋈ ego_pose` ordered
  by image timestamp) into the JSON + `<log>/CAM_F0/*.jpg` layout step1 expects.
  `poses[i] = inv(ego_i) @ ego_{i+1}` (last = identity) because step2/step3b
  multiply consecutive entries; this matches the shipped
  `configs/normalizer_nuplan` scale (mean dx 2.04 m per 0.5 s).
- `tools/pickle_gen/nuplan_pickle_gen/running_stats.py` — local replacement for
  the missing `utils.dataset.normalize_pi0`.

Patched: steps 1/2/3a/3b/4, `verify_final.py`, `run_all.py`
- paths come from `NUPLAN_ROOT`, `NUPLAN_JSON_DIR`, `NUPLAN_IMAGES_DIR`,
  `NUPLAN_INTERMEDIATE_DIR`, `NUPLAN_OUTPUT_DIR`, `NUPLAN_OUTPUT_PICKLE`,
  `NUPLAN_NORMALIZER_DIR`; no `/mnt/vdb1` left on the pipeline path
- VQ codes are optional: unset `NUPLAN_VQ_DIR` and step1 no longer requires
  `.npy`; `npy_paths` is still derived from the jpg names and is what the
  dataset rewrites to `<data_root>/<log>/CAM_F0/<frame>.jpg`
- `step3a --threshold` (or `NUPLAN_COMMAND_THRESHOLD`) skips the interactive
  prompt
- step4 reuses `configs/normalizer_nuplan/norm_stats.json` by default so the
  action-token distribution stays identical to training; set
  `NUPLAN_RECOMPUTE_NORMALIZER=1` for a full-dataset recompute

Planned commands (once the box is free):

```bash
export NUPLAN_ROOT=/mnt/sfs_turbo/lmc/data/nuplan
python tools/pickle_gen/nuplan_pickle_gen/step0_export_nuplan.py \
  --db-root $NUPLAN_ROOT/raw/splits/mini --blob-root $NUPLAN_ROOT/raw/sensor_blobs
python tools/pickle_gen/nuplan_pickle_gen/run_all.py           # steps 1..4 + verify
python tools/eval_world_model_fid.py --mode all --num_samples 20 \
  --num_inference_steps 10 --output_dir /mnt/sfs_turbo/lmc/fid_smoke ...
```

Blockers:
- **nuPlan download is credential-blocked**: no `~/.aws`, no
  `AWS_ACCESS_KEY_ID`, no `nuplan` package installed, so the official mini split
  cannot be fetched from here. The pipeline is ready for a local copy.
- FID needs `torchvision` InceptionV3 weights; if the box has no egress, pass
  `--inception_weights_path` with a pre-downloaded checkpoint.
- FID on fewer than ~2048 frames is a relative signal only (the reference
  protocol uses ~50k); use the navtest pickle for a true hold-out.

Local synthetic check (mac, no box, 90-frame fake log): step0 -> step1 (VQ
disabled) -> step2 -> step3a (`--threshold 2.0`) -> step3b -> step4 -> verify all
exit 0. It caught one format bug that is now fixed: `poses` must be nested 4x4
matrices (step2 matmuls each entry), not flattened 16-vectors. With 0.4 m per
10 Hz frame the generated action is 2.0 m per 0.5 s, and the tail is zero-padded
as expected. Nothing was run on the NPU box.

## 2026-11-10 — nuPlan v1.1 landing on a54 (in progress)

New box is `ott9-a54`. The public bucket `s3://motional-nuplan` (region
ap-northeast-1) is anonymously listable/readable from it. Total v1.1 is
**18.45 TiB** (DB zips 1.11, cameras 6.69, LiDAR 10.65) while the shared SFS has
11 TiB free, so the literal "entire dataset" cannot land. Scope was matched to
the paper: pretrain is nuPlan **front camera only** (Table 1 "1x Cam") at
256x144 with 6VA/1 s spacing; LiDAR and the other seven cameras are unused
(Sec. 4.2, Table 9).

Throughput: 8 and 16 parallel streams both cap at ~36-37 MB/s on a54, and the
jump host a51 independently caps at ~39 MB/s, so the ceiling is the cluster
uplink, not per-connection. No full China mirror exists (HF has only
`nuplan_maps` and small example repos).

Transfer reduction (valid only for the `mini_set` zips): those are 45-150 GB
zips holding all 8 channels. `tools/pickle_gen/nuplan_pickle_gen/zipfetch.py`
reads the central directory over HTTP range requests and fetches only the
requested members (CRC-checked), so the mini part drops to ~0.73 TiB of bytes.
Verified on `nuplan-v1.1_mini_camera_0.zip`: 48.6 GiB shard -> 30,290 CAM_F0
frames, 5.95 GiB transferred, 11.2 min. The `train_*`/`val_*` shards are tars
and cannot be trimmed this way (see the 2026-11-10/11 entries).

Running layout `/mnt/sfs_turbo/datasets/nuplan-v1.1/`:
- `db/<log>.db` (mini extracted: 64 logs, 14 GB; train/val in flight)
- `images/<log>/CAM_F0/<frame>.jpg`
- `_fetch/{fetch_nuplan.py,zipfetch.py,run_mini_pipeline.sh,venv}` (venv has
  numpy/tqdm/matplotlib from the Huawei mirror)
- `_state/*.done` resume markers, `_logs/*.log`, `_tmp/` in-flight zips

Queues (tmux `lmc-dl-nuplan`, both resumable):
- `--kinds db,maps`: 12 items, 1.02 TiB (mini done)
- `--kinds camera`: 64 shards, ~5.9 TiB compressed (9 mini zips + 43 train +
  12 val tars)
Original estimate of ~1.75 TiB of transfer / ~15-17 h held only for the zips;
the tar shards force the full ~5.2 TiB (see 2026-11-11). Test sensor blobs,
LiDAR, and the other 7 cameras are deliberately excluded from `images/`.

### 2026-11-10 (later) — shard format surprise and revised plan

The `train_*`/`val_*` camera shards are **tar** archives misnamed `.zip`
(`ustar` at offset 257), so the channel-skipping trick only applies to the
`mini_set` zips, which really are zips with all 8 channels. The tars **do**
carry all 8 cameras (see the correction below). Consequences:

- `fetch_nuplan.py` now probes the first 512 bytes: zip -> `zipfetch.py`,
  tar -> resumable download + `tar -xf` into `images/`.
- The 43 train + 12 val camera tars must be streamed in full: ~5.7 TiB,
  ~45-50 h at the ~31 MB/s host cap (DB queue shares the same link). The
  earlier "~1.75 TiB, ~15 h" figure assumed CAM_F0 could be range-read out of
  the tars at 1/8 the bytes; that is not possible for tar.
- One a53 tar extraction (`train_camera_0`) landed at 21:47; extraction is
  verified by tar's own exit code; frame counts are checked afterwards against
  the DBs.
- Partial `.zip` downloads from a54 were resumed by curl `-C -` on a53.

### 2026-11-11 — tar contents correction, filtered extraction, throughput cap

`tar -tf` on a partial `train_camera_16` tar shows `<shard>/<log>/CAM_L0/...`
first, and one extracted log holds all 8 channel dirs (CAM_F0 456 MB of 3.5 GB
total). So the earlier "tars contain CAM_F0 only" note was wrong: the tars are
8-channel and every log was being written out 8x larger than the paper scope
needs. Two consequences:

- Extraction is now filtered: `tar -xf <tar> -C images --strip-components=1
  --wildcards '*/CAM_F0/*'` (verified on GNU tar 1.35 with a 2-channel test
  tar). Needed disk drops from ~5.9 TiB to ~0.72 TiB; without it the remaining
  tars plus 1.0 TiB of in-flight temp would have hit the 512 GiB free guard.
  The camera queue (tmux `lmc-dl-nuplan`, `--jobs 8`) was stopped parent-first
  then workers and relaunched from the same partial `_tmp/*.tar` files, so all
  232 GB already fetched resumed.
- The ~0.53 TB already extracted before the patch keeps its extra 7 channels
  (4 train shards, ~0.42 TB). Harmless for step0; deleting it is optional.

Throughput (measured 2026-11-11 07:55-08:40 CST): a53 is at its host ceiling
with 8 workers — 31 MB/s aggregate from `_tmp` growth, and an extra 200 MB
range stream added on top got only 4.4 MB/s. The same range from idle a54 got
30.4 MB/s (28.1 MB/s via the regional endpoint, so no gain there). Per-host cap,
not per-connection: more streams or another endpoint on a53 buys nothing. The
`--shard-parity {odd,even}` flag in `fetch_nuplan.py` splits camera shards
across hosts for a ~60 MB/s aggregate; on 2026-11-11 the decision was to keep
a53 alone (a54 hosts other tenants plus the repro session), i.e. ~45-50 h for
the remaining 5.21 TiB, ~26 h if the split is enabled later.

### 2026-11-11 — subset-first camera pass

Decision: land a pretrain-sized camera subset first, then backfill. The camera
queue was replaced by `_fetch/run_cam_subset_then_rest.sh` (tmux
`lmc-dl-nuplan`), which runs
`fetch_nuplan.py --only nuplan-v1.1_train_camera_{12,13,14,15,16,18} --jobs 6`
in the foreground and then re-runs the full camera queue (`--jobs 8`) in the
same shell, so no arm is lost between passes. Those six are the most-progressed
in-flight tars (~0.70 TiB left, ~7 h with the DB queue still sharing the link);
the two least-progressed (17, 19) keep their partial tars for the backfill.

New flag: `--only <comma separated job names>` (validated against the job
list, exits on unknown names). Logs: `_logs/fetch_cam5_a53_subset.log`
(ends with `SUBSET_EXIT=<rc>`) and `_logs/fetch_cam6_a53_backfill.log`.

The 51 pending shards are 43 train + 12 val once camera_12..19 are counted;
`images/` now grows ~1/8 as fast as before because of the CAM_F0 filter.

### 2026-11-11 10:00 CST — link collapse, downloads stalled

Between 08:00 and 09:50 CST the queues fell from ~31 MB/s to ~1-6 MB/s and the
in-flight `_tmp` tars/zips stopped growing (60 s delta 0 MB, except
`train_camera_18` which still held ~5 MB/s). Evidence that this is the site's
international egress, not our code:

- Per socket: `delivery_rate` 0.22-0.26 Mbps, `cwnd` pinned at 10, RTT 88-104 ms,
  `rcv_ooopack` 0.1-1.6M; four fresh parallel range GETs from a53 got
  0.07-0.19 MB/s each, so extra streams do not add bandwidth.
- a53 NIC rx was 33 MB/s while our sockets saw ~5 MB/s, so other tenants'
  traffic holds most of the host's allowance; a54's NIC was idle (0 MB/s) yet a
  fresh S3 GET still only reached 0.12 MB/s, so the shared upstream is also a
  limit.
- Other destinations from a53: Aliyun 15.5 MB/s, hf-mirror 6.7 MB/s (was
  40-47), Cloudflare 3.6 MB/s. From `vultr-server` the same S3 object gave
  24.2 MB/s, so S3 itself is healthy.
- Only `cubic`/`reno` are available on the 6.6 openEuler kernel, so BBR is not a
  local lever.

Decision: leave both queues running untouched. `curl` hangs on a slow link
rather than failing, so in-flight shards keep their partials and resume by
themselves; restarting only resets cwnd and loses nothing but time. Re-planning
waits on a throughput re-measurement: at ~5 MB/s the six-shard subset is ~1.7
days and the full camera set ~12 days, against ~7 h / ~45 h measured at 08:00.

**Concurrency A/B, 10:45-11:05 CST.** Cut 9 flows to 2 (`--jobs 1 --only
camera_12` + DB `--jobs 1`): camera_12 measured **98 KB/s over 90 s**, vegas_5
73 KB/s, so per flow stayed at ~0.1 MB/s and the aggregate fell to ~0.17 MB/s.
Restored 8 camera flows (+1 DB): per flow 113-190 KB/s and aggregate **~1.12
MB/s** (995 KB/s tars + 128 KB/s zip). Conclusion: the streams are not
competing — every S3 flow is loss-starved (`cwnd` 10 MSS, RTT ~90 ms,
`rcv_ooopack` 0.1-1.6M) and aggregate scales with flow count at ~0.12-0.15 MB/s
per flow. Non-AWS sources are unaffected: a53 pulled 11.7 MB/s on one flow from
`hf-mirror` -> xet, and vultr pulls the same S3 object at 24 MB/s. At 1.12 MB/s
the 8-shard subset needs ~8 days and the whole set ~4-5 months; the DB queue at
one flow (0.13 MB/s) would need ~18 days, so it is now the laggard. Runner:
`_fetch/run_cam8_then_rest.sh` (subset pass then backfill, both `--jobs 8`).

### 2026-11-11 14:25 CST — orchestrators SIGKILLed, supervisors added

All four `fetch_nuplan.py` processes (camera subset, DB `val`, DB retry, camera_15
retry) were killed with SIGKILL at 14:24:54 / 14:25:53 (wrapper logs show exit
137) while their `curl` children survived as orphans. No OOM: dmesg has no kill
records, memory was 17 GB used of 1502 GB, and the tmux session and the wrapper
shells stayed alive. The signature — python gone, curls alive, shell alive — is
a `pkill -f fetch_nuplan`-style action from outside our session, not a crash.

Fallout and recovery:
- Pass 2 had restarted at 14:24:58 and spawned a second curl per shard, so seven
  shards had two writers on the same file for ~10 minutes. Both streams are the
  same byte range of the same object, so the partials should be consistent; a
  completed tar that fails to extract will be treated as suspect and re-fetched.
- Killed all 18 orphan curls before relaunching, so nothing is double-writing.
- Both queues now run under restart-on-kill wrappers,
  `_fetch/run_cam_supervised.sh` and `_fetch/run_db_supervised.sh`: they re-run
  the orchestrator every 60 s (max 50 attempts) until it reports
  `DONE finished=N missing=0`, and stream to the log with `tee` so progress stays
  visible during an attempt. Flow count: 8 camera + 2 DB.
- Partials preserved through the event: camera_12/13/14/15/16/17/18/19 =
  94.2/61.0/40.5/47.5/27.5/28.2/62.4/29.2 GB; vegas_5/vegas_6/val =
  99.9/67.9/58.7 GB.
- The kills repeated at 15:26 and 15:29 and both queues died at the *same*
  timestamps, which is an external action, not a crash. `who` shows root logins
  from 1.95.179.234 at 13:55 and 14:23 and from 10.0.16.12 at 15:11; each
  precedes a kill burst (14:24:54, 15:26+). No OOM records, memory idle.
- First supervisor version had a bug: it cleaned up every `_tmp/nuplan*` curl
  regardless of queue, so the DB wrapper killed the camera wrapper's fresh
  curls (jobs logged `curl rc=-15`), and each retry added a curl set on top of
  the previous orphans (30 flows observed). Fixed by scoping the cleanup glob
  (`*_tmp/nuplan-v1.1_*camera_*` for the camera wrapper, `*_tmp/nuplan-v1.1_*.zip`
  for DB), killing all orphans, and relaunching: exactly one writer per file,
  10 flows.
- `curl rc=56` (connection reset) was also shrinking each attempt's pool
  permanently, so `fetch_nuplan.py` now re-queues a failed job up to
  `MAX_RETRIES = 3` inside the same run (`RETRY` in the log, `requeue=N` in the
  message) instead of dropping it until the attempt ends. A job that still
  fails after three retries is reported as `FAIL`.

### 2026-11-12 00:30 CST — archives corrupted by duplicate writers, repaired by range

Symptoms: eight camera tars failed their in-job extraction with `tar rc=2`
("Skipping to next header") and both remaining DB zips failed with
`unzip rc=2` ("bad CRC"), while every file size matched S3 exactly.

Diagnosis:
- Two fresh S3 fetches of the same 64 MiB range were identical, and that range
  matched the local file, at 50 GiB; 62/63/64/65 GiB and the tail of camera_13
  also matched. The damage is therefore localised, not a global shift or a bad
  object.
- `/tmp/tar_region.py` walks headers and checks the ustar checksum: camera_13's
  first bad header is at byte 60,844,597,248 (56.66 GiB), with ~400 damaged
  patches over the following ~0.4 GiB. That is exactly where a duplicate curl
  joined the file at 14:24, so two processes were appending the same object at
  slightly different offsets; their bytes interleaved while both were alive and
  realigned once the duplicate was killed.
- Repair instead of re-downloading: `/tmp/repair_tar.py` re-fetches a 1 GiB
  window from the last good header and splices it in place, then re-scans.
  camera_13 came back clean after one round; the same driver ran over the other
  seven tars. `/tmp/repair_zip.py` does the member-level version for zips:
  decompress each member, compare CRC, splice the compressed range of any bad
  one (vegas_5's bad member is
  `2021.07.13.18.48.33_veh-37_00440_01932.db`).

The queues were stopped for the repair so nothing extracted a half-repaired
archive, and were restarted from the partials afterwards.

### 2026-11-11 — mini end-to-end validation (PASS)

Ran `_fetch/run_mini_pipeline.sh` on a53 with the data venv (numpy/tqdm/matplotlib),
64 mini DBs (symlinked in `_mini/db`) and the extracted CAM_F0 tree:

- step0: 64/64 logs exported, 0 missing images
- step1: 1,304 segments (200 frames each)
- step2: actions generated, 0 failed segments
- step3a/3b: threshold 2.0 m; commands 78.8% straight, 15.1% right, 6.1% left
- step4: reused `configs/normalizer_nuplan/norm_stats.json` (q99
  [6.729, 0.162, 0.135]); pickle 67 MB, 258,580 frames
- verify_final: PASS

Pickle schema check: `{segment_id, image, action, text}`; `image[0]` is
`<log>/<frame>.npy` (two components, as the dataset expects), `action`
(200, 8, 3) float64 in [-1, 1], `text` length == frames.

Download state at the same time: DBs 7/9 zips + mini + maps (12,066 .db),
cameras 4/43 train tars + all 9 mini shards; 51 tar shards (~5.2 TiB) queued at
8 workers (~30 MB/s aggregate, ETA ~50 h); 8.1 TiB free. Two orphan duplicate
curls writing the same tars were killed; queues relaunched cleanly.

## 2026-11-12 04:00-07:20 CST — nuPlan subset repaired, re-fetched, pickled (PASS)

### What the corruption actually was

The duplicate-writer episodes left three kinds of damage, and they need
different fixes:

1. **Banded header damage** (cam12/13/14/15/17): a stretch where two curls
   interleaved appends. `tar -tf` exits rc=2 with `Skipping to next header`,
   but the rest of the archive is canonical. Fixed by re-fetching the band plus
   padding and splicing it in place; each tar needed 1-6 windows of 0.1-4.7 GiB.
2. **Data-level divergence** (cam16/18/19): header walk is *clean* yet
   `local[off] != s3[off]` for 1 MiB windows across the first 20 GB - the
   bytes are simply wrong while the tar structure survives. Not repairable by
   band patching; re-downloaded from scratch.
3. **Mid-stream zero runs** (cam19): a run of NUL blocks inside the archive
   followed by more members. Both GNU tar and the header walker treat a zero
   block as EOF, so the walker reported "clean" while tar still printed
   `Skipping to next header` (twice). The final block layout was verified
   directly (18 trailing NUL blocks = legitimate EOF), so the run was
   mid-archive, not at the tail.

### Tools written for this (all on a53 /tmp)

- `tar_bands.py` — walks headers; on a bad block it probes forward in 1 MiB
  steps for the resync header, then refines backwards in 512 B steps. Reports
  `[prev_good, resume)` windows in seconds instead of the 512 B-at-a-time walk.
- `tar_bands2.py` — same, but treats a zero run followed by more data as damage
  (the case `tar_region.py` and `tar -tf` agree to stop on).
- `splice.sh` — fetches `[S,E]` in 4 MiB-ish chunked `curl -r` calls and
  `dd ... oflag=seek_bytes seek=$S conv=notrunc` into the file; retries from the
  partial length if a chunk fails.
- `seg_dl.sh` — N parallel byte-range downloads to `.segN` files, then `cat`
  them together; used for the fresh cam16/18/19 fetches (3 segments each).
- `resume_archive.sh` — append-only resume with stall detection.
- `sample_cmp.sh` — md5-compares a 1 MiB local window against the same S3 range;
  this is what distinguished band damage from data-level divergence.
- `zip_fix.py` — splices one member's compressed range (plus its local header
  when that is what broke); `zip_crc.py` — full CRC walk of a DB zip.

### Link behaviour that shaped the design

- `curl` **hangs** on this path instead of failing, so every fetch now carries
  `--speed-limit 2000 --speed-time 45`; before that, two splices sat at 1.4 GB
  and 87 MB for 20 min with the process still alive.
- Throughput recovered completely: 3-6 flows sustain **33-41 MB/s** aggregate
  (per-flow ~11 MB/s). The 0.1 MB/s-per-flow collapse of 2026-11-11 is over.
- Scan "resume" offsets understate a band: the damage regularly straddles the
  window edge, so patch windows are padded 32-64 MiB on both sides.

### Final state of the subset (train camera 12-19)

All eight tars are byte-exact S3 size and `tar -tf` rc=0, 0 errors:

| shard | size (GB) | members | logs | DB | CAM_F0 |
|---|---|---|---|---|---|
| cam12 | 125.8 | 551,981 | 33 | 33/33 | extracted |
| cam13 | 122.8 | 581,954 | 48 | 48/48 | extracted |
| cam14 | 129.1 | 600,080 | 44 | 44/44 | extracted |
| cam15 | 126.3 | 574,083 | 18 | 18/18 | extracted |
| cam16 | 134.4 | 648,473 |  8 |  8/8  | extracted |
| cam17 | 110.2 | 529,425 |  7 |  7/7  | extracted |
| cam18 | 132.1 | 641,195 | 17 | 17/17 | extracted |
| cam19 | 110.6 | 536,918 | 13 | 13/13 | extracted |
| total | 981.4 | 4,664,109 | 188 | 188/188 | |

cam12/13/14/15/17 match S3 md5 at sampled offsets (5/27/55 GB grid);
cam16/18/19 are fresh downloads (0 errors, full member counts restored -
cam16 gained 495 members over the patched copy, cam19 one).

Note: a shard holds the same log name in up to 8 channel dirs, and frames are
named by content hash, so identical JPEGs collapse to one filename
(5,691 tar entries -> 5,690 files on disk for one log). That is not data loss.

### Pickle run (`_fetch/run_full_pipeline.sh` variant, `run_subset_pipeline.sh`)

Only logs that have **both** a `.db` and `images/<log>/CAM_F0` are fed in
(373 of the 374 image logs; `db_subset/` is a symlink farm). Output:

- step0: 373/373 logs exported, **1,067,449 frames**, missing_images=0
- step1: **5,385 segments** (200 frames each)
- step2: 0 failed segments; step3a threshold 2.0 m (mean 0.620, sd 2.947)
- step3b: commands go straight 867,903 / left 53,252 / right 141,385
- step4 + verify_final: **PASS**
- pickle `_full/output/nuplan_processed_data.pkl` (284.7 MB): 5,385 segments,
  1,062,540 frames, keys `{segment_id, image, action, text}`,
  `action` (200, 8, 3) float64 in [-0.329, 1.000], `image[i]` =
  `<log>/<hash>.npy`. `qwenvl/dataset/data_qwen_vla.py:399-414` rewrites that to
  `<data_root>/<log>/CAM_F0/<hash>.jpg`, so `--data_root` =
  `/mnt/sfs_turbo/datasets/nuplan-v1.1/images` for `nuplan2va_ross`.

Normalizer: `_full/output/normalizer/` is empty; the repo's
`configs/normalizer_nuplan/norm_stats.json` is the one the launcher uses.

### DB zips

- vegas_5: 1 bad member (`2021.07.13.18.48.33_veh-37_00440_01932.db`, 469 MB
  compressed) - spliced, CRC ok.
- vegas_6: 3 bad members (724 MB, 107 MB, and one whose local header was gone,
  525 MB incl. header) - all spliced, all CRC ok.
- Both re-extracted into `db/` (the earlier extraction had stopped at the first
  CRC error): 13,437 -> **14,571** files, all four repaired members pass
  `sqlite3 "pragma quick_check" -> ok`. Train DB shards are now complete
  (boston, pittsburgh, singapore, vegas_1-6); `val` is in `db/` too.

### Still open

- Camera shards 20-199 are **not** downloaded (5.3 TiB, ~37 h at 40 MB/s).
- The 12-19 subset is a pretrain smoke set, not the paper's 8k-step corpus.

## 2026-11-12 — NAVSIM SFT four-arm A/B and full navtest PDMS (PASS)

Produced in **ZYC's checkout**, not in `/mnt/sfs_turbo/lmc/DriveVLA-W0`: repo
`/mnt/sfs_turbo/zyc/DriveVLA-W0`, branch `zyc-hif8`, commit `63898d1` (same commit
message as this repo's `4f34d27`, **different hash** — the two histories are not
the same), plus 8 modified + 52 untracked files. Records: exp 4-9 in the
experiment log above.

### Config (identical for all four arms except quantisation)

| Field | Value |
|---|---|
| dataset | `navsim2va_ross`, `/mnt/sfs_turbo/zjj/datasets/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl` (103,288 scenes), `--data_root .../sensor_blobs/trainval` |
| model | `/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct` + SD-1.5 `unet` (world model **on**) |
| params | total 3072.6M, trainable 2988.9M |
| schedule | 4000 steps, 8 NPUs x bs 6 x ga 1 = **gb 48**, warmup 50, lr 8e-5 -> min 1e-6, cosine_with_min_lr, wd 0.1, clip 5.0, beta (0.9, 0.95) |
| data pipeline | `cur_frame_idx 3`, `future_nums 8`, `use_previous_actions`, `max_pixels 147456`, `min_pixels 12544`, `model_max_length 1400`, seed 42 |
| deepspeed | `scripts/sft/zero1.json` (zero2 does not fit at this microbatch) |
| ckpt | `--save_strategy no`; the final model is written by `safe_save_model_for_hf_trainer` (6.4 GB incl. tokenizer + preprocessor) |

### Results (navtest, 12,146 scenes, PDMS v1, 0 failed scenarios)

| Arm | Quantisation | Final loss (step 4000) | **PDMS** | vs bf16 | Mini (21 scenes) |
|---|---|---|---|---|---|
| `night_bf16` (a52) | off, baseline | 1.8074 | **68.43** | — | 46.13 |
| `night_hif8_168` (a53) | `cts`, ignore `lm_head,o_proj` | 1.8021 | **68.69** | +0.27 | 44.55 |
| `night_hif8_196` (a51) | `cts`, ignore `lm_head` | 1.8013 | **69.89** | +1.46 | 40.93 |
| `night_hif8_edge3` (a54) | `cts`, + layers 0-2/25-27 bf16 | 1.8088 | **67.55** | -0.79 | 48.67 |

Wall clock ~2 h 01 m per arm (06:13:10 -> 08:14:22 for the bf16 arm; the other
three ended 08:14:54 / 08:15:03 / 08:15:22), i.e. ~1.82 s/step at gb 48.

Reading the table: one seed, one run per arm, no error bar. The four PDMS scores
sit in a 2.3-point band and the four final losses differ by 0.008 nats, so this
data does not rank the arms — it only shows that HiF8 (168/196/edge3) did not
degrade PDMS relative to bf16 by any amount visible above single-seed noise. Do
not quote the +1.46 on the 196 arm as a gain. The 21-scene mini run ranks the arms
completely differently (edge3 first, 196 last) and is **not** a comparison — it
only proves each checkpoint loads and the chain runs.

### Which sub-metric moves the PDMS

Recomputed from the per-scene CSVs (`exp/eval_all_<arm>_full/<ts>/<ts>.csv`); the
header plus a `token="average"` summary row inflate the row count to 12,147, and
the 12,146 real scenes reproduce the logged scores exactly (0.684286 / 0.686950 /
0.698910 / 0.675533).

| mean over 12,146 scenes | bf16 | HiF8-168 | HiF8-196 | HiF8-edge3 |
|---|---|---|---|---|
| PDMS | 0.6843 | 0.6869 | 0.6989 | 0.6755 |
| no-at-fault collision (NC) | 0.9257 | 0.9259 | 0.9293 | 0.9226 |
| drivable-area compliance (DAC) | 0.8134 | 0.8148 | **0.8265** | 0.8082 |
| ego progress (EP) | 0.6287 | 0.6327 | **0.6454** | 0.6219 |
| TTC within bound | 0.8576 | 0.8555 | 0.8581 | 0.8495 |
| comfort | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scenes with DAC = 0 | 2,266 | 2,249 | **2,107** | 2,329 |
| scenes with NC = 0 | 812 | 800 | **775** | 848 |
| scenes with DDC = 0 | 599 | 614 | 603 | 663 |

Exact decomposition by veto status (contribution to the mean-score delta, `n` =
scene count; `x` = arm, `r` = bf16):

| arm | Δ PDMS | both clean | bf16 vetoed, arm not | arm vetoed, bf16 not |
|---|---|---|---|---|
| 168 | +0.0027 | +0.0013 (8,146) | +0.0652 (971) | −0.0634 (937) |
| 196 | +0.0146 | +0.0023 (8,163) | +0.0745 (1,108) | −0.0620 (920) |
| edge3 | −0.0088 | −0.0014 (8,148) | +0.0553 (832) | −0.0625 (935) |

Inside the both-clean scenes, where no veto couples the terms, first-order
contributions for the 196 arm are EP +0.0020, TTC +0.0002, comfort 0.0000.

So the arm spread is **not** a smooth metric difference. It is ~4.6 % of scenes
gaining and ~5.4 % losing the multiplicative veto, each swing worth ~0.25 of that
scene's score, and the net is the small imbalance of the two. DAC is the dominant
flip (2,107-2,329 zeroed scenes per arm) and DDC also vetoes despite carrying
weight 0. Two consequences: comfort never fails in any arm (0 scenes), so its 2/12
weight is a constant and explains nothing; and a ~1,000-scene symmetric churn of a
0/1 flag is exactly the quantity that needs a second seed before it can be read as
a quantisation effect.

### How the eval was run

`tools/zyc/eval_ckpt.sh <ckpt> <tag> full 2` per arm (four in parallel, 2 NPUs
each, `tools/zyc/eval_all.sh`), two stages:

1. **Inference** — `venv-hif8`, `torchrun --nproc_per_node=2
   inference/qwen/inference_action_navsim_vava.py`, `--test_data_pkl
   navsim_workspace/navtest_qwen_2va.pkl`, `--token_yaml
   navsim_workspace/navtest_tokens.yaml`, `--norm_stats_path
   configs/normalizer_navsim_trainval/norm_stats.json`, `--cur_frame_idx 3
   --future_nums 8 --model_max_length 1400 --use_previous_actions --bf16
   --save_gt`, images from
   `/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/test`.
2. **PDMS** — `run_pdm_score.py train_test_split=navtest agent=emu_vla_agent`,
   run from the separate conda env `/mnt/sfs_turbo/zyc/miniconda3/envs/navsim`
   (python 3.9, nuplan-devkit + hydra), Ray pool of 192 threads, metric cache
   `navsim_workspace/exp/metric_cache`.

Artifacts per arm: `navsim_workspace/eval_all_<arm>_full/json_output`
(12,146 `<token>.json` + 12,146 `_agent_human_traj_diff.json` + 12,146 `.png`)
and `navsim_workspace/exp/eval_all_<arm>_full/<timestamp>/` with the CSV and
`run_pdm_score.log`. The outer `eval_all.log` holds only its two startup lines,
so the per-arm `run_pdm_score.log` and CSV are the record.

### Caveats that belong with these numbers

- The HiF8 code is **not committed**: `models/quant/` is untracked, and
  `argument.py`, `utils/train_qwen_vla.py`, `modeling_qwen3_vl_ross.py`,
  `denoiser_sd.py`, `unet_2d_condition.py`, both launchers and `FACTS.md` are
  modified. The determinism package in those edits is what makes the four arms
  comparable at all (seed before model build — the denoiser mlp/pos_embed are
  randomly initialised; `torch.npu.conv/matmul.allow_hf32 = False`; `.contiguous()`
  before the deterministic `F.interpolate`; UNet timestep `float64 -> float32`).
- Evaluation loads the plain `Qwen3VLForConditionalGeneration`; `denoiser.*` /
  `vae.*` load as unexpected keys, so PDMS scores the action policy only. The
  world model is exercised in training, not in this metric.
- PDMS v1 only (v1.1 devkit); EPDMS v2 is not implemented in this checkout.
- Eval inputs used here are `navtest_qwen_2va.pkl` + `navtest_tokens.yaml`
  (both 12,146 scenes / tokens, 2026-11-11 09:38) and the metric cache above —
  not the `tools/zyc/make_navtest_pkl.py` path, which builds the same schema.

### Stage-1 (nuPlan 2VA pretrain) trial

- `runs/pretrain_trial50`: first trial on the generated
  `_full/output/nuplan_processed_data.pkl` (5,385 segments), mb 12 x 8 cards
  (gb 96), 50 steps, bf16, world model on → **completed**, `TRAIN_EXIT=0`,
  loss 4.729 at step 50, 6.4 GB checkpoint. Its `train.log` stamps are 8 h
  behind host CST (01:48:39 -> 01:52:49; directory mtime 09:52 CST).
- `runs/pretrain_trial50_hif8`: same with `HIF8=cts` — killed by SIGTERM at
  ~2 min 45 s, no checkpoint. Not yet requeued.
- `drivevla-w0-zyc/nuplan_pickle/` (09:24) holds 3 exported logs with an empty
  `output/` — an aborted pickle attempt, unrelated to the working `_full` pickle.

### Box state while this was recorded

- `tools/zyc/extract_v2.sh 27 28` is still extracting `CAM_F0` from the held
  camera tars: `.../nuplan-v1.1/images` went 540 -> **708 log dirs**, 9 tars
  left, guarded by `tools/zyc/diskguard.sh`. Shared SFS **1.9 T free of 21 T
  (92 % used)**.
- No ZYC job is running. All 8 NPUs on a53 and a54 are held by another tenant's
  2-node Qwen3-VL-8B 50-step MFU run (a53 master, a54 rank 1), so new arms have
  to wait or move to a51/a52.
- `watchdog.log` stops at 08:35 and nothing supervises ZYC's jobs any more.

### Open

1. Publish or reconcile the HiF8/determinism delta — it lives only in ZYC's
   checkout, and `models/quant/` is not even tracked.
2. Requeue the HiF8 stage-1 trial once the cards free; bf16 stage-1 has only run
   to step 50.
3. Repeat at least one arm with a different seed before quoting any PDMS delta.
4. Finish the camera backfill (shards 20-199) against 1.9 T of free space.

## 2026-11-13 — 2B `aligned` family (incl. HiF8-DTS) and 7B family: full navtest PDMS + sub-metrics

Exp 10-19. Still produced in **ZYC's checkout** (`/mnt/sfs_turbo/zyc/DriveVLA-W0`,
branch `zyc-hif8` @ `63898d1` + the same uncommitted HiF8/determinism edits).
Every number below was read on 2026-11-13 09:59 CST from that box, not re-run.

Per-arm records: `one_<tag>.log` (train summary + HBM), `runs/<tag>/train.log`,
`runs/<tag>/trainer_state.json`, `eval_<tag>.log` (eval windows + final PDMS),
per-scene CSVs at `navsim_workspace/exp/eval_<tag>_full/<timestamp>/<timestamp>.csv`,
and `submetrics.py` (stdlib; reads those CSVs, intersects the scene sets, prints
means and paired t against a chosen baseline).

### The `aligned` / `w7b` families are not on the exp 4-7 recipe

Read off each run's own lr series (`trainer_state.json` -> `log_history`):

| family | first logged lr | peak lr | implied warmup |
|---|---|---|---|
| `night_*` (exp 4-7) | 1.6e-6 | 8e-5 | 50 steps |
| `aligned_*`, `w7b_*` (exp 10-19) | 5e-8 | 1e-5 | 200 steps |

So the **0.68 -> 0.81 level shift between the two families is not a quantisation
result**, and the two families must not be compared with each other. Which knob
drives the shift is **not established**: the run record carries the lr ladder
only, and the launcher (`tools/zyc/run_sft_navsim.sh`, mtime 11-13 03:46) postdates
the 11-12 arms so it cannot be cited for them. Everything below compares arms
inside one family, where data order, seed, batch and schedule are shared.

### 2B (Qwen3-VL-2B-Instruct): 4000 steps, 8 NPUs x bs 6 (gb 48), seed 42, zero1, world model on

| Arm | HiF8 | Quantised Linear | Final loss | Train | Eval (cards) | mini | **full PDMS** | vs bf16 |
|---|---|---|---|---|---|---|---|---|
| `aligned_bf16` | off | 0 | 1.8703 | 120 min | 11-12 19:36-20:15 (4) | 0.5776 | **0.81203** | — |
| `aligned_hif8_168` | `cts`, ignore `lm_head,o_proj` | 168 | 1.8910 | 120 min | 11-13 06:32-06:55 (8) | 0.5703 | **0.80748** | −0.56 % |
| `aligned_hif8_132` | `cts`, + layers 0-2/25-27 bf16 | 132 | 1.8816 | 120 min | 11-13 07:29-07:54 (8) | 0.6168 | **0.80980** | −0.28 % |
| `aligned_hif8_196` | `cts`, ignore `lm_head` | 196 | 1.8915 | 120 min | 11-12 19:36-20:15 (4) | 0.6113 | **0.81026** | −0.22 % |
| `aligned_hif8_132_dts` | `dts`, interval 10, ignore as 132 | 132 | 1.8927 | 125 min | 11-13 09:18-09:42 (8) | 0.6525 | **0.80985** | −0.27 % |
| `aligned_hif8_196_dts` | `dts`, interval 10, ignore `lm_head` | 196 | 1.8844 | 129 min | 11-13 09:11-09:35 (8) | 0.6438 | **0.81971** | **+0.95 %** |

### 7B (Qwen2.5-VL-7B-Instruct): same recipe as the 2B aligned family, 4000 steps, gb 48, seed 42

| Arm | HiF8 | Quantised Linear | Final loss | Train | Eval (cards) | mini | **full PDMS** | vs bf16 |
|---|---|---|---|---|---|---|---|---|
| `w7b_bf16` | off | 0 | 1.6984 | 143 min | 11-13 06:39-07:03 (8) | 0.7006 | **0.84066** | — |
| `w7b_hif8_132` | `cts`, + layers 0-2/25-27 bf16 | 132 | 1.6932 | 139 min | 11-13 08:04-08:27 (8) | 0.7462 | **0.83970** | −0.11 % |
| `w7b_hif8_168` | `cts`, ignore `lm_head,o_proj` | 168 | 1.7198 | 137 min | 11-13 06:14-06:37 (8) | 0.8067 | **0.83999** | −0.08 % |
| `w7b_hif8_196` | `cts`, ignore `lm_head` | 196 | 1.7092 | 137 min | 11-13 06:09-06:32 (8) | 0.7499 | **0.83738** | −0.39 % |

`w7b_hif8_132` needed two attempts (first `TRAIN_EXIT=1`, then a clean 4000/4000).

### Quantisation scope, confirmed in code

`utils/train_qwen_vla.py:346-380` turns `--hif8_apply` / `--hif8_ignore` into dotted
substring lists (defaults `model.language_model.layers.` / `lm_head,o_proj`), and
`models/quant/hif8_cts.py:163-193` / `hif8_dts.py:404-434` apply the identical rule:
a `nn.Linear` is replaced only if its FQN contains an `apply` substring, contains no
`ignore` substring (ignore wins), and `min(in_features, out_features) >= 64`;
everything else stays bf16 and is only recorded as skipped.

Consequence: `apply` never reaches the head, so **`lm_head` is bf16 in every arm**
(168, 196, 132, both DTS arms, and all four 7B arms). The `lm_head` entry in the
ignore list is a guard, not the effective switch. The arms differ as follows:

| arm | ignore list | quantised | composition |
|---|---|---|---|
| 168 | `lm_head, o_proj` | 168 | 28 layers x 6 (q, k, v, gate, up, down) |
| 196 | `lm_head` | 196 | 28 layers x 7 (adds **o_proj**) |
| 132 | `lm_head, o_proj, layers.0.,1.,2.,25.,26.,27.` | 132 | 22 layers x 6 |

So the 196 arms differ from 168 by quantising `o_proj` (28 matrices), **not** by
touching `lm_head`; the 132 arms additionally keep layers 0-2 / 25-27 in bf16.
The code's own self-check maps `{lm_head}` -> 196, `{lm_head,o_proj}` -> 168,
`{lm_head,down_proj}` -> 168, `{lm_head,o_proj,down_proj}` -> 140; the 132 arms carry
an extra ignore prefix so no expected count is asserted, which is why only the 168
and 196 logs print `[HiF8] 数量自检通过`. DTS uses the same scope with `interval=10`
(the code rejects `interval < 2`; 1 would be CTS) and `amax_sample_last` in
`1 .. interval-1`.

Within the 2B family this gives two clean single-knob contrasts: 168 -> 196 isolates
`o_proj` quantisation (0.80748 -> 0.81026), and 196 -> 196_dts isolates the refresh
schedule (0.81026 -> 0.81971).

### The dataset's window design sets the resolution of every comparison here

Loaded both pickles on 2026-11-13 (host `python3` for the eval pickle,
`docker exec zyc-w0 python` for the 335 MB train pickle):

| pickle | entries | segments | logs | windows/segment | windows/log |
|---|---|---|---|---|---|
| `navsim_emu_vla_256_144_trainval_pre_1s.pkl` | 103,288 | **1,192** | 162 | mean 86.7 (1 … 1,080) | 638 |
| `navtest_qwen_2va.pkl` (ours) | 12,146 | **136** | 44 | mean 89.3 (8 … 568) | 276 |
| `navsim_emu_vla_256_144_test_pre_1s.pkl` (zjj, official) | 12,146 | — | — | — | — |

So "12,146 scenes" and "103,288 scenes" are **window counts**, not independent
scenes: they are 136 and 1,192 NAVSIM scenes respectively, each expanded into ~87-89
sliding windows. The windows inside a segment sit 0.5 s apart (`pre_1s_action` is the
`action` array shifted by 2 rows, i.e. 2 Hz) and each carries a 12-step (6 s) horizon,
so **adjacent windows share 11 of 12 future steps and 3 of 4 history images**. Segment
and log sets of the two splits are disjoint (0 overlap on both), and our navtest pkl
matches zjj's official test pkl entry-for-entry (nearest-neighbour max-abs action
distance <= 0.017 over 150 sampled entries; only ordering and float32 rounding differ).

What that does to the 196-dts vs bf16 comparison:

| unit of analysis | n | mean Δ PDMS | t |
|---|---|---|---|
| windows (naive, what `submetrics.py` prints) | 12,146 | +0.00768 | 2.65 |
| segment means | 136 | +0.00714 | 1.87 |
| log means | 44 | +0.00693 | 1.46 |
| log block bootstrap (4,000 resamples) | 44 | +0.00765, 95 % CI [+0.0010, +0.0145] | — |

Window-level ICC of the PDMS delta is 0.01, design effect ~1.9, so clustering
inflates the window-level t by ~1.4x rather than by sqrt(89). One SE at segment
level is **0.38 PDMS points**, i.e. the eval resolves ~1-point differences at ~2.5 sigma
and cannot resolve anything under ~0.5. All the non-DTS arms (-0.22 … -0.56 %) sit
inside 1.2 SE. The +0.95 % survives the coarser units but only marginally (p ~ 0.06
at segment level, 0.15 at log level) — it moves from "outside noise" to "borderline"
and needs the second seed.

Estimator choice moves the verdict on the same two checkpoints, so quote the unit:
window-level t 2.65 (p 0.008); cluster-robust by segment t 2.50; cluster bootstrap on
the window-weighted mean 95 % CI [+0.0016, +0.0137] (P(<=0) = 0.008); equal weight per
scene CI [-0.0002, +0.0146] (P(<=0) = 0.030); sign-flip on scene means p 0.061; log
level p 0.158. Segment size does not bias the delta (corr(log windows, delta) = +0.03),
so the spread between estimators is variance, not a length effect. With a scene-level
delta SD of 0.044 and 136 scenes, 80 % power is only reached at a **1.1-point** PDMS
difference — a 0.8-point effect is expected to land on the boundary, which is exactly
where this one sits.

The hard-flag surplus behaves the same way: 68 segments net +236 vetoed scenes
against 39 segments net -100 (120 of the 136 segments involved, largest single net
+14), so it is a broad, small shift rather than a handful of drives.

### Sub-metrics, 2B family, 12,146 common scenes, paired against `aligned_bf16`

`python3 submetrics.py aligned_bf16 aligned_hif8_132 aligned_hif8_132_dts aligned_hif8_196 aligned_hif8_196_dts aligned_hif8_168`

| sub-metric | bf16 | 132 | 132-dts | 196 | **196-dts** | 168 |
|---|---|---|---|---|---|---|
| NC no-fault collision (hard) | 0.97427 | 0.97402 (−0.03 %) | 0.97411 (−0.02 %) | 0.97328 (−0.10 %) | **0.97604 (+0.18 %)** | 0.97143 (−0.29 %) |
| DAC drivable area (hard) | 0.90277 | 0.90252 (−0.03 %) | 0.90227 (−0.06 %) | 0.90482 (+0.23 %) | **0.91092 (+0.90 %)** | 0.90186 (−0.10 %) |
| DDC driving direction (hard) | 0.96690 | 0.96814 (+0.13 %) | 0.96600 (−0.09 %) | 0.96801 (+0.12 %) | **0.96913 (+0.23 %)** | 0.96723 (+0.03 %) |
| EP ego progress (soft) | 0.75325 | 0.75202 (−0.16 %) | 0.75007 (−0.42 %) | 0.75193 (−0.18 %) | **0.75937 (+0.81 %)** | 0.74930 (−0.52 %) |
| TTC within bound (soft) | 0.92985 | 0.92631 (−0.38 %) | 0.92730 (−0.27 %) | 0.92467 (−0.56 %) | **0.93101 (+0.12 %)** | 0.92351 (−0.68 %) |
| comfort (soft) | 0.99992 | 0.99984 | 0.99984 | 0.99992 | 1.00000 | 0.99984 |
| **PDMS** | 0.81203 | 0.80980 (−0.28 %) | 0.80985 (−0.27 %) | 0.81026 (−0.22 %) | **0.81971 (+0.95 %)** | 0.80748 (−0.56 %) |
| PDMS paired t | — | −0.76 | −0.74 | −0.60 | **+2.65** | −1.53 |

### Sub-metrics, 7B family, same 12,146 scenes, paired against `w7b_bf16`

`python3 submetrics.py w7b_bf16 w7b_hif8_132 w7b_hif8_168 w7b_hif8_196`

| sub-metric | bf16 | 132 | 168 | 196 |
|---|---|---|---|---|
| NC | 0.98230 | 0.98226 (−0.00 %) | 0.98012 (−0.22 %) | 0.98123 (−0.11 %) |
| DAC | 0.92475 | 0.92335 (−0.15 %) | 0.92417 (−0.06 %) | 0.92269 (−0.22 %) |
| DDC | 0.97394 | 0.97575 (+0.19 %) | 0.97477 (+0.09 %) | 0.97452 (+0.06 %) |
| EP | 0.77838 | 0.77749 (−0.11 %) | 0.77784 (−0.07 %) | 0.77615 (−0.29 %) |
| TTC | 0.94385 | 0.94443 (+0.06 %) | 0.94410 (+0.03 %) | 0.94097 (−0.31 %) |
| comfort | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| **PDMS** | 0.84066 | 0.83970 (−0.11 %) | 0.83999 (−0.08 %) | 0.83738 (−0.39 %) |
| PDMS paired t | — | −0.43 | −0.30 | −1.41 |

### Loss trajectory, 2B `aligned_bf16` vs `aligned_hif8_168`: EMA(β=0.9), 4000 steps

Added 2026-11-13. Both runs log one `{'loss': …}` line per step, 4000 each, so the
two series pair by step index:

| source | path |
|---|---|
| bf16 | `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs/aligned_bf16/train.log` |
| HiF8-168 | `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs/aligned_hif8_168/train.log` |

Pairing check: `corr(loss_bf16, loss_168) = 0.9999`; sd of the step-wise difference
0.0173 against sd of the loss 0.381 after warmup. The arms saw the same batch order,
so what follows is the quantisation difference, not data order.

Definitions, because the two "EMA" numbers are not interchangeable:

- `rel_t = (L_168,t − L_bf16,t) / L_bf16,t` (signed, per step)
- `E_1 = L_1`, `E_t = 0.9·E_{t−1} + 0.1·L_t` — EMA of the loss
- **rel of the EMA** = `(E_168,t − E_bf16,t) / E_bf16,t` — this is the "EMA diff"
- **EMA of the rel** = `EMA(rel_t)` — a different number, quoted below for contrast

| quantity | value |
|---|---|
| **EMA(0.9) rel difference, mean over 4000 steps** | **+0.3045 %** |
| EMA(0.9) rel, endpoint (step 4000) | +0.7500 % |
| EMA(0.9) rel, min … max | −0.4327 % … +1.1114 % |
| EMA(0.9) rel, mean over last 500 steps | +0.4748 % |
| EMA(0.9) rel, mean over last 100 steps | +0.5731 % |
| EMA of `rel_t`, mean / endpoint | +0.3081 % / +0.7463 % |
| EMA(β=0.99) rel, mean / endpoint | +0.2723 % / +0.5380 % |
| per-step `rel_t`, signed mean / absolute mean | +0.3108 % / 0.6728 % (ratio 2.17) |
| per-step `rel_t`, median / p95 / p99 / max | 0.5501 % / 1.7593 % / 2.3030 % / 4.2399 % |
| mean Δ in nats; `exp(Δ)−1` | +0.00579 nats; +0.5805 % |
| EMA(0.9) steps outside the 0.25 % band | 2400 / 4000 (60.0 %) |
| post-warmup (201–4000) EMA rel, mean | +0.3245 % |

Drift, mean of `rel of the EMA` inside each window: 1–500 +0.1056 % → 501–1000
+0.2922 % → 1001–2000 +0.2168 % → 2001–3000 +0.3435 % → 3001–3500 +0.4428 % →
3501–4000 +0.4748 %. Final printed loss 1.8703 (bf16) vs 1.8910 (168), i.e. +1.1068 %
on the last step.

Reading the loss: the offset is one-directional (HiF8 strictly higher,
`abs/signed = 2.17`), it grows through training, and it fails the 0.25 % EMA band on
60 % of steps. In absolute terms it is small — the step-wise difference is 4.5 % of
the loss sd — so this is the same optimisation trajectory with a slight upward bias,
not a diverged run. It is also *not* the explanation of the PDMS delta below: the
2B bf16→168 PDMS shift is −0.56 % and is carried by hard-flag membership, while the
7B 168 arm ends at a **larger** final-loss gap (+1.26 %, 1.6984 → 1.7198) yet a
**smaller** PDMS shift (−0.08 %), so loss and PDMS are not proportional here.

### Reading, and what these numbers do not support

- `aligned_hif8_196_dts` is the only arm in either family that moves PDMS outside
  noise (+0.95 %, paired t 2.65). Every other arm is a small negative
  (−0.08 … −0.56 %, |t| < 1.6); on 7B nothing beats bf16 in any direction, largest
  loss −0.39 % on the 196 arm. In the 11-12 `night` family two arms (168, 196)
  were above bf16, but that family is a different recipe, so it is only the
  **sign pattern** that repeats (the `196` config above bf16 in both families), not
  the magnitude.
- **Where the +0.95 % actually comes from** (decomposition over the per-scene CSVs;
  groups defined by whether any of NC/DAC/DDC is 0 in an arm):

  | group | n | contribution to Δ PDMS |
  |---|---|---|
  | both clean | 9,903 | −0.00100 |
  | bf16 vetoed, arm not | 801 | +0.05500 |
  | arm vetoed, bf16 not | 665 | −0.04636 |
  | both vetoed | 777 | +0.00005 |
  | **total** | 12,146 | **+0.00768** |

  Hard-flag zeros, bf16 -> arm, are NC 276 -> 254, DAC 1,181 -> 1,082,
  DDC 226 -> 206, i.e. 1,578 -> 1,442 scenes carrying any veto: **136 fewer**.
  That surplus is the entire delta. Inside the 9,903 scenes where neither arm is
  vetoed the quantised arm is slightly **worse** (EP −0.0033, TTC +0.0003,
  comfort 0.0000), so the all-scene EP +0.81 % above is a membership effect — which
  scenes carry a veto — not better progress where both arms drive cleanly.
- The other three arms show the same two large opposing groups nearly cancelling
  (196: 781 / 757; 132-dts: 771 / 762; 168: 763 / 786) and are likewise negative
  inside both-clean scenes (−0.0019 … −0.0023). So "a quantised arm beats bf16"
  is here a statement about 136 scenes out of 12,146 where bf16 takes a DAC/NC hit
  and the quantised arm does not — not about a better policy.
- Candidate causes for that surplus, **none established**: (i) training noise —
  one run per arm, so the run-to-run spread of the veto count is unknown;
  (ii) quantisation noise acting as a regulariser in a short schedule (4000
  steps x 48 = 192 k draws over 103,288 windows, i.e. 1.86 passes per distinct
  window but only **1,192 distinct driving behaviours**) — consistent with the sign
  pattern, higher teacher-forced loss (1.8844 vs 1.8703; the 168 arm shows the same
  sign at +0.30 % mean / +0.75 % final, see the loss-trajectory section) yet fewer hard-flag
  zeros; (iii) DTS scale refresh cutting late-training scale staleness — the
  cts -> dts swing on the same ignore list is 0.81026 -> 0.81971, but this presumes
  the refresh schedule ran. The single-knob contrasts from the scope table above
  (168 -> 196 = `o_proj` quantisation, +0.35 pt; 196 -> 196_dts = refresh, +1.17 pt)
  put almost all of the swing on the refresh, not on the extra 28 matrices.
  (i) vs (ii) separate with a second seed; (iii) needs the mechanism test.
- **The DTS gain is not attributable yet.** ZYC's own `verify_dts.py` (11-13 09:45)
  records that the refresh events never reached `train.log` (the `hif8_dts.py`
  logger is unconfigured), so "delayed scaling is better" and "the schedule never
  fired, so it quantised less" are both live. The mechanism test (optimizer
  post-step hook, scales constant inside a window, refreshes == steps // interval)
  was written but had produced no output file by 09:59.
- One seed, one run per arm, no error bar. The 11-12 section's finding is confirmed
  for these arms: a PDMS delta of this size is the net imbalance of ~800-scene
  hard-flag groups swinging ±0.05 each, so a +0.95 % needs a second seed before it
  is quoted as a gain.
- `mini` (21 scenes) is not a comparison: it ranks 168 last on 2B and first on 7B.
  It only proves the checkpoint loads and the chain runs.
- Every eval line prints `valid results: 0`; the CSVs hold 12,146 real scenes and
  reproduce the logged PDMS exactly, so it is a harness log quirk — cause not
  established (the 11-12 four-arm run printed the same).
- PDMS scores the action policy only: `denoiser.*`/`vae.*` load as unexpected keys
  at inference, so the world model is exercised in training, not in this metric.

### How these evals were run

`bash tools/zyc/eval_ckpt.sh <runs/<tag>> <tag> full 8` — inference with
`torchrun --nproc_per_node=8 inference/qwen/inference_action_navsim_vava.py
--test_data_pkl navsim_workspace/navtest_qwen_2va.pkl --token_yaml
navsim_workspace/navtest_tokens.yaml --norm_stats_path
configs/normalizer_navsim_trainval/norm_stats.json --cur_frame_idx 3 --future_nums 8
--model_max_length 1400 --use_previous_actions --bf16 --save_gt`, then
`run_pdm_score.py train_test_split=navtest agent=emu_vla_agent` from
`/mnt/sfs_turbo/zyc/miniconda3/envs/navsim`. Same pkl / token yaml / metric cache
as the 11-12 run (pkl and yaml 11-11 09:38, cache 11-11 09:17), 22 min per full
eval at 8 cards vs 38 min at 2 cards per arm.

### Box state while this was recorded (11-13 09:59 CST)

- No ZYC training or eval process alive on a54; nothing queued after the 09:42 eval.
- `diskguard.log` still ticking (15.3 T free of 21 T); no extraction running.
- The nuPlan data pipeline is blocked at step 3a: `pipeline/step3a_analyze_displacement_distribution.py:336`
  calls `input()` for the threshold choice, so it dies with `EOFError` under a
  non-interactive shell. Step 2 finished first (15,895 actions + waypoints).

### Open (this section)

1. Second seed on at least `aligned_hif8_196_dts` and `aligned_bf16` before quoting
   +0.95 % — and report it with a segment- or log-level interval, not the window-level
   t (see the window-design subsection: 12,146 windows are 136 segments / 44 drives).
2. Finish `verify_dts.py` and record its output: the 196-dts result is not
   interpretable until the refresh schedule is shown to have run.
3. Publish the HiF8/DTS/determinism delta — still uncommitted in ZYC's checkout.
4. Give step 3a a non-interactive threshold argument so the pipeline can run
   unattended.

## 2026-11-16 — Pretrain x SFT precision 2x2, sub-metrics, EMA loss, and two SFT stages killed mid-run

All four arms share the same outer recipe: nuPlan 2VA pretrain 8000 steps at lr 5e-5,
gbs 96 (mb 12 x 8), then SFT 4000 steps at lr 1e-5, bs 6 x 8, seed 42, then full
navtest (12,146 scenes / 36,438 json, PDMS v1). The pretrain-lr sweep that produced
the 5e-5 setting is recorded in ZYC's `drivevla-w0-hif8-handoff.md` (08 section), not
here — this section only covers the precision cross.

"HiF8" below means the DTS variant with `interval=10` and `ignore=[lm_head,o_proj]`,
which quantises **196 of 497 Linear layers** (301 left in bf16).

### The four arms

| cell | SFT run dir | pretrain init | chain / driver log |
|---|---|---|---|
| **1** BF/BF | `runs/sft_ptlr5e5` | `runs/pt_ptlr5e5` | `chain_ptlr5e5.log` |
| **2** BF/HiF | `runs/sft_bfpt5e5_dts196sft` | `runs/pt_ptlr5e5` (same as 1) | `sftpt_bfpt5e5_dts196sft.log` |
| **3** HiF/BF | `runs/sft_dtspt5e5_bfsft` | `runs/pt_dts196_lr5e5` | `sftpt_dtspt5e5_bfsft.log` |
| **4** HiF/HiF | `runs/sft_dts196_lr5e5` | `runs/pt_dts196_lr5e5` | `chain_dts196_lr5e5.log` |

Cells 2 and 4 reuse their column partner's pretrain checkpoint byte-for-byte, so the
pretrain stage has only **two** distinct trajectories (1=2 and 3=4) while SFT has four.

### Results

| Pretrain \ SFT | bf16 | HiF8-DTS-196 |
|---|---|---|
| **bf16** | **1** 0.85473 | **2** 0.85499 |
| **HiF8-DTS-196** | **3** 0.84810 | **4** 0.85044 |

| metric | 1 BF/BF | 2 BF/HiF | 3 HiF/BF | 4 HiF/HiF |
|---|---|---|---|---|
| NC | 98.493 | 98.469 | 98.242 | 98.337 |
| DAC | 93.644 | 93.776 | 93.150 | 93.430 |
| DDC | 97.855 | 97.884 | 97.859 | 97.921 |
| EP | 79.151 | 79.168 | 78.517 | 78.776 |
| TTC | 95.077 | 94.937 | 94.764 | 94.607 |
| comfort | 100.000 | 100.000 | 99.984 | 100.000 |
| **PDMS** | **85.473** | **85.499** | **84.810** | **85.044** |

The four single-knob contrasts (paired over the same 12,146 scenes, `submetrics.py`):

| contrast | variable isolated | dPDMS | paired t |
|---|---|---|---|
| 2 - 1 | quantise SFT only, bf16 pretrain | +0.03 | +0.13 |
| 4 - 3 | quantise SFT only, HiF8 pretrain | +0.23 | +1.10 |
| 3 - 1 | quantise pretrain only, bf16 SFT | -0.66 | -2.73 |
| 4 - 2 | quantise pretrain only, HiF8 SFT | -0.46 | -1.95 |

### Reading

- **Quantising the SFT stage is free.** Both SFT-only contrasts are positive and
  statistically nil (t +0.13, +1.10); per-factor movement is under 0.3 points on
  every metric. This closes the SFT-side quantisation axis rather than opening it.
- **Quantising the pretrain stage is the only directional cost**, and it is broad:
  EP -0.39 to -0.63, DAC -0.35 to -0.49, TTC -0.31 to -0.33, NC -0.13 to -0.25 all
  move together, DDC does not move (+0.04 / +0.06, noise). So it is not a comfort or
  direction artefact: the quantised-pretrain model drives slightly less far and
  clips slightly more.
- Magnitude is 0.46-0.66 points, about **one noise floor** (+-0.006 from the three
  2e-4 replicates). One seed per cell, no repeats. The paired t is legitimate as a
  within-split comparison of these two trained models; it is not a statement about
  seed-to-seed variation. Report this as a trend, and **do not quote 0.85499 as
  beating 0.85473** — they are tied (+0.03, t +0.13).
- Practical rule for the pipeline: keep HiF8 out of nuPlan pretraining, put it
  anywhere in SFT. Nothing here changes the lr finding; all four cells sit at
  pretrain lr 5e-5 and all four are 4.5-5 points above the 2e-4 arms.

### Loss trajectory: EMA(beta=0.9), action_loss

Definitions as in the 11-13 section: `E_1 = L_1`, `E_t = 0.9 E_{t-1} + 0.1 L_t`,
deviation reported as rel of the EMA against cell 1. Losses are `action_loss` from
each run's `train.log` (the `ross_loss` field in the same dict was not used).

Pretrain, two distinct trajectories:

| EMA @ step | 1 BF | 3 HiF8-DTS-196 | rel |
|---|---|---|---|
| 500 | 1.7456 | 1.7501 | +0.26 % |
| 1000 | 1.5758 | 1.5898 | +0.89 % |
| 2000 | 1.3157 | 1.3100 | -0.43 % |
| 4000 | 1.2196 | 1.2319 | +1.00 % |
| 6000 | 1.2060 | 1.2034 | -0.22 % |
| 8000 | 1.1165 | 1.1257 | +0.82 % |
| mean, last 500 | 1.1185 | 1.1231 | +0.41 % |

Signed mean over all 8000 steps +0.31 %, absolute mean 0.47 %; the sign flips six
times across the run.

SFT, four arms:

| EMA @ step | 1 BF->BF | 2 BF->HiF | 3 HiF->BF | 4 HiF->HiF |
|---|---|---|---|---|
| 100 | 1.6779 | 1.6777 | 1.6786 | 1.6798 |
| 200 | 1.5427 | 1.5443 | 1.5620 | 1.5628 |
| 500 | 1.4234 | 1.4271 | 1.4204 | 1.4243 |
| 1000 | 1.4483 | 1.4487 | 1.4418 | 1.4430 |
| 2000 | 1.4147 | 1.4127 | 1.4241 | 1.4266 |
| 3000 | 1.2242 | 1.2309 | 1.2496 | 1.2466 |
| 4000 | 1.1917 | 1.1922 | 1.2062 | 1.2054 |
| mean, last 500 | 1.2052 | 1.2101 | 1.2195 | 1.2230 |

Relative EMA deviation against cell 1 (%):

| | @100 | @500 | @2000 | @4000 | last 500 | signed mean | abs mean |
|---|---|---|---|---|---|---|---|
| cell 2 | -0.01 | +0.26 | -0.14 | **+0.05** | +0.41 | +0.17 | 0.22 |
| cell 3 | +0.04 | -0.21 | +0.67 | **+1.22** | +1.21 | +0.83 | 0.95 |
| cell 4 | +0.11 | +0.07 | +0.85 | **+1.15** | +1.51 | +0.96 | 1.06 |

- The SFT loss ranks the arms **exactly as PDMS does**: cells 1 and 2 (1.1917 / 1.1922
  -> 85.47 / 85.50) below cell 4 (1.2054 -> 85.04) below cell 3 (1.2062 -> 84.81).
  The split is inherited from the pretrain stage and is already visible at **SFT step
  200** (+1.25 / +1.30 % for cells 3 / 4), i.e. the quantised pretrain hands SFT a
  measurably worse starting point before the optimiser has done any work.
- **The pretrain loss does not detect the damage.** Cell 3's 8000-step EMA stays
  inside +-1 % of cell 1 and changes sign six times, yet the same checkpoint
  fine-tuned identically loses 0.66 PDMS. So pretrain action loss cannot be used as
  an early-stopping or acceptance signal for a quantised pretrain, and the 2x2 PDMS
  ranking cannot be predicted from the 8k-step curve. This is the opposite of the
  pretrain-lr sweep, where loss and PDMS moved together.
- The shared bump (minimum near step 1000, rise to 2000, drop at 3000) appears in all
  four SFT curves, so it is a property of the SFT recipe / data order, not of an arm.

### Two SFT stages were killed by SIGTERM this morning

| chain | host | SFT killed at | last milestone |
|---|---|---|---|
| `chain_dts132_lr5e5` | a53 | 09:04:39 | SFT step 50 |
| `chain_dts168_lr5e5` | a54 | 09:06:09 | SFT step 50 |

Both `runs/sft_dts{132,168}_lr5e5/train.log` end in `got signal: 15` +
`TRAIN_EXIT=1`, and the chain shells died with them (neither chain log has its
stage-2 exit-code line). Stage-1 checkpoints (1569 tensors / 2 shards / 6.77 GB each)
survived in `runs/pt_dts{132,168}_lr5e5`.

**Cause not established.** Two different hosts, 90 s apart, with no matching entry in
any local log: `diskguard.log` shows 14 TB free and `暂停=0` throughout, and both
`release_*.log`s had finished at 08:44. These two are the "HiF8 pretrain + HiF8 SFT"
arms at 168 / 132 granularity, i.e. not part of the 2x2 above.

### SFT-side quantisation granularity: three placements, all inside noise

The 132 arm's 08:43 eval produced only 5,292 of 36,438 json and was marked FAILED;
both were re-scored by a 4-card eval-only rerun on a54 the same morning.

| arm | SFT quantisation | PDMS |
|---|---|---|
| `bfpt5e5_dts168sft` | DTS, ignore `lm_head,o_proj` | 0.85552 |
| `bfpt5e5_dts196sft` | DTS, ignore `lm_head,o_proj` (196 Linear) | 0.85499 |
| `bfpt5e5_dts132sft` | DTS, ignore `lm_head,o_proj,layers.0-2,layers.25-27` | 0.85286 |

All three start from the same bf16 pretrain checkpoint (`pt_ptlr5e5`) and differ only
in SFT quantisation scope. Spread 0.8529-0.8555 = 0.27 points, every arm within
|t| < 1.7 of cell 1's 0.85473, and 0.85552 is not a new best for the same reason
0.85499 is not. Together with the 2x2 this says the SFT-side quantisation axis is
flat, so further granularity sweeps there are not worth 8-card time.

### All eight arms: the corrected table, and where each gap comes from

Extends the three-placement subsection above to every arm with a full navtest score on the
box, including the two the circulated table omits: `dtspt5e5_bfsft` (pretrain-only
quantisation) and the 168 / 132 HiF8-HiF8 diagonal. Machine-generated from the per-scene
CSVs, nothing re-scored — `tools/zyc/navsim_exp_stats.py <exp dirs>` for the components,
`tools/zyc/pdm_attrib.py <base> <arms>` for the split.

Units: PDMS and every component are means x100 over the 12,146 scenes. The CSV holds 12,147
rows because `run_pdm_score.py` appends a `token=average` summary row; averaging a set
together with its own mean leaves the mean unchanged, so no cell here is biased by it. A gap
quoted in **pd** is one unit of that x100 scale, i.e. 85.473 -> 85.552 is +0.079 pd.

| metric | type | bf16/bf16 | bf16/196 | bf16/168 | bf16/132 | 196/bf16 | 196/196 | 168/168 | 132/132 |
|---|---|---|---|---|---|---|---|---|---|
| NC | hard, mult | 98.49333 | 98.46863 | 98.40277 | 98.34102 | 98.24222 | 98.33690 | 98.35748 | 98.39865 |
| DAC | hard, mult | 93.64400 | 93.77573 | 93.93216 | 93.61107 | 93.15001 | 93.42994 | 93.64400 | 93.77573 |
| DDC | weight 0 | 97.85526 | 97.88408 | 97.85114 | 97.95818 | 97.85938 | 97.92113 | 97.92524 | 97.99111 |
| EP | soft 5/12 | 79.15055 | 79.16768 | 79.18537 | 78.93541 | 78.51746 | 78.77616 | 78.99305 | 79.07991 |
| TTC | soft 5/12 | 95.07657 | 94.93660 | 94.82957 | 94.92014 | 94.76371 | 94.60728 | 94.48378 | 94.55788 |
| comfort | soft 2/12 | 100.00000 | 100.00000 | 99.99177 | 100.00000 | 99.98353 | 100.00000 | 99.98353 | 100.00000 |
| **PDMS** | | **85.47303** | **85.49854** | **85.55184** | **85.28577** | **84.81018** | **85.04436** | **85.19759** | **85.35076** |
| relative to base | | — | +0.030 % | +0.092 % | -0.219 % | **-0.776 %** | -0.502 % | -0.322 % | -0.143 % |
| dPDMS (pd) | | 0.0000 | +0.0255 | +0.0788 | -0.1873 | **-0.6629** | -0.4287 | -0.2754 | -0.1223 |

**The component means do not multiply back to PDMS, so do not reconstruct the last three
rows by hand.** Feeding the column means into `NC * DAC * (5*EP + 5*TTC + 2*comfort) / 12`
gives 82.33 for the base arm against the measured 85.47. The formula holds per scene, and the
components correlate strongly (a scene that loses NC usually loses DAC too), so the mean of
the per-scene products sits 3.1 pd above the product of the means. The same nonlinearity is
why the first-order leverage split below sums to +0.095 pd against a real +0.079, and why the
Shapley column, not the leverage column, is the one to quote when the gap is a few hundredths
of a pd.

Four corrections against the circulated version, none of which changes a digit of its seven
present columns:

1. **`196/bf16` was missing.** It is the only arm that isolates pretrain quantisation behind
   a bf16 SFT, and it is the lowest score in the family (-0.66 pd). Without it the table
   halves the pretrain cost.
2. **comfort typo**: last column read `100000`, measured 100.00000.
3. **DDC is not "hard".** The score is `NC * DAC * (5*EP + 5*TTC + 2*comfort) / 12` with
   `driving_direction_weight = 0.0` in `PDMScorerConfig`, so DDC is reported but enters no
   score. Keep the row, draw nothing from it. Formula checked per scene: it reproduces the
   `score` column on 12,146 of 12,147 rows to 1e-9, the one miss being the average row.
4. The three right-hand columns are a **diagonal**, not a grid: each pairs with the same SFT
   granularity, so no off-diagonal cell (e.g. 196 pretrain + 132 SFT) exists.

Paired against `bf16/bf16` over the same scenes, on the per-scene `score` column:

| arm | dPDMS (pd) | t | 95 % CI (pd) |
|---|---|---|---|
| bf16/196 | +0.0255 | +0.13 | -0.373 … +0.424 |
| bf16/168 | **+0.0788** | **+0.40** | **-0.309 … +0.467** |
| bf16/132 | -0.1873 | -0.89 | -0.598 … +0.224 |
| 196/bf16 | **-0.6629** | **-2.73** | **-1.138 … -0.187** |
| 196/196 | -0.4287 | -1.82 | -0.891 … +0.033 |
| 168/168 | -0.2754 | -1.18 | -0.733 … +0.183 |
| 132/132 | -0.1223 | -0.52 | -0.581 … +0.337 |

Only `196/bf16` clears |t| = 2. Everything on the bf16-pretrain row sits at |t| < 0.9 with a
CI several times its own point estimate.

**Why PDMS can rise while three of five components fall.** 1 percentage point of each
component is worth, in pd (partial derivative at base values):

| NC | DAC | EP | TTC | comfort |
|---|---|---|---|---|
| 85.75 | **88.96** | 38.44 | 38.44 | 15.37 |

So for `bf16/168` the DAC gain of +0.288 pp is worth +0.256 pd while the TTC loss of
-0.247 pp costs only -0.095 pd and NC's -0.091 pp costs -0.078 pd: a mid-sized DAC gain
outweighs equal-sized EP / TTC losses by 2.3x. A score increase here is a property of the
weighting, not evidence that the arm drives better.

Exact Shapley split of each gap across the five scored components (sums to dPDMS by
construction, pd):

| arm | dPDMS | NC | DAC | EP | TTC | comfort |
|---|---|---|---|---|---|---|
| bf16/196 | +0.026 | -0.025 | +0.105 | -0.007 | -0.048 | 0.000 |
| bf16/168 | +0.079 | -0.044 | **+0.204** | -0.015 | -0.065 | -0.001 |
| bf16/132 | -0.187 | -0.075 | -0.006 | -0.061 | -0.045 | 0.000 |
| 196/bf16 | -0.663 | -0.129 (20 %) | **-0.334 (50 %)** | -0.124 (19 %) | -0.073 (11 %) | -0.003 |
| 196/196 | -0.429 | -0.097 (23 %) | -0.095 (22 %) | -0.089 (21 %) | -0.148 (35 %) | 0.000 |
| 168/168 | -0.275 | -0.069 (25 %) | +0.031 (helps) | -0.052 (19 %) | **-0.184 (67 %)** | -0.002 |
| 132/132 | -0.122 | -0.053 (43 %) | **+0.142 (helps)** | -0.026 (22 %) | **-0.185 (152 %)** | 0.000 |

Readings that follow:

- **`bf16/168` is a tie, not a win.** +0.079 pd with t = +0.40 and a CI six times the point
  estimate; scene-level it is 4,052 higher, 4,206 lower, 3,888 identical, i.e. net -154
  scenes with the mean positive only because the wins are individually larger. Quoting
  0.85552 as beating 0.85473 is the same error as quoting 0.85499.
- **The DAC and TTC moves are not one tradeoff.** corr(dDAC, dTTC) = +0.055 over the scenes,
  with 511 scenes moving on DAC only, 326 on TTC only, 30 on both. Naming a mechanism
  ("takes more margin, accepts shorter TTC") would be invention; what is measured is two
  disjoint scene sets moving independently.
- **The pretrain cost is drivable-compliance-led, not collision-led** in `196/bf16` (DAC half
  the gap), and it is the one arm with a paired scene-level collision increase: 95 scenes
  collide only with the arm vs 63 only with the base (McNemar p = 0.013), and 438 vs 378 on
  drivable-area (p = 0.039, borderline across six tests).
- **Narrowing the quantisation changes the shape, not just the size.** 196 -> 168 -> 132
  moves DAC -0.095 / +0.031 / +0.142 (above base by 132) while TTC stays -0.148 / -0.184 /
  -0.185. The residue of a narrow quantised pretrain is shorter accepted TTC with drivable
  behaviour restored.
- **Flat trajectory error and moving PDMS are compatible.** ADE 0.809-0.828 / FDE
  1.762-1.803 across all eight arms, from own post-processing of `action` vs
  `action_gt_denorm`, with the error predominantly longitudinal (lateral 0.214-0.219 m
  against longitudinal 0.729-0.750 m). PDMS does not score that planned path:
  `pdm_score.py` builds a proposal from it and runs it through `PDMSimulator` / LQR, so the
  scored path is a tracking simulation sampled at 10 Hz against agent boxes and the
  drivable map. See the erratum below: these are the corrected numbers, and the ones
  quoted before it were computed on a misaligned window.

Still missing to settle the diagonal: **PT-168 + bf16 SFT** and **PT-132 + bf16 SFT**. Both
stages move together in every right-hand column, so the improvement 196 -> 168 -> 132 cannot
currently be assigned to pretrain scope rather than SFT scope.

### Seven requested metrics, and an erratum on the earlier ADE / FDE

**The erratum first.** Every ADE / FDE / head / LDist figure quoted in this file before this
subsection was computed on the wrong thing, in two independent ways. The dumps are per-step
SE2 increments, not poses: `tools/pickle_gen/pickle_generation_navsim_pre_1s.py` builds
`relative_action_list` as `rel_all[j, j+1]`, commented "only relative", and its `pre_1s`
variant shifts that list by 2 frames, which fixes the frame spacing at 0.5 s. So

1. increments expressed in a rotating frame have to be composed
   (`pose[k+1] = pose[k] + R(theta[k]) . (dx, dy)`) before an error means anything;
   differencing them raw measures a displacement difference, not a position error;
2. the GT window is 12 increments spanning `[i-3, i+8]`, so its first three rows are *past*
   intervals and the 8-step predicted horizon aligns with `gt[3:11]`, not `gt[0:8]`.

The alignment is measured, not assumed: a shift search over 200 scenes gives a mean |d dx| of
0.450 m at shift 0 against 0.230 m at shift +3. Both defects are fixed in
`tools/zyc/traj_metrics.py` (`87f4235`), which now carries a `--selftest` covering
composition, alignment and a turning case.

Corrected, all eight arms, 12,146 scenes each. ADE / FDE / LDist in metres over the 4 s
horizon, head in degrees:

| arm | ADE | FDE | head | LDist | lateral |
|---|---|---|---|---|---|
| bf16/bf16 | 0.813 | 1.767 | 1.585 | 0.734 | 0.217 |
| bf16/196 | 0.809 | 1.762 | 1.587 | 0.730 | 0.216 |
| bf16/168 | 0.812 | 1.764 | 1.580 | 0.734 | 0.214 |
| bf16/132 | 0.813 | 1.766 | 1.582 | 0.734 | 0.216 |
| 196/bf16 | 0.828 | 1.803 | 1.590 | 0.750 | 0.217 |
| 196/196 | 0.821 | 1.786 | 1.587 | 0.741 | 0.219 |
| 168/168 | 0.813 | 1.762 | 1.578 | 0.736 | 0.214 |
| 132/132 | 0.809 | 1.763 | 1.588 | 0.729 | 0.217 |

The ranking and the spread are unchanged from the invalid version, so no conclusion in this
section moves; the magnitudes roughly double and the lateral error is 0.217 m, not 0.031 m.
One reading does change shape: `196/bf16`, the worst PDMS arm, is also the worst on ADE,
FDE, head and LDist, which the broken version flattened into 0.472-0.476.

The seven metrics for the non-DTS (CTS) arms that were asked for (`tools/zyc/metrics7.py`,
run 2026-11-17 on the 11-13 `aligned` family, where the quantisation is CTS and nothing in
the pretrain path is quantised; the two-stage CTS cells of 11-17/18 are a separate section
below and are not in this table):

| metric | bf16 | cts-168 | cts-196 | cts-168 vs base | cts-196 vs base |
|---|---|---|---|---|---|
| ADE (m) | 1.0495 | 1.0400 | 1.0491 | -0.0095, t -1.40 | -0.0004, t -0.06 |
| FDE (m) | 2.0930 | 2.0932 | 2.1025 | +0.0003, t +0.02 | +0.0095, t +0.69 |
| Head Error (deg) | 2.0312 | 2.0439 | 2.0611 | +0.0128, t +0.67 | +0.0299, t +1.57 |
| LDist (m) | 0.9455 | 0.9357 | 0.9438 | -0.0098, t -1.44 | -0.0017, t -0.25 |
| Agent Coll | 2.272 % (276) | 2.528 % (307) | 2.330 % (283) | 169 vs 138, p 0.087 | 142 vs 135, p 0.719 |
| HB Coll | 0.008 % (1) | 0.000 % (0) | 0.000 % (0) | 0 vs 1, p 1.000 | 0 vs 1, p 1.000 |
| Lane Cross | 9.723 % (1,181) | 9.814 % (1,192) | 9.518 % (1,156) | 619 vs 608, p 0.775 | 599 vs 624, p 0.493 |

Sources: `navsim_workspace/exp/eval_aligned_bf16_full` (PDMS 81.20328),
`eval_aligned_hif8_168_full` (80.74802), `eval_aligned_hif8_196_full` (81.02600), each with
its own `eval_<tag>_full/json_output`, 12,146 scenes per arm. Run dirs `runs/aligned_bf16`,
`runs/aligned_hif8_168`, `runs/aligned_hif8_196`; their launch lines read `hif8=off`,
`hif8=cts ignore=lm_head,o_proj`, `hif8=cts ignore=lm_head`, i.e. no DTS anywhere.

**This trio is not the 11-16 2x2/diagonal and must not be compared with it.** The `aligned`
family is a different SFT recipe (lr 1e-5, 200-step warmup, no pretrain stage at all), which
is why every absolute number here is ~0.2 m worse than the chain arms above; only the paired
columns are meaningful.

Paired over the same scenes: t for the continuous rows, exact McNemar for the binary ones.
The ordering does match PDMS (168 lowest of the three, 196 in between), but the resolution is
thin: 168 is 0.010 m shorter in ADE and LDist at |t| ~ 1.4, takes 31 more agent collisions
(169 vs 138 discordant, p 0.087) and 11 more DAC zeroes; 196 moves head error the wrong way
(+0.030 deg, t +1.57) while recovering both DAC (-0.21 pp, p 0.493) and collisions
(p 0.719). Nothing crosses |t| = 2 and no binary row reaches p < 0.08. Three definitional
notes are needed to read the binary rows:

- **Agent Coll is `no_at_fault_collisions == 0.0`, and the column is ternary, not binary.**
  `pdm_scorer.py:340` assigns `0.0` when the struck object is in `AGENT_TYPES` and `0.5` when
  it is not, taking the minimum over the collisions of a proposal. `AGENT_TYPES` is
  `{BICYCLE, EGO, PEDESTRIAN, VEHICLE}`, so `0.0` is exactly "vehicle, pedestrian or
  cyclist" and the `0.5` rows -- 73 on the base arm, 80 on cts-168, 83 on cts-196 -- are
  static or unknown objects. Counting `NC < 1` instead would give 349 / 387 / 366 scenes, a
  different metric; the mean-based rate, `100 - NC`, is 2.573 % / 2.857 % / 2.672 % and mixes
  the two.
- **HB Coll is close to degenerate on these arms too.** It needs a collision *and* a predicted
  deceleration of at least 3 m/s^2, and predicted deceleration has almost no mass up there
  (p99 = 1.99 on all three arms; max 4.40 base / 4.59 cts-168 / 4.97 cts-196). The single base
  event comes from a 4.40 m/s^2 collision scene; among the 307 and 283 collision scenes of the
  quantised arms the largest predicted deceleration is only 2.18 and 2.17, so they cannot
  produce one at any threshold above that. Lower thresholds: 1.5 m/s^2 gives 11 / 5 / 16
  (p 0.146 / 0.332) and 1.0 m/s^2 gives 34 / 33 / 38 (p 1.000 / 0.665). The ground truth in the
  same dumps reaches 9.32 m/s^2 with 45 scenes at or above 3.0 over all 12 GT rows (29 over the
  8 predicted rows), so the data contains emergency braking that the model never plans. Why the
  predicted range stops near 3-5 is **not established**; the action tokenizer's quantile range
  is the candidate and it is untested.
- **Lane Cross is the drivable-map proxy**, `100 * (1 - mean DAC)`, i.e. 1,181 / 1,192 / 1,156
  scenes. It is not a painted-lane-line test and a normal lane change does not trip it.

### How these evals were run

Full navtest via `bash tools/zyc/eval_ckpt.sh <runs/<tag>> <tag> full 8` (8 cards,
~22 min per arm); the two re-scores used a 4-card eval-only variant, two arms in
parallel on a54's 8 cards, ~31 min each. Sub-metrics and paired t from
`python3 submetrics.py <base> <arm>...` run under
`/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python3`, which reads the per-scene CSVs
under `navsim_workspace/exp/eval_<tag>_full/*/*.csv`. The seven metrics come from
`python3 metrics7.py <exp dir> <json_output dir>` per arm, base first
(`eval_aligned_bf16_full`, `eval_aligned_hif8_168_full`, `eval_aligned_hif8_196_full` for the
CTS table; the same command with the chain dirs reproduces the DTS trio), which is what the
paired columns above are. The three `aligned_*` directories hold 24,292 json each
(12,146 x 2: dump + `_agent_human_traj_diff`, no PNG) and the eight chain directories
36,438 (12,146 x 3), so every number above is the full split. `metrics7.py` needed one fix
on 2026-11-17: its exact McNemar converted `2.0 ** n` to float and raised `OverflowError`
whenever the discordant pairs exceeded 1,024 (Lane Cross on these arms has ~1,200), so the
denominator is now the integer `1 << n`.

### Box state while this was recorded (11-16 10:02 CST)

- a54: the two 4-card eval-only jobs had just finished; no ZYC chain or queue alive.
- a53: no ZYC process, and `npu-smi` reports 8 `Ascend950DT` chips with 0 % AICore and
  0 MB HBM. On 11-15 the same host reported 8 `python3` processes at ~62 GB per card,
  so either the box or its driver view changed; not investigated.
- a52 and a51: another tenant's `VLLMWorker_DP` on all 8 cards each (91 GB / 83 GB).
- a3 / a31 / a38: other tenants' serving containers, ~63 GB HBM resident per chip with
  no process rows visible from the host. Do not treat these as idle.
- a21 / a22: unreachable (`Permission denied (publickey)`).

### Open (this section)

1. Second seed on cells 1 and 3 before the pretrain-quantisation cost is quoted: at
   0.46-0.66 points it is ~1 noise floor, and the entire claim rests on that one pair.
2. Find the cause of the 09:04 / 09:06 SIGTERM. Two hosts 90 s apart with no log entry
   is either a deliberate stop or an external sweeper; until it is identified, any
   relaunch can be killed the same way.
3. Relaunch the 132 / 168 SFT stages from their surviving stage-1 checkpoints if that
   pretrain-quantisation placement still matters — nothing is queued for them.
4. The pretrain loss being blind to the quantisation cost means a cheap acceptance
   test for a quantised pretrain needs to be something other than its own loss curve;
   a short SFT from the checkpoint is the obvious candidate and is untested.

## 2026-11-17/18 — CTS in rows 2 and 3 of the 2x2, all three ignore sets (a51-a54)

Why this exists: the only two-stage CTS cells on record were in the **2e-4 family**
(`sftpt_cts168`, `sftpt_cts196`, `chain_sft_cts196`), while the 2x2, the diagonal and the
trajectory tables all live in the **5e-5 family** (`pt_ptlr5e5` pretrain, SFT lr 1e-5). So
the sentence "CTS exists only as single-stage SFT" was false, but "there is no CTS cell in
the family the 2x2 quotes" was true. This run closes that gap with six cells.

Read "row 2" and "row 3" as the two axes of the 11-16 2x2, not as numbered rows of any
table: row 2 = **bf16 pretrain -> quantised SFT**, row 3 = **quantised pretrain -> quantised
SFT**. Each row is run at all three ignore sets, so the result is a 2x3 block rather than
the single 196 diagonal the 2x2 has.

Everything outside quantisation is byte-identical to the 11-16 arms: same pretrain
(`pt_ptlr5e5`, lr 5e-5, 8000 steps, gbs 96 = mb 12 x 8), same SFT (lr 1e-5, bs 6 x 8, 4000
steps, `zero2.json`, seed 42, GC off), same navtest full split (12,146 scenes / 36,438
json), same launchers (`sft_from_pt.sh`, `chain_ptlr.sh`) and the same container (`zyc-w0`,
the env every comparable DTS arm ran in). The only edits are `HIF8_*=cts` and the ignore
set, made through thin wrappers (`r2_<N>.sh`, `r3_<N>.sh`) that export environment variables
and exec the unmodified launcher. One further difference, non-behavioural: row 3 sets
`SAVE_STEPS_PT=2000` instead of the launcher default 1000, which changes how often a
pretrain checkpoint is written and nothing about the training trajectory.

### Setup

Ignore set -> quantised Linear out of 497 (the arithmetic: 196 - 28 `o_proj` = 168;
168 - 36 for the six listed layers x 6 Linears = 132):

| name | ignore list | quantised | left in bf16 |
|---|---|---|---|
| 132 | `lm_head,o_proj,layers.0.,layers.1.,layers.2.,layers.25.,layers.26.,layers.27.` | 132 | 365 |
| 168 | `lm_head,o_proj` | 168 | 329 |
| 196 | `lm_head` | 196 | 301 |

| node | row | arm | run dirs | log |
|---|---|---|---|---|
| a53 | 2 | CTS-168 -> CTS-132 -> CTS-196, serial | `runs/sft_bfpt5e5_cts{168,132,196}sft` | `sftpt_bfpt5e5_cts{168,132,196}sft.log` |
| a54 | 3 | CTS-168 | `runs/pt_cts168_lr5e5`, `runs/sft_cts168_lr5e5` | `chain_cts168_lr5e5.log` |
| a51 | 3 | CTS-132 | `runs/pt_cts132_lr5e5`, `runs/sft_cts132_lr5e5` | `chain_cts132_lr5e5.log` |
| a52 | 3 | CTS-196 | `runs/pt_cts196_lr5e5`, `runs/sft_cts196_lr5e5` | `chain_cts196_lr5e5.log` |

Row 3 needs one node per arm because the pretrain dominates (2.5 s/step x 8000); row 2
reuses the existing bf16 pretrain, so three arms fit on one node. Launched 2026-11-17
23:51 CST, last score 2026-11-18 07:41 CST.

Measured stage costs, for sizing future runs: pretrain 20,643 / 20,700 / 20,880 s
(132 / 168 / 196) against 20,708 s for `chain_dts196_lr5e5`, so CTS adds nothing
measurable to pretrain; SFT stage 93-98 min at 1.43 s/step; full eval 22-25 min on 8 cards.

### Results

Row 2 — bf16 pretrain `pt_ptlr5e5` -> CTS SFT (isolates SFT-side quantisation):

| ignore set | CTS | DTS counterpart | dPDMS vs DTS | t | vs bf16/bf16 (0.85473) |
|---|---|---|---|---|---|
| 132 | 0.85402 | 0.85286 | **+0.12 pd** | +0.55 | -0.07 pd (-0.084 %) |
| 168 | 0.85435 | 0.85552 | -0.12 pd | -0.57 | -0.04 pd (-0.045 %) |
| 196 | 0.85289 | 0.85499 | -0.21 pd | -1.02 | -0.18 pd (-0.215 %) |

Row 3 — CTS pretrain (8000 steps, lr 5e-5) -> CTS SFT:

| ignore set | CTS | DTS counterpart | dPDMS vs DTS | t | vs bf16/bf16 (0.85473) |
|---|---|---|---|---|---|
| 132 | **0.76743** | 0.85351 | **-8.61 pd** | **-26.76** | **-8.73 pd (-10.21 %)** |
| 168 | 0.82626 | 0.85198 | -2.57 pd | -9.54 | -2.85 pd (-3.33 %) |
| 196 | 0.80314 | 0.85044 | -4.73 pd | -16.15 | -5.16 pd (-6.04 %) |

Because row 2 and row 3 now share the *same* SFT quantisation, a cleaner pretrain-only
contrast is available than in the 11-16 2x2 (where the SFT side also changed):

| ignore set | CTS pretrain vs bf16 pretrain, CTS SFT | DTS equivalent |
|---|---|---|
| 132 | **-8.66 pd** (0.85402 -> 0.76743) | +0.07 pd (0.85286 -> 0.85351) |
| 168 | -2.81 pd (0.85435 -> 0.82626) | -0.35 pd (0.85552 -> 0.85198) |
| 196 | -4.98 pd (0.85289 -> 0.80314) | -0.45 pd (0.85499 -> 0.85044) |

So quantising the pretrain costs **0.0-0.5 pd with DTS and 2.8-8.7 pd with CTS** in this
family. That is the headline: the CTS costs are 5-14x the +-0.6 pd noise floor of the 196
cells (8.66 / 4.98 / 2.81 pd), while every DTS cost is inside it.

> **Superseded in part (2026-11-19).** The DTS half of that sentence did not reproduce on
> re-run: DTS pretrain cost -2.66 / -0.45 / -4.11 pd at ignore sets 132 / 168 / 196 against a
> same-node row-2 control, i.e. only 168 stayed inside the noise floor. CTS is still worse at
> every ignore set (2.8-8.7 pd), but the margin is no longer an order of magnitude. Read the
> 2026-11-19 section before quoting the DTS column of any contrast here.

### Components, paired against `ptlr5e5` over the same 12,146 scenes

`submetrics.py ptlr5e5 <arm>`; means x1, `t` is the paired t of the per-scene column.

| arm | NC | DAC | DDC | EP | TTC | comfort | PDMS | t vs base |
|---|---|---|---|---|---|---|---|---|
| bf16/bf16 (base) | 0.98493 | 0.93644 | 0.97855 | 0.79151 | 0.95077 | 1.00000 | 0.85473 | — |
| bf16/DTS-132 | 0.98341 | 0.93611 | 0.97958 | 0.78935 | 0.94920 | 1.00000 | 0.85286 | -0.89 |
| bf16/DTS-168 | 0.98403 | 0.93932 | 0.97851 | 0.79185 | 0.94830 | 0.99992 | 0.85552 | +0.40 |
| bf16/DTS-196 | 0.98469 | 0.93776 | 0.97884 | 0.79168 | 0.94937 | 1.00000 | 0.85499 | +0.13 |
| bf16/CTS-132 | 0.98353 | 0.93767 | 0.97876 | 0.79071 | 0.94879 | 1.00000 | 0.85402 | -0.35 |
| bf16/CTS-168 | 0.98419 | 0.93735 | 0.97966 | 0.79119 | 0.94879 | 1.00000 | 0.85435 | -0.18 |
| bf16/CTS-196 | 0.98390 | 0.93628 | 0.97913 | 0.78993 | 0.94871 | 0.99992 | 0.85289 | -0.89 |
| DTS-132/DTS-132 | 0.98399 | 0.93776 | 0.97991 | 0.79080 | 0.94558 | 1.00000 | 0.85351 | -0.52 |
| DTS-168/DTS-168 | 0.98357 | 0.93644 | 0.97925 | 0.78993 | 0.94484 | 0.99984 | 0.85198 | -1.18 |
| DTS-196/DTS-196 | 0.98337 | 0.93430 | 0.97921 | 0.78776 | 0.94607 | 1.00000 | 0.85044 | -1.82 |
| DTS-196/bf16 SFT | 0.98242 | 0.93150 | 0.97859 | 0.78517 | 0.94764 | 0.99984 | 0.84810 | -2.73 |
| CTS-132/CTS-132 | 0.95620 | 0.87650 | 0.95818 | 0.71027 | 0.90145 | 0.99992 | **0.76743** | -26.74 |
| CTS-168/CTS-168 | 0.97736 | 0.91742 | 0.97394 | 0.76529 | 0.93389 | 1.00000 | 0.82626 | -10.60 |
| CTS-196/CTS-196 | 0.96945 | 0.90260 | 0.96925 | 0.74582 | 0.91808 | 1.00000 | 0.80314 | -17.75 |

Row 2's components are flat: every column is within 0.21 pp of base and the largest |t| is
1.58 (NC on CTS-132, DDC on CTS-168). Row 3 moves **every** component in the wrong direction
at |t| >= 4.4, which rules out a comfort, direction or single-scenario artefact — these are
uniformly worse drivers. The three DTS row-3 arms and `DTS-196/bf16 SFT` are the comparison
columns here: they lose at most 0.66 pd (t >= -2.73), an order of magnitude short of the CTS
row-3 losses.

### Trajectory and event metrics (independent of the simulator)

`metrics7.py` reads `action` and `action_gt_denorm` from each arm's `json_output`, composes
8 increments of 0.5 s into poses aligned to gt rows 3..10, and reports twelve metrics:
**ADE** (mean per-step displacement, m), **FDE** (final displacement), **head** / **headF**
(mean and final absolute heading error, deg), **LDist** / **LDistF** (mean and final
absolute longitudinal error, m), **lat** / **latF** (mean and final absolute lateral error,
m), **latBias** (mean signed lateral error, m, `+` = prediction left of GT), plus
**AgentColl** / **LaneCross** (100 x the fraction of scenes with
`no_at_fault_collisions == 0.0` / `drivable_area_compliance < 0.5`, i.e. NAVSIM's own
agent-collision and drivable-area tests read off the score csv) and **HBColl** (collision
with a predicted decel >= 3 m/s^2).

This matters because PDMS scores an LQR rollout of the plan, not the plan itself, so a second
measurement on the raw prediction is genuinely independent. It reproduces the same ordering.

Mean over the eight 0.5 s steps (ADE / head / LDist / lat / latBias):

| arm | ADE m | head deg | LDist m | lat m | latBias m |
|---|---|---|---|---|---|
| bf16/bf16 (base) | 0.8134 | 1.5852 | 0.7339 | 0.2174 | +0.0003 |
| bf16/DTS-132 | 0.8132 | 1.5816 | 0.7344 | 0.2164 | -0.0004 |
| bf16/DTS-168 | 0.8118 | 1.5803 | 0.7338 | 0.2141 | -0.0007 |
| bf16/DTS-196 | 0.8090 | 1.5870 | 0.7297 | 0.2164 | -0.0032 |
| bf16/CTS-132 | 0.8084 | 1.5758 | 0.7290 | 0.2146 | +0.0000 |
| bf16/CTS-168 | 0.8117 | 1.5822 | 0.7322 | 0.2167 | -0.0011 |
| bf16/CTS-196 | 0.8139 | 1.5888 | 0.7354 | 0.2157 | -0.0036 |
| DTS-132/DTS-132 | 0.8095 | 1.5878 | 0.7295 | 0.2170 | -0.0048 |
| DTS-168/DTS-168 | 0.8134 | 1.5778 | 0.7363 | 0.2143 | -0.0065 |
| DTS-196/DTS-196 | 0.8212 | 1.5871 | 0.7409 | 0.2186 | -0.0033 |
| DTS-196/bf16 SFT | 0.8284 | 1.5900 | 0.7503 | 0.2174 | -0.0029 |
| CTS-132/CTS-132 | **1.0791** | **2.2895** | **0.9680** | **0.2792** | +0.0001 |
| CTS-168/CTS-168 | 0.9198 | 1.8358 | 0.8293 | 0.2392 | -0.0056 |
| CTS-196/CTS-196 | 0.9705 | 1.9415 | 0.8770 | 0.2497 | +0.0060 |

Final step only (FDE / headF / LDistF / latF), and the events:

| arm | FDE m | headF deg | LDistF m | latF m | AgentColl | HBColl | LaneCross |
|---|---|---|---|---|---|---|---|
| bf16/bf16 (base) | 1.7672 | 2.7395 | 1.5481 | 0.5439 | 1.375 % | 0.000 % | 6.356 % |
| bf16/DTS-132 | 1.7661 | 2.7367 | 1.5480 | 0.5422 | 1.515 % | 0.000 % | 6.389 % |
| bf16/DTS-168 | 1.7642 | 2.7501 | 1.5488 | 0.5355 | 1.490 % | 0.000 % | 6.068 % |
| bf16/DTS-196 | 1.7617 | 2.7568 | 1.5443 | 0.5413 | 1.416 % | 0.000 % | 6.224 % |
| bf16/CTS-132 | 1.7553 | 2.7329 | 1.5381 | 0.5359 | 1.515 % | 0.000 % | 6.233 % |
| bf16/CTS-168 | 1.7644 | 2.7517 | 1.5458 | 0.5430 | 1.466 % | 0.000 % | 6.265 % |
| bf16/CTS-196 | 1.7672 | 2.7574 | 1.5507 | 0.5411 | 1.482 % | 0.000 % | 6.372 % |
| DTS-132/DTS-132 | 1.7630 | 2.7528 | 1.5440 | 0.5417 | 1.507 % | 0.000 % | 6.224 % |
| DTS-168/DTS-168 | 1.7625 | 2.7366 | 1.5505 | 0.5353 | 1.490 % | 0.000 % | 6.356 % |
| DTS-196/DTS-196 | 1.7858 | 2.7605 | 1.5651 | 0.5476 | 1.540 % | 0.000 % | 6.570 % |
| DTS-196/bf16 SFT | 1.8034 | 2.7594 | 1.5882 | 0.5450 | 1.638 % | 0.000 % | 6.850 % |
| CTS-132/CTS-132 | **2.4433** | **4.3133** | **2.1282** | **0.7241** | **4.059 %** | 0.000 % | **12.350 %** |
| CTS-168/CTS-168 | 2.0174 | 3.2393 | 1.7699 | 0.5999 | 2.067 % | 0.000 % | 8.258 % |
| CTS-196/CTS-196 | 2.1445 | 3.5205 | 1.8844 | 0.6348 | 2.808 % | 0.008 % | 9.740 % |

Paired against base over the same 12,146 scenes, delta (t), from
`metrics7.py <base> <arm> ...` in one invocation:

| arm | dADE | dhead | dLDist | dlat | dlatBias | AgentColl | LaneCross |
|---|---|---|---|---|---|---|---|
| bf16/DTS-132 | -0.0003 (-0.07) | -0.0036 (-0.34) | +0.0005 (+0.13) | -0.0010 (-0.62) | -0.0007 (-0.36) | p 0.129 | p 0.904 |
| bf16/DTS-168 | -0.0016 (-0.44) | -0.0049 (-0.44) | -0.0001 (-0.04) | -0.0033 (-2.11) | -0.0010 (-0.52) | p 0.215 | p 0.144 |
| bf16/DTS-196 | -0.0044 (-1.23) | +0.0018 (+0.16) | -0.0042 (-1.16) | -0.0010 (-0.62) | -0.0035 (-1.78) | p 0.679 | p 0.538 |
| bf16/CTS-132 | -0.0051 (-1.40) | -0.0095 (-0.82) | -0.0049 (-1.37) | -0.0028 (-1.73) | -0.0003 (-0.14) | p 0.125 | p 0.568 |
| bf16/CTS-168 | -0.0017 (-0.48) | -0.0030 (-0.28) | -0.0017 (-0.47) | -0.0007 (-0.46) | -0.0015 (-0.78) | p 0.315 | p 0.693 |
| bf16/CTS-196 | +0.0005 (+0.14) | +0.0036 (+0.31) | +0.0015 (+0.41) | -0.0017 (-1.08) | -0.0039 (-2.03) | p 0.228 | p 0.968 |
| DTS-132/DTS-132 | -0.0040 (-0.80) | +0.0025 (+0.15) | -0.0044 (-0.91) | -0.0004 (-0.20) | -0.0052 (-1.99) | p 0.236 | p 0.584 |
| DTS-168/DTS-168 | -0.0001 (-0.01) | -0.0074 (-0.44) | +0.0023 (+0.48) | -0.0030 (-1.50) | -0.0068 (-2.60) | p 0.292 | p 1.000 |
| DTS-196/DTS-196 | +0.0077 (+1.61) | +0.0018 (+0.12) | +0.0069 (+1.46) | +0.0013 (+0.64) | -0.0036 (-1.46) | p 0.128 | p 0.363 |
| CTS-132/CTS-132 | +0.2657 (+33.39) | +0.7042 (+28.06) | +0.2340 (+29.27) | +0.0618 (+22.12) | -0.0002 (-0.07) | p 0.0000 | p 0.0000 |
| CTS-168/CTS-168 | +0.1064 (+17.07) | +0.2506 (+12.99) | +0.0954 (+15.32) | +0.0219 (+9.37) | -0.0059 (-1.92) | p 0.0000 | p 0.0000 |
| CTS-196/CTS-196 | +0.1571 (+23.86) | +0.3562 (+18.03) | +0.1431 (+21.67) | +0.0323 (+13.29) | +0.0057 (+1.75) | p 0.0000 | p 0.0000 |
| DTS-196/bf16 SFT | +0.0150 (+3.09) | +0.0047 (+0.31) | +0.0164 (+3.40) | +0.0001 (+0.04) | -0.0032 (-1.29) | p 0.013 | p 0.039 |

The three CTS pretrains are outside anything the other eleven arms reach on every axis:
ADE t = +33.39 / +17.07 / +23.86, all four LDist / lat columns at |t| >= 8.5, and both
events at exact-McNemar p = 0.0000. **The row-3 damage is not one-dimensional**: lateral
error rises by 0.022 / 0.031 / 0.062 m (t +9.4 to +22.1) alongside the longitudinal rise, and
`latBias` stays at or below 6 mm on all three (|t| <= 1.9 against base), so the added error is
scatter around GT rather than a directional offset.

`DTS-196/bf16 SFT` is the other arm that leaves the noise floor, and the split explains it:
ADE +0.0150 m at t 3.09 with LDist +0.0164 (t 3.40) and lat +0.0001 (t 0.04) — the bf16-SFT
cost of the DTS-196 pretrain is longitudinal only. Every other arm stays at |t| <= 1.67 on
ADE / FDE / head / headF / LDist / LDistF, and the only readings above |t| = 2 anywhere else
are lateral: bf16/DTS-168 lat -0.0033 (t -2.11), bf16/CTS-196 latBias -0.0039 (t -2.03) and
DTS-168/DTS-168 latBias -0.0068 (t -2.60). Those are 3 % of the 0.21 m lateral error and
should not be read as a driving difference.

Paired CTS-against-DTS within the same stage pair, over the same scenes:

| pair | dADE | t | dFDE | t | AgentColl p | LaneCross p |
|---|---|---|---|---|---|---|
| row 2, 132 | -0.0048 | -1.32 | -0.0108 | -1.34 | 1.0000 | 0.4658 |
| row 2, 168 | -0.0001 | -0.03 | +0.0002 | +0.03 | 0.8358 | 0.3494 |
| row 2, 196 | +0.0049 | +1.37 | +0.0055 | +0.68 | 0.4926 | 0.4870 |
| row 3, 132 | +0.2697 | +34.09 | +0.6803 | +35.07 | 0.0000 | 0.0000 |
| row 3, 168 | +0.1064 | +16.94 | +0.2549 | +17.22 | 0.0000 | 0.0000 |
| row 3, 196 | +0.1494 | +22.89 | +0.3588 | +23.18 | 0.0000 | 0.0000 |

The same pairs on the remaining columns (delta, t):

| pair | dhead | dheadF | dLDist | dLDistF | dlat | dlatF | dlatBias |
|---|---|---|---|---|---|---|---|
| row 2, 132 | -0.0058 (-0.51) | -0.0037 (-0.15) | -0.0054 (-1.49) | -0.0099 (-1.26) | -0.0018 (-1.16) | -0.0063 (-1.46) | +0.0004 (+0.20) |
| row 2, 168 | +0.0019 (+0.18) | +0.0017 (+0.07) | -0.0016 (-0.43) | -0.0030 (-0.38) | +0.0026 (+1.71) | +0.0076 (+1.85) | -0.0005 (-0.25) |
| row 2, 196 | +0.0018 (+0.15) | +0.0005 (+0.02) | +0.0057 (+1.58) | +0.0064 (+0.82) | -0.0007 (-0.46) | -0.0002 (-0.06) | -0.0004 (-0.19) |
| row 3, 132 | +0.7017 (+27.94) | +1.5605 (+27.42) | +0.2385 (+30.10) | +0.5842 (+30.49) | +0.0622 (+22.03) | +0.1823 (+22.60) | +0.0049 (+1.32) |
| row 3, 168 | +0.2580 (+13.34) | +0.5028 (+11.78) | +0.0931 (+14.81) | +0.2194 (+15.13) | +0.0249 (+10.72) | +0.0645 (+9.76) | +0.0009 (+0.28) |
| row 3, 196 | +0.3544 (+18.10) | +0.7600 (+17.34) | +0.1361 (+20.85) | +0.3193 (+20.94) | +0.0311 (+12.88) | +0.0872 (+12.91) | +0.0093 (+2.96) |

### Seven-column additions: how they were checked

Three definitional defects were found and fixed on 2026-11-18 while extending the report from
seven columns to twelve; none of them changed a number that was already published.

1. **`latBias` sign was documented backwards.** `traj_metrics.py` said `+ = pred right of GT`.
   The dumps are in the ego frame of the current step, x forward and y left
   (`convert_absolute_to_relative_se2_array` scores `(p - origin) @ R(-heading).T`, which sends
   a point left of the ego to `(0, +1)`), so `+` is **left**. Checked on the dumps, not only in
   the algebra: GT scenes whose mean predicted-window `dtheta` exceeds 0.05 rad end at a mean
   final lateral offset of **+2.95 m** over 50 such scenes, i.e. a left turn is +y.
2. **`AgentColl` was documented as `100 x (1 - mean no_at_fault_collisions)`.** The code counts
   `NC == 0.0` only. `pdm_scorer.py:340` gives 0.0 for agent types, 0.5 for static/unknown
   objects and 1.0 for clear, so the mean-based rate would add the 0.5 rows (73 on the
   aligned base arm) and is a different metric. The docstring now says which one is computed.
3. **`load_csv` could parse a different file than the one it selected.** It kept the last
   file's rows but the last *scoring* file's path. With one score csv per eval dir nothing
   changed, but a dir holding a second csv would have mixed the two.
4. The `--selftest` in `traj_metrics.py` gains a lateral case (0.4 m of left offset per step
   gives `lat` 1.8 m mean / 3.2 m final and `latBias` +1.8 m), and an independent numpy
   rotation walk over 300 scenes reproduces `ADE` and `lat` to 7.8e-15.

Two caveats on this table. **`HBColl` carries no information on this data** — every arm reads
0.000 % except CTS-196 at a single scene, because a predicted decel of 3 m/s^2 appears in at
most one scene per arm; that is a property of the threshold, and the column should not be
quoted as a result. And `cts132_recheck` has PDMS only: its trajectory metrics were not
recomputed, so the ~0.2 pd eval-reproducibility seen on PDMS is assumed, not measured, to
apply here too — immaterial against a 0.27 m ADE gap.

### The 132 arm is reproducible, and it inverts the ordering

Two facts, both measured:

1. **It is not an eval artefact.** The original eval of `cts132_lr5e5` ran concurrently with
   the 168 and 196 evals, three nodes writing into one shared `navsim_workspace/exp/`, so it
   was re-scored alone on an idle a51 with nothing else on the box
   (`eval_cts132_recheck_full`, 36,438 json): **0.76538 vs 0.76743**, a gap of 0.002 that
   matches the run-to-run noise of one eval. The damage is in the checkpoint.
2. **It is not the quantisation count.** 132 quantises the *fewest* layers (365 left in
   bf16) and scores the *worst*; 168 quantises 36 more and scores 5.8 pd higher. The old
   ordering 196 < 168 < 132 in the DTS diagonal does not carry over to CTS pretraining,
   where the order is 132 << 196 < 168.

What is **not** established: the mechanism. The candidate — and this is my hypothesis, not a
measurement — is that keeping layers 0-2/25-27 in bf16 changes the activation range reaching
the quantised layers, and the *shape* of that range is what per-tensor scaling is sensitive
to. Note the direction of the two failure modes, since an earlier draft of this paragraph had
it backwards:

- CTS has no interval by construction: `npu_dynamic_quant` derives the scale from the current
  tensor's amax on every call (`models/quant/hif8_cts.py`), so the largest element lands at
  `dst_type_max` and **nothing clips**. The cost of a wide or outlier-heavy range is
  *resolution*: one scale per tensor means a large element coarsens everything below it.
- DTS is the opposite: the scale is frozen from a sampled step and reused for up to
  `interval - 1` more (`models/quant/hif8_dts.py`, `history_len=1`), so values that grow
  above the sampled amax **do** clip before the next refresh.

So "CTS clips harder" is not available as an explanation — if anything the ordering is
reversed, and the CTS-132 collapse (the arm with the *most* layers left in bf16) would have to
come from precision loss rather than saturation. Nothing in this run tests either story; it
needs the per-layer amax distributions of the 132 and 168 pretrains compared side by side.

### What these numbers do and do not support

- **Supported:** in the 5e-5 family, CTS in the SFT stage alone is equivalent to bf16 and to
  DTS: all three CTS arms sit within 0.18 pd of the bf16 base and within 0.21 pd of their DTS
  counterparts, largest |t| = 1.02 across those six comparisons. This reproduces the 11-16
  2x2's SFT-axis conclusion in a second quantisation scheme, which is the strongest form of
  that claim so far.
- **Supported:** CTS in the pretrain stage is not equivalent to DTS. The gap is 2.6-8.6 pd
  at |t| up to 26.8, i.e. far outside noise, and it holds at all three ignore sets.
- **Not supported:** any claim that CTS is generally worse than DTS. Rows 2 and 3 differ only
  in which stage is quantised, so the two-stage result says the *pretrain* stage is where CTS
  breaks, not the scheme.
- **Not supported:** attributing the row-3 ordering (168 > 196 > 132) to quantisation
  breadth, for the inversion reason above.
- One seed per cell. The row-2 differences are all within one noise floor and should not be
  ranked; the row-3 differences do not need a second seed to be directional, but a second
  seed would be needed before quoting the exact 8.61 pd.

### How this was run and how to reproduce

- Wrappers on `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/`: `r2_{132,168,196}.sh`,
  `r3_{132,168,196}.sh`, `a53_queue.sh` (runs the three row-2 arms back to back). They only
  export `PT_DIR`/`TAG`/`HIF8_*`/`IG_*`/`SAVE_STEPS_PT` and exec `sft_from_pt.sh` or
  `chain_ptlr.sh`; both launchers are unmodified from the DTS runs.
- Container: `docker exec zyc-w0 ...`. The `drivevla-w0` image was tried first and cannot run
  these launchers at all — `venv-hif8` is built on `/usr/local/python3.12.15`, which that
  image does not ship, so `bin/torchrun` fails with exit 127 after 0 steps. `zyc-w0` has it.
- Scores: `submetrics.py <base> <arm>...` and `metrics7.py <exp dir> <json_output dir>...`
  under `/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python3`, reading
  `navsim_workspace/exp/eval_<tag>_full/*/*.csv` and `navsim_workspace/eval_<tag>_full/json_output`.
  `metrics7.py` + `traj_metrics.py` were copied byte-identically (md5 checked) from the repo's
  `tools/zyc/` into `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/m7tools/`, because no checkout on the
  box carries them. The 14-arm run is saved there as `m7_all12.txt` (6 s per arm, all 14 arms
  in one invocation with the base first, so every pair is printed against it); the earlier
  seven-column run is `m7_all.txt`. The twelve-column version reproduces the seven numbers of
  the earlier run exactly, which is the regression check for the extension.
- Eval tags: `bfpt5e5_cts{132,168,196}sft`, `cts{132,168,196}_lr5e5`, plus the recheck
  `cts132_recheck` (kept separate, not folded into any table).

### Open (this section)

1. **Compare per-layer amax between the CTS-132 and CTS-168 pretrains.** This is the direct
   test of the hypothesis above and needs no retraining: instrument `npu_dynamic_quant`'s
   returned scale (or log `amax` per quantised module) on a few hundred steps of each arm and
   ask whether 132's ranges are wider or more outlier-dominated where CTS then loses
   resolution. Minutes to hours, not a full arm.
   Deliberately **not** on this list: "re-run with a different `interval`". `interval` is a DTS
   parameter only (`convert_to_hif8_cts` does not accept one, and `hif8_dts.py` rejects
   `interval < 2` because 1 *is* CTS) — re-deriving the scale from the current tensor is the
   definition of CTS, not a setting to sweep.
2. Only if (1) shows a range problem: a CTS variant with a delayed or running-MAX scale, or a
   wider `dst_type_max` (both currently hard-coded in `hif8_cts.py` as 15 forward / 224 grads).
   That is a new recipe, and a full 132 arm (pretrain + SFT + eval, ~8 h) to score it.
3. **`pt_cts132 -> bf16 SFT`**, the analogue of `runs/sft_dtspt5e5_bfsft`, to separate "CTS
   pretrain damages the weights" from "CTS pretrain + CTS SFT interact". ~2 h if the
   checkpoint is reused; nothing is queued.
4. The quantisation self-check in `utils/train_qwen_vla.py:368` has an expected-count map of
   `{196, 168, 168, 140}` and therefore prints **nothing** for the 132 combination. The count
   still matched (132 = 168 - 6x6), but a silent skip in the one assertion that guards
   quantisation scope is a trap worth closing by adding the 132 entry.
5. Second seed on row 3 CTS-168 before the 2.57 pd figure is quoted as a fixed number.


## 2026-11-19 — DTS row 2 and row 3 re-run at pretrain lr 5e-5, refresh-clock verification, one unstable cell

Why this exists: three reasons, in the order they forced the work.

1. The DTS cells of the 5e-5 family had exactly one measurement each. Every CTS claim in the
   11-18 section is read *against* those DTS cells, so the comparison column had no error bar.
   Re-running the six arms turns the DTS column into a measured quantity instead of a number
   carried over from 11-16.
2. `models/quant/hif8_dts.py` carried a guard that aborted the run when the clock source was
   not one of the two expected values. The first launch of all six arms at 23:30 died in
   ~60 s on it. The guard had to be resolved and, with it, the refresh cadence proven rather
   than assumed.
3. Result on (1): the re-run does **not** reproduce on two of the three DTS-pretrain arms, so
   the DTS column of the 11-18 contrast needs correcting rather than this section just being
   appended to it.

Everything outside quantisation is byte-identical to the 11-16 arms: pretrain `pt_ptlr5e5`
(lr 5e-5, 8000 steps, gbs 96 = mb 12 x 8) reused for row 2 and re-trained for row 3, SFT lr 1e-5
(bs 6 x 8, 4000 steps, `zero2.json`, seed 42, GC off, `HIF8_INTERVAL=10`), navtest full split
(12,146 scenes / 36,438 json), container `zyc-w0`, interpreter `/mnt/sfs_turbo/lmc/venv-hif8`.
New tags only, so nothing overwrites a baseline: `sft_r2_bfpt5e5_dts{132,168,196}sft`,
`pt_r2_dts{X}_lr5e5`, `sft_r2_dts{X}_lr5e5`, `eval_<tag>_full`.

### Setup

Each node ran one row-2 arm and then the matching row-3 arm, so every pretrain-only contrast
below is within a single node on a single schedule.

| node | row 2 (bf16 pretrain -> DTS SFT) | row 3 (DTS pretrain -> DTS SFT) | log |
|---|---|---|---|
| a52 | `sft_r2_bfpt5e5_dts168sft` | `pt_r2_dts168_lr5e5` -> `sft_r2_dts168_lr5e5` | `r2_a52.log` |
| a53 | `sft_r2_bfpt5e5_dts132sft` | `pt_r2_dts132_lr5e5` -> `sft_r2_dts132_lr5e5` | `r2_a53.log` |
| a54 | `sft_r2_bfpt5e5_dts196sft` | `pt_r2_dts196_lr5e5` -> `sft_r2_dts196_lr5e5` | `r2_a54.log` |

Ignore sets are the 11-17 ones (132 = 132 quantised / 365 in bf16; 168 = 168 / 329; 196 = 196 /
301). Six arms, six `阶段三 退出码 0`, three `ALLDONE`, 36,438 json each.

Measured stage costs: pretrain 20,984.5 / 21,185.8 / 21,251.6 s (132 / 168 / 196) against
20,708 s for `chain_dts196_lr5e5` on 11-16, so the re-run costs the same; SFT ~1.65 h;
full eval 22-25 min on 8 cards.

### The two-line deletion, and why it cannot move a number

`hif8_dts.py:393-394` held a stray `else:` plus `raise ValueError("HIF8_DTS_CLOCK_SRC 只支持
trainer / optimizer ...")` attached to the *static-recipe* branch chain. With `CLOCK_SRC=trainer`
the preceding `elif` is taken, so the removed `else` was unreachable on the path every arm uses:
the deletion changes control flow only for the configurations that were meant to abort.

Evidence, both on a51 and both on the patched file:

- Guard restored (control): aborts within 60 s at `hif8_dts.py:394` with **0** refreshes,
  `VERDICT=FAIL` (`guard_restore_test.log`).
- Patched code, 80-step smoke, `HIF8_INTERVAL=10`: refreshes #1-#8 at steps 10, 20, ... 80,
  consecutive deltas all exactly 10, every refresh logging `588/588 points had pending`,
  `VERDICT=PASS` (`smoke_clock196_a51_guard.log`, driver `smoke_clock_now.sh`).

So the deletion decides proceed-vs-abort and nothing else. File state afterwards: md5
`27d30a7cf15108101b34403f9b8e5b76`, 23,946 B, zero `raise` occurrences in the region, and the
diff against the pre-edit backup `hif8_dts.py.bak-r2before` (md5 `b838188b...`) is exactly the
two deleted lines. **The correct repair, not applied:** move the `else` back onto the
clock-source chain so an unknown `HIF8_DTS_CLOCK_SRC` still fails loudly. As it stands an
unknown source now falls through silently.

Note for reading pretrain logs: they do not print the `hif8=...` header; the scheme is in
`[HiF8] 模式 dts（间隔 N 步）`, and per-refresh `refresh #N at step=M` lines appear only under
`HIF8_DTS_CLOCK=1`.

### Results

Row 2 — bf16 pretrain -> DTS SFT, re-run against the same cells on 11-16:

| ignore set | r2 | 11-16 | dPDMS | vs bf16/bf16 base (0.85473) |
|---|---|---|---|---|
| 132 | 0.85720 | 0.85286 | **+0.43 pd** | +0.25 pd |
| 168 | 0.85641 | 0.85552 | +0.09 pd | +0.17 pd |
| 196 | 0.85438 | 0.85499 | -0.06 pd | -0.04 pd |

All three reproduce inside +-0.5 pd, which fixes the eval pipeline's practical noise floor at
about 0.4 pd and confirms that bf16-pretrain -> DTS-SFT is insensitive to which layers carry
the quantisation.

Row 3 — DTS pretrain -> DTS SFT, same comparison:

| ignore set | r2 | 11-16 | dPDMS | vs bf16/bf16 base (0.85473) |
|---|---|---|---|---|
| 132 | 0.83055 | 0.85351 | **-2.30 pd** | -2.42 pd |
| 168 | 0.85194 | 0.85198 | -0.00 pd | -0.28 pd |
| 196 | 0.81324 | 0.85044 | **-3.72 pd** | -4.15 pd |

### The pretrain-only contrast does not reproduce on 132 and 196

Same-node, same-run pairing (row 3 minus row 2, both from this re-run) is the cleanest form —
same tree, same clock, same node, only the pretrain differs:

| ignore set | DTS pretrain cost, r2 | DTS cost recorded 11-16 |
|---|---|---|
| 132 | **-2.66 pd** (0.85720 -> 0.83055) | +0.07 pd |
| 168 | -0.45 pd (0.85641 -> 0.85194) | -0.35 pd |
| 196 | **-4.11 pd** (0.85438 -> 0.81324) | -0.45 pd |

168 reproduces to 0.10 pd. 132 and 196 move by 2.7 and 3.7 pd against a row-2 noise floor of
0.4 pd. **This directly contradicts the sentence in the 11-18 section that quantising the
pretrain costs "0.0-0.5 pd with DTS"**, and that sentence is what made the CTS row-3 losses
look like 5-14x the DTS cost. With these numbers the honest form is: DTS pretrain costs
0.4-4.1 pd depending on ignore set, CTS pretrain costs 2.8-8.7 pd, so CTS is still worse at
every ignore set but the margin is no longer an order of magnitude.

What is *not* established: why 132 and 196 moved while 168 did not. The obvious candidates are
ruled out by the same data:

- Not the eval pipeline: row 2 reproduced on all three nodes, and 168's row-3 eval reproduced to
  0.003 pd.
- Not per-node instability: on a53 and a54 the row-2 arm of the *same* node reproduced (a53
  +0.43 pd, a54 -0.06 pd) while only the row-3 arm moved.
- Not the eval restarts recorded below: 168 restarted and held; 132 ran first-try and moved.
- Not the ignore-set breadth: 168 quantises the middle count and is the one that held, so
  "more or fewer bf16 layers" has no monotone story here either.

Candidates that remain, untested: (a) DTS-pretrain run-to-run variance is genuinely much
larger than the bf16-pretrain floor and one seed per cell cannot see it; (b) the refresh clock
in the pretrain stage behaves differently from the SFT stage in a way the 80-step smoke did not
exercise at 8000 steps. Only a second seed separates them.

### Components, `average` row of each arm's newest score csv, same 12,146 valid scenes

These are unpaired means read off the csv, **not** `submetrics.py` output, so there is no `t`
column here — the paired version of this table has not been run for the r2 arms. Column order
follows the 11-18 table.

| arm | NC | DAC | DDC | EP | TTC | comfort | PDMS |
|---|---|---|---|---|---|---|---|
| bf16/DTS-132 (11-16) | 0.98341 | 0.93611 | 0.97958 | 0.78935 | 0.94920 | 1.00000 | 0.85286 |
| bf16/DTS-132 (r2) | 0.98481 | 0.93907 | 0.97933 | 0.79321 | 0.95126 | 1.00000 | 0.85720 |
| bf16/DTS-168 (11-16) | 0.98403 | 0.93932 | 0.97851 | 0.79185 | 0.94830 | 0.99992 | 0.85552 |
| bf16/DTS-168 (r2) | 0.98460 | 0.93899 | 0.97909 | 0.79334 | 0.94928 | 1.00000 | 0.85641 |
| bf16/DTS-196 (11-16) | 0.98469 | 0.93776 | 0.97884 | 0.79168 | 0.94937 | 1.00000 | 0.85499 |
| bf16/DTS-196 (r2) | 0.98407 | 0.93759 | 0.97958 | 0.79096 | 0.94920 | 0.99984 | 0.85438 |
| DTS-132/DTS-132 (11-16) | 0.98399 | 0.93776 | 0.97991 | 0.79080 | 0.94558 | 1.00000 | 0.85351 |
| DTS-132/DTS-132 (r2) | 0.97616 | 0.92220 | 0.97509 | 0.77031 | 0.93290 | 1.00000 | 0.83055 |
| DTS-168/DTS-168 (11-16) | 0.98357 | 0.93644 | 0.97925 | 0.78993 | 0.94484 | 0.99984 | 0.85198 |
| DTS-168/DTS-168 (r2) | 0.98345 | 0.93603 | 0.97991 | 0.78898 | 0.94665 | 1.00000 | 0.85194 |
| DTS-196/DTS-196 (11-16) | 0.98337 | 0.93430 | 0.97921 | 0.78776 | 0.94607 | 1.00000 | 0.85044 |
| DTS-196/DTS-196 (r2) | 0.97180 | 0.90927 | 0.97085 | 0.75601 | 0.92327 | 1.00000 | 0.81324 |

Two things the shape of that table settles:

- **The degradation is not one metric breaking.** Every component of the two moved arms falls
  together — 132: NC -0.78 pp, DAC -1.56, EP -2.05, TTC -1.27, DDC -0.48; 196: NC -1.16,
  DAC -2.50, EP -3.18, TTC -2.23, DDC -0.84. This is a uniformly worse driver, not a collision
  or a drivable-area artefact, and it mirrors how the CTS row-3 arms fail.
- **Comfort is untouched everywhere.** Six arms, one scene's worth of deviation (0.99984) and
  otherwise exactly 1.0. It is a discrete jerk/latency test that the policy either clears or
  does not, and no amount of quantisation drift crossed it — the same conclusion the 11-13 and
  11-17 tables reached.
- EP loses the most in both moved arms, which is the expected signature: it is the continuous
  progress term, so it absorbs small trajectory deviations that the discrete metrics tolerate.

### Two anomalies recorded, not explained

1. **`chain_ptlr.sh: line 159: in_bf16: command not found`, once on a52 and once on a54, zero
   times on a53.** In both cases it was followed by a silent restart of `阶段三` ~50 s later, and
   the retried eval produced a valid score. Cause not established: `in_bf16` is not defined
   anywhere in `chain_ptlr.sh`, so the reference is either a leftover from an edited copy or an
   inherited shell function; the run does not fail because the command is only used for a
   branch label. Worth 10 minutes of looking, since a branch conditioned on an undefined
   variable always takes the same side and that side is currently "not bf16".
2. **Score-csv mtimes are exactly 8 h ahead of the timestamps in their own filenames**
   (e.g. `2026.11.19.01.31.07.csv` with mtime 09:31). Every r2 score dir shows the offset, so
   any wall-clock reasoning about these evals has to use the filename, not `ls -l`. The same
   skew makes `find -newermt` useless for deciding whether a dir was written recently: a
   09:30 cutoff matched every r2 csv. The real eval windows are 01:26-02:11 and the row-2 evals
   17:35-17:38 on 11-18.

### What these numbers do and do not support

- **Supported:** bf16-pretrain -> DTS-SFT at lr 5e-5 is reproducible to +-0.4 pd at all three
  ignore sets, on three different nodes, on a tree that has moved since 11-16. That is now the
  noise floor to quote against.
- **Supported:** the DTS refresh clock fires every `interval` steps as configured, verified at
  10 steps over 8 consecutive refreshes with all tracked points pending each time.
- **Supported:** the 11-16 DTS-pretrain costs (0.0-0.5 pd) are wrong by 2-7x for the 132 and
  196 ignore sets. The 11-18 CTS contrast needs its DTS column replaced with the r2 numbers.
- **Not supported:** any claim that DTS pretrain is safe. Two of three cells now show a real
  regression; 168 alone is clean, and 168 is a single seed.
- **Not supported:** any ranking between 132 and 196 as "how much the pretrain tolerates".
  Both moved, one run each, and the 11-16 ordering that made 196 look worst is not reproduced
  in magnitude even though the direction matches.
- **Still unknown:** whether the 11-16 numbers or the r2 numbers are the outlier for 132 and
  196. One seed on each side cannot decide it.
- The row-2 arms in this re-run are also subject to the 11-18 open item 4: the quantisation
  scope self-check in `utils/train_qwen_vla.py:368` has no 132 entry and prints nothing for that
  combination, so the 132 arms again ran that assertion silently.

### How this was run and how to reproduce

- Driver: `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/r2_sched.sh <host> <ignore set> ...`, whose 9th
  argument is the pretrain start time (`START`, default 2315) so a rerun can be shifted without
  editing a running script. It execs the unmodified `chain_ptlr.sh`; per-host status from
  `r2stat.sh`, per-host logs `r2_a52.log` / `r2_a53.log` / `r2_a54.log`.
- Clock evidence: `smoke_clock_now.sh` (takes label and batch size), run on a51 via
  `docker exec zyc-w0`, patched code then guard-restored control; logs
  `smoke_clock196_a51_guard.log`, `guard_restore_test.log`.
- Scores read from `navsim_workspace/exp/eval_<tag>_full/<timestamp>/<timestamp>.csv`, the
  `average` row. Newest timestamp per tag, which matters: a52 and a54 each hold an abandoned
  first attempt (`01.32.31` on a52, `01.43.23` on a54) next to the scored one.
- Cancelled, for the record: a54 had a clock smoke queued behind these arms. It was stopped
  (`tmux kill-session -t smoke-clock`, PIDs 1003952 / 1227532 / 1003953) before it launched
  anything, so no container side effects and no partial run to clean up.

### Open (this section)

1. **Second seed on DTS-132 and DTS-196 pretrain.** This is the only thing that separates
   "variance" from "these two cells really are worse", and it decides whether the 11-18 DTS
   column gets rewritten upward or downward. ~8.0-8.5 h per arm at the measured stage costs;
   a52 and a54 are free (8 idle NPUs each), a53 was fully occupied by a non-ours job at the time
   of writing.
2. **Run the paired `submetrics.py` for the six r2 arms** so this section's component table
   gains `t` values and becomes comparable, cell for cell, with the 11-18 table. Minutes.
3. **Close the `in_bf16` reference** in `chain_ptlr.sh:159` and the missing `132` entry in the
   `expected-count` map of `utils/train_qwen_vla.py:368` (same item as 11-18 open 4, now
   exercised twice).
4. **Restore the `else` onto the clock-source chain** in `hif8_dts.py` so an unknown
   `HIF8_DTS_CLOCK_SRC` fails loudly again. The backup `hif8_dts.py.bak-r2before` holds the
   pre-edit text; the current file is the two-line deletion only.
5. **Untriaged WIP.** ZYC's tree still carries ~123 uncommitted lines across `hif8_dts.py` and
   `train_qwen_vla.py` on top of these edits. Nothing in this run isolates their effect, so any
   cross-date comparison on that tree carries that caveat.

## 2026-11-24 — consolidated full-navtest result inventory

This is a scan of Yichen's workspace (`/mnt/sfs_turbo/zyc/drivevla-w0-zyc`) for the requested
single-stage and two-stage result sets. All PDMS values are the full 12,146-scene navtest result,
read from the `average` row of
`navsim_workspace/exp/eval_<tag>_full/<timestamp>/<timestamp>.csv`.

**There is no 192-layer arm anywhere in the workspace.** A source scan of run directories, train
logs, launch scripts, and eval tags found no `192` Linear count. The third arm in this workspace
is **196 layers**; the user confirmed 192 was a typo for 196.

### Single-stage: 4,000-step NAVSIM SFT only

No nuPlan pretraining is used in this block.

| Model / layers | HiF8 mode | Run | PDMS |
|---|---|---|---:|
| Qwen2.5-VL-7B | BF16 | `w7b_bf16` | 0.840658 |
| Qwen3-VL-2B | BF16 reference | `aligned_bf16` | 0.812033 |
| Qwen3-VL-2B / 132 | CTS | `aligned_hif8_132` | 0.809797 |
| Qwen3-VL-2B / 168 | CTS | `aligned_hif8_168` | 0.807480 |
| Qwen3-VL-2B / 196 | CTS | `aligned_hif8_196` | 0.810260 |
| Qwen3-VL-2B / 132 | DTS i10 | `aligned_hif8_132_dts` | 0.809851 |
| Qwen3-VL-2B / 168 | DTS i10 | **missing** | — |
| Qwen3-VL-2B / 196 | DTS i10 | `aligned_hif8_196_dts` | 0.819715 |

### Two-stage: 8,000-step nuPlan pretrain -> 4,000-step NAVSIM SFT

**BF16 pretrain -> BF16 SFT, different seeds**

| Seed | Runs | PDMS |
|---:|---|---:|
| 42 | `pt_ptlr5e5` -> `sft_ptlr5e5` | 0.854730 |
| 43 | `pt_bf16_s43` -> `sft_bf16_s43` | 0.805825 |
| 44 | `pt_bf16_s44` -> `sft_bf16_s44` | 0.810343 |

**BF16 pretrain -> DTS HiF8 SFT, interval 10**

| Layers | Original SFT run | PDMS | Rerun SFT run | PDMS |
|---:|---|---:|---|---:|
| 132 | `sft_bfpt5e5_dts132sft` | 0.852858 | `sft_r2_bfpt5e5_dts132sft` | 0.857197 |
| 168 | `sft_bfpt5e5_dts168sft` | 0.855518 | `sft_r2_bfpt5e5_dts168sft` | 0.856413 |
| 196 | `sft_bfpt5e5_dts196sft` | 0.854985 | `sft_r2_bfpt5e5_dts196sft` | 0.854380 |

Both columns start from the same BF16 pretrain checkpoint, `pt_ptlr5e5`.

**DTS HiF8 pretrain -> DTS HiF8 SFT, interval 10**

| Layers | Original SFT run | PDMS | Rerun score(s) |
|---:|---|---:|---|
| 132 | `sft_dts132_lr5e5` | 0.853508 | 0.830552 |
| 168 | `sft_dts168_lr5e5` | 0.851976 | 0.853160, 0.851942 |
| 196 | `sft_dts196_lr5e5` | 0.850444 | 0.814681, 0.813238 |

The i10 DTS-pretrain rerun is unstable for 132, and especially for 196. The 168 rerun is stable
within roughly 0.1 pd.

**DTS HiF8 pretrain -> DTS HiF8 SFT, interval 200**

| Layers | Seed | SFT run | PDMS |
|---:|---:|---|---:|
| 132 | 42 | `sft_dts132_i200_s42` | 0.755560 |
| 168 | 42 | `sft_dts168_i200_s42` | 0.848391 mean; four evals 0.845462-0.850181 |
| 196 | 42 | `sft_dts196_i200` | 0.855851 |
| 196 | 43 | `sft_dts196_i200_s43` | 0.852255 |

Additional seed coverage found:

- `sft_dts168_i10_s43` exists and scored 0.722588.
- No seed-43 i10 result exists for 132 or 196.
- No seed-43 i200 result exists for 132 or 168.

### The 132-layer i200 anomaly

`sft_dts132_i200_s42` is the only result in this block that is far outside both the interval-10
controls and the 168/196 i200 arms. Its 0.755560 PDMS is not an incomplete-eval artefact: the
directory has the expected 36,438 total JSONs, including 12,146 model-output JSONs, and the CSV
contains 12,146 valid scenes.

The loss gap also exists before the final evaluation:

- Pretrain action loss, 132 vs 168: signed mean +5.96%, +0.0711 nats, `exp(delta)-1` +7.37%.
- SFT action loss, 132 vs 168: signed mean +11.09%, +0.1470 nats, `exp(delta)-1` +15.84%.
- Relative to the 132 i10 control, i200 adds +3.48% in pretrain and +8.07% in SFT.
- Relative to the 168 i10 control, i200 adds only +1.01% in pretrain and +1.32% in SFT.

The final PDMS drop is mostly additional catastrophic failures rather than uniformly lower
quality:

- 132 has 2,068 zero-score scenes (17.02%) versus 956 for 168 (7.87%).
- The excess zero-score contribution is -0.08448 of the total -0.09462 PDMS difference (~89%).
- Mean score among nonzero scenes is 0.9106 for 132 versus 0.9228 for 168.
- Score zero coincides exactly with `ego_progress == 0`.

The observed mechanism is therefore a 132/i200 interaction that causes much more frequent
closed-loop progress failures. Layer-scope versus refresh-interval causality is not yet isolated;
the two-stage arms also ran on different hosts and the 168 arm used
`dataloader_pin_memory=False` as an operational workaround. That setting changes host-memory
transfer behavior, not tensor values or optimizer math.

### Missing result cells

1. No 192-layer runs; use 196.
2. No single-stage Qwen3-VL-2B DTS-i10 168 arm.
3. No i10 seed-43 runs for 132 or 196.
4. No i200 seed-43 runs for 132 or 168.
