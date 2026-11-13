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
  pattern, higher teacher-forced loss (1.8844 vs 1.8703) yet fewer hard-flag
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

