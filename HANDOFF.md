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

Download status (2026-11-09, sessions on container `drivevla-w0-hif8-lmc`):
- `lmc-dl2` runs `navsim_openscene/parallel_dl.sh` (6 workers): trainval camera
  splits 2..199 → `sensor_blobs/trainval`, then test metadata → `navsim_logs/test`,
  then test camera 0..31 → `sensor_blobs/test`, then `normalize_test` and 3 retry
  passes over its own failed list.
- `lmc-dl-retry` runs `navsim_openscene/retry_failed.sh` (4 workers) over splits
  that failed in parallel_dl's first pass: per-file `wget -o _wget_logs/<name>.log`,
  `--tries=3`, random backoff; success writes `_done/<name>.done` and removes the
  `.fail` entry so parallel_dl's retry passes skip it.
- Failure burst 13:06:40–13:07:14 UTC: two of six workers failed every fetch in
  ~1 s and burned splits 31→77 into `_failed/`; both recovered at split 78.
  Re-fetching the same splits individually succeeds (HTTP 206 from the xet CDN), so
  this is transient per-connection throttling, not a bad object. The instrumented
  retry pool has not reproduced it.
- Throughput is an aggregate **~40 MB/s per-IP ceiling** (6 connections → 47 MB/s,
  8 → 40, 10 → 39), so worker counts only decide which pool finishes last; 6 main
  + 4 retry puts both at ~8 h. Remaining at 13:30 UTC ≈ 1154 GB.
- Splits 0 and 1 came from the earlier sequential run; `_done` markers exist only
  for splits finished by these two scripts.
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

Still missing for a navtest run: the test pickle from
`tools/pickle_gen/pickle_generation_navsim_pre_1s.py` over `navsim_logs/test`, and
the `{tokens: [...]}` yaml naming output files (derive from the navtest scene
filter). Image sizing is inherited from the checkpoint's `preprocessor_config.json`
(`max_pixels` 150528, `min_pixels` 12544), so no CLI flags are needed.

## Experiment log

| # | Date | What | Config | Log | Result |
|---|------|------|--------|-----|--------|
| 1 | 2026-11-09 | Qwen3-VL-2B + ROSS startup smoke, 20 steps | NAVSIM subset (467 samples), 8 NPUs x bs 2, `zero2.json`, `max_pixels` 150528, `cur_frame_idx` 3, `future_nums` 8, lr 5e-5, grad ckpt on | `/mnt/sfs_turbo/lmc/smoke_qwen3vl_2b/train.log` | **PASS**: 20/20 steps, `TRAIN_EXIT=0`, ~1.26 it/s (~21 s); loss 21.14 -> 6.41, grad_norm 556 -> 38, `action_loss` 7.8 -> 6.1, `ross_loss` 0.03-0.34 |

Blockers found and fixed on the way to the smoke (all landed on `qwen-hif8`):
- `b691a7e` vendored `transformers.data` collator shim (the 4.56 tree omits it)
- `e614e09` ROSS negative-prompt path resolved relative to the module
- `6598667` negative-prompt `.pt` loaded with `map_location="cpu"`, `weights_only=False`

## Next step

When the download completes, run both smokes from `venv-hif8` in the `lmc-w0-hif8`
pane: (1) the 20-step training smoke again, (2) an evaluation smoke — load the smoke
checkpoint with the ported runner, generate on a handful of navtest scenes, then run
`run_pdm_score.py` on those JSONs.

