# DriveVLA-W0 事实清单（查这个，不要凭记忆）

> 2026-11-11。每条都标了证据位置。**改动前先核这里。**

---

## 0. 一句话

**我们只做 Qwen3-VL-2B + ViT + 扩散世界模型（ROSS）这一条线，窗口是 2VA，
预训练和 SFT 都是 2VA。Emu3 / VQ / 自回归 / 6VA 全部不跑。**

---

## 1. 两条分支（别混）

| | **Emu3 / VQ + 自回归** | **Qwen / ViT + 扩散** |
|---|---|---|
| **我们跑吗** | ❌ **不跑** | ✅ **只跑这条** |
| 骨干 | Emu3，9.3 B，视觉是离散 VQ 码字 | **Qwen3-VL-2B**，视觉是连续特征 |
| 世界模型 | 同一个 `lm_head` 预测下一帧码字，交叉熵 | 独立 SD-1.5 UNet 预测 latent 噪声，MSE |
| 额外参数 | 0 | 约 860M（UNet）+ 83.7M（VAE，冻结） |
| 训练入口 | `train/train_moe.py` | **`utils/train_qwen_vla.py`** |
| 预训练窗口 | **6VA** | **2VA** |
| SFT 窗口 | 2VA | **2VA** |
| 谁在做 | zjj（MFU 性能基准，bf16，不动 FP8） | lmc + 我们 |

**6VA 只存在于 Emu3 分支。** 三条证据：

1. ROSS 头硬限两帧，且是**原作者**写的：
   `reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py:335-337`
   三条 `assert image_hidden.shape[1] == 2`，注释 "Currently only support 2 images for
   each sample"，引入 commit `3c10d99 init`。lmc 的 Qwen3-VL 版（`ba4ce89`）照抄了。
2. Qwen 路径只有两个 dataset，都是 2VA：`train_qwen_vla.py:381-384`
   → `navsim2va_ross` / `nuplan2va_ross`。**没有 6VA 的 Qwen dataset。**
3. 作者自己的 Qwen checkpoint 名两头都是 2VA：
   `qwen25vl_fast_147456_4_000_2va_nuplan_pretrained_8k_2va`（见 `inference/qwen/inference_2va.sh`）

---

## 2. 官方超参（六个训练脚本审计，2026-11-11）

| 脚本 | 分支 | 阶段 | lr | warmup | 步数 | bs | **全局 batch** | `max_pos_emb` |
|---|---|---|---|---|---|---|---|---|
| `train_nuplan_6va_multi_v0.2_master.sh` | Emu3 | 预训练 | 2e-4 | 400 | 8000 | 4 | **96**（3 机×8 卡） | 4000 |
| `train_nuplan_6va_multi_v0.2_worker.sh` | Emu3 | 预训练(多机) | 2e-4 | 400 | 8000 | 4 | 96 | 4000 |
| **`train_base_ar_withou_moe.sh`** | Emu3 | **SFT 无 MoE** | **8e-5** | **50** | **4000** | **6** | **48** | **1400** |
| `train_navsim_ar.sh` | Emu3 | SFT + AR expert | 5e-5 | 50 | 10000 | 6 | 48 | 1500 |
| `train_navsim_flow_matching.sh` | Emu3 | SFT + 流匹配 | 5e-5 | 50 | 10000 | 12 | 96 | 1400 |
| `train_navsim_query_based.sh` | Emu3 | SFT + query | 5e-5 | 50 | 4000 | 12 | 96 | — |

**六个脚本全部相同**：`null_prompt_prob 0.15` · `weight_decay 0.1` ·
`min_learning_rate 1e-6` · `max_grad_norm 5.0` · `gradient_accumulation_steps 1` ·
`seed 42` · 世界模型开（`apply_loss_on_only_vision True` /
`apply_loss_on_only_action False`）。

⚠️ **Emu3 预训练是 3 机**：`master.sh` / `worker.sh` 里 `WORLD_SIZE=3`、`--nnodes=${WORLD_SIZE}`，
所以全局 batch = 4 × 8 × 3 = **96**，比 SFT 的 48 大一倍（预训练大 batch、微调小 batch）。
worker 脚本只写了 `RANK=1`，第三台（`RANK=2`）作者没给脚本，要手改 RANK。

⚠️ **仓库里六个脚本全是 Emu3 的，没有一个 Qwen 的。** Qwen 路径的唯一启动方式是
手写启动器直接调 `utils/train_qwen_vla.py`（lmc 的 `run_smoke.sh`、我们的 `tools/run_*.sh`）。

---

## 3. 我们的配置

### SFT（NAVSIM 2VA）—— 有依据，可直接跑

超参全部取自 `train_base_ar_withou_moe.sh`（Emu3 的 SFT 无 MoE 那一档，
和我们"不挂 action expert"对应）：

| 参数 | 值 |
|---|---|
| 数据 | `navsim2va_ross`，官方 trainval pickle（103,288 场景） |
| lr / warmup / 步数 | **8e-5 / 50 / 4000** |
| bs / 卡 / ga / **全局 batch** | **6 / 8 / 1 / 48** |
| `model_max_length` | **1400** |
| `max_pixels` / `min_pixels` | 147456 / 12544（`inference_2va.sh`） |
| 世界模型 | **开** |
| 其余 | 同第 2 节"六个脚本全部相同"那一行 |

脚本：`tools/run_sft_navsim.sh`

### 预训练（nuPlan 2VA）—— 与 Emu3 释放的预训练配置对齐

决定：**超参照搬 `train_nuplan_6va_multi_v0.2_master.sh`，只有窗口不照搬。**

| 参数 | 值 | 依据 |
|---|---|---|
| 数据 | `nuplan2va_ross` | Qwen 路径的 nuPlan dataset |
| **lr** | **2e-4** | Emu3 预训练脚本 |
| **warmup** | **400** | 同上 |
| **步数** | **8000** | 同上（也对上 ckpt 名 `nuplan_pretrained_8k`） |
| **全局 batch** | **96** | 同上（那边 3 机×8 卡×mb4=96） |
| **微批 / ga** | **mb12 × ga1**（8 卡），跑不下退 **mb6 × ga2** | 单机 8 卡凑出同样的 96 |
| `model_max_length` | **1400** | ⚠️ 2VA，和 SFT 同构。**不是 4000** —— 4000 是 Emu3 6VA 装六对 VA 用的 |
| 其余 | `null_prompt_prob 0.15` / wd 0.1 / min_lr 1e-6 / clip 5.0 / β(0.9,0.95) / eps 1e-6 / seed 42 / 世界模型开 | 六脚本一致 |

**"借超参、不借窗口"**：窗口必须是 2VA（ROSS 头硬限两帧），其余数值全部对齐。

脚本：`tools/run_stage1_pretrain.sh`（⚠️ `dataset_type` 还是 `navsim2va_ross`，
等 nuPlan pickle 就绪后改）

---

## 4. 我犯过的错（别再犯）

| 错误 | 正确 | 怎么发现的 |
|---|---|---|
| 以为预训练要 6VA | **2VA**。6VA 是 Emu3 分支 | 查 ROSS 头的 assert + dataset 清单 |
| 把 6VA 的 `max_pos_emb=4000` 搬到我们脚本 | **1400** | 2VA 序列短，4000 是装六对 VA 用的 |
| 把 SFT 的 warmup 从 50 改成 400 | **50** | 四个 SFT 脚本一致；"冷启动要更长 warmup"是我的推断、无依据 |
| 推荐 SFT 关掉世界模型 | **开** | 五个脚本（含预训练）全部 `apply_loss_on_only_vision True` |
| 用 lmc smoke 的 lr 5e-5 当预训练 lr | 预训练 2e-4 / SFT 8e-5 | 5e-5 是带 action expert 那三个脚本的值 |
| 说 "OpenScene = nuPlan 相机，lmc 在下重复数据" | **不是**。日志名集合同（1457），但同一日志 nuPlan 1990 帧（10 Hz）、OpenScene 398 帧（2 Hz） | 逐字节对比：456 M vs 92 M |
| 算 Emu3 预训练全局 batch 时漏了机器数 | **3 机 × 8 卡 × bs4 = 96**，不是 32。`master.sh` / `worker.sh` 里 `WORLD_SIZE=3` | 用户问"确定不是三机吗" |
| 说下载慢是"并发太多/被限流" | **S3 时段性波动**。两边方式完全不同却同步涨跌 | 两台机器同时刻实测 20.05 vs 21.44 MB/s |

**共同毛病：拿单点测量或自己的推断当结论。** 要么给证据位置，要么标明"这是判断"。

---

## 5. 动作头：不挂 action expert

`utils/train_qwen_vla.py` **零处**引用 `models/policy_head`。对应 Emu3 侧的
`train_base_ar_withou_moe.sh` 那一档。

论文对标值：**无 action expert 85.6 → 加 expert 88.4 → 精调 90.2**。
我们这条线（Qwen 2VA/2VA、无 expert）对标 Table 6 的 **84.2**，不是 90.2。

---

## 6. 已验证的事实（实测，非推断）

| | 结果 |
|---|---|
| human agent PDMS | **94.55**（论文 94.8），12,146 场景 |
| 20 步 ckpt 走完推理→PDMS | 21 场景 → 21 JSON → **41.13** |
| 官方 trainval pickle 可用于 Qwen 路径 | 103,288 场景，1192 log 全在，抽查 7199 张图 **100% 命中** |
| 世界模型开着训、推理不用它 | 已走通（20 步 ckpt 就是 `world_model: True` 训的） |
| `--enable_world_model false` 开关 | 参数 3072.6M → 2127.5M，无 ross_loss，存单片权重 |
| bs4 八卡 HBM | 23 / 98 GB |
| 速度 | bs4 约 1.13 it/s（8 卡，a52/a53） |
