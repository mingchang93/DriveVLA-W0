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

## 确定性排查的判据（2026-11-11 修正）

**不要用 `loss` 判断逐位一致。** HF Trainer 把它 round 到 4 位小数
（日志里就是 `{loss: 18.7769}`），分辨率只有 1e-4。
`grad_norm` 是全精度打印的（`513.1156005859375`），用它。

即便如此，grad_norm 是标量归约：513 在 float32 下的间隔约 3e-5，
所以"前两步 grad_norm 相同"只能把梯度差约束到 <6e-8 相对，不等于逐位相同。
实测分歧量级在第 3 步才升到 7.7e-6（可分辨），之后每步涨约两个数量级
（6.6e-3 → 5.2e-1），是早期大梯度 + clip 5.0 下的混沌放大。

比对脚本：`tools/zyc/cmp_det.py <tagA> <tagB>`

## 确定性：根因与修法（2026-11-11 定案）

**根因是播种时机，不是任何算子、也不是 NPU。**

`utils/train_qwen_vla.py` 里模型在第 314 行构造，`set_reproducibility()` 在第 404 行
才调用。denoiser 的 `mlp` / `mlp_a` / `pos_embed` / `factor` / `factor_a`
**不在 Qwen 的 checkpoint 里**（HF 自己会打印 "newly initialized" 列出它们），
是建模时随机初始化的；那一刻 torch 默认生成器还没被播种——它首次使用时从系统熵
取种，所以两把同配置跑的扩散头权重从一开始就不同。

**修法**（一行位置改动）：在 `train()` 解析完参数、建模之前先播种。
注意只能提前「播种」这一件：实测把整套 `set_reproducibility` 提前会让
`ross_loss` 变成 `nan`，所以提前那次传 `deterministic=False`，
确定性算法 / HF32 开关仍留在建模之后那次。

```python
set_reproducibility(seed, deterministic=False)   # 建模前：只播种
...  from_pretrained(...)                        # 随机初始化 denoiser
set_reproducibility(seed, deterministic=True)    # 原有那次：确定性算法 + 关 HF32
```

**验证**（单卡 ZeRO-1 bs6 12 步，世界模型开，lr 8e-5）：
`grad_norm` / `ross_loss` / `action_loss` **12/12 全部逐位相同**。
健康性锚点：`action_loss` 第 1 步 `18.45903205871582`、
`ross_loss` `0.31791043281555176`。

**8 卡 ZeRO-1 bs6（真实训练配置）同样 12/12 逐位相同**，`grad_norm` 前四步锚点：
`467.6930847168 / 470.3354797363 / 475.2049865723 / 412.1593627930`。
清理掉全部排查开关后复跑，这 12 个值与清理前逐位一致——即清理没动任何数值。
复验脚本：`tools/zyc/verify_determinism_8card.sh`。

### 为什么伪装成「数值噪声」

初始化差异只有一条泄漏路径，且第 1 步还被堵着：
- `action_loss` 走 LM 通路，不经过 mlp → 第 1 步逐位相同
- `ross_loss` 走 UNet，而条件张量 `zin = factor × z`，`factor` 初值严格为 0
  → 也不依赖 mlp → 第 1 步逐位相同
- 只有 `z` 依赖 mlp，而 `z` 唯一的去处是
  `factor.grad = Σ(G ⊙ z)` —— 1700 万项的强抵消求和，
  把权重差放大成 **+21.375 vs −80.0**（符号都不同）
- 第 2 步 `factor` 一更新，世界模型通路打开，全盘发散

所以「前两步完全一致、第 3 步开始分歧」是这个结构的必然，不是巧合。
`warmup=0` 时 lr 第 1 步就非零，分歧如期提前到第 2 步——这条预言当时就验证过。

### 排查中被证伪的假设（都做过实测，别再试）

| 假设 | 结果 |
|---|---|
| 主干融合注意力（`ZYC_ATTN_IMPL=eager`） | 照样分歧 |
| UNet 自身注意力（朴素 `AttnProcessor`） | 照样分歧 |
| 梯度检查点（`GC=False`） | 照样分歧 |
| ZeRO-0 / 1 / 2 | 三档都一样 |
| 多卡通信 | 单卡就能复现 |
| dropout | 模型里本来就没有 |
| float64 | 整个 run 只告警 1 次，是一次性标量 cast |
| HF32（`conv.allow_hf32`，默认 True） | 照样分歧（但精度确有改善，已保留关闭） |
| 显存分配器 `expandable_segments` | 照样分歧 |
| denoiser 走 fp32 | 照样分歧 |
| jit 编译模式 | 照样分歧 |
| 「大轴 sum 归约不确定」 | 微基准证伪：1700 万元素强抵消求和，进程内 20 次 + 进程间两遍全一致 |

### 判据（重要）

**不要用 `loss` 判断逐位一致**：HF Trainer 把它 `round(,4)`，日志里就是
`{'loss': 18.7769}`，分辨率只有 1e-4。用 `grad_norm`（全精度打印），
以及我们自己 log 的 `action_loss` / `ross_loss`（没被 round）。

**先看损失是否健康再看一致性**：跑挂时 `loss` 恒为 0、`grad_norm` 恒为 √2，
比对脚本会报「12/12 逐位相同」——那是假的成功。

比对脚本：`tools/zyc/cmp_det.py <tagA> <tagB>`
梯度逐位定位仪器：`ZYC_DUMP_GRAD=1 ZYC_GRAD_OUT=<json>`（每参数 .grad 的 crc32）

## HiF8-CTS（2026-11-11 实现）

**CTS = Current Tensor Scaling**：每次前向/反向都用当前张量的 amax 现算 per-tensor
缩放因子（对应 DTS 用历史 amax）。实现在 `models/quant/hif8_cts.py`，
移植自 MindSpeed 的 `te/pytorch/fp8/recipes/current_scaling_recipe.py`，
剥掉 Megatron `get_args()` 与 TransformerEngine 依赖。

**不需要 FSDPTurbo。** 只依赖两个 torch_npu 原语，我们 venv 里都有
（torch_npu 2.11.0，`hifloat8` dtype id = 290）：
`npu_dynamic_quant` / `npu_quant_matmul`。

### 量化范围（客户规格，不是我们的取舍）

来源：`/mnt/sfs_turbo/zyc/hif8/hif8-cts-code-map.md` 第 4 节。

```yaml
quant_apply_modules:   ["model.language_model.layers.{*}"]
quant_ignored_modules: ["*lm_head", "*o_proj"]
```

| 投影 | 数量 | |
|---|---|---|
| q/k/v/gate/up/down_proj | **168** | 量化（28 层 × 6） |
| o_proj | 28 | 排除 |
| 视觉塔 | 104 | **不量化** |

外加 `_SMALL_LINEAR_DIM = 64` 的最小维门槛。
接入点会自检「数量 != 168 就打警告」。

⚠️ **我犯过的错**：第一版按「先只量化 QwenVL，UNet 不用」理解成连视觉塔一起量化，
换了 300 个，还漏了 `o_proj` 的排除。用户质疑「vit 应该量化吗」才去翻文档发现。
**改范围前先查这一节。**

### 关键数值

per-role 的 `dst_type_max`（照搬，不是调的）：

| role | dst_type_max |
|---|---|
| inputs | 15.0 |
| weight | 15.0 |
| grads | 224.0 |

kernel 只接受 `{0.0, 15.0, 56.0, 224.0, 32768.0}`。

per-tensor 的标量 scale **转置不变**，所以前向量化出的 codes 可直接 `.t()` 给反向两个
GEMM 用 —— 每层每步只量化 3 次。（MX 不行：块 scale 依赖轴向，要 dual-axis，每层 6 次。）

### 不要做权重量化缓存

对方踩过一个**静默正确性 bug**：缓存键里有 `storage.data_ptr()`，而 FSDP2 复用同一块
all-gather 输出缓冲，导致 28 层同形状投影键碰撞，**每层都在用第 1 层的权重 codes**。
不崩、不 NaN、不告警，只是 loss 从 1.66 变 17.76，作废了 3 次运行。

我们这版**每次前向重新量化权重、完全不缓存**，天然没有这个风险；
`gradient_accumulation_steps=1` 下缓存本来也没收益。

### 用法

```bash
HIF8=cts bash tools/zyc/run_sft_navsim.sh      # HiF8 臂
bash tools/zyc/run_sft_navsim.sh               # bf16 基线臂
```

对比脚本：`tools/zyc/cmp_arms.py <hif8_tag> <bf16_tag>`，
口径是「带符号的平均相对偏差」。**噪声底是 0**（确定性修好后同配置两把 bf16 逐位一致），
所以偏差全部来自量化，不需要再扣噪声。

### 待办

- `TASK_QUEUE_ENABLE`：对方环境用 2，我们确定性那套设的是 0，是吞吐冲突点，待量化。
- 20 步内 `ross_loss` 偏差恒为 0：denoiser 不量化，且零初始化门控 `factor` 还只有 ~3e-5，
  世界模型通路基本没打开。要看量化对它的影响得跑到 `factor` 长起来之后。

## 4000 步四臂对比（2026-11-12）

NAVSIM 2VA SFT，Qwen3-VL-2B + ROSS，8 卡 ZeRO-1，mb6×ga1=全局 48，
4000 步（1.86 epoch），lr 8e-5 / warmup 50 / cosine_with_min_lr，seed 42。
四条臂除量化外逐键相同。**噪声底 = 0**（同配置两把 bf16 已验证逐位一致），
所以偏差全部来自量化，不需要扣噪声。

一机一把并行：a51=hif8_196 / a52=bf16 / a53=hif8_168 / a54=hif8_edge3。

### action_loss 带符号平均相对偏差（n=4000）

| 配置 | 量化层数 | 平均偏差 | 标准差 |
|---|---|---|---|
| hif8_168（客户规格） | 168 | **+0.0149%** | 2.08% |
| hif8_196 | 196 | −0.1289% | 2.15% |
| hif8_edge3（前后三层留 bf16） | 132 | +0.1394% | 2.00% |

三者跨度 0.27pp，远小于单步标准差 2%，**从 loss 看没有实质差别**。

`ross_loss` 偏差全部在 ±0.003%：denoiser 不量化，且零初始化门控 `factor`
增长极慢，世界模型通路对主干量化基本不敏感。

### 吞吐：HiF8 没有收益

| 配置 | 4000 步 |
|---|---|
| bf16 | 7139.98 s |
| hif8_196 | 7157.49 s (+0.25%) |
| hif8_168 | 7176.16 s (+0.51%) |
| hif8_edge3 | 7176.80 s (+0.52%) |

三条臂都比 bf16 慢 0.25–0.52%，在噪声内（同配置噪声底约 1.7%）。
原因：只量化 168/497 个线性层、模型仅 2B、且确定性变量
（`TASK_QUEUE_ENABLE=0` 等）本身在抑制异步派发。

### ⚠️ LLaVA 那套敏感性先验在这里不成立

`/mnt/sfs_turbo/zyc/hif8/hif8-sensitivity-results.md`（LLaVA + MindSpeed）
的结论是「偏差主要由 down_proj 决定，少量化就低偏差」。据此 edge3（132 个，
排掉前后三层）应该偏差最小。**实测它最大（+0.139%），量化最多的 196 反而是
负偏差。** 那组数不同数据、不同框架、还少一个世界模型通路，**只能当先验，
不能照搬**。引用它之前先在本任务上复测。

### 判据

loss 偏差三者在噪声内，**真正的判据是 PDMS**（全量 navtest 12146 场景）。
mini（21 场景）只用于确认链路通，统计上无意义，不要用来比较。

## nuPlan 预训练数据：清单、缺口、泄漏（2026-11-12）

### 清单（S3 枚举得来，不是逐个探名字）

```
curl "https://motional-nuplan.s3.amazonaws.com/?list-type=2&prefix=public/nuplan-v1.1/sensor_blobs/train_set/&max-keys=1000"
```

`train_set/` 下**恰好**只有两类，`IsTruncated=false`（完整，无分页）：

| 类别 | 编号 | 数量 | 我们要吗 |
|---|---|---|---|
| `nuplan-v1.1_train_camera_N.zip` | 0–42 | **43** | 要 |
| `nuplan-v1.1_train_lidar_N.zip` | 0–42 | 43 | 不要（点云） |

`val_set/` 与 `test_set/` 下有独立的 `val_camera_*` / `test_camera_*`，**不要下**（见泄漏一节）。

⚠️ **我犯过的错**：最初逐个探到 28 发现都存在就停了，把「29 个包」当成全部，
据此算出「3.5 TB 总量」「29/29 全部到位」等结论，实际只有 2/3。
**查清单用枚举（ListObjects），不要用逐个探名字。**

### db 侧有缺口（step0 必需）

S3 上 12 个 sqlite 包，我们完成 9 个：

| 包 | 状态 |
|---|---|
| mini / boston / pittsburgh / singapore / vegas_1–4 / val | 已完成 |
| **train_vegas_5** | **部分（128 G，未下完）** |
| **train_vegas_6** | **部分（164 G，未下完）** |
| test | 未下（也不该下，是评测集） |

本地 `.db` 共 **14571** 个。现有 744 个相机 log 里只有 1 个缺 db，影响很小；
但 **camera 29–42 的 log 若落在 vegas_5/6 区域就会没有 db**——有图无位姿等于废，
下完后要逐个核对。step0 读 `.db` 取位姿，没有 db 的 log 无法进预训练数据。

### ⚠️ 评测集泄漏（必须在预训练前过滤）

**nuPlan 的 train 划分与 OpenScene 的 train/test 划分不是同一套。**

```
已解压的 744 个 train_camera log
  ∩ OpenScene test     = 10
  ∩ navtest 评测 log   = 10     ← 本该是 0
现有预训练 pickle：214 / 5385 片段（3.97%）来自这 10 个 log
navtest 的 136 个评测 log 中 7.4% 会在预训练时被看到
```

泄漏的 10 个 log：
```
2021.05.25.14.16.10_veh-35_01690_02183   2021.06.03.12.02.06_veh-35_00233_00609
2021.06.03.13.55.17_veh-35_00073_00426   2021.06.28.15.02.02_veh-38_02398_02848
2021.06.28.16.29.11_veh-38_01415_01821   2021.06.28.16.29.11_veh-38_03263_03766
2021.06.28.16.57.59_veh-26_00016_00484   2021.08.30.14.54.34_veh-40_00439_00835
2021.09.16.15.12.03_veh-42_01037_01434   2021.10.06.07.26.10_veh-52_00006_00398
```

官方流水线 step1 里有 `EXCLUDE_DIR = "/mnt/vdb1/.../navsim/processed_data/test_vq_codes"`
正是为此设计，但路径写死在作者机器上，我们这儿不存在 → **排除数为 0，过滤失效**。

**正确做法**：不依赖那个目录，直接拿 `metric_cache` 下的 136 个评测 log 名做黑名单过滤。
显式、可验证，而且对后续新下的包同样有效（按比例推算 camera 29–42 里还会有约 5 个）。

**已报的 SFT 结果不受影响**：用的是官方 `navsim_emu_vla_256_144_trainval_pre_1s.pkl`，
1192 个 log 与 navtest 交集为 **0**，已验证。
