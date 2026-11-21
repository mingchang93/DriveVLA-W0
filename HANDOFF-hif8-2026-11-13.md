# DriveVLA-W0 上的 HiF8 验证 — 交接

最后更新 2026-11-15 10:00。范围：Qwen3-VL-2B/7B + ROSS（ViT + SD-1.5 扩散头）的 2VA 路线，
HiF8 训练验证与 PDMS 精度交付，以及正在进行的 nuPlan 预训练。
**不含** DriveLM / LLaVA 那条线（那是 MindSpeed-MM 的另一套，代码在
`/mnt/sfs_turbo/zyc/MindSpeed-MM`，与本文无关）。

---

## 1. 这个任务在做什么

在 DriveVLA-W0 上验证 HiF8（昇腾的 8 位浮点格式，`torch_npu.hifloat8 = 290`）能不能
替代 bf16 训练而不掉精度，指标是 navsim 的 PDMS。路线分两阶段：

1. **阶段一 预训练**：nuPlan 数据，8000 步，产出骨干权重
2. **阶段二 SFT**：navsim 数据，4000 步，从阶段一权重起
3. **评测**：navtest 全量 12,146 场景算 PDMS

目前 2B 与 7B 的 **SFT + 评测已完成**（跳过预训练，直接从 Qwen3-VL 骨干做 SFT），
结果见第 6 节。

**2026-11-14 结论（十条全链条实测，三组交叉设计，全部出分）：**

- **HiF8 端到端与 bf16 等价。** 十个 PDMS 全落在 0.6776–0.6870（极差 1.38%、std 0.0032），
  bf16 基线排在第 7、上面 6 个下面 3 个；组间均值差 −0.10% / +0.60% / −0.06% 与组内
  极差 0.24–0.83% 同量级，**没有任何阶段能分离出可辨识的量化代价**。吞吐代价 1.8–4.3%。
- **「加预训练掉 13 个点」已结案：预训练 lr 借错了参照系。** 2e-4 来自官方
  **Emu3-8B / VQ / 6VA** 的预训练脚本，官方从未公开 Qwen 任一阶段的预训练 lr，借过来高了 4 倍。
  降到 **5e-5** 后预训练给出 **+4.3 个点**（0.8547 vs 无预训练 0.8120），方向与论文一致；
  2.5e-5 给 +1.7 点（0.8293），说明该区间**尚未见顶**。规律只有两条：
  **SFT lr 必须 1e-5**（官方 ROSS 版的值，8e-5 / 2e-4 都塌到 0.68），
  **预训练 lr 要在 5e-5 量级**。
  本文先前提出的「骨干位移是自变量」假设已被一次**预注册预测**证伪（见第 6 节）。
- **因此此前十条 HiF8 vs bf16 全部测于坏工况**（预训练 lr 2e-4，PDMS 0.68 带）。
  结论内部仍有效，但那是在模型本身已被打坏的操作点上得到的，说服力有限。
  **已在修正后的超参上重做出第一个数**：HiF8-DTS-196 **0.8504** vs bf16 **0.8547**
  （−0.50%），而它相对无预训练仍是 **+3.84 点** —— **量化没有吃掉预训练的收益**。
  那 0.50% 是否真实尚不能定（本操作点没有方差基线）。
  预训练阶段几乎无差异（七个窗口 +0.06%~+0.26%、`grad_norm` 三位小数相同），
  **差异在 SFT 之后才出现**。CTS-196 在 a53 跑中，约 17:45 出第三个点。
- **7B 在 2B 的同配置下装不下**（`BS=12/GC=False` 实测 OOM，88.78/91.75 GiB）。
  要测必须同时定 batch，`BS=6/GA=2` 可保持全局 96。见第 6 节。

细节见第 6 节；实验编排与待办见第 8 节；抽帧与续训的根因与解法见第 7 节。

---

## 2. 机器与路径

### 2.0 怎么连进去

```
你的机器 ──► ecs-gy (1.95.88.43:22, root, ~/.ssh/KeyPair-de38.pem)
                 │  反向隧道 127.0.0.1:23680
                 └──► ott9-a51   ← 这台就是 a51（内网 10.0.16.12），同时是内网跳板
                          ├──► a52  10.0.16.75
                          ├──► a53  10.0.16.231
                          └──► a54  10.0.16.118
```

`~/.ssh/config`：

```sshconfig
Host ecs-gy
    HostName 1.95.88.43
    User root
    IdentityFile ~/.ssh/KeyPair-de38.pem
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes

Host ott9-a51
    HostName 127.0.0.1
    Port 23680
    User root
    ProxyJump ecs-gy
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes

Host a52
    HostName 10.0.16.75
    User root
    ProxyJump ott9-a51
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes

Host a53
    HostName 10.0.16.231
    User root
    ProxyJump ott9-a51
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes

Host a54
    HostName 10.0.16.118
    User root
    ProxyJump ott9-a51
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

要注意的四点：

1. **跳板机 `ecs-gy` 用的是 pem，不是 ed25519**（`KeyPair-de38.pem`，权限 600）。
   四台机器的 `authorized_keys` 里有 ed25519 公钥，但 `ecs-gy` 不认它 —— 两样都要。
2. **`ott9-a51` 走的是 `127.0.0.1:23680` 的反向隧道**，需要 a51 主动在 ecs-gy 上维持。
   隧道断了四台全连不上（a52–a54 都 ProxyJump 它）。**这是单点。**
3. **服务器之间完全不能 ssh/scp —— 包括 a51。** a52/a53/a54 的 `authorized_keys`
   里确实有 `root@ott9-a51`，但 a51 上**没有** `a53`/`a54` 这些 Host 条目、也没有出站密钥，
   实测 `ssh a53` 直接 `Could not resolve hostname`。那些名字只是**本地** ssh config 的别名。
   跨机取文件走共享盘 `/mnt/sfs_turbo`；**跨机判断进程状态只能在本地做**（见第 10 节坑 2）。
4. hostname 与别名的对应：`a53 = liteserver-a5-00003`、`a54 = liteserver-a5-00004`。
   日志里出现 `liteserver-a5-0000N` 时按这个对。

| | |
|---|---|
| 机器 | `a52` `a53` `a54`（经 a51 跳板），`ott9-a51`（**注意不是 `a51`**） |
| 共享盘 | `/mnt/sfs_turbo`，33 T，当前用 55% |
| 我们的代码 | `/mnt/sfs_turbo/zyc/DriveVLA-W0` |
| 我们的工作目录 | `/mnt/sfs_turbo/zyc/drivevla-w0-zyc`（脚本、日志、`runs/`） |
| 容器 | **`zyc-w0`**（每台都有）。`lyz-cann920` 是同事的，别用 |
| 训练环境 | 容器内 `source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate` |
| navsim 环境 | `/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python3`（有 numpy/tqdm/ray/nuplan） |

### 环境上三个必须知道的坑

1. **宿主机的 `python3` 没有 numpy/tqdm。** 数据流水线（`tools/pickle_gen/nuplan_pickle_gen/step0~4`）
   必须用 navsim conda 的解释器。脚本里写的是裸 `python3`，会直接 `ModuleNotFoundError`。
2. **`/mnt/sfs_turbo/lmc/venv-hif8/bin/python3` 在宿主机上是断链**（指向容器内路径），
   所以凡是依赖它的东西（`torchrun`、`eval_ckpt.sh` 的推理部分）**必须 `docker exec zyc-w0` 跑**。
   PDMS 部分用的是 navsim conda 的 python，宿主机和容器里都能跑。
3. **运行时 `import transformers` 解析到 `/mnt/sfs_turbo/lmc/DriveVLA-W0/reference/transformers`**，
   不是我们自己那份 —— 我们脚本的 `PYTHONPATH` 只加了 `$ROOT` 和 `qwen-vl-finetune`。
   **所有已有结果都是在 lmc 那份上跑出来的**，换成我们自己那份会换掉数值基线，
   要换先逐文件 diff。

---

## 3. 数据

### 3.1 预训练数据（nuPlan）

**官方的现成产物可以直接用，这是对齐的最优解。**

| | 路径 | 规模 |
|---|---|---|
| 官方 pickle | `/mnt/sfs_turbo/zyc/official_pretrain_meta/meta/nuplan_processed_data_ndarray.pkl` | **18,191 段 / 1,310 log / 3,593,670 帧** |
| 我们复现的 | `/mnt/sfs_turbo/datasets/nuplan-v1.1/_full3/output/nuplan_processed_data_01.pkl` | 18,257 段 / 1,310 log |
| 相机图 | `/mnt/sfs_turbo/datasets/nuplan-v1.1/images` | 709 G / 1,321 log |

- 官方 pickle 来自 HF `liyingyan/DriveVLA-W0` 的 `nuplan_pickle_meta_and_vq_codes.tar.gz`（6.8 G）。
  **服务器直连 huggingface.co 不通，走 `https://hf-mirror.com/...`。**
- pickle 里**只有元数据**：`image` 是 200 个路径字符串、`action` 是 (200,8,3) 已归一化的路点、
  `text` 是 200 条指令、`segment_id`。**没有图** —— 709 G 相机数据必须自己从 nuPlan 站点下，省不掉。
- 原始官方 pickle 的 `action` 是嵌套 list，Qwen 2VA 的 dataset 对它调 `.ndim` 会炸
  （那份 pickle 是给 Emu3 6VA 的 dataset 做的）。`_ndarray.pkl` 是转过容器类型的版本，
  **数值一字未改**（抽 200 条逐值校验过）。转换脚本 `/mnt/sfs_turbo/zyc/fix_official_pickle.py`。
- 我们的 `_full3` 是官方的**严格超集**：官方 18,191 个 `segment_id` 一条不缺，多 66 段，
  log 集合完全相同。多的 66 段来自官方丢掉每个 log 的最后一帧（相对位姿是单位矩阵，
  没有下一帧可算）。`_full3` 的价值是证明流水线能复现官方数据，正式训练用官方那份。
- **「官方 17,376 段」是错的**，官方 `Training.md` 里没有这个数，实测 18,191。
- 归一化器：训练侧**不读**（pickle 里的 action 已归一化）。评测反归一化用
  `configs/normalizer_navsim_trainval/norm_stats.json`（SFT 是 navsim 数据）。
  我们算的 `_full3/normalizer_nuplan_01` 与官方 `configs/normalizer_nuplan` 吻合到
  mean 0.10% / std 0.04%。
- navtest 泄漏：官方排除 147 个 log、我们排除 137 个，但落在我们数据里的是**同样的 10 个**，
  机制等效。

### 3.2 SFT 与评测数据（navsim）

- SFT pickle 与 navtest：`/mnt/sfs_turbo/zyc/drivevla-w0-zyc/navsim_workspace/`
- 评测原图：`/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/test`（147 log）
- `metric_cache`：`navsim_workspace/exp/metric_cache`（137 个）
- mini 评测 = 单个 log 的 21 场景，约 1.5 分钟；full = 12,146 场景，约 22 分钟

---

## 4. 代码

### 4.1 HiF8 实现

`/mnt/sfs_turbo/zyc/DriveVLA-W0/models/quant/` ——

| 文件 | 内容 |
|---|---|
| `hif8_cts.py` | current tensor scaling：每步用当前张量的 amax 算 scale |
| `hif8_dts.py` | delayed tensor scaling，439 行，是 FSDPTurbo `hif8_dts.py` 的忠实移植 |

DTS 的要点：

- 调度 `observing_now()`：`pos = step % interval`，`pos != 0 and pos >= interval - sample_last`
- 稳态走 `torch_npu.npu_quantize(x, scale, ..., div_mode=True)` —— **一个 kernel、零 reduction**
- 要记 amax 那步走 `torch_npu.npu_quant_max(x, inv_scale)` —— 一次扫出 codes + amax
- **注意乘除方向**：`npu_quantize` 用 scale **除**，`npu_quant_max` 用 inv_scale **乘**
- `_refresh()` 批量 `all_reduce(MAX)`，用 `[n,-n]` 探针守护；scale 按 `_SCALE_FLOOR` 夹
- **roll 时 rebind 新 tensor，不做 in-place 写回** —— 持有旧 scale 的量化张量必须保持
  它被量化时用的那个 scale
- scale 存成普通属性、**不进 `state_dict`**。已验证：bf16 / CTS-196 / DTS-196 / CTS-168
  四个 ckpt 都是 **1,569 个张量、张量名完全相同**，带 `quant` 字样的只有 SD-1.5 VAE 自带的
  `vae.quant_conv` / `post_quant_conv`。所以推理侧加载不需要 scale，HiF8 的 ckpt 与 bf16 可互换。
- per-role `dst_type_max`：inputs/weight 15、grads 224（TE 那边是 32768）

另一套 HiF8（MindSpeed-MM + FSDPTurbo，`/mnt/sfs_turbo/zyc/hif8/`）的代码结构见
`C:\works\didi\hif8-handover\docs\hif8-cts-code-map.md`（标题是 CTS 但内容含 DTS，第 6 节整节）。
**那份是主副本,改动与运行都在 `/mnt/sfs_turbo/zyc/hif8/`。**

### 4.2 执行路径上的模型文件

模型是 Qwen3-VL（`model_type: qwen3_vl`，架构 `Qwen3VLForConditionalGenerationROSS`），
所以跑的是 `reference/transformers/src/transformers/models/**qwen3_vl**/modeling_qwen3_vl_ross.py`。
`qwen2_5_vl/` 下的同名文件**不在执行路径上** —— 曾有人（我）去修那份里的
`torch.amp.autocast('cuda')`（在 NPU 上是空操作），那是个真实缺陷但**改了不生效**，
`qwen3_vl` 那份两个副本早就是 `image_hidden.device.type` 的正确写法。

ROSS 头硬限两帧：`modeling_qwen3_vl_ross.py` 里三条 assert
（`image_hidden.shape[1] == 2` 等，报错信息 "Currently only support 2 images for each sample"）。
Qwen 训练入口只认 `navsim2va_ross` / `nuplan2va_ross` 两个 dataset，**都是 2VA**；
6VA 只存在于 Emu3/VQ 分支。官方自己的 Qwen ckpt 名
`qwen25vl_fast_147456_4_000_2va_nuplan_pretrained_8k_2va` 也证明两个阶段都是 2VA、预训练 8k 步。

---

## 5. 怎么跑

脚本都在 `/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc/`。

| 脚本 | 用途 |
|---|---|
| `run_stage1_pretrain.sh` | 阶段一预训练（nuPlan） |
| `run_sft_navsim.sh` | 阶段二 SFT（navsim） |
| `run_sft_nodet.sh` | SFT 的副本，带 `DETERMINISM=off` 预设，用于采非确定性 profiling |
| `eval_ckpt.sh` | `bash eval_ckpt.sh <ckpt目录> <标签> <mini\|full> [卡数]` |
| `run_stage1_pretrain_lrtest.sh` | 预训练脚本的副本，多了 `LR` / `SAVE_STRATEGY` / `SAVE_STEPS` 开关 |

### 环境变量开关

```
MODEL          起始权重（骨干或上一阶段输出）
HIF8           空=bf16 / cts / dts
HIF8_IGNORE    lm_head,o_proj = 168 个量化 Linear；lm_head = 196 个
HIF8_INTERVAL  DTS 的 amax 刷新间隔，默认 10
GC             gradient checkpointing，**默认已改为 False**
BS GA NPROC    微批 / 梯度累积 / 卡数
STEPS LOGSTEP  步数 / 日志间隔
OUT PORT       输出目录 / master_port
DS             deepspeed 配置，默认 scripts/sft/zero1.json
HCCL_RANGE     HCCL 端口段。SFT 脚本默认 17500-17599；预训练脚本原来写死
               17100-17199，已参数化（a53 上同事容器有七八个，写死容易撞端口）
LR             仅副本脚本有，默认 2e-4
SAVE_STRATEGY  仅副本脚本有，默认 no
SAVE_STEPS     仅副本脚本有，默认 1000；仅 SAVE_STRATEGY=steps 时生效
```

**长跑务必开 `SAVE_STRATEGY=steps SAVE_STEPS=1000`。** `save_strategy=no` 只在训练
结束时写权重，中途被停就全部作废 —— 已经为此丢过 915 步。开了之后崩了最多丢 1000 步，
而且验续训可以直接用长跑自己的 ckpt，不必单独做实验。中间 ckpt 带优化器状态、
调度器和 RNG 状态，所以 resume 能真正接上。

### 启动方式（照抄，别改）

```bash
docker exec zyc-w0 env STEPS=8000 BS=12 GA=1 GC=False NPROC=8 \
  HIF8=dts HIF8_IGNORE=lm_head HIF8_INTERVAL=10 \
  OUT=/path/to/out PORT=23701 \
  bash /mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc/run_stage1_pretrain.sh
```

长跑一律 `setsid nohup ... < /dev/null &`，断连不影响。

### 当前超参（与官方 Emu3 nuPlan 预训练的 `training_args.bin` 逐项核对一致）

lr 2e-4 · wd 0.1 · warmup 400 · `cosine_with_min_lr`（min 1e-6）· 8000 步 ·
max_grad_norm 5.0 · adamw_torch · β(0.9, 0.95) · eps 1e-6 · seed 42 ·
全局 batch 96（mb12 × 8 卡 × ga1）· bf16 · **model_max_length 8192** ·
max_pixels 147456 / min_pixels 12544 · 2VA · ZeRO-1

与官方的差异只有三处且都不改数值：ZeRO-1（官方 `zero3_offload`，他们 3 机 24 卡跑更大的
Emu3 才需要 offload）、tf32（NPU 上无意义）、save_strategy。

**SFT 侧（与官方 ROSS 版逐项核对一致）**：官方 `Qwen_PDMS_85.3`（**ROSS 变体**，`pintang` 镜像，run_name 带 `lr1e-5_unfreezevit`）实测 **lr 1e-5** · wd 0.001 · warmup 200 · `cosine`（无 min_lr）· 4000 步 · per-device bs 6 × ga 1 = **全局 48** · model_max_length 8192 · max_grad_norm 5.0 · seed 42 · num_workers 12 —— **与我们完全一致**，只差 `GC`(True/False)、ZeRO 阶段(3/2)、`save_strategy`，前两者合计实测 0.79%。
**注意另有一个同名目录的非 ROSS run（`huyhung411991` 镜像，lr 2e-4），那是另一个实验，不要拿它当参照** —— 本文曾为此搞错过一次，详见第 6 节。

---

## 6. 已经确立的事实（都有实测依据）

### 精度（2B / 7B，lr 1e-5，SFT 4000 步，无预训练）

结果表与分项差异在 `C:\Users\...\scratchpad\hif8_pdms_results.md`（已交付）。要点：

- HiF8 各配置与 bf16 的 PDMS 差异都在 ±1% 量级
- **DTS-196 的 +0.946% 是唯一通过配对 t 检验显著性的**（配对 SE ≈ 0.0025）
- **但 PDMS 的 run-to-run 方差从未测过** —— 这是所有「A 比 B 好」结论的共同短板，
  包括上面那条。要测就跑 `bf16 seed43` + `CTS-196 seed43`（约 3.5 h）

### 确定性

- 同配置重跑**逐位一致**。之前非确定性的根因是**播种顺序**（扩散头的 mlp/pos_embed/factor
  不在 Qwen ckpt 里，建模时随机初始化发生在 `set_reproducibility` 之前），不是环境变量
- `TASK_QUEUE_ENABLE=0` 与 `=1` **逐位一致**（bf16 与 HiF8 都验过，含跨设置比对）。
  它数值上是中性的，但吞吐价值未测（50 步噪声 7.5–9.2%，淹掉了它）
- 其余八个确定性开关**从未逐个验证**。`FLAGS_npu_storage_format` 值约 162 ms/步

### gradient checkpointing（重计算）

**开和关不逐位一致。** a53 实测（2B、mb12、同 seed、50 步）：

| 指标 | 首次分叉 | 分叉量级 | 50 步内最大相对差 |
|---|---|---|---|
| `ross_loss` | 第 3 步 | 1.5e-7 | 1.8e-6 |
| `grad_norm` | 第 6 步 | 7.1e-8 | **5.3e-2 @ 第 13 步** |
| `action_loss` | 第 7 步 | 1.2e-3 | 1.2e-3 |

起点是 float32 最低位的舍入（计算图不同，规约顺序与落位会差最低位），但会随优化步放大。
**所以 bf16 与 HiF8 两条对比臂必须用同一个 GC 设置。** 三个脚本的 `GC` 默认已改为 `False`。

**后果：已有的全部 SFT / PDMS 结果都是开着重计算跑的，关掉之后的新结果不能与它们同表比较。**

### 性能（预训练配置，ML 8192、关重计算、gbs 96）

| | HBM 峰值 | 占比 | s/步 |
|---|---|---|---|
| bf16 | 66,669 MB | 68% | 2.832 |
| HiF8-CTS-196 | 67,325 MB | 68% | 2.862 |
| HiF8-DTS-196 | 67,346 MB | 69% | 2.878 |

CTS 与 DTS 差 0.6%、显存差 21 MB —— 落在 50 步噪声里，不能当结论。
关重计算比开着快 8.7%（3.134 → 2.863 s/步）。显存还有 30% 余量，mb12 不是上限。

### profiling 的结论（SFT 配置、确定性模式）

单步约 2204 ms 的构成：`aten::index` 回退 AI_CPU 约 265 ms（24% 设备耗时，根因是
确定性模式下 `F.interpolate` 被分解）、conv 因 `FLAGS_npu_storage_format=0` 慢 2–5× 约 162 ms、
空转 203 ms、通信 51 ms。HiF8 比 bf16 多约 **567 ms/步的主机开销**，来自 Python 层的量化
`autograd.Function`（+1,875 个 0.1–1 ms 空隙）。关掉确定性能快 35%，但那会改变数值。

### lr 2e-4 在 Qwen 2VA 上可用（915 步实测）

Qwen 2VA 预训练的 lr 没有任何公开记录，2e-4 是从 Emu3（VQ 离散 token + 6 对 VA）借的。
2026-11-13 在 a54 实测 915 步（bf16、mb12、gbs 96、关重计算）。

**先说一个计数陷阱**：日志里 `'epoch'` 在两种行里都出现 —— 我们自己打的
`{'action_loss','ross_loss','epoch'}` 和 HF 打的 `{'loss','grad_norm','learning_rate','epoch'}`
—— 所以数 `'epoch'` 会**双计**。我曾因此把 915 步报成 1,812 步，连带把进度表的步号
全标错一倍。可靠的计数是数 `'learning_rate'`（只在 HF 行里）或 `'action_loss'`（只在我们的行里）。

单步 loss 抖动大，要用窗口均值（官方 trainer_state 的每条记录本身就是 10 步平均）。
100 步窗口均值：

| 区间 | 我们 action_loss | grad_norm | 官方 Emu3 loss |
|---|---|---|---|
| 1–100 | 6.222 | 68.9 | 6.325 |
| 101–200 | 2.216 | 7.43 | 5.265 |
| 201–300 | 1.942 | 4.42 | 5.016 |
| 301–400 | 1.819 | 3.16 | 5.076 |
| 401–500 | 1.717 | 2.44 | 4.923 |
| 501–600 | 1.664 | 2.08 | 4.744 |
| 601–700 | 1.646 | **4.65** | 4.665 |
| 701–800 | 1.627 | 1.90 | 4.600 |
| 801–900 | **1.547** | 1.79 | 4.537 |

**lr 调度与官方逐步重合**（只差一步的增量：我们 4.5e-6 / 2.45e-5 / 9.95e-5 / 1.995e-4，
官方 5e-6 / 2.5e-5 / 1e-4 / 2e-4）。两边窗口均值都单调下降、无反弹；grad_norm 从 68.9
降到 1.79，只在 601–700 有一个瞬时尖峰。**两边的 loss 不是同一个量**（我们是 FAST 动作
token 的交叉熵，Emu3 是 VQ 码本的），幅度不可比，可比的形状都健康。

一个曾被我误报的点：按单步看第 600 步（1.164）到第 900 步（1.669）像是涨了 43%，
那是单步噪声；窗口均值是单调下降的。

**这把在 915 步时被人为停掉（卡要让给别的验证），而当时 `save_strategy=no` 所以没有任何
ckpt，915 步全部作废** —— 这就是「长跑务必开中间 ckpt」那条的由来。

### 端到端 PDMS：HiF8 与 bf16 等价（十条链实测），但预训练路径掉 13 个点

2026-11-14 01:49–06:00，**十条完整链条全部出分**（每条都是预训练 8000 步 →
SFT 4000 步 → navtest 全量 12,146 场景）。三组交叉设计把「量化用在哪个阶段」隔离开：

| 组 | 预训练 | SFT | 输出目录 | PDMS | 相对 bf16 |
|---|---|---|---|---|---|
| 组二 | bf16 | CTS-196 | `sftpt_cts196` | 0.686973 | +0.99% |
| 组二 | bf16 | DTS-196 | `sftpt_dts196` | 0.686626 | +0.94% |
| 组二 | bf16 | CTS-168 | `sftpt_cts168` | 0.682535 | +0.34% |
| 组三 | DTS-168 | bf16 | `ptsft_dts168pt_bfsft` | 0.682188 | +0.28% |
| 组二 | bf16 | DTS-168 | `sftpt_dts168` | 0.681324 | +0.16% |
| 组一 | DTS-168 | DTS-168 | `chain_sft_dts168` | 0.680366 | +0.02% |
| **组一** | **bf16** | **bf16** | `chain_sft_bf16` | **0.680256** | **基线** |
| 组一 | DTS-196 | DTS-196 | `chain_sft_dts196` | 0.678924 | −0.20% |
| 组一 | CTS-196 | CTS-196 | `chain_sft_cts196` | 0.678729 | −0.23% |
| 组三 | DTS-196 | bf16 | `ptsft_dts196pt_bfsft` | 0.677553 | −0.40% |

十个数：**极差 0.00942（1.38%）、std 0.0032**。bf16 基线排第 7，上面 6 个、下面 3 个。

| 组 | 含义 | n | 均值 | 相对 bf16 | 组内极差 |
|---|---|---|---|---|---|
| 组一 | 两阶段同精度 | 4 | 0.679569 | −0.10% | 0.24% |
| 组二 | 只 SFT 用 HiF8 | 4 | 0.684364 | +0.60% | 0.83% |
| 组三 | 只预训练用 HiF8 | 2 | 0.679870 | −0.06% | 0.68% |

**结论：HiF8 端到端与 bf16 等价。**

组二的均值比基线高 0.60%，一度看着像「量化在 SFT 起了类似正则的作用」。
**数据内部有两条证据把它否掉了**：

1. **组三内部自己就有 0.68% 的落差。** `ptsft_dts168pt_bfsft` 0.682188 vs
   `ptsft_dts196pt_bfsft` 0.677553 —— 两臂的 SFT 都是 bf16，唯一区别是预训练用了哪个
   HiF8 模式。要么「预训练的 `dst_type_max`」能撬动 0.68%（没有机制支持：四条预训练
   曲线在 32 个窗口上全部落在 ±0.23% 内），要么 **0.7% 量级的噪声本来就存在**。
   是后者。而 0.7% 正好是组二那个效应的尺度。

2. **168 / 196 的高低在三组之间翻转：**

   | 组 | 168 | 196 | 谁高 |
   |---|---|---|---|
   | 组一 | 0.680366 | 0.678924 | **168** 高 0.21% |
   | 组三 | 0.682188 | 0.677553 | **168** 高 0.68% |
   | 组二 | 0.682535 | 0.686973 | **196** 高 0.65% |

   如果 `dst_type_max` 的 15 vs 224（即 168 / 196）真有方向性影响，三组应当同向。
   翻了，就是噪声。

所以**组二那 0.60% 不写成增益**。正确的说法是「十条链全落在 1.4% 的带内，
与 bf16 不可区分」。配合预训练 loss 曲线 ±0.23%、吞吐代价 1.8–4.3%，
**HiF8 在这个任务上可用**。

**仍然缺的那把尺子**：run-to-run 方差没有直接测量值。上面两条是**数据自洽性论证** ——
用「量化无害」推出「0.7% 是噪声」，再用它说明组二不显著，逻辑上比一次直接测量弱。
`variance_run.sh`（同配置换 `SEED=43`，复用现成的 `chain_pt_bf16` 权重，2.6 小时）
能把 0.7% 从推论变成测量。它现在的价值是**补强**，不再是**卡结论**。

**但十条链同时暴露一个更大的问题：加了预训练，PDMS 从 0.8120 掉到 0.6803。**
十条臂一起掉，所以**与量化无关**。但「掉的原因是预训练」这个归因是错的 —— 见下一节。

### 超参已定稿（2026-11-15）

```
预训练  lr 5e-5 · 8000 步 · warmup 400 · cosine_with_min_lr(min 1e-6) · wd 0.1
        mb12 × 8 卡 × ga1 = 全局 96 · GC=False · ZeRO-1 · ML 8192 · seed 42
SFT     lr 1e-5 · 4000 步 · warmup 200 · cosine（无 min_lr）· wd 0.001
        bs6 × 8 卡 × ga1 = 全局 48 · GC=False · ZeRO-2 · ML 8192 · seed 42
```

SFT 侧与官方 ROSS 版（`pintang` 镜像）逐项一致。预训练 lr 由实测扫出：

| 预训练 lr | PDMS | 相对无预训练 0.8120 |
|---|---|---|
| 2.5e-5 | 0.8293 | +1.7 点 |
| **5e-5** | **0.8547** | **+4.3 点** ← 定稿 |
| 1e-4 | 0.7054 | −10.7 点 |
| 2e-4 | 0.6803 | −13.2 点 |

**5e-5 → 1e-4 之间就是断崖**，不必再往上探。2e-4 是从官方 Emu3-8B/VQ/6VA
预训练脚本借的，官方从未公开 Qwen 任一阶段的预训练 lr。

#### lr 已写死在脚本里，不再由环境变量传

| 文件 | lr | 说明 |
|---|---|---|
| `tools/zyc/run_stage1_pretrain.sh` | `--learning_rate 5e-5` 写死 | **预训练正本**。已把 `SAVE_STRATEGY` / `SAVE_STEPS` / `HCCL_RANGE` 从 `_lrtest` 版搬进来（默认值与原行为一致，旧调用方不受影响） |
| `tools/zyc/run_sft_navsim.sh` | `--learning_rate 1e-5` 写死 | **SFT 正本** |
| `chain_ptlr.sh` | `LR_PT=5e-5` 写死 | **链条正本**：预训练 → SFT → full 评测 |
| `chain_ptlr_hif8.sh` | 同上 + HiF8 参数 | 截至本次更新仍在跑，`LR_PT` 待写死 |
| `chain_ptlr_7b.sh` | 同上 + `BASE`/`GC_PT`/`GC_SFT` + 20 步显存烟测 | 7B 用 |

`run_stage1_pretrain_lrtest.sh` / `run_sft_navsim_lr.sh` 是过渡期的 lr 可配副本，
待两台跑完即删。**改脚本前先确认它没有进程在执行**（见下方第 2 条坑）。

### HiF8 vs bf16 在**正确超参**下的第一个端到端结果（2026-11-15）

此前十条 HiF8 对比全测在预训练 lr 2e-4 的坏工况上（PDMS 0.68 带），
模型本身已被打坏，等价性测不出意义。在定稿超参（预训练 5e-5 / SFT 1e-5）上重做：

| 配置 | PDMS | 相对 bf16 | 相对无预训练 0.8120 |
|---|---|---|---|
| **bf16** | **0.8547** | — | +4.27 点 |
| **HiF8-DTS-196** | **0.8504** | **−0.50%** | **+3.84 点** |
| HiF8-CTS-196 | 跑中（a53，约 17:45） | — | — |

**两件事同时成立**：HiF8 比 bf16 低 0.50%；而 HiF8 带预训练仍拿到 **+3.84 点**
的预训练收益 —— **量化没有吃掉预训练的好处**，这是比「差 0.5%」更重要的一句。

**这 0.50% 是否构成真实差异，目前不能定。** 手上唯一的噪声估计约 0.68%
（组三两臂只差预训练 HiF8 模式却相差 0.68%），而那是在坏工况下测的，
拿来当本操作点的噪声尺度很弱。要定性得在 5e-5 上测方差 ——
**做法现在很简单：跑两遍 `chain_ptlr.sh`，只改 `SEED`。**

#### 差异出现在 SFT 之后，不在预训练阶段

同 lr 下 bf16 与 DTS-196 的预训练曲线配对比较（500 步窗口均值，`pair.py`）：

| 区间 | lr 5e-5 bf16 | lr 5e-5 DTS-196 | grad_norm bf16 / DTS |
|---|---|---|---|
| 1–500 | 3.4605 | 3.4647 (+0.12%) | 29.515 / 29.341 |
| 1001–1500 | 1.4986 | 1.5024 (+0.26%) | 2.929 / 2.929 |
| 2501–3000 | 1.3230 | 1.3265 (+0.26%) | 2.473 / 2.460 |
| 3501–3636 | 1.2794 | 1.2801 (+0.06%) | — |

**七个窗口全在 +0.06% ~ +0.26%，`grad_norm` 到小数点后三位相同。**
所以预训练阶段几乎无差异，PDMS 上那 0.50% 是在 SFT 之后才出现的。
这个定位缩小了后续排查范围。

#### 顺带排除了「数据流失同步」这个混淆因素

lr 2.5e-5 那一对曾出现单调扩大的偏差（窗口均值 −0.00% → −5.36%），
我一度判为噪声（错）、又怀疑是抽帧的私有 RNG 失同步。**逐步对比排除了后者**：

- 两对的**前 10 步都贴到 1e-3 量级**，且两对的第 1、2 步相对差完全相同
  （7.8e-03、4.4e-03）—— warmup 初期模型还几乎是骨干，这个差就是同一份数据、
  同一份权重下的纯量化误差，**四个 run 喂的是同一条数据流**
- 全程**没有任何一步超过 16.4%**；数据错开会让 96 样本上的 loss 立刻差几十个百分点
- 增长**平滑无突跳**；失同步必然表现为某一步的阶跃

形态差异：lr 2.5e-5 的逐步相对差中位数单调增长到 5.5e-2，
lr 5e-5 则从 1.3e-2 起就**饱和不动**。一个自洽解释是
**量化扰动是固定的相对幅度，而优化步长随 lr 缩小** —— lr 越小，
扰动相对优化信号越大，两条轨迹分得越开。
注意方向：HiF8 的 loss 是**更低**，不是更差。

### 7B：BS=12 在 96 GB 卡上装不下（实测）

`chain_ptlr_7b.sh` 的阶段零烟测 20 步，a53 实测**真 OOM**：

```
NPU out of memory. Tried to allocate 1.45 GiB
(NPU 4; 91.75 GiB total capacity; 88.78 GiB already allocated; 1.48 GiB free)
```

对照：同配置 2B 的 HBM 峰值只有 66.9 GB / 96 GB。7B 光权重 + 梯度 +
ZeRO-1 分片优化器就约 42 GB，加上 mb12 的激活就超了。

**要测 7B 必须同时定 batch**：`BS=6 / GA=2` 可保持全局 96 且大概率装得下
（7B 的 SFT 正是 BS=6 跑成的，`w7b_bf16` 无预训练 PDMS 0.8407）。
烟测这一步值得保留 —— 2 分钟就定性，不会白占整夜。

### 工作目录已收敛到 7 个脚本

删掉 33 个一次性/已废弃脚本（判据三条全中：0 真调用方 + 当前未在执行 +
不在保留名单；第二轮补删因第一轮而变成孤儿的两个）。剩下：

| 脚本 | 作用 |
|---|---|
| `chain_ptlr.sh` | **链条正本**：已并入 HiF8 参数（`HIF8_PT`/`HIF8_SFT` 置空即 bf16、`IG_*` 控制 196/168、`SAVE_STEPS_PT`、`HIF8_INTERVAL`），一个脚本覆盖 bf16/DTS/CTS × 196/168 全部组合 |
| `chain_ptlr_7b.sh` | 链条 + `BASE`/`GC_PT`/`GC_SFT` + 20 步显存烟测 |
| `queue_dts196.sh` | 通用排队器（等标记 + 等卡 + 起链条），已指向正本 |
| `queue_7b.sh` | 7B 排队器，已指向 `chain_ptlr_7b.sh` |
| `run_full3.sh` | 生成预训练 pickle |
| `pipestate.sh` | 流水线状态（给监控用） |

`tools/zyc/` 侧：`run_stage1_pretrain.sh`（预训练正本，lr 写死 5e-5）、
`run_sft_navsim.sh`（SFT 正本，1e-5）、`run_sft_nodet.sh`（关确定性做 profiling）、
`eval_ckpt.sh`。`run_one.sh` / `run_pipeline*.sh` 是更早遗留，未动。

**已删的过渡副本**：`chain_ptlr_hif8.sh`、`chain_ptlr_sv1000.sh`、
`run_stage1_pretrain_lrtest.sh`、`run_sft_navsim_lr.sh`。
删除后两个排队器里还残留着对它们的引用（会断链），已一并改指向正本 ——
**删脚本之后务必 `grep` 一遍谁还在引用它。**

**「测 PDMS 方差」现在不需要专门脚本**：跑两遍 `chain_ptlr.sh`、只改 `SEED` 即可。

**中间 ckpt 已清理（2026-11-15，释放 583 GB）**：已跑完 8000 步的预训练 run
只保留 `checkpoint-8000` 与根目录的最终权重（6.77 GB，SFT 的起始权重）；
中间 ckpt 每个 46 GB（含优化器状态与 8 个 `rng_state`）。
两道门槛：`train.log` 必须有 `TRAIN_EXIT`（否则认为还在跑）、
`checkpoint-8000` 与根权重索引都必须存在。
**代价**：待办里「用中间 ckpt 画『预训练步数 vs PDMS』曲线」已不可做，要做得重跑预训练。

也删掉了错配置 run 的权重约 714 GB（预训练 lr ≠ 5e-5 或 SFT lr ≠ 1e-5 的产物）。
**每个 run 的 `train.log` / `training_args.bin` / `trainer_state.json` / `config.json`
都保留了** —— 本文所有 PDMS 与超参结论的原始依据都在里面，可复核。
注意：`pt_ptlr1e4` / `pt_ptlr2p5e5` 的权重已不在，那两条链不能再重新评测或续训。

### 「加预训练掉 13 个点」已结案：预训练 lr 借错了参照系

**结论**：不是预训练有害，是预训练的 lr 从 Emu3 借了 2e-4，比合适值高约 4 倍。
把它降到 5e-5，预训练给出 **+4.3 个点**的正收益，方向与论文一致。

#### 全部 bf16 实测点（2B · Qwen3-VL-2B + ROSS · 4000 步 SFT · navtest 全量）

| 预训练 lr | SFT lr | 其他 | 骨干位移 | `merger.fc2.bias` | **PDMS** |
|---|---|---|---|---|---|
| **5e-5** | 1e-5 | `sft_ptlr5e5` | 3.61% | 34.2% | **0.8547** |
| **2.5e-5** | 1e-5 | `sft_ptlr2p5e5` | 2.13% | 23.2% | **0.8293** |
| 无 | 1e-5 | GC=True·ZeRO-1 `aligned_bf16` | 0.89% | 5.8% | 0.8120 |
| 无 | 1e-5 | GC=False·ZeRO-2 `sftonly_bf16_newcfg` | — | — | 0.8056 |
| 2e-4 | 2e-4 | `lr2e4_chain` | 10.59% | 79.1% | 0.6891 |
| 无 | 8e-5 | `night_bf16` | 3.51% | 47.6% | 0.6843 |
| 无 | 2e-4 | `lr2e4_sftonly` | 6.35% | 50.0% | 0.6833 |
| 2e-4 | 1e-5 | `chain_sft_bf16` | 8.84% | 70.3% | 0.6803 |

#### 规律只有两条，都很朴素

```
PDMS 0.80+ ：SFT lr = 1e-5  且  预训练 lr ≤ 5e-5（或不做预训练）
PDMS 0.68  ：SFT lr ≥ 8e-5  或  预训练 lr = 2e-4
```

- **SFT lr 必须是 1e-5** —— 官方 ROSS 版（`pintang` 镜像）的值。8e-5 与 2e-4 都塌到 0.68。
- **预训练 lr 要在 5e-5 量级** —— 2e-4 是官方 **Emu3-8B / VQ / 6VA** 预训练脚本的值，
  官方**从未公开 Qwen 任一阶段的预训练 lr**，借过来高了 4 倍。
- **预训练 lr 在 2.5e-5 → 5e-5 这一段还没见顶**：翻一倍多拿 2.5 个点
  （0.8293 → 0.8547）。合适区间在 5e-5 附近或更高，但 2e-4 肯定过了。

#### 被证伪的「骨干位移」假设（含一次失败的预注册预测）

本文曾提出：PDMS 由**骨干离 Qwen 多远**决定，过了约 3.5% 就塌。
2026-11-15 01:10，在 a53/a54 两把出分**之前**，依据该假设写下两个预测：

| 臂 | 位移 | 预测 | 实际 | 结果 |
|---|---|---|---|---|
| `sft_ptlr2p5e5` | 2.13% | 接近 0.80 | 0.8293 | 勉强算对 |
| `sft_ptlr5e5` | **3.61%** | **接近 0.684**（因 `night_bf16` 位移 3.51% 得 0.6843） | **0.8547** | **差 17 个点，证伪** |

**几乎相同的骨干位移（3.51% vs 3.61%），PDMS 差 17 个点。** 更致命的是符号：
在预训练这条线上位移与 PDMS **正相关**（2.13%→0.8293，3.61%→0.8547），
在 SFT lr 那条线上**负相关**。同一个量在两条路径上符号相反，
说明它不是原因，只是随 lr 变化的伴生量 —— lr 大则权重走得远，
但伤害来自优化过程本身，不是终点离骨干多远。

**仍然站得住的那半句**：相近总位移下，温和预训练对**视觉桥**的扰动明显更小
（`merger.fc2.bias` 34.2% vs `night_bf16` 的 47.6%）。这条观察有效，
但不足以单独解释 PDMS。

#### 这件事对 HiF8 结论的影响（重要）

**此前那十条 HiF8 vs bf16 的链条，全部是在预训练 lr 2e-4 这个坏工况下测的**
（PDMS 0.6776–0.6870）。它们内部仍然有效（两臂同配置），
但「HiF8 ≡ bf16」这个结论是在一个**模型本身已经被打坏**的操作点上得到的，
说服力有限。**HiF8 的对比需要在修正后的超参上重做** ——
已排进 a53/a54 的队列（`dts196_lr2p5e5` / `dts196_lr5e5`，预训练 lr 分别 2.5e-5 / 5e-5）。

#### 一个要谨慎对待的数字

0.8547 与官方 Qwen ROSS 版的 **85.3** 数值上相当，但**官方那份是 Qwen2.5-VL-7B，
我们这是 2B**，不能当作「追平官方」。真正可比的是我们自己的 7B
（`w7b_bf16`，无预训练，PDMS 0.8407）—— 7B 带预训练的链条已排进队列。

### 官方 Qwen SFT 的参照：两个同名目录、两个不同的 run（务必分清）

**这一节是为了防止重犯一个已经犯过的错。** HF 上有多个 DriveVLA-W0 的第三方镜像，
`Qwen_PDMS_85.3/` 这个目录名下放的**不是同一个实验**：

| | `huyhung411991/DriveVLA-W0` | `pintang/DriveVLA-W0` |
|---|---|---|
| `run_name` | `logs/qwen25vl_fast_147456_4_000_2va_nuplan_pretrained_8k_2va` | `logs2/ross_sd15_img_action_qwen25vl_fast_147456_4_000_2va_`**`lr1e-5`**`_unfreezevit` |
| 日志日期 | Sep10 | Sep20 |
| `architectures` | `Qwen2_5_VLForConditionalGeneration`（**裸的**） | `Qwen2_5_VLForConditionalGeneration`**`ROSS`** |
| `enable_ross` | 无此键 | **True** |
| `sd_model_path` | 无 | `stable-diffusion-v1-5/unet` |
| **learning_rate** | **2e-4** | **1e-5** |
| per-device bs × 进程 | 8 × 6 = 48 | 6 × ? = 48 |
| `training_args.bin` md5 | `0c173b81b2ac36e3…` | `ce3f953fd5bcb6f9…` |

两份 md5 不同，是**两个真实存在的不同实验**。
**`pintang` 那份（ROSS）才是我们的同族** —— 我们跑的是
`Qwen3VLForConditionalGenerationROSS` + SD-1.5 扩散头。
作者本人的仓库 `liyingyan/DriveVLA-W0` 里**没有** `Qwen_PDMS_85.3`（只有
`Emu3_Flow_Matching_Action_Expert_PDMS_87.2`），所以这两份都来自第三方镜像。

#### 我们的 SFT 配置与官方 ROSS 版逐项一致

| 项 | 官方 ROSS 版 | 我们 |
|---|---|---|
| `learning_rate` | **1e-5** | **1e-5** |
| `weight_decay` | 0.001 | 0.001 |
| `warmup_steps` | 200 | 200 |
| `lr_scheduler_type` | `cosine`（无 min_lr） | `cosine` |
| `max_steps` | 4000 | 4000 |
| `per_device_train_batch_size` × `ga` | 6 × 1 | 6 × 1 |
| 全局 batch | 48 | 48 |
| `model_max_length` | 8192 | 8192 |
| `max_grad_norm` | 5.0 | 5.0 |
| adam β / eps | (0.9, 0.95) / 1e-6 | 同 |
| `seed` · `dataloader_num_workers` | 42 · 12 | 42 · 12 |
| **只差三处** | `GC=True` · ZeRO-3 · `save_strategy=steps` | `GC=False` · ZeRO-2 · `no` |

GC + ZeRO 这两处合计实测只值 **0.79%**（`sftonly_bf16_newcfg` 0.8056 vs
`aligned_bf16` 0.8120）。**所以 SFT 侧本来就是对齐的。**

<div></div>

#### 要撤回的一条错误链（2026-11-14 当天犯并当天纠正）

1. **「官方 SFT 是 2e-4、我们低 20 倍」—— 撤回。** 我只查到 `huyhung411991`
   那份（非 ROSS 的另一个实验），没有继续核对其他镜像，就下了结论。
2. **「本文原先写的 lr 1e-5 是抄错列」—— 撤回。** 那句话原本就是对的，
   是后来的「更正」错了。
3. **方法上的错误比结论更值得记**：我当时为「抄错列」构造的判据是
   「同时记下的 wd 0.001 + warmup 200 + 裸 cosine + 4000 步 + ML 8192 + ZeRO-3
   在五份官方配置里只有那一份满足」。**但 pintang 那份同样全部满足** ——
   这个判据对两个候选根本没有区分力，我却用它去「定位」错误来源。
   **教训：用来排除候选的判据，必须先确认它在候选之间真的不同。**
4. **据此起的两把实验**（`lr2e4_chain` 0.6891、`lr2e4_sftonly` 0.6833）
   结果本身有效，而且恰好证明 2e-4 对本模型有害 —— 但那约 5.8 小时卡时本可不花。

#### 这件事对「13 个点」的影响

**SFT 侧被排除掉了**：我们的 SFT 与官方 ROSS 版一致，lr 不是原因。
所以剩下的自变量仍然是**预训练**，而官方**从未公开 Qwen 任一阶段的预训练 lr** ——
我们用的 2e-4 唯一出处是官方 **Emu3-8B / VQ / 6VA** 的预训练脚本
（gbs 96 / warmup 400 / 8k 步），那是另一个模型族。
**后续**：降预训练 lr 这个处方被实测证实有效（5e-5 得 0.8547），
但推出它的那条「位移」推理本身已被证伪（见上一节的预注册预测）。
处方对、理由错 —— 两者要分开记。

#### 进行中：预训练 lr 扫描（用户起的，2026-11-14 17:42/17:46）

| 机器 | 输出 | 预训练 lr | save_steps |
|---|---|---|---|
| a53 | `runs/pt_ptlr2p5e5` | **2.5e-5** | 1000 |
| a54 | `runs/pt_ptlr5e5` | **5e-5** | 2000 |

与我们那四把 8000 步预训练**只差两处**：lr（2e-4 → 降 8× / 4×）与
`save_strategy=steps`（之前是 `no`）。其余与官方 Emu3 nuPlan 预训练逐项一致的那套
超参未动（全局 batch 96、warmup 400、`cosine_with_min_lr` min 1e-6、wd 0.1、
max_grad_norm 5.0、ZeRO-1、`GC=False`、ML 8192、seed 42、确定性全开）。

**预训练曲线已经能看出 2e-4 偏大**（500 步窗口均值，`ptcmp.py`）：

| 区间 | lr 2e-4 | lr 5e-5 | lr 2.5e-5 |
|---|---|---|---|
| 1–500 | 2.7831 | 3.4605 (+24.3%) | 3.9953 (+43.6%) |
| 1001–1500 | 1.5064 | 1.4986 (−0.5%) | 1.5794 (+4.9%) |
| 3001–3500 | 1.3794 | 1.2994 (−5.8%) | 1.3875 (+0.6%) |
| **5501–6000** | **1.3030** | **1.1681 (−10.4%)** | **1.2470 (−4.3%)** |

低 lr 在 warmup 期劣 24–44%，从 1500 步起反超且差距持续扩大；
lr 2e-4 已基本走平（1.3112 → 1.3030）。`grad_norm` 更直接：
lr 2e-4 从第 3000 步起**回升**（1.296 → 1.466）而 loss 不再降 —— lr 过大的特征。

**但预训练 loss 低不等于 PDMS 高。** 本任务已经踩过这个坑：
`chain_sft_bf16` 的 action_loss 比 `aligned_bf16` 低 5.3%，PDMS 却差 13 个点。
所以上表只是旁证，定论要等接上 SFT + 评测后的 PDMS。

**`save_steps` 这一改很有用**：跑完能直接拿 1k/2k/…/8k 步的 ckpt 做
「骨干位移 vs PDMS」曲线，一把预训练换一条曲线，不必每个 lr 都重跑。
注意开中间 ckpt 会让**续训**不逐位一致（抽帧的私有 RNG 不进 ckpt，见第 7 节），
但对「只存快照、不续训」没有影响。

#### 仍未排除的其他可能

- **归一化器尺度不匹配**（预训练用 `configs/normalizer_nuplan`、SFT 与评测用
  `configs/normalizer_navsim_trainval`，第一维 std 2.1266 vs 1.5279）。
  已降为次要假设 —— 位移这条线能解释「无预训练、只调 SFT lr 也掉 13 点」，
  归一化器假设解释不了。
  <br>**一条曾用来支持它的旁证是无效的**：`*_agent_human_traj_diff.json` 里的
  `diff_x` / `diff_y` **存的是绝对值**（有符号均值与 |误差| 均值每行完全相等），
  拿它做「系统性尺度偏差」的回归无效。要验尺度必须去取 navsim 的真值轨迹。
- **预训练与 SFT 的数据分布差异**（nuPlan 20 秒驾驶段 vs navtest 场景）。
- **一个要纠正的前提**：「预训练该涨分」对 Qwen 2VA **没有直接依据**。
  论文 Table 7 的 80.7 → 85.6 是 VQ 变体（Emu3 那条线）；官方 Qwen ckpt 名里有
  `nuplan_pretrained_8k`，证明他们做过，但那是 7B 非 ROSS，增益没有公开数字。

### 一个顺带翻出来的事实：ROSS 扩散分支几乎不提供学习信号

<span>比 `aligned_bf16` 与 `chain_sft_bf16` 两把 SFT 的逐步 loss：</span>

| 指标 | 两把的相对差（中位） | 说明 |
|---|---|---|
| `action_loss` | **1.49e-1**（第 1 步差 8 倍：18.46 vs 2.33） | 对权重高度敏感 |
| `ross_loss` | **1.79e-4**（p99 1.6e-3） | 几乎与权重无关 |

两把的起始权重差到「一把的 denoiser/VAE 是随机初始化的」，`ross_loss` 却只差 0.02%。
同 seed → 同数据顺序 → 同采样时间步与噪声，而模型对这个 loss 的贡献小到被淹没。
**也就是说 8000 步预训练实际上只训了动作 token 那一项**，扩散分支的目标函数
基本是个常数。这一点对「预训练到底在学什么」的解释很重要，也值得单独查
（VAE 全程 0.000% 位移 = 冻结，而 denoiser 在 SFT 里挪了 10.9% 却不改变 loss）。

### 输出轨迹的诊断：形状对、指向错

3000 个共同场景，比两把的输出轨迹（`action` 是每步增量 `[dx, dy, dyaw]`，8 步 × 0.5 s）：

| 指标 | `aligned_bf16` | `chain_bf16` | 变化 |
|---|---|---|---|
| 纵向总进度（均值） | 19.215 m | 18.802 m | −2.1% |
| 横向活动量 Σ\|dy\|（均值） | 0.337 m | 0.333 m | −1.1% |
| 转向活动量 Σ\|dyaw\|（均值） | 0.272 rad | 0.272 rad | −0.0% |
| 近乎直行的场景比例 | 74.5% | 74.8% | +0.3pp |
| **末点纵向误差 \|dx\|** | 2.002 m | 2.827 m | **+41.2%** |
| **末点横向误差 \|dy\|** | 0.702 m | 1.192 m | **+69.7%** |

**不是模式崩塌** —— 轨迹的统计形状几乎没变。变的是对人类轨迹的误差。
轨迹依然像人开的，只是**不对应当前这个场景**，也就是条件化断了。
PDMS 分项与之吻合：

| 分项 | `aligned_bf16` | `chain_bf16` | 相对 |
|---|---|---|---|
| EP 行进度 | 0.75325 | 0.62752 | **−16.7%** |
| DAC 可行驶区域 | 0.90277 | 0.81130 | −10.1% |
| TTC | 0.92985 | 0.85213 | −8.4% |
| DDC 行驶方向 | 0.96690 | 0.91577 | −5.3% |
| NC 无责碰撞 | 0.97427 | 0.92479 | −5.1% |
| C 舒适度 | 0.99992 | **1.00000** | +0.008% |
| PDMS | 0.81203 | 0.68026 | −16.2% |

**六项里五项一起掉，舒适度反而满分** —— 又平又顺、但不跟场景走。
（脚本 `submetrics.py` / `trajdiag.py`，都不占卡。）

### 四把 8000 步预训练的吞吐

| 配置 | runtime | s/步 | vs bf16 |
|---|---|---|---|
| bf16 | 19,847.8 s | 2.481 | — |
| HiF8-CTS-196 | 20,213.6 s | 2.527 | +1.8% |
| HiF8-DTS-168 | 20,361.3 s | 2.545 | +2.6% |
| HiF8-DTS-196 | 20,708.8 s | 2.589 | +4.3% |

两点要注意：

- **CTS-196（+1.8%）比 DTS-196（+4.3%）快**，而 DTS 本该更省（稳态单遍
  `npu_quantize`，CTS 要先求 amax）。可能是 DTS 的簿记开销（每步判断
  `observing_now`、roll 时批量 all_reduce）盖过了省下的那一遍，也可能是机器间差异
  （四把跑在四台不同机器上）。**要分清只能同机背靠背重跑，目前这组数据分不出来。**
- DTS-168 比 DTS-196 快，量化的 Linear 少 28 个，方向符合预期。

### 四条预训练曲线的全长度对比（32 个窗口）

四条链各跑满 8000 步，`action_loss` 的 1000 步窗口均值，相对 bf16：

| 区间 | bf16 | DTS-196 | DTS-168 | CTS-196 |
|---|---|---|---|---|
| 1–1000 | 2.2001 | +0.07% | +0.23% | +0.20% |
| 1001–2000 | 1.4892 | −0.12% | −0.05% | −0.18% |
| 2001–3000 | 1.4071 | −0.05% | −0.10% | −0.08% |
| 3001–4000 | 1.3742 | −0.00% | −0.04% | −0.03% |
| 4001–5000 | 1.3344 | +0.05% | +0.07% | +0.03% |
| 5001–6000 | 1.3071 | +0.07% | +0.10% | +0.02% |
| 6001–7000 | 1.2829 | +0.07% | +0.03% | +0.01% |
| 7001–8000 | 1.2686 | +0.16% | +0.17% | +0.14% |

**32 个窗口的差全部在 ±0.23% 以内。** 中段（2000–7000）三条 HiF8 一致贴在 bf16 上，
正负交替、量级 0.0–0.1%。

**一个值得记录的观察（不是结论）**：末段 7001–8000 三条 HiF8 **同向**偏高
0.14–0.17%，方向一致而非随机。可能是量化误差在 lr 衰减到 1e-6 附近、
loss landscape 变平时的轻微体现。但 0.15% 的量级加上没有方差基线，
只能当观察记录。**而且它没有传导到 PDMS** —— 端到端四个数全距 0.25%，
DTS-168 甚至比 bf16 高 0.01%。

脚本：`cmp4.py <窗口> <tag>=<log> ...`（`/mnt/sfs_turbo/zyc/`）。

### 预训练全长度对比：HiF8-DTS-196 与 bf16 等价（8000 步实测）

2026-11-13/14，两把各跑满 8000 步（a54 bf16 / a53 DTS-196，同配置同数据，
唯一变量是量化）。**这是目前 HiF8 等价性最强的证据** —— 之前的验证全在 SFT 阶段
（4000 步、lr 1e-5），预训练是更长、lr 高 20 倍、数据量大得多的场景，
量化误差有更多机会累积。

`action_loss` 的 1000 步窗口均值：

| 区间 | bf16 | DTS-196 | 差 | bf16 grad_norm | DTS grad_norm |
|---|---|---|---|---|---|
| 1–1000 | 2.2001 | 2.2016 | +0.1% | 9.846 | 9.603 |
| 1001–2000 | 1.4892 | 1.4874 | −0.1% | 1.513 | 1.508 |
| 2001–3000 | 1.4071 | 1.4065 | −0.0% | 1.337 | 1.332 |
| 3001–4000 | 1.3742 | 1.3742 | −0.0% | 1.296 | 1.292 |
| 4001–5000 | 1.3344 | 1.3350 | +0.0% | 1.341 | 1.336 |
| 5001–6000 | 1.3071 | 1.3081 | +0.1% | 1.433 | 1.431 |
| 6001–7000 | 1.2829 | 1.2838 | +0.1% | 1.565 | 1.558 |
| 7001–8000 | 1.2686 | 1.2707 | +0.2% | 1.651 | 1.644 |

**八个窗口的差都在 ±0.2% 以内，没有累积漂移**，`grad_norm` 全程贴合。
后半段有极轻微的扩大趋势（−0.0% → +0.2%），量级太小，单条 run 说明不了什么。

两把的 ckpt 都是 1,569 张量 / 2 分片 / 6.77 GB、前缀分布相同 ——
再次印证 HiF8 的 ckpt 与 bf16 结构一致。

### 吞吐：HiF8-DTS 比 bf16 慢 4.3%（8000 步实测）

| | runtime | s/步 |
|---|---|---|
| bf16 | 19,847.8 s（5.51 h） | **2.481** |
| HiF8-DTS-196 | 20,708.8 s（5.75 h） | **2.589** |

**这更正了先前的 50 步估计。** 50 步测出 2.832 vs 2.878（1.6%），但 50 步的吞吐噪声
本身就有 7.5–9.2%，足以把 1.6% 完全淹掉。8000 步的 4.3% 可信得多。

保留：两把跑在不同机器（a54 vs a53），同规格但机器间差异无法排除。
要彻底干净得同机背靠背，那要 11 小时。当前这个 4.3% 当「量级」用是可靠的。

**排期参考：8000 步预训练 5.5 小时**（bf16）/ 5.75 小时（HiF8），
不必再按先前的 6.3 小时留余量。

### token 长度（回答 max_length 该设多少）

真实样本实测：最小 342、中位 346、p99 354、**最大 360**。所以 1400 和 8192 **都不截断**，
之前写 1400 没有污染任何结果。但训练侧统一用 8192（与 SFT、官方一致），评测侧 1400
（与官方 `inference_2va.sh` 一致）。

### 链条（预训练 → 读 ckpt 续 SFT → 评测）

**已走通**（bf16，预训练 20 步 → SFT 20 步 → mini 评测 PDMS 0.3629；0.36 只证明管线出分，
不代表精度）。过程中修掉三个真实问题：

1. **预训练输出缺 processor 配置** → `AutoProcessor` 退到 `AutoVideoProcessor → AutoConfig`
   兜底路径，而那条路径的 video processor 注册表里没有 `qwen3_vl`，报
   "Transformers does not recognize this architecture"。**修法**：训练成功后从骨干目录补
   `video_preprocessor_config.json` / `chat_template.json` / `configuration.json`
   —— 已加进 `run_stage1_pretrain.sh`
2. **评测必须进容器**（见第 2 节第 2 条坑）
3. **评测门槛不能看退出码** —— `eval_ckpt.sh` 末尾 `echo EVALDONE_` 是无条件执行的。
   真实判据是 `Final average score of valid results: <数>`，拿不到就是失败

ckpt 结构：1,569 个张量、2 个分片、6.77 GB，顶层前缀 `denoiser 695 / model 625 / vae 248 / lm_head 1`。
`train_qwen_vla.py:620` 的逻辑是「`output_dir` 里有 `checkpoint-*` 就自动
`resume_from_checkpoint=True`」。

---

## 7. 抽帧与续训（根因 + 解法，已闭环）

### 观测：续训没有逐位接上

A 跑 50 步（`save_steps=30`）→ B 从 `checkpoint-30` 续到 50 步。B 确实走了 resume
（日志有 `Checkpoint found, resuming`，且只跑 20 步，说明 `global_step` 恢复正确）。

| 指标 | A[31..50] 均值 | B 均值 | A[1]（初始） | 逐位一致 |
|---|---|---|---|---|
| action_loss | 5.866 | 5.830 | 19.16 | 0/20 |
| ross_loss | 0.15238 | 0.15269 | 0.2289 | 0/20 |
| grad_norm | 43.96 | 43.71 | 515.98 | 0/20 |

水平差 0.6%、离初始极远 → **权重和优化器状态确实恢复了**。但第 1 步就分叉。

### 那把尺子：分叉量级说明问题不在数值

重计算开关那次实验给了标定值：**纯浮点舍入的首次分叉是 1.5e-7**。
这次是 **2.2e-2**，高四五个数量级 —— 不是数值误差，是输入不同。

再分一层：`ross_loss` 不同可以用扩散头采样噪声解释，但 `action_loss` 是 LM head 在
FAST 动作 token 上的交叉熵、**不依赖扩散噪声**，它也差了 → 只能是喂进去的那批数据不同。

### 根因（定位到行）

```
data_qwen_vla.py:760   self.rng = random.Random(seed)         # dataset 的私有 RNG
data_qwen_vla.py:1006  cur_idx = self.rng.randint(low, high)  # 从 200 帧的段里抽当前帧
```

这个状态随 `__getitem__` 的调用次数推进，而且三条都成立：

- 是**私有实例**，不是全局 `random` 模块 —— trainer 的 `_load_rng_state` 里
  `random.setstate()` 碰不到它
- **从不进 checkpoint**
- `num_workers=12` 时被 fork 成 12 份，各自独立推进

所以 B 重启后 worker 带着「种子 42 的初始状态」开工，第一个批次抽的帧是
「第 1 批次该抽的」，而不是 A 第 31 批次那些 —— **同样的段、不同的帧**。

顺带确认：HF 的 `_load_rng_state` 是在取出 inputs **之后**、`training_step` **之前**
执行的，所以扩散头采样用的是已恢复的 RNG，**模型侧没问题**。

### 一个被这次调查翻出来的事实：原实现的数据流依赖 num_workers

`self.rng` 在 `__init__` 里播种、fork 在那之后，所以每个 worker 拿到一份**起始状态相同**
的副本，全局序列是 W 份相同序列的交织。换 W 就换了交织方式。实测（顺序取前 24 个样本，
比 `input_ids` 指纹）：

| | num_workers 0 vs 2 | 0 vs 4 |
|---|---|---|
| 原实现 | 相同 **8/24** | 相同 **6/24** |
| 无状态模式 | **24/24** | **24/24** |

**后果：「与论文逐字节对齐」在原实现下本来就不可达** —— 还要同时匹配作者的
`num_workers`（官方 Emu3 预训练的 `training_args.bin` 里是 **16**，我们用 **12**），
以及 worker 的调度顺序（不保证确定）。所以「和论文对齐」只能是**同数据、同窗口、
同分布、同超参**，不可能是同字节流。

### 解法：做成开关，默认不动原实现

```
FRAME_SAMPLING_STATELESS=0（默认）  原实现：推进式私有 RNG
FRAME_SAMPLING_STATELESS=1          无状态：按 (种子, epoch, 索引) 现算

_fs = (int(self.base_seed) * 1000003 + int(self.epoch)) * 1000003 + int(i)
cur_idx = random.Random(_fs).randint(low, high)
```

配一个十行的 `_DatasetEpochCallback`（`train_qwen_vla.py`），在 `on_epoch_begin` 把
`state.epoch` 写进 dataset。`persistent_workers` 未设（HF 默认 False），worker 每个
epoch 重新 fork，所以值能传进去。

备份：`data_qwen_vla.py.bak-prerngfix`、`train_qwen_vla.py.bak-prerngfix`。

### 验证结果（全部实测）

| 验证项 | 结果 |
|---|---|
| **续训逐位接上**（A[31..50] vs B[1..20]，`FRAME_SAMPLING_STATELESS=1`） | **20/20** × 三项指标 |
| **同配置重跑逐位一致**（A vs A2，50 步） | **50/50** × 三项指标 |
| 开关生效（新 A 第 1 步 18.876 ≠ 原实现 19.1603） | 是 |
| **抽帧开关对 SFT 是空操作**（两把 12 步只差开关） | **12/12** × 三项指标 |
| 纯 CPU：乱序 + 插入额外调用后同索引内容相同 | 模式 0 **否** / 模式 1 **是** |
| 分布与原实现无法区分 | 均值 74.40 vs 74.50、卡方 13.1 vs 18.6（15 桶，0.1% 临界 36.1）、lag1–5 自相关 <0.01、跨 epoch corr 0.0003 |

只改了抽帧这一处就逐位一致 → **反证权重、优化器一二阶矩、lr 调度位置、RNG 状态
本来就都恢复好了**，唯一的缺口就是那个私有 RNG。

### 两个判据上的提醒

- **不要验「和改前 loss 一致」** —— 那是必然失败且无信息的判据，改动改的正是抽哪一帧，
  连全新一把的第 1 步都会不同。
- `cmp_resume2.py` **不通用**：它的逻辑是「第二个文件是 20 步的续训」，拿 A[31..50] 对
  B[1..20]。比两把同步数的跑要用 `cmp_same.py`（按同一步号对齐）。我曾因为混用这两个
  脚本，把 A↔A2 的 50/50 误读成 0/20。

### 还没验到的边界

分布检验测的是边缘分布和沿样本索引的 lag-k 自相关。shuffled sampler 下一个 batch 里的
96 个样本**不是相邻索引**，所以 batch 层面的联合结构没有直接实测。两边都是独立均匀抽取
的话应当等价，但那是推理。真要排查预训练结果不如预期时，这一项可以列进候选 ——
不过「`num_workers` 12 vs 官方 16」那一项的嫌疑更大，而它在任何模式下都存在。

---

## 8. 实验编排（已全部完成）与下一步

三组交叉设计，共十条链，2026-11-13 17:54 起、2026-11-14 06:00 全部出分。
PDMS 结果见第 6 节「端到端 PDMS」。

### 第一组：两阶段同精度（四条，唯一变量是量化配置）

| 机器 | 配置 | 量化 Linear | 启动 | 脚本 / 日志 |
|---|---|---|---|---|
| a54 | bf16 | — | 17:54:58 | `chain_bf16_full.sh` → `chain_bf16.log` |
| a53 | HiF8-DTS-196 | 196 / 未量化 301 | 17:58:28 | `chain_dts196_full.sh` → `chain_dts196.log` |
| ott9-a51 | HiF8-DTS-168 | 168 / 未量化 329 | 20:10:01 | `chain_dts168_full.sh` → `chain_dts168.log` |
| a52 | HiF8-CTS-196 | 196 / 未量化 301 | 20:12:54 | `chain_cts196_full.sh` → `chain_cts196.log` |

每条都是 **预训练 8000 步 → SFT 4000 步 → full 评测**，约 9.2 小时。
共同配置：`GC=False` 两阶段统一、`SAVE_STRATEGY=no`、抽帧用默认（原实现）、
数据同一份官方 pickle、全部超参一致。
预训练阶段四条曲线几乎重合，32 个窗口均值差都在 ±0.23% 内。

### 第二组：共用 bf16 预训练权重，只在 SFT 换 HiF8（四条）

| 机器 | 队列 | 脚本 |
|---|---|---|
| a54 | DTS-196 → DTS-168 | `sft_from_bf16pt.sh dts196:dts:lm_head dts168:dts:lm_head,o_proj` |
| a53 | CTS-196 → CTS-168 | `sft_from_bf16pt.sh cts196:cts:lm_head cts168:cts:lm_head,o_proj` |

起始权重都是 `runs/chain_pt_bf16`（a54 第一组产出的那份，在共享盘上）。
每个配置 SFT 4000 步 + full 评测，两个串行约 5.8 小时。
日志 `sft_from_bf16pt_<hostname>.log`，输出 `runs/sftpt_<tag>`。

### 第三组：HiF8 预训练权重 + bf16 SFT（两条）

| 机器 | 配置 | 脚本 |
|---|---|---|
| ott9-a51 | DTS-168 预训练 → bf16 SFT | `pt_then_sft.sh dts168pt_bfsft:chain_pt_dts168::lm_head,o_proj` |
| a52 | DTS-196 预训练 → bf16 SFT | `pt_then_sft.sh dts196pt_bfsft:chain_pt_dts196::lm_head,o_proj` |

日志 `pt_then_sft_<hostname>.log`，输出 `runs/ptsft_<tag>`。
这一组是**噪声量级的意外来源** —— 两臂只差预训练的 HiF8 模式却相差 0.68%，
成了否掉第二组「0.6% 增益」的关键证据（见第 6 节）。

### 三组交叉能回答什么

| 想问 | 看哪两组 |
|---|---|
| 全程 HiF8 的代价 | 组一 vs 组一的 bf16 臂 |
| 量化只在 SFT 的代价 | 组二 vs 组一的 bf16 臂 |
| 量化只在预训练的代价 | 组三 vs 组一的 bf16 臂 |
| 误差主要在哪个阶段累积 | 组二 vs 组三（都与组一 bf16 臂比） |

实际答案：三组的均值分别是 −0.10% / +0.60% / −0.06%，而组内极差 0.24–0.83%
与组间差同量级 —— **没有任何一个阶段能分离出可辨识的量化代价**。

### 一个事后才看清的方法论问题

**十个数不能直接当十个独立样本。** 组二四条共用同一份 `chain_pt_bf16` 预训练权重，
组三两条各自共用自己的预训练权重，只有组一四条是从头独立的。
所以「十条链 std 0.0032」描述的是这个设计下的离散度，不是 PDMS 的 run-to-run 方差。
要后者必须同配置换种子重跑（`variance_run.sh`）。

### `SAVE_STRATEGY=no` 的风险（十条跑全都这么跑的）

十条跑全都没有中间 ckpt，中途被停就从零重来。实际发生过：a54 的卡被抢过两次、
a51/a52 是从别人手里清出来的、a52 上还因为我自己重复启动把一条链的日志冲掉
（结果数据活着，分数是后来从 `/tmp/chain_eval_*.out` 里捞回来的）。
想加保险的话，`SAVE_STRATEGY=steps SAVE_STEPS=1000` 与抽帧开关**独立** ——
开它不需要改抽帧（续训虽不逐位一致，但水平差 0.6%、轨迹正常接着走）。

**捞分的两条退路**（日志被冲掉时）：
`/tmp/chain_eval_<tag>.out`、`/tmp/eval_<sftpt|ptsft>_<tag>.out` 里
`grep "Final average score of valid results:"`。

### 待办（按重要性）

1. **在修正后的超参上重做 HiF8 vs bf16** —— 此前十条全测于预训练 lr 2e-4 的坏工况。
   已排进队列：a53 `dts196_lr2p5e5`（预训练 2.5e-5）、a54 `dts196_lr5e5`（预训练 5e-5），
   都排在各自的 7B 链条之后，靠 `QUEUE_7b_*_(DONE|FAILED)` 标记触发
2. **7B 全链条**（已排队，`chain_ptlr_7b.sh`）—— 骨干 Qwen2.5-VL-7B，
   与 2B 同配置，预训练 lr 取各机对应值。阶段零有 20 步烟测只验显存，
   装不下就写 `CHAIN_*_FAILED` 让 DTS-196 照跑，**不做回退**。
   对照是 `w7b_bf16`（7B 无预训练 0.8407）
3. **继续往上探预训练 lr** —— 2.5e-5 → 5e-5 还在涨（+1.7 → +4.3 点），尚未见顶。
   下一档可试 1e-4（注意 2e-4 已确认过头）
4. ~~用中间 ckpt 画「预训练步数 vs PDMS」曲线~~ —— **已不可做**：中间 ckpt
   已于 2026-11-15 清理（释放 583 GB），只留了每个 run 的 `checkpoint-8000`。
   要画得重跑预训练。lr 已定稿、主问题不依赖这条曲线，取舍合理
5. **测 PDMS 的 run-to-run 方差**（`variance_run.sh`，2.6 小时）——
   现在各点差距（+1.7/+4.3 点）远大于此前估计的噪声量级，优先级下降，
   但要给「HiF8 ≡ bf16」定量仍然需要它
6. 查 CANN 有没有 `npu_rms_norm_quant_v2`（见第 9 节 BaiZe）—— 吞吐最直接的一招
7. 把 `_HiF8CTSMatMul` 改成注册的自定义算子 + `torch.compile`（见第 9 节 torchtitan）
8. 逐个验证剩下八个确定性开关，尤其 `FLAGS_npu_storage_format`
9. 正经测 `TASK_QUEUE_ENABLE` 的吞吐价值（500 步 × 2，同机）
10. 考虑把 `dataloader_num_workers` 从 12 改成官方的 16（独立的对齐项，
    而且它直接影响原实现下的数据流）
11. 改掉 `run_sft_navsim_lr.sh` 里那条错注释（「官方实测 lr=2e-4、非 ROSS」）——
    正确说法是两个镜像两个 run，ROSS 版是 1e-5。**改之前确认没有作业正在执行它**

### 已不再需要做的

- ~~lr 2e-4 下的 HiF8 对比~~ —— `lr2e4_dts196`(1373 步) 与 `lr2e4_sft_dts196`(1492 步)
  当时被外部 SIGTERM 杀掉（不是崩、不是 OOM、不是发散）。既然 2e-4 已被否掉，
  不必补跑
- ~~「位移 vs PDMS」曲线~~ —— 位移假设已被证伪，画它没有意义；
  改为画「预训练步数 / 预训练 lr vs PDMS」

---

## 9. 两个参考实现

两份都读过了，结论是**它们各自解决了一半,而且都没有实测吞吐数字**。

### 9.1 torchtitan（本地 `C:\works\didi\torchtitan-npu_0902`，分支 `hif8-dts-1020`）

文档 `docs/hif8/hif8_optimization_summary.md`（462 行）。它的办法是**降低派发层级**：

- 一个**不透明的自定义算子** `hif8_kiwi::delayed_scale` + `torch.compile`
- 簿记按 `(state.data_ptr(), slot)` 索引
- scale 用裸张量做载体（不是 buffer，所以不进 ckpt）
- `_save_scale_op_under_selective_ac` —— 让 AC 重算时复用 forward 算出的 scale，
  **这是它处理"重计算导致量化做两遍"的办法**

**但那份文档自己写明：没有性能数字，"本分支的任何机制都还没有在 NPU 上跑出过完整的 loss 曲线"。**

### 9.2 BaiZe（私有 repo `EthanZhangYC/llm_train_system_snapshot`，分支 `develop`，
路径 `train_bf16_mxfp8_hif8/BaiZe`）

这是在 NPU 上真跑过的实现（PanGu 血统，`from PanGu.models.low_precision...`）。
拉取方式：Windows 凭据管理器里有 token，`git` 能直接握手，
`git clone --filter=blob:none --no-checkout -b develop` + `sparse-checkout set train_bf16_mxfp8_hif8/BaiZe`
（8.4 MB）。**服务器上没有 `gh`、也没有 GitHub connector，只能在本机拉。**

核心在 `baize/core/low_precision/`：

| 文件 | 内容 |
|---|---|
| `functions/linear_function.py` | `QuantLinearFunction` 等，**全是 `torch.autograd.Function`** |
| `recipe/quantizer.py` | 686 行，`HiF8DTSQuantizer`（**`HiF8CTSQuantizer.__init__` 直接 `raise NotImplementedError` —— BaiZe 的 HiF8 只有 DTS**） |
| `fusions/fused_rmsnorm_quant.py` | **把量化折进 RMSNorm 的 kernel** |
| `fusions/fused_swiglu_quant.py` | 把量化折进 SwiGLU 的 kernel |
| `state_manager.py` | 757 行，scale 的全局状态与 ckpt 存取 |
| `layer_policy.py` | 按 module 名/类型拉黑，粒度与我们的 `HIF8_IGNORE` 相同 |

**它和 torchtitan 不是一条路：BaiZe 全程 `autograd.Function`，没有 `torch.library` /
`custom_op` / `torch.compile`，派发层级和我们完全一样。** 它降低开销的办法是
**让量化不以独立算子存在**：

1. **量化折进前一个算子的 kernel**
   - `torch_npu.npu_rms_norm_quant_v2(x, w, scale, div_mode=True, dst_dtype=hifloat8)`
   - `torch_npu.npu_swiglu_group_quant(..., quant_mode=2, scale=inv_scale)`
   - `_npu_rms_norm_backward_quant`（norm 反向把 dx 量化折进去）

   Linear 收到的已经是 `HiF8Tensor`，`QuantLinearFunction.forward` 里
   `if not isinstance(input_, QuantizedTensor)` 直接跳过量化。
   **稳态下每个 Linear 的输入量化是 0 个额外 kernel、0 次额外 Python 下发。**
   我们现在是每个 Linear 独立调量化算子，profiling 数到 980 个量化 kernel/步
   —— 这一项可以基本归零，**是目前性价比最高的一招**。

2. **`--cache-quantized-weight`（默认开）：权重每迭代只量化一次并缓存。**
   权重常驻 HiF8（`parameter.data` 就是 `HiF8Tensor`，bf16 影子放 `_bf16_weight_for_ckpt`
   只用于存 ckpt）。我们现在每次 forward 量化一遍权重，重算再来一遍。

**重计算导致量化做两遍，BaiZe 对 HiF8 没有任何处理。** 它的
`use_full_recompute` 分支（只产 rowwise 不产 colwise）和 `rowwise/colwise` 参数
**都只在 MXFP8 路径上生效** —— `HiF8DTSQuantizer.quantize` 和
`FusedHiFQuantRMSNormFunction.forward` 的签名里有这些参数，**函数体一次都没用**。
原因：HiF8 是 per-tensor scale，同一份字节转置着读就行；MXFP8 沿固定轴分块，
转置后分块边界变了才必须单独量化一份。

未融合那条路（`_delayed_quantize`）的 kernel 数和我们一样（1 个/张量），
它也没有特殊手段 —— 稳态单遍 `npu_quantize`、记 amax 那步单遍 `npu_quant_max`、
amax 只在播种时单独 all_reduce、之后 roll 时用 `_foreach_copy_` 批量下发。
**这三条我们已经一样了。**

两个顺带的印证：BaiZe 的乘除方向（`div_mode=True` 用 scale 除、`quant_mode=2` 用
inv_scale 乘）和我们查到的完全一致；"roll 时 rebind 新 tensor 而不是 in-place 写回"
它注释写得很细（in-place 一开重算就炸：`state["scale"]` 会被持有者
`untyped_storage().resize_(0)` 释放），和我们 `_refresh()` 的做法相同。

**BaiZe 也没有实测吞吐数字** —— 仓库里只有 MFU 的采集管线，没有 bf16/HiF8 对照结果。

---

## 10. 踩过的坑（别再踩）

1. **不要改正在被执行的脚本。** bash 是边读边执行的。给 `run_sft_navsim.sh` 加 `SEED` 时
   四把训练正在跑它，结果全部变成假的 `TRAIN_EXIT=1`（训练其实成功了）。
   要改就改副本（`run_sft_nodet.sh`、`run_stage1_pretrain_lrtest.sh` 就是为此存在的）。
2. **`pkill -f` 会杀掉自己。** ssh 命令行里含那个模式串，`pkill -f` 连自己一起杀。
   用 `ps -eo pid=,args= | awk '!/awk|ps -eo/ && ...'` 并显式排除自身，或 `pgrep -x`。
   `grep -c` 同理会把自己数进去。
3. **`npu-smi` 的输出格式别猜。** 我用 awk 猜格式，读出"0 个占卡"，而其实 lh 的 vLLM
   正压着 8 张卡每张 92 GB，白烧两分钟卡时。可靠写法：
   占卡数 `awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+/{c++} END{print c+0}'`；
   HBM `grep -oE '[0-9]+ *\/ *98304' | grep -oE '^[0-9]+' | sort -rn | head -1`。
   **OOM 日志要看"已分配"那个数** —— 如果它很小而"剩余"也很小，是别人占着，不是我们太大。
4. **退出码不等于成功。** 脚本末尾的无条件 `echo` 会让 `$?` 变成 0。门槛要用业务判据
   （`TRAIN_EXIT=0` **且** 训练进程数为 0；评测要有 `Final average score`）。
5. **两个 run 不要共用输出目录。** A 的 `TRAIN_EXIT=1` 被追加进 B 的新日志，骗过了评测门槛。
6. **后台采样进程必须重定向 stdout。** `hp=$(hbm_start)` 让采样进程继承了命令替换的
   stdout，`$(...)` 永不返回，整条链卡死。写 `>/dev/null 2>&1 &`。
7. **tqdm 的瞬时 ETA 不能当结论。** 我据此说 a54 要到 10:10，实际 08:00 结束。
   同一行里的平均速率（s/it）才是稳定的尺子。
8. **HF 的 `loss` 四舍五入到 4 位小数**，分辨率只有 1e-4，会盖掉真实分歧。
   比对要用我们自己打的 `action_loss` / `ross_loss`（全精度）和 `grad_norm`。
9. **内联 heredoc 里的反斜杠会被吞。** 补丁脚本写成文件 `scp` 过去跑，
   别内联;Python 字符串末尾也不能有反斜杠（会转义收尾引号），用 `chr(92)` 拼。
10. **删东西之前先验判据。** 我曾在未验证判据的情况下自动删掉 lmc 的 6 个归档。
    清理 `runs/` 时的做法是：判据按 `trainer_state.json` 的 `global_step` 和
    「有没有对应的 `eval_*` 产物」算出来，先 dry-run 列出每个文件，
    反向检查"要删的里面有没有步数在 200~999 的"，确认为空再执行。
    脚本 `/mnt/sfs_turbo/zyc/cleanup_runs.py`（默认 dry-run，`--apply` 才删）。
11. **别和别人跑同一张卡；只 kill lh 的程序、不动容器；别 clone 到服务器上;
    占卡程序启动前要先获得用户许可。**
    `lh-vllm` 是跨 a53/a54 的 DP 部署 —— kill 一台会让另一台残缺,要提前说明。
    `zf-test` 是 zf 的（a51/a52 上出现过），不是 lh 的，别动。
12. **共享盘上的文件不要用 glob 删。** 队列脚本的日志放在 `/mnt/sfs_turbo/.../`、
    文件名带 hostname，但启动命令里写了 `rm -f sft_from_bf16pt_*.log` ——
    结果 a53 启动时把 a54 的日志删了（内容会被后续写入重建，没丢数据，但一度看起来
    像两台在跑同一个配置）。共享目录里只删自己那一个确切文件名。
13. **跨机取文件走共享盘，别用 scp。** 服务器之间没有互相授权（a51 也不行，见坑 2），
    从 a54 scp a53 的日志会静默失败（文件不存在、也不报错）。日志都在
    `/mnt/sfs_turbo` 上，直接读路径。
14. **`ls <共享目录>/*.log | head -1` 不可靠。** 多台机器写同一个目录时，
    它挑到的可能是别的机器的文件。要按 hostname 精确取。
15. **门槛的必需文件列表要和产出方一致。** 我给 `chain_*_full.sh` 的 `check_ckpt`
    加了 `video_preprocessor_config.json`，但自动补这个文件的步骤只加在
    `run_stage1_pretrain.sh`，没加到 `run_sft_navsim.sh` —— bf16 链条在阶段二
    ckpt 完好（1569 张量 / 6.77 GB）的情况下被自己的门槛拒掉，停在评测之前。
    而以前成功评测过的 `aligned_bf16`、`w7b_bf16` 也都没有这个文件，
    **证明评测并不需要它，是门槛加错了**。
16. **改不动正在跑的脚本时，反过来让文件出现。** 三条还在跑的链条，
    `check_ckpt` 在脚本开头就已解析进 bash 内存，改文件对已运行的实例无效；
    而 `run_sft_navsim.sh` 正被两个作业执行，改它会把它们搞炸。
    解法是挂一个 watcher（`fix_sft_cfg_watcher.sh`）每 5 秒扫 `runs/` 下的 SFT 输出，
    `index.json` 一落地就把配置补进去 —— 纯追加、不碰任何运行中的东西。已验证生效。
17. **监控的判据不要放裸 `Error`。** ray 每次评测都会打
    `Failed to establish connection to the metrics exporter agent`、
    `ERROR 1: PROJ: proj_create_from_database`、`No module named 'acl'` ——
    全是无害的。放了裸 `Error` 的监控把这些当成失败连报两次假警。
    评测的正确判据只有两个：`Final average score of valid results: <数>` 表示成功，
    `RuntimeError` / `FileNotFoundError` / `RayTaskError` / `AssertionError` 表示真失败。
18. **容器时区比宿主机晚 8 小时**（UTC vs Asia/Shanghai）。容器里打的
    `2026-11-13 17:36` 对应宿主机 `11-14 01:36`。看日志时间戳别被这个误导。
19. **评测的 json_output 每个场景产出 3 个文件**（`<token>.json` 动作、`<token>.png`
    BEV 图、`<token>_agent_human_traj_diff.json` 轨迹差）。所以 12,146 个场景
    对应 36,438 个文件 —— 我一度以为是跑了三遍。数场景要只数 `*.json` 里不带
    `_agent_human_traj_diff` 的那部分，或直接看日志里的
    `Starting pdm scoring of N scenarios`。

---

### 2026-11-15 新增的四条坑（都是当天实际踩的）

**1. Bash 工具报「超时转后台」不等于命令没执行。**
a53 上我据此补发了一次链条启动，结果两个实例相差 28 秒同时起跑：
第二个实例的 `rm -rf "$PT"` 删掉了第一个正在写的 `train.log`（第一个的文件句柄
指向已 unlink 的 inode，之后它写的东西谁也看不见），然后第二个撞
`EADDRINUSE 23941` 秒死，把 `TRAIN_EXIT=1` 和 `CHAIN_*_FAILED` 写进新建的日志里。
于是：第一个白跑 9 分钟占 8 张卡、注定被自己的链条判死；监控读到那个 FAILED
报出假的「全部完成」。
**规矩：发一次 → 只验证实例数 → 绝不补发。** 验证用
`ps -eo args= | grep '[x]xx.sh' | grep -vE 'bash -c|setsid' | wc -l`。

**2. 服务器之间不能 ssh —— 包括 a51。**
本文先前写「除了 a51→其他三台」，**那是错的**。实测：
```
a51 上: ssh a53 "echo OK"
  -> ssh: Could not resolve hostname a53: Name or service not known
```
`a53`/`a54` 只是**本地** ssh config 的别名，a51 上没有对应 Host 条目，
`~/.ssh/` 里也只有 `authorized_keys`、没有出站密钥。
**后果：服务器端脚本无法跨机判断「某脚本是否正在别的机器上执行」。**
所有跨机判定必须在本地做。曾有一版守候器把这个判断放在 a51，
它的 `busy()` 恒返回 99+99，永远等不到 0（失败安全，但完全无效）。

**3. 多条队列不要等同一个标记。**
a54 上 `CHAIN_ptlr5e5_DONE` 一出，三条队列同时醒来抢卡：
`gbs48_lr5e5`（01:25 卡空 → 01:26 静置后又被占 → FAILED）、
`7b_lr5e5`（01:28 卡空 → 01:30 静置后又被占 → FAILED）、
`dts196_lr5e5`（等的是 `QUEUE_7b_*`，01:30 拿到标记、抢到卡、正常跑）。
**三者互相搅掉了对方的静置窗口，最后只有一条活下来。**
正确做法：首尾相接，而且等**排队器**的 `QUEUE_*_(DONE|FAILED)`
而不是链条的 `CHAIN_*` —— 排队器无论成败一定写标记，
而链条在排队器就失败时根本不会产生标记（`gbs48_lr1e4` 就因此空等到超时）。

**4. 要「只跑阶段一、不跑后续」，必须在阶段一还在训练时就杀父脚本。**
我写的守候器是「等 `TRAIN_EXIT` 出现再杀父脚本」，15 秒轮询一次 —— 没赶上：
链条从 `TRAIN_EXIT` 到启动阶段二只用了不到 15 秒（`ckcheck` 几秒 +
`wait_free` 卡立刻就空）。结果阶段二起来了，而且**杀父脚本不会杀掉
`docker exec` 启动的容器内进程**，于是留下一个无人收尾的孤儿 SFT 继续占卡。
**正确做法**：趁阶段一还在跑就杀父脚本 —— 被孤立的 `docker exec` 会自己跑完、
正常写出 ckpt，而没有父脚本就不会有阶段二。

## 11. 分析脚本索引

都在 `/mnt/sfs_turbo/zyc/drivevla-w0-zyc/`（除注明外）：

| 脚本 | 用途 |
|---|---|
| `ck.py` `ckdiff.py` | ckpt 完整性（index/分片/逐字节）与差异比对 |
| `cmp_det.py`（`tools/zyc/`） | 确定性比对（逐位） |
| `paired2.py` `explain_t.py` | 配对 t 检验与显著性说明 |
| `submetrics.py` | PDMS 分项（NC / DAC / DDC / EP / TTC / C） |
| `prof_analyze.py` `prof_pipeline.py` `prof_free.py` | profiling 分析（算子、流水、空隙） |
| `verify_dts.py` | DTS 机制验证 |
| `inventory.py` `cleanup_runs.py`（`/mnt/sfs_turbo/zyc/`） | `runs/` 盘点与清理 |
| `cmp_gc.py` `cmp_resume2.py`（`/mnt/sfs_turbo/zyc/`） | 重计算开关、ckpt 续训的 loss 比对 |
| `measure_len2.py`（`/mnt/sfs_turbo/zyc/`） | 真实样本 token 长度 |
| `read_args2.py`（`/mnt/sfs_turbo/zyc/`） | 读官方 `training_args.bin`（带桩模块绕过缺失的类） |
| `fix_official_pickle.py`（`/mnt/sfs_turbo/zyc/`） | 官方 pickle 的 action 转 ndarray |

---

## 附：2026-11-14 凌晨补记的两条坑

20. **启动命令超时被转后台 ≠ 没启动。** 在 a52 上拉 CTS-196 链条时，组合命令超时、
    日志看不到，我判断"没起来"就重启了一次 —— 实际两个实例都在跑。两个写同一个
    输出目录和日志，第一个跑完全程，第二个几小时后醒来发现卡被下一个队列占着才退出，
    顺手把链条日志清了（脚本开头有 `rm -f`）。**结果数据没丢**（评测 12,146 场景、
    PDMS、ckpt 都在，各阶段的 `train.log` 也在），丢的是链条自己的流程日志。
    判断"有没有起来"要看进程表（`ps -eo pid=,args=`），不要只看日志文件。
21. **链条末尾的 FAILED 可能来自重复实例。** `!! 链条中断: 有 N 个进程占卡`
    这句只出现在脚本**开头**的卡检查里。如果它出现在结果已经拿到之后，
    那是第二个实例在后来启动并退出，不是结果有问题。
