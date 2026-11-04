## 附录 —— Qwen3-VL backbone 训练中保留的加速特性（2026-11-04）

范围：MFU 最优点上的 Qwen3-VL-8B ROSS 配置（L≈1774，
`--max_pixels 903168`，单卡 bs 24–27，真实 NAVSIM 20× 子集）。Qwen3
wrapper 复用 Qwen2.5-VL 的 `npu_fused_ops` wrapper 和启动器
默认值，所以在 Qwen2.5 路径上测得的特性在这里同样生效，并标记为
*（Q2.5）*；只有融合 gate/up 在 Qwen3-VL 上重新测过 *（Q3）*。

### 保留

| 特性 | 开关 | 位置 | 实测效果 |
|---|---|---|---|
| Valid-token CE | `ROSS_VALID_CE=1`（启动器默认） | LM head | *（Q2.5）* bs40 上 +2.6%；fp32 logits 瞬态 10.3 GB → 0.55 GB；稳定 batch 40 → 64（200/200；bs72 在 logging all-gather 上失败） |
| NPU FlashAttention-2 | `--attn_type fa2` | text TND + ViT dense | *（Q2.5）* bs40 上 26.58 → 48.05 samples/s（+83%）；*（Q3）* L>2048 的 causal-mask 修复（`72fd940`） |
| 融合 RMSNorm | `ROSS_NPU_RMSNORM=1` | text RMSNorm | *（Q2.5，bs48）* 中位步时 −6.4%，吞吐量 +6.8%（A-B-A） |
| 融合 RoPE | `ROSS_NPU_ROPE=1` | text + ViT RoPE | *（Q2.5，bs48）* 步时 −5.9%，吞吐量 +6.3% |
| 融合 gate/up | `ROSS_FUSED_GATEUP=1` | text MLP | *（Q3，bs24）* 12.399 → 12.325 s，47.8 → 48.1% MFU；在 bs27 最佳点保留（13.779 s，48.4%） |
| Expandable segments | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 分配器 | *（Q2.5）* 轨迹中通信 4.317 → 0.234 s，步时 12.194 → 6.996 s；未剖析均值 +0.4%，CV 20.7 → 15.8% |
| CPU 亲和性 | `CPU_AFFINITY_CONF=1` | 主机 | 吞吐量相同，步时 CV 12.88% → 2.38%（约低 5×） |
| Pinned loader + memlock | `--pin_memory true`，`ulimit -l unlimited` | dataloader | 消除 pageable 的主机到设备停顿 |
| ZeRO-2 | `--zero_stage 2`，`zero2.json`（`overlap_comm`、`contiguous_gradients`） | DeepSpeed | 在相同 batch 下胜于 ZeRO-3：没有每步参数 all-gather；参数复制，梯度/优化器分片 |
| 不写检查点 | `--save_model false` | trainer | 在扫描运行中避免 ~18 GB 最终模型 + ~137 GB 优化器检查点 |
| 步时统计 | `ROSS_STEP_TIMING=1` | trainer | 仅测量（设备同步的 `step_time_s`） |

启动器设置的标准环境：`CPU_AFFINITY_CONF=1`、
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、`ulimit -l unlimited`、
`ASCEND_LAUNCH_BLOCKING=0`、`ROSS_STEP_TIMING=1`、`ROSS_NPU_RMSNORM=1`、
`ROSS_NPU_ROPE=1`、`ROSS_VALID_CE=1`、`--attn_type fa2`、`--zero_stage 2`。

### 有意关闭

| 特性 | 开关 | 原因 |
|---|---|---|
| 融合 SwiGLU | `ROSS_NPU_SWIGLU` | −3.2%（Step 14），−2.6%（Step 33） |
| 融合 QKV | `ROSS_FUSED_QKV=1` | *（Q3，bs24）* 12.464 s / 47.6% 对 12.399 s / 47.8% — 无收益 |
| GC-skip 最后 N 层 | `ROSS_GC_SKIP_LAST_LAYERS=N` | *（Q3）* bs16 上 +0.5 pp 换 +13.8 GB；bs24 时 OOM；不在最佳配置中 |
| 关闭重计算 | `--gradient_checkpointing False` | bs8 以上 OOM；bs8 时慢 1.51×（Step 31） |
| BSH attention layout | — | 会让 padding token 参与注意力；未移植 |
| 融合 AdamW | `adamw_torch_fused` | −3.6%（Step 14） |
| Add-RMSNorm v2 / selective CE / `TASK_QUEUE_ENABLE=2` | — | 中性（Step 36） |

### 这些特性下的最佳点

L≈1774、`--max_pixels 903168`、融合 gate/up、fa2、ZeRO-2、valid CE、融合
RMSNorm+RoPE、expandable segments、pinning、真实 20× 子集：**bs 24–27，
48.4% MindSpeed MFU，27.8k tokens/s**。若把检查点前向也计入，
`utils/flops_mindspeed.py --include_recompute` 可把这个数字提到 **62.19%
实际执行 MFU（bs27）**（+13.77 pp）— 这是关闭重计算无法
达到的上限，因为它在 bs8 以上放不下。
