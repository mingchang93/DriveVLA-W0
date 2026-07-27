import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os

def load_trainer_state(filepath):
    with open(filepath) as f:
        data = json.load(f)
    log_history = data.get("log_history", [])
    records = {"step": [], "loss": [], "grad_norm": [], "learning_rate": []}
    for entry in log_history:
        if "loss" in entry:
            for k in records:
                records[k].append(entry.get(k, 0))
    return {k: np.array(v) for k, v in records.items()}

def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ── Args ──
parser = argparse.ArgumentParser(
    description="Compare two trainer_state.json files (e.g. CUDA vs NPU) and plot loss/gradient metrics."
)
parser.add_argument("--cuda_json", required=True, help="Path to trainer_state.json from CUDA run")
parser.add_argument("--npu_json",  required=True, help="Path to trainer_state.json from NPU run")
parser.add_argument("--output_dir", "-o", default=None,
                    help="Directory for output plots (default: same dir as cuda_json)")
args = parser.parse_args()

LABEL_CUDA = "CUDA"
LABEL_NPU  = "NPU"

OUT = args.output_dir or os.path.dirname(args.cuda_json) or "."
os.makedirs(OUT, exist_ok=True)

# ── Load data ──
d_cuda = load_trainer_state(args.cuda_json)
d_npu  = load_trainer_state(args.npu_json)

steps = d_cuda["step"]
loss_cuda = d_cuda["loss"]
loss_npu  = d_npu["loss"]
grad_cuda = d_cuda["grad_norm"]
grad_npu  = d_npu["grad_norm"]

# ── Derived metrics ──
abs_diff = loss_cuda - loss_npu
rel_error = (loss_cuda - loss_npu) / loss_npu * 100
smoothed_loss_cuda = smooth(loss_cuda, 5)
smoothed_loss_npu  = smooth(loss_npu, 5)
smoothed_rel_error = smooth(rel_error, 10)

# ════════════════════════════════════════════════════════
# Figure 1: Comprehensive comparison (2x2)
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(f"DriveVLA-W0: {args.label_cuda} vs {args.label_npu} Loss Comparison",
             fontsize=16, fontweight="bold", y=0.98)

# ── (0,0) Loss curves ──
ax = axes[0, 0]
ax.plot(steps, loss_cuda, alpha=0.35, color='#e74c3c', linewidth=1, label=f"{args.label_cuda} raw")
ax.plot(steps, loss_npu,  alpha=0.35, color='#2ecc71', linewidth=1, label=f"{args.label_npu} raw")
ax.plot(steps[:len(smoothed_loss_cuda)], smoothed_loss_cuda, color='#c0392b', linewidth=2.5,
        label=f"{args.label_cuda} (smoothed)")
ax.plot(steps[:len(smoothed_loss_npu)], smoothed_loss_npu, color='#27ae60', linewidth=2.5,
        label=f"{args.label_npu} (smoothed)")
stable_mask = steps > 5
ymin_loss = min(loss_cuda[stable_mask].min(), loss_npu[stable_mask].min())
ymax_loss = max(loss_cuda[stable_mask].max(), loss_npu[stable_mask].max())
margin = (ymax_loss - ymin_loss) * 0.15
ax.set_ylim(ymin_loss - margin, ymax_loss + margin)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

# ── (0,1) Absolute difference ──
ax = axes[0, 1]
diff_label = f"{args.label_cuda} - {args.label_npu}"
colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in abs_diff]
ax.bar(steps, abs_diff, color=colors, alpha=0.6, width=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)
mean_diff = np.mean(abs_diff)
ax.axhline(y=mean_diff, color='#3498db', linestyle='--', linewidth=1.5,
           label=f"Mean: {mean_diff:.4f}")
stable_diff = abs_diff[stable_mask]
ymin_diff = min(0, stable_diff.min() * 1.2)
ymax_diff = stable_diff.max() * 1.3
ax.set_ylim(ymin_diff, ymax_diff)
ax.set_xlabel("Step")
ax.set_ylabel(f"Loss Difference ({diff_label})")
ax.set_title("Absolute Loss Difference per Step")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── (1,0) Relative error ──
ax = axes[1, 0]
ax.plot(steps, rel_error, alpha=0.3, color='#3498db', linewidth=1)
ax.plot(steps[:len(smoothed_rel_error)], smoothed_rel_error, color='#2c3e50', linewidth=2.5,
        label="Smoothed (window=10)")
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=2, color='#e67e22', linestyle=':', linewidth=1.5)
ax.axhline(y=-2, color='#e67e22', linestyle=':', linewidth=1.5)
stable_rel = rel_error[stable_mask]
ymin_rel = min(-2, stable_rel.min() * 1.3)
ymax_rel = max(2, stable_rel.max() * 1.5)
ax.set_ylim(ymin_rel, ymax_rel)
ax.set_xlabel("Step")
ax.set_ylabel("Relative Error (%)")
ax.set_title(f"Relative Error: ({diff_label}) / {args.label_npu}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── (1,1) Gradient norm relative error ──
ax = axes[1, 1]
grad_rel_diff = (grad_cuda - grad_npu) / grad_npu * 100
smoothed_grad_diff = smooth(grad_rel_diff, 10)
ax.plot(steps, grad_rel_diff, alpha=0.3, color='#9b59b6', linewidth=1)
ax.plot(steps[:len(smoothed_grad_diff)], smoothed_grad_diff, color='#8e44ad', linewidth=2.5,
        label="Smoothed (window=10)")
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=10, color='#e67e22', linestyle=':', linewidth=1.5, label='±10% threshold')
ax.axhline(y=-10, color='#e67e22', linestyle=':', linewidth=1.5)
stable_grad_diff = grad_rel_diff[stable_mask]
ymin_grad = min(-10, stable_grad_diff.min() * 1.3)
ymax_grad = max(10, stable_grad_diff.max() * 1.5)
ax.set_ylim(ymin_grad, ymax_grad)
ax.set_xlabel("Step")
ax.set_ylabel(f"({diff_label}) / {args.label_npu} (%)")
ax.set_title("Gradient Norm Relative Error")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path1 = os.path.join(OUT, "loss_comparison.png")
plt.savefig(path1, dpi=150)
print(f"Saved {path1}")

# ════════════════════════════════════════════════════════
# Figure 2: Relative error detail (standalone, high-res)
# ════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.fill_between(steps, rel_error, alpha=0.15, color='#3498db')
ax2.plot(steps, rel_error, alpha=0.4, color='#3498db', linewidth=1)
ax2.plot(steps[:len(smoothed_rel_error)], smoothed_rel_error, color='#2c3e50', linewidth=2.5,
         label="Smoothed (window=10)")
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.axhline(y=2, color='#e67e22', linestyle=':', linewidth=1.5, label='±2% threshold')
ax2.axhline(y=-2, color='#e67e22', linestyle=':', linewidth=1.5)
max_rel = np.max(rel_error)
min_rel = np.min(rel_error)
ax2.annotate(f"Max: {max_rel:.2f}%", xy=(steps[np.argmax(rel_error)], max_rel),
             xytext=(10, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
ax2.annotate(f"Min: {min_rel:.2f}%", xy=(steps[np.argmin(rel_error)], min_rel),
             xytext=(10, -15), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Relative Error (%)", fontsize=12)
ax2.set_title(f"DriveVLA-W0: {args.label_cuda} vs {args.label_npu} Relative Error "
              f"({args.label_npu} as baseline)", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
path2 = os.path.join(OUT, "relative_error.png")
plt.savefig(path2, dpi=150)
print(f"Saved {path2}")

# ════════════════════════════════════════════════════════
# Figure 3: Relative error after step 50 (zoom-in)
# ════════════════════════════════════════════════════════
mask = steps > 50
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(steps[mask], rel_error[mask], alpha=0.5, color='#3498db', linewidth=1,
         marker='o', markersize=4)
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax3.axhline(y=2, color='#e67e22', linestyle=':', linewidth=1.5, label='±2% threshold')
ax3.axhline(y=-2, color='#e67e22', linestyle=':', linewidth=1.5)
ax3.set_xlabel("Step", fontsize=12)
ax3.set_ylabel("Relative Error (%)", fontsize=12)
ax3.set_title(f"DriveVLA-W0: Relative Error After Step 50", fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
path3 = os.path.join(OUT, "relative_error_after50.png")
plt.savefig(path3, dpi=150)
print(f"Saved {path3}")

# ════════════════════════════════════════════════════════
# Print summary stats
# ════════════════════════════════════════════════════════
print("\n===== Summary =====")
print(f"Steps: {steps[0]} - {steps[-1]}")
print(f"Loss {args.label_cuda}:  {loss_cuda[0]:.4f} -> {loss_cuda[-1]:.4f}")
print(f"Loss {args.label_npu}:   {loss_npu[0]:.4f} -> {loss_npu[-1]:.4f}")
print(f"Mean relative error: {np.mean(rel_error):.2f}%")
print(f"Max relative error:  {max_rel:.2f}% (step {steps[np.argmax(rel_error)]})")
print(f"Min relative error:  {min_rel:.2f}% (step {steps[np.argmin(rel_error)]})")
print(f"Mean abs difference: {mean_diff:.4f}")