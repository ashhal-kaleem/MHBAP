"""
generate_figures.py — Publication-quality figures for MHBAP v5 final.

Outputs (all in docs/figures/):
  fig1_confusion_matrix.png      — 4x4 emotion confusion matrix (normalised)
  fig2_per_class_f1.png          — Per-class F1 bar chart (v5)
  fig3_benchmark_comparison.png  — v2->v5 macro-F1 and stress-R2 trend
  fig4_regression_heads.png      — Regression R2 grouped bar chart
  fig5_roc_auc.png               — ROC-AUC curve (v5 OvR)
  fig6_training_curve.png        — Training loss curve (v5)

Run: python scripts/generate_figures.py
"""
import os, sys, json, re
sys.path.insert(0, r"D:\MHBAP")
import numpy as np

OUTDIR = r"D:\MHBAP\docs\figures"
os.makedirs(OUTDIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load v5 metrics ──────────────────────────────────────────────────────────
with open(r"D:\MHBAP\ml\models\weights\tcmt_eval_metrics.json") as f:
    v5 = json.load(f)

LOG_DIR = r"D:\MHBAP\scripts"

def parse_log(path):
    try:
        text = open(path, encoding="utf-8").read()
        m = re.search(r"Final test metrics[^\{]*(\{.*\})", text, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(1))
    except Exception:
        return None

def parse_training_curve(path):
    """Extract epoch, loss, val_f1 from training log."""
    epochs, losses, val_f1s = [], [], []
    try:
        for line in open(path, encoding="utf-8"):
            m = re.search(r"Epoch\s+(\d+)/\d+\s+loss=(\S+)\s+val_f1=(\S+)", line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                val_f1s.append(float(m.group(3)))
    except Exception:
        pass
    return epochs, losses, val_f1s

v2 = parse_log(f"{LOG_DIR}\\training_log_v2.txt")
v3 = parse_log(f"{LOG_DIR}\\training_log_v3.txt")
v4 = parse_log(f"{LOG_DIR}\\training_log_v4.txt")
# v5 already loaded from JSON checkpoint

versions = {"v2": v2, "v3": v3, "v4": v4, "v5": v5}
version_labels = ["v2\n(4-class fix)", "v3\n(+shuffle)", "v4\n(+sampler\nbad map)", "v5\n(FINAL)"]
colors = ["#d62728", "#ff7f0e", "#e377c2", "#2ca02c"]

print("Loaded versions:", {k: (v is not None) for k, v in versions.items()})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 1 — Confusion Matrix (normalised)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cm_raw = np.array(v5["emotion"]["confusion_matrix"], dtype=float)
cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)
class_names = ["Neutral\n(cls-0)", "Happy\n(cls-1)", "Sad/Fear\n(cls-2)", "Angry\n(cls-3)"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(class_names, fontsize=9)
ax.set_yticklabels(class_names, fontsize=9)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("TCMT v5 — Emotion Confusion Matrix\n(row-normalised, held-out test set, n=1200)")

for i in range(4):
    for j in range(4):
        raw_val = int(cm_raw[i, j])
        norm_val = cm_norm[i, j]
        color = "white" if norm_val > 0.55 else "black"
        ax.text(j, i, f"{norm_val:.2f}\n({raw_val})", ha="center", va="center",
                fontsize=8.5, color=color)

plt.tight_layout()
out = f"{OUTDIR}\\fig1_confusion_matrix.png"
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 2 — Per-class F1 bar chart (v5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pf = v5["emotion"]["per_class_f1"]
f1_vals = [pf[str(i)] for i in range(4)]
bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(class_names, f1_vals, color=bar_colors, width=0.5, edgecolor="white")
ax.axhline(v5["emotion"]["macro_f1"], color="black", linestyle="--", linewidth=1.2,
           label=f'Macro-F1 = {v5["emotion"]["macro_f1"]:.3f}')
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1 Score")
ax.set_title("TCMT v5 — Per-class F1 (Emotion)\nHeld-out test set (n=1200)")
ax.legend(fontsize=9)

for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
out = f"{OUTDIR}\\fig2_per_class_f1.png"
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 3 — Benchmark comparison: macro-F1 trend v2->v5
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
macro_f1s = []
stress_r2s = []
for vdata in [v2, v3, v4, v5]:
    if vdata is None:
        macro_f1s.append(None); stress_r2s.append(None)
    else:
        macro_f1s.append(vdata["emotion"]["macro_f1"])
        stress_r2s.append(vdata["stress"]["r2"])

x = np.arange(4)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax2 = ax.twinx()

def plot_with_none(ax_obj, vals, color, marker, label, zorder=2):
    xs, ys = zip(*[(xi, yi) for xi, yi in zip(x, vals) if yi is not None])
    ax_obj.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=7,
                label=label, zorder=zorder)
    for xi, yi in zip(xs, ys):
        ax_obj.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5, color=color)

plot_with_none(ax,  macro_f1s,  "#2ca02c", "o", "Emotion Macro-F1 (left)")
plot_with_none(ax2, stress_r2s, "#1f77b4", "s", "Stress R² (right)")

ax.set_xticks(x)
ax.set_xticklabels(version_labels, fontsize=9)
ax.set_ylabel("Macro-F1", color="#2ca02c")
ax2.set_ylabel("Stress R²", color="#1f77b4")
ax.set_ylim(0, 1.0)
ax2.set_ylim(0, 1.0)
ax.set_title("TCMT Training Run Comparison: v2 → v5\n(held-out test set)")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

# Shade v4 as regression
ax.axvspan(2.5, 3.5, alpha=0.07, color="red")
ax.text(3, 0.05, "v4\nregressed", ha="center", fontsize=8, color="red")

plt.tight_layout()
out = f"{OUTDIR}\\fig3_benchmark_comparison.png"
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 4 — Regression R² grouped bar chart across versions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
heads = ["stress", "engagement", "attention", "fatigue"]
head_labels = ["Stress", "Engagement\n(proxy)", "Attention\n(proxy)", "Fatigue\n(proxy)"]
n_heads = len(heads)
n_vers = 4
width = 0.18
x_head = np.arange(n_heads)

fig, ax = plt.subplots(figsize=(9, 5))
ver_datas = [v2, v3, v4, v5]
ver_names = ["v2", "v3", "v4", "v5 (final)"]
ver_colors = colors

for vi, (vdata, vname, vc) in enumerate(zip(ver_datas, ver_names, ver_colors)):
    r2_vals = []
    for h in heads:
        if vdata and h in vdata:
            r2_vals.append(vdata[h].get("r2", 0.0))
        else:
            r2_vals.append(0.0)
    offset = (vi - n_vers/2 + 0.5) * width
    bars = ax.bar(x_head + offset, r2_vals, width=width, label=vname, color=vc, alpha=0.85)

ax.set_xticks(x_head)
ax.set_xticklabels(head_labels, fontsize=9)
ax.set_ylabel("R² Score")
ax.set_ylim(-0.1, 1.05)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Regression Head R² Across Training Runs (v2–v5)\n[Engagement/Attention/Fatigue are proxy labels — not real GT]")
ax.legend(fontsize=9)

plt.tight_layout()
out = f"{OUTDIR}\\fig4_regression_r2.png"
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 5 — ROC AUC (OvR, from saved scalar value)
# We'll draw the macro-average as a diagonal + point annotation since we
# don't have raw probability vectors saved. Plot class-F1 vs support instead.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Class support from confusion matrix row sums
cm_raw2 = np.array(v5["emotion"]["confusion_matrix"], dtype=float)
support = cm_raw2.sum(axis=1)
f1_vals2 = [v5["emotion"]["per_class_f1"][str(i)] for i in range(4)]
roc_auc = v5["emotion"]["roc_auc_ovr"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: ROC-AUC summary bubble chart (F1 vs class size)
ax = axes[0]
scatter_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
for i, (sup, f1, cls) in enumerate(zip(support, f1_vals2, class_names)):
    ax.scatter(sup, f1, s=sup*0.5, color=scatter_colors[i], alpha=0.75,
               zorder=3, label=f"{cls.split(chr(10))[0]} (n={int(sup)})")
    ax.annotate(f"F1={f1:.2f}", (sup, f1), xytext=(6, 3),
                textcoords="offset points", fontsize=8.5)

ax.set_xlabel("Test-set support (# samples)")
ax.set_ylabel("F1 Score")
ax.set_xlim(0, max(support)*1.25)
ax.set_ylim(-0.05, 1.0)
ax.set_title(f"Class Support vs F1\n(macro OvR ROC-AUC = {roc_auc:.3f})")
ax.legend(fontsize=8, loc="upper right")
ax.axhline(v5["emotion"]["macro_f1"], color="black", linestyle="--",
           linewidth=1, label="macro-F1")

# Right: RMSE bar chart per regression head
ax2 = axes[1]
rmse_heads = ["stress", "engagement", "attention", "fatigue"]
rmse_labels = ["Stress", "Engagement\n(proxy)", "Attention\n(proxy)", "Fatigue\n(proxy)"]
rmse_vals = [v5[h]["rmse"] for h in rmse_heads]
rmse_colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
ax2.bar(rmse_labels, rmse_vals, color=rmse_colors, edgecolor="white", width=0.5)
for i, rv in enumerate(rmse_vals):
    ax2.text(i, rv + 0.002, f"{rv:.4f}", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("RMSE")
ax2.set_title("Regression Head RMSE (v5 final)\nTest set (n=1200)")
ax2.set_ylim(0, max(rmse_vals)*1.35)

plt.tight_layout()
out = f"{OUTDIR}\\fig5_class_support_rmse.png"
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 6 — Training loss curve (v5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
epochs, losses, val_f1s = parse_training_curve(
    f"{LOG_DIR}\\training_log_v5.txt")

if epochs:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()
    ax.plot(epochs, losses, color="#d62728", linewidth=2, label="Train loss")
    ax2.plot(epochs, val_f1s, color="#2ca02c", linewidth=2,
             linestyle="--", label="Val macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined loss", color="#d62728")
    ax2.set_ylabel("Val macro-F1", color="#2ca02c")
    ax.set_title("TCMT v5 Training Curve (50 epochs, real data)")
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, l1+l2, fontsize=9)
    plt.tight_layout()
    out = f"{OUTDIR}\\fig6_training_curve.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")
else:
    print("WARNING: Could not parse training curve from v5 log.")

print("\nAll figures written to:", OUTDIR)
