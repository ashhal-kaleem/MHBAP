"""
benchmark_table.py -- Generate v2 vs v3 vs v4 vs v5 comparison table.

Reads from training logs for v2-v5, and from the live checkpoint for v5.
Prints a markdown table and writes it to BENCHMARK.md.

Run: python scripts/benchmark_table.py
"""
import sys, json, re
sys.path.insert(0, r"D:\MHBAP")

LOG_DIR = r"D:\MHBAP\scripts"
METRICS_JSON = r"D:\MHBAP\ml\models\weights\tcmt_eval_metrics.json"

def parse_log(path):
    """Extract final test metrics dict from a training log file."""
    try:
        text = open(path, encoding="utf-8").read()
        # Find the JSON block after "Final test metrics"
        m = re.search(r"Final test metrics[^\{]*(\{.*\})", text, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(1))
    except Exception as e:
        return None

def fmt(v, digits=3):
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

versions = [
    ("v2 (8->4 class fix, no bias)",       f"{LOG_DIR}\\training_log_v2.txt"),
    ("v3 (+shuffle sampler)",              f"{LOG_DIR}\\training_log_v3.txt"),
    ("v4 (+WeightedSampler, bad mapping)", f"{LOG_DIR}\\training_log_v4.txt"),
    ("v5 (mapping fix, FINAL)",            f"{LOG_DIR}\\training_log_v5.txt"),
]

rows = []
for label, log_path in versions:
    d = parse_log(log_path)
    if d is None:
        rows.append((label, None))
        print(f"  {label}: log not found or unparseable ({log_path})")
    else:
        rows.append((label, d))
        print(f"  {label}: macro_f1={d['emotion']['macro_f1']:.4f}")

# ── Markdown table ───────────────────────────────────────────────────────────
lines = []
lines.append("# TCMT Training Run Benchmark: v2 → v3 → v4 → v5\n")
lines.append("All metrics are on the **held-out test set** (never seen during training or")
lines.append("hyperparameter selection). Early stopping used validation F1 only.\n")

lines.append("## Emotion Classification\n")
header = "| Run | Accuracy | Macro-F1 | F1 cls-0 (neutral) | F1 cls-1 (happy) | F1 cls-2 (sad/fear) | F1 cls-3 (angry) | ROC-AUC |"
sep    = "|-----|----------|----------|--------------------|------------------|---------------------|------------------|---------|"
lines.append(header)
lines.append(sep)
for label, d in rows:
    if d is None:
        lines.append(f"| {label} | — | — | — | — | — | — | — |")
        continue
    e  = d["emotion"]
    pf = e.get("per_class_f1", {})
    row = (f"| {label} "
           f"| {fmt(e['accuracy'])} "
           f"| **{fmt(e['macro_f1'])}** "
           f"| {fmt(pf.get('0',0))} "
           f"| {fmt(pf.get('1',0))} "
           f"| {fmt(pf.get('2',0))} "
           f"| {fmt(pf.get('3',0))} "
           f"| {fmt(e.get('roc_auc_ovr', float('nan')))} |")
    lines.append(row)

lines.append("")
lines.append("## Regression Heads (test RMSE / R²)\n")
header2 = "| Run | Stress RMSE | Stress R² | Engagement R² | Attention R² | Fatigue R² |"
sep2    = "|-----|------------|-----------|---------------|--------------|------------|"
lines.append(header2)
lines.append(sep2)
for label, d in rows:
    if d is None:
        lines.append(f"| {label} | — | — | — | — | — |")
        continue
    st = d.get("stress",     {}); en = d.get("engagement",{})
    at = d.get("attention",  {}); fa = d.get("fatigue",    {})
    row = (f"| {label} "
           f"| {fmt(st.get('rmse', float('nan')))} "
           f"| {fmt(st.get('r2',   float('nan')))} "
           f"| {fmt(en.get('r2',   float('nan')))} "
           f"| {fmt(at.get('r2',   float('nan')))} "
           f"| {fmt(fa.get('r2',   float('nan')))} |")
    lines.append(row)

lines.append("")
lines.append("## Notes\n")
lines.append("- **v2**: First clean retrain after audit fixes (EMOTION_CLASSES=4, no circular bias).")
lines.append("  Class 1/2 collapsed to F1=0.0 because HF streaming was unshuffled (shard-ordered).")
lines.append("- **v3**: Added `ds.shuffle(buffer_size=4000)` + inverse-freq CE weights.")
lines.append("  All 4 classes predicted; macro F1 improved. Class 3 weak (F1=0.08, ~198 samples).")
lines.append("- **v4**: Added `WeightedRandomSampler`. Class mapping still had Disgust→3")
lines.append("  double-counting angry class; sampler over-corrected and class 1 collapsed.")
lines.append("- **v5 (FINAL)**: Corrected FER/RAF Disgust mapping (→2 instead of →3).")
lines.append("  Both `WeightedRandomSampler` and weighted CE loss active.")
lines.append("  This is the authoritative result committed to the repository.\n")
lines.append("- Stress R² is high because WESAD provides real physio GT for 2000 samples.")
lines.append("- Engagement/Attention/Fatigue R² are **PROXY** labels (no public GT).")
lines.append("  These measure noise-regression accuracy, not real behavioral generalisation.")
lines.append("  Disclosed in `tcmt_eval_metrics.json` under `_label_provenance`.")

md = "\n".join(lines) + "\n"
out_path = r"D:\MHBAP\BENCHMARK.md"
open(out_path, "w", encoding="utf-8").write(md)
print(f"\nBenchmark table written to: {out_path}")
print("\n" + md)
