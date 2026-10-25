"""
train_eval.py — train + evaluate VideoTCMT on DAiSEE with ablations.

Uses the official subject-disjoint splits produced by extract_features.py.
Model selection uses the VALIDATION split only; the TEST split is touched
once per trained model.

Experiments (each run over --seeds, reported as mean ± std):
  majority     majority-class baseline (from train labels)
  full         face+gaze+pose, temporal
  static       face+gaze+pose, clip mean (T=1)       -> temporal ablation
  subsets      every non-empty modality subset        -> modality ablation

Usage:
  python -m scripts.daisee.train_eval --features data/daisee_features \
      --out results/daisee --seeds 0 1 2 --epochs 40
  python -m scripts.daisee.train_eval --synthetic   # smoke test, no data needed
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ml.fusion.VideoTcmt import VideoTCMT
from ml.training.DaiseeDataset import (
    AFFECTIVE_STATES, N_LEVELS, VISUAL_MODALITIES,
    apply_normaliser, class_weights, fit_normaliser, synthetic_split,
)


# ── metrics (numpy, no sklearn) ─────────────────────────────────────────────

def macro_f1(y: np.ndarray, p: np.ndarray, k: int = N_LEVELS) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        if tp + fp + fn == 0:
            continue  # class absent in both -> skip
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def state_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for s, name in enumerate(AFFECTIVE_STATES):
        out[name] = {
            "accuracy": float(np.mean(y[:, s] == p[:, s])),
            "macro_f1": macro_f1(y[:, s], p[:, s]),
        }
    return out


# ── training ────────────────────────────────────────────────────────────────

def _tensors(split: Dict[str, np.ndarray], mu, sd, device):
    X = apply_normaliser(split["X"], split["valid"], mu, sd)
    return (torch.tensor(X, device=device),
            torch.tensor(split["valid"], device=device),
            torch.tensor(split["y"], device=device))


@torch.no_grad()
def predict(model, X, V, bs: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        preds.append(model(X[i:i + bs], V[i:i + bs]).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def train_one(
    data: Dict[str, Dict[str, np.ndarray]],
    modalities: Sequence[str],
    temporal: bool,
    seed: int,
    epochs: int,
    lr: float,
    bs: int,
    device: str,
    patience: int = 8,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = fit_normaliser(data["train"]["X"], data["train"]["valid"])
    Xtr, Vtr, Ytr = _tensors(data["train"], mu, sd, device)
    Xva, Vva, Yva = _tensors(data["val"], mu, sd, device)
    Xte, Vte, Yte = _tensors(data["test"], mu, sd, device)
    W = torch.tensor(class_weights(data["train"]["y"]), device=device)

    model = VideoTCMT(modalities=modalities, max_t=Xtr.shape[1], temporal=temporal).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            logits = model(Xtr[idx], Vtr[idx])
            loss = sum(
                F.cross_entropy(logits[:, s], Ytr[idx, s], weight=W[s])
                for s in range(len(AFFECTIVE_STATES))
            ) / len(AFFECTIVE_STATES)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # model selection: mean macro-F1 over the 4 states on VALIDATION
        pv = predict(model, Xva, Vva)
        score = float(np.mean([m["macro_f1"] for m in
                               state_metrics(Yva.cpu().numpy(), pv).values()]))
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    pt = predict(model, Xte, Vte)
    return {
        "val_mean_macro_f1": best,
        "test": state_metrics(Yte.cpu().numpy(), pt),
        "epochs_run": ep + 1,
        "model": model,
        "normaliser": (mu, sd),
    }


def majority_baseline(data) -> Dict:
    ytr, yte = data["train"]["y"], data["test"]["y"]
    maj = np.array([np.bincount(ytr[:, s], minlength=N_LEVELS).argmax()
                    for s in range(ytr.shape[1])])
    return state_metrics(yte, np.tile(maj, (len(yte), 1)))


def aggregate(runs: List[Dict]) -> Dict:
    out = {}
    for name in AFFECTIVE_STATES:
        out[name] = {}
        for met in ("accuracy", "macro_f1"):
            vals = np.array([r["test"][name][met] for r in runs])
            out[name][met] = {"mean": float(vals.mean()), "std": float(vals.std()),
                              "runs": vals.tolist()}
    return out


def to_markdown(results: Dict) -> str:
    states = AFFECTIVE_STATES
    lines = ["| Setting | " + " | ".join(f"{s} Acc / F1" for s in states) + " |",
             "|---|" + "---|" * len(states)]

    def fmt(cell):
        if isinstance(cell["accuracy"], dict):
            a, f = cell["accuracy"], cell["macro_f1"]
            return f"{100*a['mean']:.1f}±{100*a['std']:.1f} / {f['mean']:.3f}±{f['std']:.3f}"
        return f"{100*cell['accuracy']:.1f} / {cell['macro_f1']:.3f}"

    for setting, res in results.items():
        if setting.startswith("_"):
            continue
        lines.append(f"| {setting} | " + " | ".join(fmt(res[s]) for s in states) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/daisee_features"))
    ap.add_argument("--out", type=Path, default=Path("results/daisee"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--no-subsets", action="store_true", help="skip modality-subset ablation")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on synthetic data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.synthetic:
        data = {"train": synthetic_split(256, seed=0), "val": synthetic_split(64, seed=1),
                "test": synthetic_split(64, seed=2)}
        args.epochs = min(args.epochs, 3)
        args.seeds = args.seeds[:1]
    else:
        data = {s: dict(np.load(args.features / f"{s}.npz", allow_pickle=False))
                for s in ("train", "val", "test")}

    for s in ("train", "val", "test"):
        print(f"[{s}] n={len(data[s]['y'])}  face-detected frames="
              f"{100*data[s]['valid'].mean():.1f}%")
    args.out.mkdir(parents=True, exist_ok=True)

    all_mods = list(VISUAL_MODALITIES)
    settings = {"full (face+gaze+pose, temporal)": (all_mods, True),
                "static (clip mean, T=1)": (all_mods, False)}
    if not args.no_subsets:
        for r in range(1, len(all_mods)):
            for combo in itertools.combinations(all_mods, r):
                settings["+".join(combo) + " (temporal)"] = (list(combo), True)

    results: Dict = {"majority-class baseline": majority_baseline(data)}
    t0 = time.time()
    for name, (mods, temporal) in settings.items():
        runs = []
        for seed in args.seeds:
            r = train_one(data, mods, temporal, seed, args.epochs, args.lr, args.bs, device)
            print(f"  {name} seed={seed} val={r['val_mean_macro_f1']:.3f} "
                  f"test-eng-acc={r['test']['Engagement']['accuracy']:.3f}", flush=True)
            if name.startswith("full") and seed == args.seeds[0]:
                torch.save({"state_dict": r["model"].state_dict(),
                            "modalities": mods, "mu": r["normaliser"][0],
                            "sd": r["normaliser"][1]},
                           args.out / "video_tcmt_daisee.pt")
            runs.append({k: v for k, v in r.items() if k not in ("model", "normaliser")})
        results[name] = aggregate(runs)

    results["_meta"] = {
        "dataset": "DAiSEE official splits (subject-disjoint)" if not args.synthetic else "SYNTHETIC",
        "n": {s: int(len(data[s]["y"])) for s in data},
        "seeds": args.seeds, "epochs_max": args.epochs, "lr": args.lr,
        "selection": "best mean macro-F1 on validation; test evaluated once per run",
        "device": device, "minutes": round((time.time() - t0) / 60, 1),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    md = to_markdown(results)
    (args.out / "results.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
