"""
xai_eval.py — faithfulness evaluation of attributions for the DAiSEE VideoTCMT.

Loads the checkpoint written by train_eval.py (full model, first seed) and a
fixed random subset of the TEST split, then for each affective state:
  * deletion / insertion AUC for Integrated Gradients, Gradient x Input and a
    random ordering (Petsiuk et al., 2018)
  * model-randomisation sanity check (Adebayo et al., 2018): Spearman rank
    correlation between |attribution| on the trained and a re-initialised model
  * IG completeness gap
  * modality shares of |IG| (what the model actually uses)

Usage:
  python -m scripts.daisee.xai_eval --features data/daisee_features \
      --ckpt results/daisee/video_tcmt_daisee.pt --out results/daisee --n 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.fusion.VideoTcmt import VideoTCMT
from ml.training.DaiseeDataset import AFFECTIVE_STATES, VISUAL_SLICES, apply_normaliser
from ml.xai import Faithfulness as FX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/daisee_features"))
    ap.add_argument("--ckpt", type=Path, default=Path("results/daisee/video_tcmt_daisee.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/daisee"))
    ap.add_argument("--n", type=int, default=300, help="test clips to evaluate")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, weights_only=False)
    test = dict(np.load(args.features / "test.npz", allow_pickle=False))
    model = VideoTCMT(modalities=ck["modalities"], max_t=test["X"].shape[1], temporal=True)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    rand_model = FX.randomise_model(model, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(test["y"]), size=min(args.n, len(test["y"])), replace=False))
    X = torch.tensor(apply_normaliser(test["X"][idx], test["valid"][idx], ck["mu"], ck["sd"]))
    V = torch.tensor(test["valid"][idx])
    slices = {m: VISUAL_SLICES[m] for m in ck["modalities"]}

    methods = {
        "integrated_gradients": lambda m, x, v, s, l: FX.integrated_gradients(m, x, v, s, l, steps=args.ig_steps),
        "gradient_x_input": FX.gradient_x_input,
        "saliency": FX.saliency,
        "random": FX.random_attribution,
    }
    res = {"_meta": {"n_clips": int(len(idx)), "split": "test", "ig_steps": args.ig_steps,
                     "baseline": "train mean (0 after normalisation)",
                     "target": "softmax prob of the predicted level"}}
    for si, state in enumerate(AFFECTIVE_STATES):
        r = {m: {"deletion": [], "insertion": [], "rand_spearman": [], "input_spearman": []} for m in methods}
        comp, shares = [], {m: [] for m in slices}
        for b in range(0, len(idx), args.bs):
            x, v = X[b:b + args.bs], V[b:b + args.bs]
            lvl = FX.predicted_level(model, x, v, si)
            lvl_r = FX.predicted_level(rand_model, x, v, si)
            for name, fn in methods.items():
                a = fn(model, x, v, si, lvl)
                curves = FX.deletion_insertion(model, x, v, si, lvl, a)
                r[name]["deletion"] += FX.auc(curves["deletion"]).tolist()
                r[name]["insertion"] += FX.auc(curves["insertion"]).tolist()
                if name != "random":
                    ar = fn(rand_model, x, v, si, lvl_r)
                    for k in range(len(x)):
                        mk = v[k].unsqueeze(-1).expand_as(x[k]).reshape(-1).numpy()
                        ak = a[k].abs().reshape(-1).detach().numpy()[mk]
                        r[name]["rand_spearman"].append(FX.spearman(
                            ak, ar[k].abs().reshape(-1).detach().numpy()[mk]))
                        r[name]["input_spearman"].append(FX.spearman(
                            ak, x[k].abs().reshape(-1).numpy()[mk]))
                if name == "integrated_gradients":
                    comp += FX.completeness_gap(model, x, v, si, lvl, a).tolist()
                    for m, val in FX.modality_shares(a, slices).items():
                        shares[m] += val.tolist()
        out = {}
        for name, d in r.items():
            out[name] = {k: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                         for k, vals in d.items() if vals}
        out["ig_completeness_gap"] = {"mean": float(np.mean(comp)), "max": float(np.max(comp))}
        out["ig_modality_share"] = {m: float(np.mean(v)) for m, v in shares.items()}
        res[state] = out
        print(state, json.dumps({k: out[k] for k in ("integrated_gradients", "random")}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "xai_faithfulness.json").write_text(json.dumps(res, indent=2))
    lines = ["| State | Deletion AUC ↓ (IG / G×I / Grad / random) | Insertion AUC ↑ (IG / G×I / Grad / random) "
             "| Randomised-model Spearman ↓ (IG / G×I / Grad) | Spearman with \\|x\\| (IG / G×I / Grad) | IG share face / gaze / pose |",
             "|---|---|---|---|---|---|"]
    for state in AFFECTIVE_STATES:
        o = res[state]
        f = lambda k, met: f"{o[k][met]['mean']:.3f}"
        sh = " / ".join(f"{o['ig_modality_share'].get(m, 0):.2f}" for m in ("face", "gaze", "pose"))
        tri = lambda met: " / ".join(f(k, met) for k in ("integrated_gradients", "gradient_x_input", "saliency"))
        lines.append(f"| {state} | {tri('deletion')} / {f('random','deletion')} "
                     f"| {tri('insertion')} / {f('random','insertion')} "
                     f"| {tri('rand_spearman')} | {tri('input_spearman')} | {sh} |")
    md = "\n".join(lines)
    (args.out / "xai_faithfulness.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
