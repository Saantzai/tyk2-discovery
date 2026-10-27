"""
Step 4 — Baseline QSAR models with an honest split.

Trains Random Forest and LightGBM on ECFP4 + physicochemical descriptors and
evaluates three ways:

  1. random split      - the optimistic number most projects report
  2. scaffold split    - the number that actually predicts prospective use
  3. external Ki set   - compounds measured by a different assay type, with
                         any structure already seen in training removed

Reporting 1 and 2 side by side is the point. The gap between them is the
single most informative figure in the whole project.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RESULTS = ROOT / "results"

SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
DESCRIPTORS = ["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors", "TPSA",
               "NumRotatableBonds", "RingCount", "FractionCSP3",
               "NumAromaticRings", "HeavyAtomCount"]

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def featurise(smiles_list):
    """ECFP4 bits plus a short physicochemical block."""
    X, keep = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.zeros(FP_BITS, dtype=np.int8)
        for b in _gen.GetFingerprint(mol).GetOnBits():
            fp[b] = 1
        desc = np.array([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol), Descriptors.FractionCSP3(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.HeavyAtomCount(mol),
        ], dtype=np.float64)
        X.append(np.concatenate([fp, desc]))
        keep.append(i)
    return np.vstack(X), np.array(keep)


def scaffold_split(smiles_list, frac=(0.8, 0.1, 0.1), seed=SEED):
    """Bemis-Murcko split. Largest scaffold groups go to train, so the test
    set is made of chemistry the model has genuinely not seen."""
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""
        groups[scaf].append(i)
    sets = sorted(groups.values(), key=len, reverse=True)
    n = len(smiles_list)
    n_tr, n_va = frac[0] * n, (frac[0] + frac[1]) * n
    train, val, test = [], [], []
    for s in sets:
        if len(train) + len(s) <= n_tr:
            train += s
        elif len(train) + len(val) + len(s) <= n_va:
            val += s
        else:
            test += s
    return np.array(train), np.array(val), np.array(test)


def random_split(n, frac=(0.8, 0.1, 0.1), seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def metrics(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def fit_models(Xtr, ytr):
    models = {}
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED,
                               min_samples_leaf=1)
    rf.fit(Xtr, ytr)
    models["RandomForest"] = rf
    try:
        import lightgbm as lgb
        gbm = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03,
                                num_leaves=63, subsample=0.8,
                                colsample_bytree=0.4, random_state=SEED,
                                n_jobs=-1, verbose=-1)
        gbm.fit(Xtr, ytr)
        models["LightGBM"] = gbm
    except Exception as exc:
        print(f"  (LightGBM unavailable: {exc})")
    return models


def parity_plot(results, out):
    panels = [(k, v) for k, v in results.items() if v.get("y_true") is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, panels):
        yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
        ax.scatter(yt, yp, s=8, alpha=0.4, color="#3b6ea5")
        lo, hi = min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color="#c0392b", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("observed pIC50"); ax.set_ylabel("predicted pIC50")
        m = r["metrics"]
        ax.set_title(f"{name}\nRMSE {m['RMSE']:.2f}  rho {m['Spearman']:.2f}  "
                     f"n={m['n']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--external", default="tyk2ki",
                    help="curated set used as an external test (set to '' to skip)")
    args = ap.parse_args()

    src = PROC / f"{args.target}_curated.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}; run 02_curate.py first")
    df = pd.read_csv(src)
    print(f"loaded {len(df):,} compounds from {src.name}")

    X, keep = featurise(df["std_smiles"].tolist())
    df = df.iloc[keep].reset_index(drop=True)
    y = df["pActivity"].to_numpy()
    print(f"featurised -> {X.shape[0]:,} x {X.shape[1]:,}")

    results, summary = {}, []

    for split_name, idx in [("random", random_split(len(df))),
                            ("scaffold", scaffold_split(df["std_smiles"].tolist()))]:
        tr, va, te = idx
        print(f"\n--- {split_name} split: train {len(tr):,} / val {len(va):,} "
              f"/ test {len(te):,}")
        models = fit_models(X[tr], y[tr])
        for mname, model in models.items():
            pred = model.predict(X[te])
            m = metrics(y[te], pred)
            summary.append({"split": split_name, "model": mname, **m})
            print(f"  {mname:<13} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  "
                  f"R2 {m['R2']:.3f}  rho {m['Spearman']:.3f}")
            results[f"{mname} ({split_name})"] = {
                "metrics": m, "y_true": y[te].tolist(), "y_pred": pred.tolist()}

    # --- external validation on the other assay type ----------------------
    if args.external:
        ext_path = PROC / f"{args.external}_curated.csv"
        if ext_path.exists():
            ext = pd.read_csv(ext_path)
            before = len(ext)
            seen = set(df["std_smiles"])
            ext = ext[~ext["std_smiles"].isin(seen)].reset_index(drop=True)
            print(f"\n--- external set {args.external}: {before:,} compounds, "
                  f"{before - len(ext):,} already in training, {len(ext):,} truly external")
            if len(ext) > 20:
                Xe, ke = featurise(ext["std_smiles"].tolist())
                ye = ext.iloc[ke]["pActivity"].to_numpy()
                models = fit_models(X, y)          # refit on everything
                for mname, model in models.items():
                    pred = model.predict(Xe)
                    m = metrics(ye, pred)
                    summary.append({"split": f"external-{args.external}",
                                    "model": mname, **m})
                    print(f"  {mname:<13} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  "
                          f"R2 {m['R2']:.3f}  rho {m['Spearman']:.3f}")
                    results[f"{mname} (external)"] = {
                        "metrics": m, "y_true": ye.tolist(), "y_pred": pred.tolist()}
        else:
            print(f"\n(external set {ext_path.name} not found; skipping)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sm = pd.DataFrame(summary)
    sm.to_csv(RESULTS / f"{args.target}_model_metrics.csv", index=False)
    (RESULTS / f"{args.target}_model_predictions.json").write_text(json.dumps(
        {k: v["metrics"] for k, v in results.items()}, indent=2))
    parity_plot(results, FIG / f"{args.target}_parity.png")

    print("\n" + "=" * 62)
    print(sm.to_string(index=False))
    print("=" * 62)
    rnd = sm[(sm.split == "random")]["RMSE"].min()
    scf = sm[(sm.split == "scaffold")]["RMSE"].min()
    print(f"\nOptimism gap: best random-split RMSE {rnd:.3f} vs "
          f"best scaffold-split RMSE {scf:.3f}  (+{scf - rnd:.3f} log units)")
    print(f"figure -> figures/{args.target}_parity.png")


if __name__ == "__main__":
    main()
