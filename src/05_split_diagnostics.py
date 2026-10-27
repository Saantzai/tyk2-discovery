"""
Step 5 — Interrogate the evaluation before trusting it.

Step 4 produced a scaffold-split R2 of about -2.4 with a rank correlation near
zero, while RMSE was only ~1.0 log unit. Those two facts are hard to reconcile
unless something about the split itself is pathological, so this script checks
three specific hypotheses rather than accepting the headline number:

  H1  The greedy split (largest scaffold groups to train) leaves a test set
      made entirely of singleton scaffolds, with a narrow potency range.
      R2 = 1 - SSE/SST collapses when SST is small, so a modest RMSE can
      produce a large negative R2 without the model being as bad as it looks.

  H2  A single split is one draw. Repeating with shuffled scaffold assignment
      shows whether the result is stable or an artefact of one partition.

  H3  The external Ki set may carry a systematic offset against IC50
      (Cheng-Prusoff: Ki and IC50 differ by an assay-dependent factor).
      A constant bias inflates RMSE and MAE while leaving ranking intact.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "base", Path(__file__).with_name("04_baseline_model.py"))
base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RESULTS = ROOT / "results"


def scaffold_groups(smiles):
    g = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        g[MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""].append(i)
    return g


def shuffled_scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=0):
    """Scaffold split with groups assigned in random order rather than by size.
    Still leak-free (no scaffold spans two sets) but the test set is no longer
    guaranteed to be all singletons."""
    g = scaffold_groups(smiles)
    sets = list(g.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(sets)
    n = len(smiles)
    n_tr, n_va = frac[0] * n, (frac[0] + frac[1]) * n
    tr, va, te = [], [], []
    for s in sets:
        if len(tr) + len(s) <= n_tr:
            tr += s
        elif len(tr) + len(va) + len(s) <= n_va:
            va += s
        else:
            te += s
    return np.array(tr), np.array(va), np.array(te)


def quick_rf(Xtr, ytr, Xte):
    m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--external", default="tyk2ki")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(PROC / f"{args.target}_curated.csv")
    X, keep = base.featurise(df["std_smiles"].tolist())
    df = df.iloc[keep].reset_index(drop=True)
    y = df["pActivity"].to_numpy()
    smiles = df["std_smiles"].tolist()

    print("=" * 66)
    print("H1 - is the greedy test set unrepresentative?")
    print("=" * 66)
    g = scaffold_groups(smiles)
    tr, va, te = base.scaffold_split(smiles)
    rnd_tr, rnd_va, rnd_te = base.random_split(len(df))
    sizes = {i: len(v) for v in g.values() for i in v}
    te_sizes = np.array([sizes[i] for i in te])
    print(f"whole set   : mean pIC50 {y.mean():.2f}  sd {y.std():.2f}")
    print(f"greedy train: mean {y[tr].mean():.2f}  sd {y[tr].std():.2f}  n={len(tr)}")
    print(f"greedy TEST : mean {y[te].mean():.2f}  sd {y[te].std():.2f}  n={len(te)}")
    print(f"random test : mean {y[rnd_te].mean():.2f}  sd {y[rnd_te].std():.2f}")
    print(f"greedy test scaffold-group sizes: min {te_sizes.min()}, "
          f"median {np.median(te_sizes):.0f}, max {te_sizes.max()}, "
          f"singletons {(te_sizes == 1).sum()}/{len(te)}")
    print(f"\n  -> R2 uses the TEST sd as its baseline. A test sd of "
          f"{y[te].std():.2f} vs whole-set {y.std():.2f} means R2 is "
          f"{'strongly ' if y[te].std() < 0.8 * y.std() else ''}penalised here.")

    print("\n" + "=" * 66)
    print(f"H2 - is one split representative? ({args.repeats} shuffled scaffold splits)")
    print("=" * 66)
    rows = []
    for s in range(args.repeats):
        a, b, c = shuffled_scaffold_split(smiles, seed=s)
        pred = quick_rf(X[a], y[a], X[c])
        rows.append({"seed": s, "n_test": len(c), "test_sd": y[c].std(),
                     "RMSE": np.sqrt(mean_squared_error(y[c], pred)),
                     "MAE": mean_absolute_error(y[c], pred),
                     "R2": r2_score(y[c], pred),
                     "Spearman": spearmanr(y[c], pred).statistic})
        print(f"  seed {s}: test n={len(c):>4}  sd {y[c].std():.2f}  "
              f"RMSE {rows[-1]['RMSE']:.3f}  R2 {rows[-1]['R2']:>6.3f}  "
              f"rho {rows[-1]['Spearman']:.3f}")
    rep = pd.DataFrame(rows)
    print(f"\n  mean RMSE {rep.RMSE.mean():.3f} +/- {rep.RMSE.std():.3f}")
    print(f"  mean R2   {rep.R2.mean():.3f} +/- {rep.R2.std():.3f}")
    print(f"  mean rho  {rep.Spearman.mean():.3f} +/- {rep.Spearman.std():.3f}")
    rep.to_csv(RESULTS / f"{args.target}_repeated_scaffold_splits.csv", index=False)

    print("\n" + "=" * 66)
    print("H3 - systematic Ki vs IC50 offset?")
    print("=" * 66)
    ext_path = PROC / f"{args.external}_curated.csv"
    bias = None
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        n_before = len(ext)
        ext = ext[~ext["std_smiles"].isin(set(df["std_smiles"]))].reset_index(drop=True)
        print(f"  {n_before:,} external compounds, {n_before - len(ext):,} overlap "
              f"with training, {len(ext):,} usable")
    if ext_path.exists() and len(ext) >= 20:
        Xe, ke = base.featurise(ext["std_smiles"].tolist())
        ye = ext.iloc[ke]["pActivity"].to_numpy()
        pred = quick_rf(X, y, Xe)
        resid = pred - ye
        bias = float(np.median(resid))
        print(f"  external n={len(ye)}")
        print(f"  observed Ki   : mean {ye.mean():.2f}  sd {ye.std():.2f}")
        print(f"  predicted     : mean {pred.mean():.2f}  sd {pred.std():.2f}")
        print(f"  median signed error (pred - obs): {bias:+.3f} log units")
        raw = {"RMSE": np.sqrt(mean_squared_error(ye, pred)),
               "MAE": mean_absolute_error(ye, pred),
               "R2": r2_score(ye, pred),
               "Spearman": spearmanr(ye, pred).statistic}
        corr = {"RMSE": np.sqrt(mean_squared_error(ye, pred - bias)),
                "MAE": mean_absolute_error(ye, pred - bias),
                "R2": r2_score(ye, pred - bias),
                "Spearman": spearmanr(ye, pred - bias).statistic}
        print(f"\n  {'':<16}{'raw':>10}{'bias-corrected':>18}")
        for k in raw:
            print(f"  {k:<16}{raw[k]:>10.3f}{corr[k]:>18.3f}")
        print(f"\n  -> Ranking is unchanged by a constant offset. If rho is "
              f"respectable but RMSE is not, the model orders compounds "
              f"correctly and mis-calibrates their absolute scale.")
        pd.DataFrame([{"metric": k, "raw": raw[k], "bias_corrected": corr[k]}
                      for k in raw]).to_csv(
            RESULTS / f"{args.target}_external_bias.csv", index=False)
    else:
        print("  external set unavailable or too small after overlap removal; skipped")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].hist([y[tr], y[te]], bins=30, label=["greedy train", "greedy test"],
               color=["#3b6ea5", "#c0392b"], density=True)
    ax[0].set_xlabel("pIC50"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
    ax[0].set_title("H1: potency distribution by split")
    ax[1].errorbar(rep.seed, rep.Spearman, fmt="o-", color="#3b6ea5", label="rho")
    ax[1].axhline(rep.Spearman.mean(), ls="--", color="#c0392b", lw=1)
    ax[1].set_xlabel("shuffled scaffold split seed"); ax[1].set_ylabel("Spearman rho")
    ax[1].set_title(f"H2: stability across {args.repeats} splits")
    if bias is not None:
        ax[2].hist(resid, bins=40, color="#3b6ea5")
        ax[2].axvline(0, color="k", lw=0.8)
        ax[2].axvline(bias, color="#c0392b", ls="--", lw=1.2)
        ax[2].set_xlabel("predicted - observed (log units)")
        ax[2].set_title(f"H3: external residuals (median {bias:+.2f})")
    fig.tight_layout()
    out = FIG / f"{args.target}_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> figures/{out.name}")


if __name__ == "__main__":
    main()
