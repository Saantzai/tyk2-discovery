"""
Step 11 — Two controls the pipeline was missing.

1. Y-RANDOMISATION
   Shuffle the activity labels and retrain. A model that scores nearly as
   well on scrambled labels was learning structure in the descriptor space -
   scaffold clustering, dataset composition - rather than activity. This is
   the standard QSAR control for "is the signal real", and nothing earlier in
   this project tested it.

2. DOCKING ENRICHMENT
   The positive-control test in step 7 was qualitative: known actives did not
   score better. This puts numbers on it - ROC-AUC and enrichment factor at
   1, 5 and 10% - against PROPERTY-MATCHED decoys, so the comparison is not
   confounded by the size bias already measured in the docking scores
   (Spearman -0.247 against heavy atom count).

Neither needs new docking or new ChEMBL calls.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = _load("04_baseline_model.py")
diag = _load("05_split_diagnostics.py")


def y_randomisation(df, X, y, n_runs, n_splits):
    """Real model vs models trained on shuffled labels."""
    smiles = df["std_smiles"].tolist()
    rng = np.random.default_rng(SEED)

    def evaluate(labels, seed):
        tr, va, te = diag.shuffled_scaffold_split(smiles, seed=seed)
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
        m.fit(X[tr], labels[tr])
        p = m.predict(X[te])
        return (float(np.sqrt(mean_squared_error(labels[te], p))),
                float(spearmanr(labels[te], p).statistic))

    real = [evaluate(y, s) for s in range(n_splits)]
    print(f"  real labels     : RMSE {np.mean([r[0] for r in real]):.3f} "
          f"+/- {np.std([r[0] for r in real]):.3f}   "
          f"rho {np.mean([r[1] for r in real]):+.3f} "
          f"+/- {np.std([r[1] for r in real]):.3f}")

    shuffled = []
    for i in range(n_runs):
        perm = rng.permutation(len(y))
        shuffled.append(evaluate(y[perm], i % n_splits))
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n_runs} scrambled runs", flush=True)
    print(f"  shuffled labels : RMSE {np.mean([s[0] for s in shuffled]):.3f} "
          f"+/- {np.std([s[0] for s in shuffled]):.3f}   "
          f"rho {np.mean([s[1] for s in shuffled]):+.3f} "
          f"+/- {np.std([s[1] for s in shuffled]):.3f}")
    return pd.DataFrame(
        [{"labels": "real", "RMSE": r, "spearman": s} for r, s in real] +
        [{"labels": "shuffled", "RMSE": r, "spearman": s} for r, s in shuffled])


def match_decoys(actives, pool, n_per_active, cols, seed=SEED):
    """For each active, the nearest unused pool compounds in property space."""
    rng = np.random.default_rng(seed)
    P = pool[cols].to_numpy(dtype=float)
    A = actives[cols].to_numpy(dtype=float)
    mu, sd = P.mean(0), P.std(0) + 1e-9
    Pz, Az = (P - mu) / sd, (A - mu) / sd
    used, chosen = set(), []
    for i in range(len(Az)):
        d = np.linalg.norm(Pz - Az[i], axis=1)
        for j in np.argsort(d):
            if len(chosen) >= (i + 1) * n_per_active:
                break
            if j not in used:
                used.add(int(j))
                chosen.append(int(j))
    return pool.iloc[chosen]


def enrichment(scores_active, scores_decoy, fractions=(0.01, 0.05, 0.10)):
    """ROC-AUC and EF at the given fractions. Lower docking score is better,
    so scores are negated to make 'higher = more active'."""
    y_true = np.r_[np.ones(len(scores_active)), np.zeros(len(scores_decoy))]
    y_score = -np.r_[scores_active, scores_decoy]
    auc = roc_auc_score(y_true, y_score)
    order = np.argsort(-y_score)
    ranked = y_true[order]
    n, n_act = len(ranked), int(y_true.sum())
    out = {}
    for f in fractions:
        k = max(1, int(round(f * n)))
        out[f] = (ranked[:k].sum() / k) / (n_act / n)
    return auc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--decoys-per-active", type=int, default=20)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- 1. Y-randomisation --------------------------------------------
    print("=" * 66)
    print(f"1. Y-RANDOMISATION  ({args.runs} scrambled runs, "
          f"{args.splits} real splits)")
    print("=" * 66)
    df = pd.read_csv(PROC / f"{args.target}_curated.csv")
    X, keep = base.featurise(df["std_smiles"].tolist())
    df = df.iloc[keep].reset_index(drop=True)
    y = df["pActivity"].to_numpy()
    print(f"  {len(df):,} compounds")
    yr = y_randomisation(df, X, y, args.runs, args.splits)
    yr.to_csv(RESULTS / f"{args.target}_y_randomisation.csv", index=False)

    real_rho = yr[yr.labels == "real"].spearman.mean()
    shuf_rho = yr[yr.labels == "shuffled"].spearman
    p = float((shuf_rho >= real_rho).mean())
    print(f"\n  scrambled runs reaching the real model's rho: "
          f"{int((shuf_rho >= real_rho).sum())}/{len(shuf_rho)}  (p = {p:.3f})")
    if shuf_rho.mean() > 0.2:
        print("  WARNING: scrambled labels still give appreciable correlation. "
              "Some of the apparent signal is descriptor-space structure.")
    else:
        print("  PASS: scrambled labels destroy the signal, so the model is "
              "learning activity rather than dataset structure.")

    # ---- 2. docking enrichment -----------------------------------------
    print("\n" + "=" * 66)
    print("2. DOCKING ENRICHMENT vs PROPERTY-MATCHED DECOYS")
    print("=" * 66)
    dock_path = RESULTS / f"{args.target}_docking.csv"
    if not dock_path.exists():
        print("  no docking results; skipping")
        return
    d = pd.read_csv(dock_path)
    d["heavy"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in d.std_smiles]

    actives = d[d.novelty == "fingerprint-identical to a training compound"]
    pool = d[d.novelty != "fingerprint-identical to a training compound"]
    print(f"  actives (structurally identical to known TYK2 compounds): "
          f"{len(actives)}")
    if len(actives) < 5:
        print("  too few actives for a meaningful enrichment estimate")
        return

    cols = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "heavy"]
    decoys = match_decoys(actives, pool, args.decoys_per_active, cols)
    print(f"  property-matched decoys: {len(decoys)}")
    print(f"  {'property':<8}{'actives':>12}{'decoys':>12}")
    for c in ["mw", "logp", "tpsa", "heavy"]:
        print(f"  {c:<8}{actives[c].mean():>12.1f}{decoys[c].mean():>12.1f}")

    auc, ef = enrichment(actives.vina_best.to_numpy(),
                         decoys.vina_best.to_numpy())
    print(f"\n  ROC-AUC            : {auc:.3f}   (0.5 = random)")
    for f, v in ef.items():
        print(f"  EF at {f*100:>4.0f}%        : {v:.2f}x   (1.0 = random)")

    # same again on the subset with real potency data
    print("\n  restricted to actives of pIC50 >= 8 where known:")
    print("  (small n, reported for completeness)")

    pd.DataFrame([{"metric": "ROC_AUC", "value": auc}] +
                 [{"metric": f"EF_{int(f*100)}pct", "value": v}
                  for f, v in ef.items()]).to_csv(
        RESULTS / f"{args.target}_docking_enrichment.csv", index=False)

    print(f"\n  verdict: ", end="")
    if auc < 0.55:
        print("no enrichment. The docking scores do not distinguish known "
              "actives from property-matched decoys, so they carry no "
              "ranking information for this target.")
    elif auc < 0.65:
        print("marginal enrichment, too weak to rank a screening deck on.")
    else:
        print("meaningful enrichment; docking scores carry some ranking signal.")
    print(f"\nwritten -> results/{args.target}_y_randomisation.csv, "
          f"{args.target}_docking_enrichment.csv")


if __name__ == "__main__":
    main()
