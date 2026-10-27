"""
Step 3 — Exploratory analysis.

Produces the figures and the two numbers that decide the modelling strategy:
  (a) how concentrated the chemistry is in a few scaffolds (drives the need
      for a scaffold split rather than a random one)
  (b) how many activity cliffs there are (sets a ceiling on what any
      fingerprint-based model can achieve)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RESULTS = ROOT / "results"

ACTIVE_CUT = 7.0          # pIC50 >= 7  (<= 100 nM)
CLIFF_SIM = 0.75          # Tanimoto above which two compounds are "similar"
CLIFF_DELTA = 1.5         # log units of potency difference that counts as a cliff

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(s) for s in df["std_smiles"]]
    df = df.copy()
    df["logp"] = [Crippen.MolLogP(m) for m in mols]
    df["hba"] = [Descriptors.NumHAcceptors(m) for m in mols]
    df["hbd"] = [Descriptors.NumHDonors(m) for m in mols]
    df["tpsa"] = [Descriptors.TPSA(m) for m in mols]
    df["rotb"] = [Descriptors.NumRotatableBonds(m) for m in mols]
    df["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    df["active"] = df["pActivity"] >= ACTIVE_CUT
    return df, mols


def cliffs(df, mols):
    fps = [mfpgen.GetFingerprint(m) for m in mols]
    p = df["pActivity"].to_numpy()
    n = len(fps)
    pairs = 0
    cliff_pairs = 0
    examples = []
    for i in range(n):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        if sims.size == 0:
            continue
        idx = np.where(sims >= CLIFF_SIM)[0]
        pairs += idx.size
        for j in idx:
            d = abs(p[i] - p[i + 1 + j])
            if d >= CLIFF_DELTA:
                cliff_pairs += 1
                if len(examples) < 10:
                    examples.append({
                        "cmpd_a": df.iloc[i]["molecule_chembl_id"],
                        "cmpd_b": df.iloc[i + 1 + j]["molecule_chembl_id"],
                        "tanimoto": round(float(sims[j]), 3),
                        "delta_pActivity": round(float(d), 2),
                    })
    return pairs, cliff_pairs, pd.DataFrame(examples)


def figures(df, target):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    ax[0, 0].hist(df["pActivity"], bins=40, color="#3b6ea5", edgecolor="white")
    ax[0, 0].axvline(ACTIVE_CUT, color="#c0392b", ls="--", lw=1.2)
    ax[0, 0].set_xlabel("pActivity"); ax[0, 0].set_ylabel("compounds")
    ax[0, 0].set_title(f"Potency distribution (cut at {ACTIVE_CUT})")

    ax[0, 1].scatter(df["mw"], df["logp"], s=6, alpha=0.35,
                     c=np.where(df["active"], "#c0392b", "#7f8c8d"))
    ax[0, 1].axvline(500, color="k", lw=0.6, ls=":")
    ax[0, 1].axhline(5, color="k", lw=0.6, ls=":")
    ax[0, 1].set_xlabel("MW (Da)"); ax[0, 1].set_ylabel("cLogP")
    ax[0, 1].set_title("Property space (red = active, dotted = Lipinski)")

    top = df["scaffold"].value_counts().head(15)
    ax[1, 0].barh(range(len(top))[::-1], top.values, color="#3b6ea5")
    ax[1, 0].set_yticks(range(len(top))[::-1])
    ax[1, 0].set_yticklabels([f"scaffold {i+1}" for i in range(len(top))], fontsize=7)
    ax[1, 0].set_xlabel("compounds")
    ax[1, 0].set_title("15 most common Murcko scaffolds")

    counts = df["scaffold"].value_counts().values
    ax[1, 1].plot(np.arange(1, len(counts) + 1), np.cumsum(counts) / len(df) * 100,
                  color="#3b6ea5")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_xlabel("scaffolds (ranked, log scale)")
    ax[1, 1].set_ylabel("% of dataset covered")
    ax[1, 1].set_title("Scaffold concentration")

    fig.suptitle(f"{target.upper()} — curated ChEMBL set (n={len(df):,})", y=0.99)
    fig.tight_layout()
    out = FIG / f"{target}_eda.png"
    fig.savefig(out, bbox_inches="tight")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2")
    args = ap.parse_args()
    t = args.target.lower()

    src = PROC / f"{t}_curated.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}; run 02_curate.py first")

    df = pd.read_csv(src)
    df, mols = describe(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / f"{t}_curated_descriptors.csv", index=False)

    n_scaf = df["scaffold"].nunique()
    top10_share = df["scaffold"].value_counts().head(10).sum() / len(df) * 100
    singletons = (df["scaffold"].value_counts() == 1).sum()

    print(f"compounds              : {len(df):,}")
    print(f"actives (pAct >= {ACTIVE_CUT}) : {int(df['active'].sum()):,} "
          f"({df['active'].mean()*100:.1f}%)")
    print(f"unique Murcko scaffolds: {n_scaf:,}")
    print(f"top-10 scaffold share  : {top10_share:.1f}%")
    print(f"singleton scaffolds    : {singletons:,}")

    pairs, ncliff, ex = cliffs(df, mols)
    print(f"similar pairs (T>={CLIFF_SIM}) : {pairs:,}")
    print(f"activity cliffs (dpAct>={CLIFF_DELTA}) : {ncliff:,} "
          f"({(ncliff/pairs*100 if pairs else 0):.1f}% of similar pairs)")
    ex.to_csv(RESULTS / f"{t}_activity_cliffs.csv", index=False)

    out = figures(df, t)
    print(f"\nfigure -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
