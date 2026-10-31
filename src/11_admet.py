"""
Step 10 — ADMET liabilities for the defensible candidate set.

Scope: the candidates whose selectivity margin rests on a MEASURED JAK value,
not a predicted one. Running ADMET on the rest would add a third layer of
model error on top of a margin that is already an artefact (step 9).

Predictions come from ADMET-AI, trained on the Therapeutics Data Commons
suite. Each is a model output, not a measurement: used here to rank and to
flag, never as a property value to quote. Thresholds are stated rather than
implied, so a reader can disagree with them.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Liability thresholds. Classification models return a probability; the
# regression ones are in the units named. These are conventional screening
# cut-offs, not bright lines - a flag means "look at this", not "reject".
FLAGS = {
    "hERG":                 (">", 0.5,   "hERG block (cardiotoxicity)"),
    "DILI":                 (">", 0.5,   "drug-induced liver injury"),
    "AMES":                 (">", 0.5,   "mutagenicity"),
    "CYP3A4_Veith":         (">", 0.5,   "CYP3A4 inhibition (interactions)"),
    "CYP2D6_Veith":         (">", 0.5,   "CYP2D6 inhibition (interactions)"),
    "ClinTox":              (">", 0.5,   "clinical toxicity"),
    "Carcinogens_Lagunin":  (">", 0.5,   "carcinogenicity"),
    "Bioavailability_Ma":   ("<", 0.5,   "poor oral bioavailability"),
    "Solubility_AqSolDB":   ("<", -5.0,  "poor aqueous solubility (log mol/L)"),
    "Caco2_Wang":           ("<", -5.15, "low permeability (log cm/s)"),
}

REPORT = ["hERG", "DILI", "AMES", "CYP3A4_Veith", "CYP2D6_Veith",
          "Bioavailability_Ma", "Solubility_AqSolDB", "Caco2_Wang",
          "Half_Life_Obach", "QED"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--all", action="store_true",
                    help="score every candidate, not just the defensible set")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    src = RESULTS / f"{args.target}_selectivity.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}; run 10_selectivity.py first")
    d = pd.read_csv(src)
    print(f"loaded {len(d):,} candidates")

    # which JAK value drove the margin, and was it measured?
    def worst_source(r):
        col = f"{r['JAK_worst']}_source"
        return r[col] if col in r else "predicted"
    d["worst_measured"] = d.apply(worst_source, axis=1) == "measured"

    if args.all:
        sel = d.copy()
        print("scoring ALL candidates (margins may be prediction artefacts)")
    else:
        sel = d[d.worst_measured & (d.selectivity > 0)].copy()
        print(f"defensible set (measured worst-JAK, positive margin): {len(sel):,}")
    if sel.empty:
        raise SystemExit("nothing to score")

    print("\nloading ADMET-AI models (first run downloads them) ...")
    from admet_ai import ADMETModel
    model = ADMETModel()

    print(f"predicting {len(sel):,} compounds ...")
    preds = model.predict(smiles=sel["std_smiles"].tolist())
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    preds = preds.reset_index(drop=True)
    sel = sel.reset_index(drop=True)
    out = pd.concat([sel, preds[[c for c in preds.columns
                                 if c not in sel.columns]]], axis=1)

    # apply the flags
    print("\nliability flags (threshold in brackets):")
    flag_cols = []
    for col, (op, cut, label) in FLAGS.items():
        if col not in out.columns:
            print(f"  {label}: column missing, skipped")
            continue
        hit = out[col] > cut if op == ">" else out[col] < cut
        name = f"flag_{col}"
        out[name] = hit
        flag_cols.append(name)
        print(f"  {label} [{op}{cut}]: {int(hit.sum()):,}/{len(out):,} "
              f"({hit.mean()*100:.0f}%)")

    # A flag that fires on almost every compound carries no information about
    # any of them - it separates nothing. Rather than let such a flag reject
    # the entire set, it is reported and then excluded from the count, with
    # the exclusion stated. (On this set the DILI model fires on 96% of
    # compounds, median probability 0.90; it is not discriminating.)
    NON_DISCRIMINATIVE = 0.90
    informative, constant = [], []
    for name in flag_cols:
        (constant if out[name].mean() >= NON_DISCRIMINATIVE
         else informative).append(name)
    if constant:
        print(f"\n  excluded from the liability count as non-discriminative "
              f"(fires on >={NON_DISCRIMINATIVE:.0%} of the set):")
        for name in constant:
            col = name[5:]
            print(f"    {col}: {out[name].mean()*100:.0f}% flagged, median "
                  f"probability {out[col].median():.2f} — reported per "
                  f"compound, not used as a filter")
    out["n_flags"] = out[informative].sum(axis=1) if informative else 0
    print(f"\nliability count per compound:")
    for n, c in sorted(out.n_flags.value_counts().items()):
        print(f"  {n} flag(s): {c:,} compounds")

    clean = out[out.n_flags == 0]
    print(f"\ncompounds with no informative liability flag: {len(clean):,}")

    out = out.sort_values(["n_flags", "selectivity"],
                          ascending=[True, False]).reset_index(drop=True)
    out.to_csv(RESULTS / f"{args.target}_admet.csv", index=False)

    cols = (["id", "selectivity", "pred_pIC50", "n_flags"]
            + [c for c in REPORT if c in out.columns])
    print(f"\n=== TOP {args.top}: fewest liabilities, then best margin ===")
    print(out.head(args.top)[cols].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nwritten -> results/{args.target}_admet.csv")
    print("\nNOTE: every ADMET value here is a model prediction, typically "
          "0.7-0.9 AUC on held-out data. A flag is a reason to look, not a "
          "reason to reject, and an absent flag is not a safety claim.")


if __name__ == "__main__":
    main()
