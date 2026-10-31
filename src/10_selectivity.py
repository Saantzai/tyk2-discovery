"""
Step 9 — Rank candidates on TYK2 selectivity rather than TYK2 potency.

The problem this fixes
----------------------
All 20 top candidates from the potency-ranked shortlist come from JAK1. The
model was trained on TYK2 data alone, so it has learned "JAK-family chemotype"
as much as "TYK2 activity", and it cannot tell a TYK2-selective compound from
a pan-JAK one. Since selectivity over JAK1/2/3 is the entire clinical rationale
for targeting TYK2 - pan-JAK inhibition carries anaemia, neutropenia and
infection risk - a shortlist of probable pan-JAK compounds is a shortlist of
probable failures.

The approach
------------
Build the same kind of model for JAK1, JAK2 and JAK3, then score each
candidate on the margin between its TYK2 activity and its best JAK activity.

Crucially, measured values are preferred over predicted ones. The screening
library was drawn from ChEMBL kinase actives, so for many candidates the JAK
activity has actually been measured - using a prediction where a measurement
exists would throw away the better evidence. Each candidate's JAK figure is
therefore labelled as measured or predicted, and predictions carry their own
per-model applicability-domain flag.
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

ANTI_TARGETS = {"JAK1": "CHEMBL2835", "JAK2": "CHEMBL2971", "JAK3": "CHEMBL2148"}
SEED = 42
DOMAIN_THRESHOLD = 0.52        # as measured in step 6 for the TYK2 model
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = _load("04_baseline_model.py")
curate = _load("02_curate.py")

FIELDS = ["molecule_chembl_id", "canonical_smiles", "standard_type",
          "standard_relation", "standard_value", "standard_units",
          "pchembl_value", "assay_chembl_id", "assay_type",
          "target_chembl_id", "document_chembl_id",
          "data_validity_comment", "potential_duplicate"]


CHUNK = 1000


def fetch(target_id: str, retries: int = 4) -> pd.DataFrame:
    """Retrieve one target's IC50 records, in slices.

    Asking ChEMBL for a well-studied kinase's entire activity set in one
    request reliably fails with HttpApplicationError - the response is simply
    too large for the server to assemble. Slicing the query keeps each request
    small; a slice that still fails is retried, then skipped, so one bad slice
    costs a few hundred records rather than the whole target.
    """
    from chembl_webresource_client.new_client import new_client
    act = new_client.activity
    q = act.filter(target_chembl_id=target_id, standard_type="IC50",
                   standard_relation="=", standard_units="nM").only(FIELDS)

    total = None
    for attempt in range(1, retries + 1):
        try:
            total = len(q)
            break
        except Exception as exc:
            print(f"    counting records, attempt {attempt} "
                  f"({type(exc).__name__})", flush=True)
            time.sleep(5 * attempt)
    if total is None:
        raise RuntimeError("could not count records")
    print(f"    {total:,} records to retrieve", flush=True)

    rows, skipped = [], 0
    for start in range(0, total, CHUNK):
        got = None
        for attempt in range(1, retries + 1):
            try:
                got = list(q[start:start + CHUNK])
                break
            except Exception as exc:
                if attempt == retries:
                    print(f"    slice {start:,}-{start+CHUNK:,} failed "
                          f"({type(exc).__name__}), skipping", flush=True)
                time.sleep(4 * attempt)
        if got is None:
            skipped += CHUNK
            continue
        rows.extend(got)
        if (start // CHUNK) % 5 == 0 or start + CHUNK >= total:
            print(f"    {len(rows):,}/{total:,}", flush=True)
    if not rows:
        raise RuntimeError("no records retrieved")
    if skipped:
        print(f"    note: ~{skipped:,} records skipped after repeated failures")
    return pd.DataFrame(rows)


def build_model(name: str, target_id: str, refresh: bool):
    """Curated dataset plus a trained model for one anti-target."""
    raw_path = RAW / f"{name.lower()}_chembl_raw.csv"
    cur_path = PROC / f"{name.lower()}_curated.csv"

    # Only touch the network when the curated set is actually missing: the
    # raw file is an intermediate, and re-fetching because it was cleaned up
    # wastes ten minutes of API calls for a file already derived from it.
    if refresh or not cur_path.exists():
        if refresh or not raw_path.exists():
            print(f"  fetching {name} ({target_id}) from ChEMBL ...", flush=True)
            fetch(target_id).to_csv(raw_path, index=False)
        print(f"  curating {name} ...", flush=True)
        # 02_curate.curate gained an activity-type argument partway through the
        # project. Call it either way rather than requiring a particular
        # revision of a script whose earlier output is already committed.
        import inspect
        raw_df = pd.read_csv(raw_path)
        if len(inspect.signature(curate.curate).parameters) >= 3:
            clean, attrition = curate.curate(raw_df, name.lower(), "IC50")
        else:
            raw_df = raw_df[raw_df["standard_type"] == "IC50"].copy()
            clean, attrition = curate.curate(raw_df, name.lower())
        clean.to_csv(cur_path, index=False)
        attrition.to_csv(RESULTS / f"{name.lower()}_attrition.csv", index=False)
    else:
        print(f"  {name}: using cached {cur_path.name}", flush=True)

    df = pd.read_csv(cur_path)
    X, keep = base.featurise(df["std_smiles"].tolist())
    df = df.iloc[keep].reset_index(drop=True)
    y = df["pActivity"].to_numpy()
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED)
    model.fit(X, y)
    fps = [_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in df["std_smiles"]]
    print(f"  {name}: {len(df):,} compounds, pIC50 "
          f"{y.min():.2f}-{y.max():.2f}", flush=True)
    return {"name": name, "df": df, "model": model, "fps": fps,
            "lookup": dict(zip(df["inchikey"], df["pActivity"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the anti-target data from ChEMBL")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    dock_path = RESULTS / f"{args.target}_docking.csv"
    src = dock_path if dock_path.exists() else RESULTS / f"{args.target}_shortlist.csv"
    cand = pd.read_csv(src)
    print(f"candidates: {len(cand):,} from {src.name}")

    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    print("\nbuilding anti-target models ...")
    anti = {}
    for n, t in ANTI_TARGETS.items():
        try:
            anti[n] = build_model(n, t, args.refresh)
        except Exception as exc:
            # A selectivity margin over two JAKs is worth far more than no
            # result at all; which ones were used is recorded alongside it.
            print(f"  {n}: FAILED ({type(exc).__name__}: {exc}) - continuing "
                  f"without it", flush=True)
    if not anti:
        raise SystemExit("no anti-target model could be built")
    if len(anti) < len(ANTI_TARGETS):
        missing = sorted(set(ANTI_TARGETS) - set(anti))
        print(f"\n  WARNING: selectivity computed against {sorted(anti)} only; "
              f"{missing} unavailable. Re-run later to include them.")

    # featurise the candidates once
    Xc, kc = base.featurise(cand["std_smiles"].tolist())
    cand = cand.iloc[kc].reset_index(drop=True)
    cand_fps = [_gen.GetFingerprint(Chem.MolFromSmiles(s))
                for s in cand["std_smiles"]]

    print("\nscoring candidates against each anti-target ...")
    for name, a in anti.items():
        measured, source, sim = [], [], []
        pred = a["model"].predict(Xc)
        for i, row in enumerate(cand.itertuples()):
            key = getattr(row, "inchikey", None)
            if key in a["lookup"]:
                measured.append(a["lookup"][key])
                source.append("measured")
                sim.append(1.0)
            else:
                measured.append(float(pred[i]))
                source.append("predicted")
                sim.append(float(max(DataStructs.BulkTanimotoSimilarity(
                    cand_fps[i], a["fps"]))))
        cand[f"{name}_pIC50"] = measured
        cand[f"{name}_source"] = source
        cand[f"{name}_nnsim"] = sim
        n_meas = source.count("measured")
        n_dom = sum(s >= DOMAIN_THRESHOLD for s in sim)
        print(f"  {name}: {n_meas:,} measured, "
              f"{len(source)-n_meas:,} predicted "
              f"({n_dom:,} of those in domain)")

    jak_cols = [f"{n}_pIC50" for n in anti]
    cand["JAK_max"] = cand[jak_cols].max(axis=1)
    cand["JAK_worst"] = cand[jak_cols].idxmax(axis=1).str.replace("_pIC50", "")
    cand["selectivity"] = cand["pred_pIC50"] - cand["JAK_max"]
    cand["all_measured"] = (cand[[f"{n}_source" for n in anti]]
                            == "measured").all(axis=1)
    cand["anti_targets_used"] = "|".join(sorted(anti))

    cand = cand.sort_values("selectivity", ascending=False).reset_index(drop=True)
    cand.to_csv(RESULTS / f"{args.target}_selectivity.csv", index=False)

    print(f"\nselectivity margin (TYK2 minus best JAK), all {len(cand):,}:")
    print(f"  median {cand.selectivity.median():+.2f}  "
          f"range {cand.selectivity.min():+.2f} to {cand.selectivity.max():+.2f}")
    print(f"  TYK2-selective (margin > 0): {(cand.selectivity>0).sum():,} "
          f"({(cand.selectivity>0).mean()*100:.1f}%)")
    print(f"  margin > 1 log unit        : {(cand.selectivity>1).sum():,}")
    print(f"  all three JAKs measured    : {cand.all_measured.sum():,}")

    cols = (["id", "pred_pIC50"] + jak_cols +
            ["JAK_max", "JAK_worst", "selectivity", "all_measured", "vina_best"])
    cols = [c for c in cols if c in cand.columns]
    print(f"\n=== TOP {args.top} BY SELECTIVITY MARGIN ===")
    print(cand.head(args.top)[cols].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n=== for comparison, the TOP {args.top} BY TYK2 POTENCY ===")
    pot = cand.nlargest(args.top, "pred_pIC50")
    print(pot[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    overlap = len(set(cand.head(args.top).id) & set(pot.id))
    print(f"\noverlap between the two rankings: {overlap}/{args.top}")
    print(f"\nwritten -> results/{args.target}_selectivity.csv")
    print("\nNOTE: a positive margin means the model expects MORE TYK2 activity "
          "than JAK activity. Where the JAK figure is measured this is a real "
          "comparison; where predicted, it inherits that model's error.")


if __name__ == "__main__":
    main()
