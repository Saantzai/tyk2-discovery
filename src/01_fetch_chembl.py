"""
Step 1 — Fetch raw bioactivity data from ChEMBL.

Primary target : TYK2        (CHEMBL3553)
Comparator     : RORgamma-t  (CHEMBL1163101)

Writes one CSV per target to data/raw/. No filtering beyond the query itself —
all curation decisions are made in 02_curate.py so they are auditable.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TARGETS = {
    "TYK2": "CHEMBL3553",
    "RORGT": "CHEMBL1163101",
}

# Only these activity types; only exact measurements, only nM.
ACTIVITY_TYPES = ["IC50", "Ki"]

FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "assay_type",
    "target_chembl_id",
    "document_chembl_id",
    "data_validity_comment",
    "potential_duplicate",
]


def fetch(target_id: str, retries: int = 3) -> pd.DataFrame:
    from chembl_webresource_client.new_client import new_client

    act = new_client.activity
    frames = []
    for atype in ACTIVITY_TYPES:
        for attempt in range(1, retries + 1):
            try:
                q = act.filter(
                    target_chembl_id=target_id,
                    standard_type=atype,
                    standard_relation="=",
                    standard_units="nM",
                ).only(FIELDS)
                rows = list(q)
                print(f"  {atype}: {len(rows)} records", flush=True)
                frames.append(pd.DataFrame(rows))
                break
            except Exception as exc:  # network / server hiccup
                wait = 5 * attempt
                print(f"  {atype} attempt {attempt} failed ({exc}); retry in {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"could not fetch {atype} for {target_id}")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="TYK2", choices=list(TARGETS) + ["ALL"])
    args = ap.parse_args()

    names = list(TARGETS) if args.target == "ALL" else [args.target]
    RAW.mkdir(parents=True, exist_ok=True)

    for name in names:
        tid = TARGETS[name]
        print(f"Fetching {name} ({tid}) from ChEMBL ...", flush=True)
        df = fetch(tid)
        out = RAW / f"{name.lower()}_chembl_raw.csv"
        df.to_csv(out, index=False)
        print(f"  -> {len(df):,} rows written to {out.relative_to(ROOT)}\n", flush=True)


if __name__ == "__main__":
    main()
