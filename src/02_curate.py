"""
Step 2 — Curate and standardise.

Every filter logs how many compounds it removed. The attrition table it writes
(results/<target>_attrition.csv) goes straight into the methods section: a
reviewer can see exactly what was discarded and why.

Pipeline:
  parse SMILES -> drop ChEMBL validity-flagged rows -> organic-elements only
  -> largest fragment -> neutralise -> normalise -> canonical tautomer
  -> MW window -> pIC50 conversion -> deduplicate by InChIKey (median)
  -> flag compounds whose replicate measurements disagree by >1 log unit
"""
import argparse
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
MW_MIN, MW_MAX = 150.0, 900.0
MIN_HEAVY_ATOMS = 6
# Replicate handling. ChEMBL routinely holds the same compound measured in
# several assays, and spreads of ~1 log unit between labs are normal rather
# than erroneous. Discarding every such compound would strip out exactly the
# best-studied (and often most potent) chemistry, biasing the training set.
# Policy: take the median, keep the compound, record the spread, and exclude
# only irreconcilable measurements.
FLAG_LOG_UNITS = 1.0      # spread above this is recorded as unreliable
DROP_LOG_UNITS = 2.0      # spread above this is excluded outright

_lfc = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()
_norm = rdMolStandardize.Normalizer()
_taut = rdMolStandardize.TautomerEnumerator()


def organic_only(mol) -> bool:
    """True if every atom is a commonly drug-like element."""
    return all(a.GetSymbol() in ORGANIC for a in mol.GetAtoms())


def standardise(smiles: str):
    """Return (canonical_smiles, inchikey, mol) or None if the molecule fails.

    Order matters. We select the parent fragment FIRST (so a legitimate salt
    such as sodium acetylsalicylate keeps its organic parent), then apply the
    element and size checks to that parent. Checking elements on the molecule
    as supplied would throw away every salt form; checking only elements after
    fragment selection would let a platinum diammine complex through as
    ammonia. Both filters together are what make this safe.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _norm.normalize(mol)
        mol = _lfc.choose(mol)          # parent fragment, counterions dropped
        mol = _unc.uncharge(mol)
        mol = _taut.Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    if not organic_only(mol):           # parent must be an organic molecule
        return None
    if mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:   # not a drug-like entity
        return None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol), mol


def to_pic50(value_nm: float) -> float:
    """nM -> pIC50 (= -log10[M])."""
    if value_nm is None or value_nm <= 0:
        return math.nan
    return -math.log10(value_nm * 1e-9)


def curate(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = []

    def step(name, frame):
        log.append({"step": name, "rows": len(frame),
                    "unique_compounds": frame["molecule_chembl_id"].nunique()
                    if "molecule_chembl_id" in frame else None})
        return frame

    df = step("0. raw records from ChEMBL", df)

    df = step("1. drop ChEMBL validity-flagged",
              df[df["data_validity_comment"].isna()].copy())

    df = step("2. drop missing SMILES or value",
              df.dropna(subset=["canonical_smiles", "standard_value"]).copy())

    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = step("3. drop non-numeric / non-positive values",
              df[df["standard_value"] > 0].copy())

    # --- standardisation -------------------------------------------------
    recs = []
    for row in df.itertuples(index=False):
        res = standardise(row.canonical_smiles)
        if res is None:
            continue
        can, ikey, mol = res
        mw = Descriptors.MolWt(mol)
        recs.append({
            "molecule_chembl_id": row.molecule_chembl_id,
            "std_smiles": can,
            "inchikey": ikey,
            "mw": mw,
            "standard_type": row.standard_type,
            "standard_value_nm": row.standard_value,
            "pActivity": to_pic50(row.standard_value),
            "assay_chembl_id": row.assay_chembl_id,
            "document_chembl_id": row.document_chembl_id,
        })
    std = pd.DataFrame(recs)
    step("4. RDKit standardisation + organic-element filter", std)

    std = step("5. MW window %d-%d Da" % (MW_MIN, MW_MAX),
               std[(std["mw"] >= MW_MIN) & (std["mw"] <= MW_MAX)].copy())

    # --- deduplicate by structure ---------------------------------------
    g = std.groupby("inchikey")
    agg = g.agg(
        molecule_chembl_id=("molecule_chembl_id", "first"),
        std_smiles=("std_smiles", "first"),
        mw=("mw", "first"),
        pActivity=("pActivity", "median"),
        p_min=("pActivity", "min"),
        p_max=("pActivity", "max"),
        n_measurements=("pActivity", "size"),
        n_assays=("assay_chembl_id", "nunique"),
    ).reset_index()
    agg["p_spread"] = agg["p_max"] - agg["p_min"]
    step("6. deduplicate by InChIKey (median pActivity)", agg)

    agg["replicate_flag"] = agg["p_spread"] > FLAG_LOG_UNITS
    n_flag = int(agg["replicate_flag"].sum())
    clean = agg[agg["p_spread"].fillna(0) <= DROP_LOG_UNITS].copy()
    log.append({"step": f"7. drop irreconcilable replicates >{DROP_LOG_UNITS} log units "
                        f"({n_flag} of the remainder flagged >{FLAG_LOG_UNITS})",
                "rows": len(clean), "unique_compounds": len(clean)})

    clean = clean.sort_values("pActivity", ascending=False).reset_index(drop=True)
    attrition = pd.DataFrame(log)
    return clean, attrition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2")
    args = ap.parse_args()
    t = args.target.lower()

    src = RAW / f"{t}_chembl_raw.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}; run 01_fetch_chembl.py first")

    PROC.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(src)
    clean, attrition = curate(raw, t)

    clean.to_csv(PROC / f"{t}_curated.csv", index=False)
    attrition.to_csv(RESULTS / f"{t}_attrition.csv", index=False)

    print(attrition.to_string(index=False))
    if clean.empty:
        print("\nWARNING: curation produced an empty set - check the raw input.")
        return
    print(f"\nFinal curated set: {len(clean):,} unique compounds")
    print(f"  flagged as replicate-inconsistent: "
          f"{int(clean['replicate_flag'].sum()):,} (kept, column replicate_flag)")
    print(f"pActivity range {clean['pActivity'].min():.2f} - {clean['pActivity'].max():.2f}")
    print(f"written -> {(PROC / f'{t}_curated.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
