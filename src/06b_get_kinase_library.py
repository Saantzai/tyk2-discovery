"""
Step 6a (alternative) — Build a kinase-focused screening library from ChEMBL.

Why this instead of a generic catalogue:

The first screen used a general-purpose ZINC library and found that ONE of
192,573 compounds fell inside the model's validated applicability domain. The
maximum Tanimoto similarity to any training compound across the whole library
was 0.533. A QSAR model trained on one target's inhibitors knows about that
target's chemotypes; a random catalogue does not contain them, whatever its
molecular weight.

A library of compounds with measured activity against OTHER human kinases
does contain them - hinge binders, aminopyrimidines, the privileged scaffolds
that kinase programmes converge on. It is therefore inside the chemistry the
model actually learned.

It also asks a better question. "Which known kinase inhibitors might also
inhibit TYK2?" is a real pharmacological question about off-target activity
and repurposing, not a hypothetical about molecules nobody has made.

Targets are resolved by gene symbol through the ChEMBL API rather than by
hard-coded CHEMBL IDs, which go stale and are easy to mistype.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data" / "library"
PROC = ROOT / "data" / "processed"

# A spread of human kinases across the kinome: JAK family (closest relatives
# of TYK2), tyrosine kinases, serine/threonine kinases, and lipid kinases.
# TYK2 itself is deliberately absent.
KINASES = [
    "JAK1", "JAK2", "JAK3",
    "EGFR", "ERBB2", "ABL1", "SRC", "LCK", "LYN", "FYN", "BTK", "SYK",
    "KDR", "FLT3", "KIT", "RET", "MET", "ALK", "FGFR1", "IGF1R", "AXL",
    "CDK1", "CDK2", "CDK4", "CDK6", "CDK9",
    "MAPK14", "MAPK1", "MAPK8", "GSK3B", "CSNK2A1",
    "AURKA", "AURKB", "PLK1", "CHEK1", "WEE1",
    "AKT1", "PDPK1", "MTOR", "PIK3CA", "PIK3CD",
    "ROCK1", "ROCK2", "PRKACA", "PIM1", "IRAK4", "TBK1", "RIPK2",
    "BRAF", "RAF1", "MAP2K1", "LRRK2", "DYRK1A", "HCK", "ITK", "ZAP70",
]

EXCLUDE_TARGETS = {"TYK2"}


def resolve_target(gene: str, client):
    """Find the single-protein human target for a gene symbol."""
    try:
        hits = client.target.filter(
            target_synonym__iexact=gene,
            target_type="SINGLE PROTEIN",
            organism="Homo sapiens",
        ).only(["target_chembl_id", "pref_name", "organism"])
        hits = list(hits)
        if not hits:
            hits = list(client.target.search(gene).filter(
                target_type="SINGLE PROTEIN", organism="Homo sapiens"
            ).only(["target_chembl_id", "pref_name", "organism"]))
        if not hits:
            return None
        return hits[0]["target_chembl_id"], hits[0].get("pref_name", gene)
    except Exception as exc:
        print(f"    {gene}: lookup failed ({type(exc).__name__})", file=sys.stderr)
        return None


def fetch_actives(target_id: str, client, max_per_target: int, pchembl_min: float):
    """Compounds with a measured, reasonably potent activity at this target."""
    try:
        q = client.activity.filter(
            target_chembl_id=target_id,
            pchembl_value__gte=pchembl_min,
        ).only(["molecule_chembl_id", "canonical_smiles", "pchembl_value"])
        out = []
        for rec in q:
            smi = rec.get("canonical_smiles")
            if not smi:
                continue
            out.append((rec["molecule_chembl_id"], smi))
            if len(out) >= max_per_target:
                break
        return out
    except Exception as exc:
        print(f"    fetch failed ({type(exc).__name__})", file=sys.stderr)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50",
                    help="curated training set, used to exclude known compounds")
    ap.add_argument("--max-per-target", type=int, default=4000)
    ap.add_argument("--pchembl-min", type=float, default=5.0,
                    help="minimum pChEMBL (5.0 = 10 uM) for a compound to count "
                         "as a kinase inhibitor at all")
    ap.add_argument("--kinases", nargs="*", default=None,
                    help="override the gene list")
    args = ap.parse_args()

    from chembl_webresource_client.new_client import new_client as client

    genes = [g for g in (args.kinases or KINASES) if g not in EXCLUDE_TARGETS]
    LIB.mkdir(parents=True, exist_ok=True)

    print(f"resolving {len(genes)} kinase targets in ChEMBL ...")
    resolved = {}
    for g in genes:
        r = resolve_target(g, client)
        if r:
            resolved[g] = r
            print(f"  {g:<10} {r[0]:<16} {r[1][:46]}")
        else:
            print(f"  {g:<10} NOT FOUND")
        time.sleep(0.1)
    print(f"\nresolved {len(resolved)}/{len(genes)} targets")
    if not resolved:
        raise SystemExit("no targets resolved; is ChEMBL reachable?")

    # compounds already in the TYK2 training set must not reappear here
    known_smiles = set()
    src = PROC / f"{args.target}_curated.csv"
    if src.exists():
        known = pd.read_csv(src)
        known_smiles = set(known["std_smiles"])
        print(f"excluding {len(known_smiles):,} compounds already in training")

    print(f"\nfetching actives (pChEMBL >= {args.pchembl_min}) ...")
    seen, rows = set(), []
    for g, (tid, name) in resolved.items():
        got = fetch_actives(tid, client, args.max_per_target, args.pchembl_min)
        added = 0
        for cid, smi in got:
            if cid in seen:
                continue
            seen.add(cid)
            rows.append({"id": cid, "smiles": smi, "from_kinase": g})
            added += 1
        print(f"  {g:<10} {len(got):>6,} records, {added:>6,} new "
              f"(running total {len(rows):,})", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n{len(df):,} unique compounds before standardisation")

    # canonicalise so the overlap check against training is meaningful
    print("canonicalising and removing training compounds ...")
    keep = []
    for r in df.itertuples(index=False):
        m = Chem.MolFromSmiles(r.smiles)
        if m is None:
            continue
        cs = Chem.MolToSmiles(m)
        if cs in known_smiles:
            continue
        keep.append({"id": r.id, "smiles": cs, "from_kinase": r.from_kinase})
    out_df = pd.DataFrame(keep).drop_duplicates("smiles").reset_index(drop=True)
    print(f"  {len(out_df):,} remain after canonicalisation and overlap removal")

    out = LIB / "kinase_focused.smi"
    with out.open("w") as fh:
        for r in out_df.itertuples(index=False):
            fh.write(f"{r.smiles}\t{r.id}\n")
    out_df.to_csv(LIB / "kinase_focused_provenance.csv", index=False)

    print(f"\n{len(out_df):,} compounds -> {out}")
    print(f"provenance (which kinase each came from) -> "
          f"{(LIB / 'kinase_focused_provenance.csv').name}")
    print("\nNext:")
    print(f"  python src/07_screen.py --target {args.target} "
          f"--library data/library/kinase_focused.smi --top 2000")


if __name__ == "__main__":
    main()
