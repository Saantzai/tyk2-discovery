"""
Step 7a — Prepare the TYK2 JH2 receptor for docking.

Structure: PDB 6NZQ, TYK2 JH2 pseudokinase domain at 2.11 A with the
allosteric inhibitor LB4 bound. This is the deucravacitinib-class binding
site, not the ATP pocket of the JH1 kinase domain - which is the whole point
of the target, since JH2 binding is what gives TYK2 inhibitors their
selectivity over JAK1/2/3.

Produces:
  receptor.pdbqt    protein, hydrogens added, charges assigned
  native_ligand.sdf the crystallographic ligand, for redocking validation
  box.json          docking box centred on the crystallographic ligand
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCK = ROOT / "data" / "docking"

PDB_ID = "6NZQ"
LIGAND_CODE = "LB4"          # the co-crystallised allosteric inhibitor
CHAIN = "A"
BOX_PADDING = 5.0            # Angstrom beyond the native ligand's extent

SOURCES = [
    "https://files.rcsb.org/download/{pdb}.pdb",
    "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{lower}.ent",
]


def fetch_pdb(pdb_id: str, dest: Path) -> Path:
    import requests
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  using cached {dest.name}")
        return dest
    for tmpl in SOURCES:
        url = tmpl.format(pdb=pdb_id.upper(), lower=pdb_id.lower())
        try:
            r = requests.get(url, timeout=(10, 60))
            if r.status_code == 200 and len(r.content) > 10_000:
                dest.write_bytes(r.content)
                print(f"  downloaded {len(r.content):,} bytes from {url}")
                return dest
            print(f"  {url}: HTTP {r.status_code}")
        except Exception as exc:
            print(f"  {url}: {type(exc).__name__}")
    raise SystemExit(f"could not download {pdb_id} from any source")


def split_structure(pdb_path: Path, chain: str, lig_code: str):
    """Separate protein from the named ligand. Waters, ions and other
    heteroatoms are discarded - docking into a structure that still contains
    its crystallographic waters silently changes the pocket."""
    protein, ligand = [], []
    for line in pdb_path.read_text().splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        ch = line[21]
        resn = line[17:20].strip()
        altloc = line[16]
        if altloc not in (" ", "A"):          # keep one conformer only
            continue
        if ch != chain:
            continue
        if rec == "ATOM":
            # Strip hydrogens. Crystal structures at this resolution have
            # none, but a structure that does carry them makes Meeko fail
            # with an unhelpful "residue has no concrete definition" error.
            # Meeko adds hydrogens itself during preparation.
            el = line[76:78].strip()
            if el == "H" or (not el and line[12:16].strip().startswith("H")):
                continue
            protein.append(line)
        elif resn == lig_code:
            ligand.append(line)
    return protein, ligand


def coords(lines):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in lines])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=PDB_ID)
    ap.add_argument("--ligand", default=LIGAND_CODE)
    ap.add_argument("--chain", default=CHAIN)
    ap.add_argument("--padding", type=float, default=BOX_PADDING)
    args = ap.parse_args()

    DOCK.mkdir(parents=True, exist_ok=True)
    raw = DOCK / f"{args.pdb.lower()}.pdb"

    print(f"fetching {args.pdb} ...")
    fetch_pdb(args.pdb, raw)

    print(f"splitting chain {args.chain}, ligand {args.ligand} ...")
    protein, ligand = split_structure(raw, args.chain, args.ligand)
    if not protein:
        raise SystemExit(f"no protein atoms found in chain {args.chain}")
    if not ligand:
        raise SystemExit(f"ligand {args.ligand} not found in chain {args.chain}; "
                         f"check the PDB entry")
    print(f"  protein atoms {len(protein):,}, ligand atoms {len(ligand)}")

    prot_pdb = DOCK / "receptor_clean.pdb"
    prot_pdb.write_text("\n".join(protein) + "\nTER\nEND\n")
    lig_pdb = DOCK / "native_ligand.pdb"
    lig_pdb.write_text("\n".join(ligand) + "\nTER\nEND\n")

    # ---- docking box from the crystallographic ligand -------------------
    c = coords(ligand)
    centre = c.mean(axis=0)
    size = (c.max(axis=0) - c.min(axis=0)) + 2 * args.padding
    size = np.maximum(size, 18.0)            # never smaller than a usable box
    box = {"center_x": round(float(centre[0]), 3),
           "center_y": round(float(centre[1]), 3),
           "center_z": round(float(centre[2]), 3),
           "size_x": round(float(size[0]), 2),
           "size_y": round(float(size[1]), 2),
           "size_z": round(float(size[2]), 2),
           "pdb": args.pdb, "ligand": args.ligand, "chain": args.chain,
           "padding": args.padding}
    (DOCK / "box.json").write_text(json.dumps(box, indent=2))
    print(f"  box centre ({box['center_x']}, {box['center_y']}, "
          f"{box['center_z']})  size ({box['size_x']}, {box['size_y']}, "
          f"{box['size_z']})")

    # ---- receptor PDBQT via Meeko ---------------------------------------
    print("preparing receptor PDBQT (Meeko) ...")
    out_base = DOCK / "receptor"
    cmd = [sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
           "--read_pdb", str(prot_pdb), "-o", str(out_base), "-p"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # older/newer Meeko expose the script rather than the module
        cmd = ["mk_prepare_receptor.py", "--read_pdb", str(prot_pdb),
               "-o", str(out_base), "-p"]
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(
            "Meeko could not prepare the receptor. Most often this is a "
            "residue it does not recognise (a modified residue, or a gap in "
            "the chain). The message above names it; -d <resid> deletes it.")

    pdbqt = out_base.with_suffix(".pdbqt")
    if not pdbqt.exists():
        cand = list(DOCK.glob("receptor*.pdbqt"))
        if cand:
            pdbqt = cand[0]
    if not pdbqt.exists():
        raise SystemExit("receptor PDBQT was not produced")
    n_atoms = sum(1 for l in pdbqt.read_text().splitlines()
                  if l.startswith(("ATOM", "HETATM")))
    print(f"  receptor PDBQT: {n_atoms:,} atoms -> {pdbqt.name}")

    # ---- native ligand as SDF for redocking validation ------------------
    from rdkit import Chem
    mol = Chem.MolFromPDBBlock(lig_pdb.read_text(), removeHs=False)
    if mol is not None:
        Chem.MolToMolFile(mol, str(DOCK / "native_ligand.sdf"))
        print(f"  native ligand: {mol.GetNumAtoms()} atoms -> native_ligand.sdf")
    else:
        print("  WARNING: could not read the native ligand with RDKit; "
              "redocking validation will be skipped")

    print(f"\nready. Files in {DOCK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
