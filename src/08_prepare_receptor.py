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
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCK = ROOT / "data" / "docking"

PDB_ID = "6NZQ"
LIGAND_CODE = "LB4"          # the co-crystallised allosteric inhibitor
CHAIN = "A"
# Padding beyond the crystallographic ligand's extent, per side. 5 A leaves a
# large flexible ligand barely able to reorient inside the box: with 9
# rotatable bonds and an 18 A longest dimension, LB4 could not rotate within
# a 19 A box axis, and the top-ranked redocked pose came out 10.5 A from the
# crystal structure. 9 A per side gives the search room to work.
BOX_PADDING = 9.0
# A rejected residue is unsafe to delete if it could contact the ligand.
# Measured against the CRYSTALLOGRAPHIC LIGAND, not the box: the box is
# deliberately padded for sampling, so "inside the box" says nothing about
# proximity to the pocket. In 6NZQ the residues lining the site sit 1.9-2.7 A
# from the ligand, while the nearest incomplete residue is 9.7 A away.
NEAR_SITE_CUTOFF = 6.0       # Angstrom from the nearest ligand atom

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


def _formula_counts(pdb_lines) -> dict:
    """Element counts from the element column of PDB ATOM/HETATM lines."""
    from collections import Counter
    c = Counter()
    for l in pdb_lines:
        el = l[76:78].strip() or l[12:16].strip()[:1]
        c[el.capitalize()] += 1
    return dict(c)


def _mol_formula_counts(mol) -> dict:
    """Element counts for an RDKit molecule, hydrogens included."""
    from collections import Counter
    from rdkit import Chem
    m = Chem.AddHs(mol)
    return dict(Counter(a.GetSymbol() for a in m.GetAtoms()))


def _fetch_ligand_template(code: str):
    """The PDB Chemical Component Dictionary's definition of a ligand,
    which carries the bond orders a PDB coordinate file cannot."""
    import requests
    from rdkit import Chem
    urls = [
        f"https://files.rcsb.org/ligands/download/{code.upper()}_ideal.sdf",
        f"https://files.rcsb.org/ligands/download/{code.upper()}_model.sdf",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=(10, 60))
            if r.status_code != 200 or not r.content:
                print(f"    {url.rsplit('/', 1)[-1]}: HTTP {r.status_code}")
                continue
            mol = Chem.MolFromMolBlock(r.text, removeHs=False)
            if mol is not None:
                print(f"    template from {url.rsplit('/', 1)[-1]}")
                return mol
            print(f"    {url.rsplit('/', 1)[-1]}: unparseable")
        except Exception as exc:
            print(f"    {url.rsplit('/', 1)[-1]}: {type(exc).__name__}")
    return None


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
    size = np.maximum(size, 25.0)            # never smaller than a usable box
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
    # Crystal structures routinely contain residues with unresolved side-chain
    # atoms - flexible surface lysines, glutamates, arginines. Meeko cannot
    # parameterise an incomplete residue and refuses the whole structure.
    #
    # Deleting them is standard practice, but ONLY when they are far from the
    # binding site: deleting a pocket residue silently changes what is being
    # docked. So this runs Meeko, reads back which residues it rejected,
    # checks every one against the docking box, and refuses to continue if
    # any lies inside or close to it.
    print("preparing receptor PDBQT (Meeko) ...")
    out_base = DOCK / "receptor"
    base_cmd = [sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
                "--read_pdb", str(prot_pdb), "-o", str(out_base), "-p",
                "--default_altloc", "A"]

    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return None

    r = run(base_cmd)
    if r is None:
        raise SystemExit(
            "Could not run Meeko. Install it with:  pip install meeko gemmi")

    if r.returncode != 0:
        blob = (r.stdout or "") + (r.stderr or "")
        bad = sorted({m for m in re.findall(r"residue_key='([^']+)'", blob)})
        if not bad:
            print(blob[-2500:])
            raise SystemExit("Meeko failed and named no specific residue.")

        print(f"  {len(bad)} residues could not be parameterised "
              f"(incomplete side chains in the crystal structure)")

        # how close is each to the docking box?
        res_xyz = {}
        for line in prot_pdb.read_text().splitlines():
            if line.startswith("ATOM"):
                key = f"{line[21]}:{line[22:26].strip()}"
                res_xyz.setdefault(key, []).append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])])

        lig_xyz = coords(ligand)
        near = []
        for key in bad:
            pts = np.array(res_xyz.get(key, []))
            if pts.size == 0:
                continue
            d_lig = float(np.linalg.norm(
                pts[:, None, :] - lig_xyz[None, :, :], axis=-1).min())
            flag = "   CONTACTS LIGAND" if d_lig < NEAR_SITE_CUTOFF else ""
            print(f"    {key:<10} {d_lig:>6.1f} A from the ligand{flag}")
            if d_lig < NEAR_SITE_CUTOFF:
                near.append((key, d_lig))

        if near:
            raise SystemExit(
                f"\n{len(near)} incomplete residue(s) lie within "
                f"{NEAR_SITE_CUTOFF} A of the crystallographic ligand:\n" +
                "\n".join(f"  {k}  {d:.1f} A" for k, d in near) +
                "\n\nThese line the pocket, so deleting them would change "
                "what is being docked. Rebuild their side chains first "
                "(PDBFixer or equivalent), or choose another structure. "
                "Not proceeding.")

        print(f"  all {len(bad)} are >= {NEAR_SITE_CUTOFF} A from the ligand "
              f"and none line the pocket; deleting them and retrying")
        (DOCK / "deleted_residues.txt").write_text("\n".join(bad) + "\n")
        r = run(base_cmd + ["-d", ",".join(bad)])

    if r is None or r.returncode != 0:
        blob = ((r.stdout or "") + (r.stderr or "")) if r else ""
        print(blob[-2500:])
        raise SystemExit("Meeko could not prepare the receptor.")

    pdbqt = out_base.with_suffix(".pdbqt")
    if not pdbqt.exists():
        cand = sorted(DOCK.glob("receptor*.pdbqt"))
        if cand:
            pdbqt = cand[0]
    if not pdbqt.exists():
        raise SystemExit("receptor PDBQT was not produced")
    n_atoms = sum(1 for l in pdbqt.read_text().splitlines()
                  if l.startswith(("ATOM", "HETATM")))
    print(f"  receptor PDBQT: {n_atoms:,} atoms -> {pdbqt.name}")

    # ---- native ligand as SDF for redocking validation ------------------
    # A PDB file stores coordinates and element symbols, and nothing about
    # bond orders - there is no field for them, and CONECT records only give
    # connectivity. Reading a ligand straight from PDB therefore produces a
    # molecule with every bond single: aromatic rings become saturated, amide
    # C=O becomes C-OH. Redocking THAT is a test of the wrong molecule.
    #
    # So the bond orders are taken from the PDB Chemical Component
    # Dictionary's definition of this ligand and transferred onto the
    # crystallographic coordinates, and the result is checked against the
    # formula actually present in the structure before it is trusted.
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    lig_mol = Chem.MolFromPDBBlock(lig_pdb.read_text(), removeHs=False,
                                   sanitize=False)
    if lig_mol is None:
        print("  WARNING: could not read the ligand; redocking will be skipped")
    else:
        obs = _formula_counts(ligand)
        print(f"  ligand in structure: "
              f"{' '.join(f'{k}{v}' for k, v in sorted(obs.items()))}")

        template = _fetch_ligand_template(args.ligand)
        if template is None:
            print(f"  WARNING: could not fetch the definition of "
                  f"{args.ligand} from the PDB chemical dictionary.")
            print(f"  Bond orders are unknown, so redocking validation would "
                  f"test the wrong molecule. Run 09_dock.py with "
                  f"--skip-redock if you accept that, but say so in the "
                  f"write-up.")
        else:
            tmpl_counts = _mol_formula_counts(template)
            if tmpl_counts != obs:
                print(f"  WARNING: the fetched template "
                      f"({' '.join(f'{k}{v}' for k, v in sorted(tmpl_counts.items()))}) "
                      f"does not match the structure. Not using it.")
            else:
                try:
                    fixed = AllChem.AssignBondOrdersFromTemplate(template, lig_mol)
                    Chem.SanitizeMol(fixed)
                    arom = sum(1 for a in fixed.GetAtoms() if a.GetIsAromatic())
                    dbl = sum(1 for b in fixed.GetBonds()
                              if b.GetBondTypeAsDouble() == 2)
                    Chem.MolToMolFile(fixed, str(DOCK / "native_ligand.sdf"))
                    print(f"  bond orders assigned from the PDB definition: "
                          f"{arom} aromatic atoms, {dbl} double bonds")
                    print(f"  SMILES: {Chem.MolToSmiles(Chem.RemoveHs(fixed))}")
                    print(f"  -> native_ligand.sdf")
                except Exception as exc:
                    print(f"  WARNING: bond-order assignment failed ({exc}); "
                          f"redocking validation will be unreliable")

    print(f"\nready. Files in {DOCK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
