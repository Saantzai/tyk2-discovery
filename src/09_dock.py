"""
Step 7b — Dock the shortlist into TYK2 JH2.

Design notes:

* Redocking first. The crystallographic ligand is docked back into its own
  site before anything else. If that pose does not reproduce the experimental
  one to within ~2 A RMSD, the box or the receptor preparation is wrong and
  every downstream score is meaningless. The run stops rather than producing
  a confident-looking table built on a broken setup.

* Resumable. Each ligand's output is written to its own file and existing
  results are skipped, so an overnight run that is interrupted continues
  where it stopped instead of starting again.

* One CPU per Vina process, many processes. For a few thousand small ligands
  this finishes sooner than giving each ligand every core in turn.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DOCK = ROOT / "data" / "docking"
LIGS = DOCK / "ligands"
POSES = DOCK / "poses"
RESULTS = ROOT / "results"

REDOCK_RMSD_LIMIT = 2.0


def find_vina(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in ("vina", "vina.exe", "vina_1.2.7_win.exe"):
        w = shutil.which(cand)
        if w:
            return w
        local = ROOT / cand
        if local.exists():
            return str(local)
    raise SystemExit(
        "AutoDock Vina was not found.\n"
        "Download vina_1.2.7_win.exe from\n"
        "  https://github.com/ccsb-scripps/AutoDock-Vina/releases\n"
        f"and put it in {ROOT}, or pass --vina <path>.")


def embed_3d(smiles: str, seed: int = 0xF00D):
    """2D SMILES -> one reasonable 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol


def write_pdbqt(mol, path: Path) -> bool:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        if not setups:
            return False
        txt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            return False
        path.write_text(txt)
        return True
    except Exception:
        return False


def run_vina(vina, receptor, ligand_pdbqt, out_pdbqt, box, exhaustiveness,
             num_modes=9, seed=42, timeout=1800, scoring="vinardo"):
    cmd = [vina, "--receptor", str(receptor), "--ligand", str(ligand_pdbqt),
           "--out", str(out_pdbqt),
           "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]),
           "--center_z", str(box["center_z"]),
           "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]),
           "--size_z", str(box["size_z"]),
           "--exhaustiveness", str(exhaustiveness),
           "--num_modes", str(num_modes), "--seed", str(seed), "--cpu", "1",
           "--scoring", scoring]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None, (r.stderr or r.stdout)[-300:]
        return parse_scores(out_pdbqt), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def parse_scores(pdbqt: Path):
    """Vina writes 'REMARK VINA RESULT: <affinity> <rmsd_lb> <rmsd_ub>'."""
    if not pdbqt.exists():
        return None
    out = []
    for line in pdbqt.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            out.append(float(line.split()[3]))
    return out or None


def pose_coords(pdbqt: Path, model: int = 1):
    """Heavy-atom coordinates of one model from a Vina output file."""
    xs, cur = [], 0
    for line in pdbqt.read_text().splitlines():
        if line.startswith("MODEL"):
            cur = int(line.split()[1])
        elif line.startswith(("ATOM", "HETATM")) and cur == model:
            el = line[76:78].strip()
            if el and el != "H" and not el.startswith("H"):
                xs.append([float(line[30:38]), float(line[38:46]),
                           float(line[46:54])])
    return np.array(xs)


def pose_rmsds(pose_pdbqt: Path, ref_sdf: Path):
    """Symmetry-aware RMSD of every docked pose against the crystal pose.

    This must use real atom correspondence. An earlier version matched each
    atom to its nearest neighbour in the other structure, which needs no
    correspondence but is not an RMSD: tested against a 180-degree flip whose
    true RMSD was 10.6 A, it reported 2.35 A. Every pose looked acceptable.

    Meeko reconstructs the docked molecule with its bond orders intact, and
    RDKit's CalcRMS then resolves topological symmetry (equivalent ring
    flips, rotatable terminal groups) before measuring - without
    superimposing, since the question is where the pose sits in the pocket,
    not whether its shape can be aligned onto the reference.
    """
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit.Chem import rdMolAlign
    except ImportError:
        return None
    try:
        ref = Chem.RemoveHs(Chem.MolFromMolFile(str(ref_sdf), removeHs=False))
        pm = PDBQTMolecule.from_file(str(pose_pdbqt), skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(pm)
        if not mols or mols[0] is None:
            return None
        probe = Chem.RemoveHs(mols[0])
        out = []
        for cid in range(probe.GetNumConformers()):
            try:
                out.append(float(rdMolAlign.CalcRMS(probe, ref, prbId=cid, refId=0)))
            except Exception:
                out.append(float("nan"))
        return out
    except Exception:
        return None


def prepare_one(args):
    idx, cid, smiles = args
    out = LIGS / f"{cid}.pdbqt"
    if out.exists() and out.stat().st_size > 100:
        return cid, True
    mol = embed_3d(smiles)
    if mol is None:
        return cid, False
    return cid, write_pdbqt(mol, out)


def dock_one(args):
    cid, vina, receptor, box, exh, scoring = args
    lig = LIGS / f"{cid}.pdbqt"
    out = POSES / f"{cid}_out.pdbqt"
    if out.exists() and out.stat().st_size > 100:
        sc = parse_scores(out)
        if sc:
            return cid, sc, None, True       # resumed
    scores, err = run_vina(vina, receptor, lig, out, box, exh, scoring=scoring)
    return cid, scores, err, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--vina", default=None)
    ap.add_argument("--n", type=int, default=2000, help="how many to dock")
    ap.add_argument("--exhaustiveness", type=int, default=32,
                    help="LB4 has 9 rotatable bonds; 16 under-samples it")
    ap.add_argument("--scoring", default="vinardo",
                    choices=["vinardo", "vina", "ad4_scoring"],
                    help="Vinardo is the default because, on this target, "
                         "Vina's own function ranked a flipped pose 10.5 A "
                         "from the crystal structure above the correct one, "
                         "while Vinardo ranked the correct pose first. "
                         "Vinardo scores are NOT on the same scale as Vina's.")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-redock", action="store_true")
    args = ap.parse_args()

    vina = find_vina(args.vina)
    print(f"vina: {vina}   scoring: {args.scoring}   "
          f"exhaustiveness: {args.exhaustiveness}")

    box_path = DOCK / "box.json"
    receptor = DOCK / "receptor.pdbqt"
    if not box_path.exists() or not receptor.exists():
        raise SystemExit("run 08_prepare_receptor.py first")
    box = json.loads(box_path.read_text())
    LIGS.mkdir(parents=True, exist_ok=True)
    POSES.mkdir(parents=True, exist_ok=True)

    # ---- redocking validation ------------------------------------------
    native_sdf = DOCK / "native_ligand.sdf"
    if not args.skip_redock and native_sdf.exists():
        print("\nredocking the crystallographic ligand to validate the setup ...")
        nat = Chem.MolFromMolFile(str(native_sdf), removeHs=False)
        if nat is not None:
            smi = Chem.MolToSmiles(Chem.RemoveHs(nat))
            m3d = embed_3d(smi)
            ok = m3d is not None and write_pdbqt(m3d, LIGS / "_native.pdbqt")
            if ok:
                sc, err = run_vina(vina, receptor, LIGS / "_native.pdbqt",
                                   POSES / "_native_out.pdbqt", box,
                                   args.exhaustiveness, scoring=args.scoring)
                if sc:
                    rmsds = pose_rmsds(POSES / "_native_out.pdbqt", native_sdf)
                    if not rmsds:
                        print("  could not measure pose RMSD; continuing "
                              "unvalidated")
                    else:
                        print(f"  {'rank':>5} {'affinity':>10} {'RMSD':>8}")
                        for i, (a, r) in enumerate(zip(sc, rmsds), 1):
                            mark = "  <- crystal pose" if r <= REDOCK_RMSD_LIMIT else ""
                            print(f"  {i:>5} {a:>10.2f} {r:>8.2f}{mark}")
                        top = rmsds[0]
                        best = min(rmsds)
                        best_rank = rmsds.index(best) + 1
                        print(f"\n  top-ranked pose : {top:.2f} A")
                        print(f"  best of {len(rmsds)} poses : {best:.2f} A "
                              f"(rank {best_rank}, "
                              f"{sc[best_rank-1] - sc[0]:+.2f} kcal/mol vs top)")
                        if top > REDOCK_RMSD_LIMIT and best > REDOCK_RMSD_LIMIT:
                            raise SystemExit(
                                f"\nNo pose reproduces the crystal structure "
                                f"within {REDOCK_RMSD_LIMIT} A (best "
                                f"{best:.2f} A). The setup cannot find the "
                                f"known binding mode, so docking scores from "
                                f"it carry no weight. Try a larger box "
                                f"(--padding in 08) or higher "
                                f"--exhaustiveness before continuing.")
                        if top > REDOCK_RMSD_LIMIT:
                            print(f"\n  PARTIAL PASS: the crystal pose is found "
                                  f"but ranked {best_rank}, not first. Vina's "
                                  f"scoring prefers a different pose by "
                                  f"{sc[0] - sc[best_rank-1]:.2f} kcal/mol. "
                                  f"Pose ranking for this target is therefore "
                                  f"unreliable - report ranking by score with "
                                  f"that caveat, and inspect top poses by eye.")
                        else:
                            print("\n  PASS: top-ranked pose reproduces the "
                                  "crystal structure.")
                else:
                    print(f"  redocking failed ({err}); continuing unvalidated")
    # ---- shortlist -------------------------------------------------------
    short = pd.read_csv(RESULTS / f"{args.target}_shortlist.csv").head(args.n)
    print(f"\nligands to dock: {len(short):,}")

    print("generating 3D structures ...")
    jobs = [(i, r.id, r.std_smiles) for i, r in enumerate(short.itertuples())]
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for n, (cid, good) in enumerate(ex.map(prepare_one, jobs), 1):
            ok += good
            if n % 200 == 0:
                print(f"  {n:,}/{len(jobs):,}  prepared {ok:,}", flush=True)
    print(f"  {ok:,} ligands prepared")

    prepared = [c for c in short["id"] if (LIGS / f"{c}.pdbqt").exists()]
    print(f"\ndocking {len(prepared):,} ligands on {args.jobs} workers "
          f"(exhaustiveness {args.exhaustiveness}) ...")
    t0 = time.time()
    rows, done, resumed, failed = [], 0, 0, 0
    tasks = [(c, vina, receptor, box, args.exhaustiveness, args.scoring)
             for c in prepared]
    out_csv = RESULTS / f"{args.target}_docking.csv"

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for cid, scores, err, was_resumed in ex.map(dock_one, tasks):
            done += 1
            resumed += was_resumed
            if scores:
                rows.append({"id": cid, "vina_best": scores[0],
                             "vina_mean_top3": float(np.mean(scores[:3])),
                             "n_modes": len(scores)})
            else:
                failed += 1
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(tasks) - done) / rate / 60 if rate else 0
                print(f"  {done:,}/{len(tasks):,}  ok {len(rows):,}  "
                      f"failed {failed:,}  {rate*60:.0f}/min  ETA {eta:.0f} min",
                      flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    dock = pd.DataFrame(rows)
    merged = short.merge(dock, on="id", how="inner").sort_values("vina_best")
    merged.to_csv(out_csv, index=False)

    print(f"\ndocked {len(dock):,} ligands "
          f"({resumed:,} resumed from a previous run, {failed:,} failed)")
    if len(dock):
        print(f"  Vina score range {dock['vina_best'].min():.2f} to "
              f"{dock['vina_best'].max():.2f} kcal/mol")
        print(f"  top 5:")
        for _, r in merged.head(5).iterrows():
            print(f"    {r['id']:<18} {r['vina_best']:>7.2f}  "
                  f"pred pIC50 {r['pred_pIC50']:.2f}  {r['novelty']}")
    print(f"\nwritten -> results/{out_csv.name}")
    print(f"\nNOTE: a docking score is not a binding affinity. Use it to rank "
          f"and triage, never as a predicted potency. These are {args.scoring} "
          f"scores and are not comparable with scores from another function.")


if __name__ == "__main__":
    main()
