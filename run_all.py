#!/usr/bin/env python3
"""
Run the whole pipeline, in order, skipping stages whose outputs already exist.

    python run_all.py                 # everything that is not already done
    python run_all.py --force         # ignore existing outputs, rerun all
    python run_all.py --from 7        # start at stage 7
    python run_all.py --only 4 5      # just those stages
    python run_all.py --list          # show the stages and what is done

Stages that need the network (ChEMBL, ZINC, the PDB) or the AutoDock Vina
binary are declared as such and skipped with an explanation rather than a
traceback when the requirement is missing.

Total runtime from nothing is several hours, nearly all of it docking.
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PY = sys.executable


@dataclass
class Stage:
    n: int
    script: str
    args: list = field(default_factory=list)
    produces: list = field(default_factory=list)
    needs_net: bool = False
    needs_vina: bool = False
    note: str = ""

    @property
    def done(self) -> bool:
        return bool(self.produces) and all((ROOT / p).exists()
                                           for p in self.produces)


T = "tyk2ic50"
STAGES = [
    Stage(1, "01_fetch_chembl.py", ["--target", "ALL"],
          ["data/raw/tyk2_chembl_raw.csv"], needs_net=True,
          note="TYK2 and RORgt bioactivity from ChEMBL"),
    Stage(2, "02_curate.py", ["--target", T],
          [f"data/processed/{T}_curated.csv"],
          note="standardise, deduplicate, attrition table"),
    Stage(3, "03_eda.py", ["--target", T],
          [f"figures/{T}_eda.png"],
          note="property distributions, scaffolds, activity cliffs"),
    Stage(4, "04_baseline_model.py", ["--target", T, "--external", "tyk2ki"],
          [f"results/{T}_model_metrics.csv"],
          note="random, scaffold and external-set evaluation"),
    Stage(5, "05_split_diagnostics.py", ["--target", T, "--repeats", "10"],
          [f"results/{T}_repeated_scaffold_splits.csv"],
          note="split representativeness and external bias"),
    Stage(6, "06b_get_kinase_library.py", [],
          ["data/library/kinase_focused.smi"], needs_net=True,
          note="kinase-focused screening library from ChEMBL"),
    Stage(7, "07_screen.py",
          ["--target", T, "--library", "data/library/kinase_focused.smi"],
          [f"results/{T}_shortlist.csv"],
          note="applicability domain, predict, diverse shortlist"),
    Stage(8, "08_prepare_receptor.py", [],
          ["data/docking/receptor.pdbqt", "data/docking/native_ligand.sdf"],
          needs_net=True, note="PDB 6NZQ, bond orders, docking box"),
    Stage(9, "09_dock.py", ["--target", T, "--n", "2000"],
          [f"results/{T}_docking.csv"], needs_vina=True,
          note="redocking gate, then dock the shortlist (hours)"),
    Stage(10, "10_selectivity.py", ["--target", T],
          [f"results/{T}_selectivity.csv"], needs_net=True,
          note="JAK1/2/3 models, measured-first selectivity margin"),
    Stage(11, "11_admet.py", ["--target", T],
          [f"results/{T}_admet.csv"],
          note="ADMET liabilities on the defensible set"),
    Stage(12, "12_validation.py", ["--target", T],
          [f"results/{T}_y_randomisation.csv"],
          note="Y-randomisation and docking enrichment controls"),
    Stage(13, "13_figure.py", ["--target", T],
          ["figures/selectivity_illusion.png"],
          note="the headline figure"),
]


def have_vina() -> bool:
    if any(shutil.which(c) for c in ("vina", "vina.exe")):
        return True
    return any((ROOT / c).exists()
               for c in ("vina.exe", "vina", "vina_1.2.7_win.exe"))


def check_imports() -> list:
    missing = []
    for mod, pkg in [("rdkit", "rdkit"), ("pandas", "pandas"),
                     ("sklearn", "scikit-learn"), ("lightgbm", "lightgbm"),
                     ("matplotlib", "matplotlib"), ("scipy", "scipy"),
                     ("chembl_webresource_client", "chembl_webresource_client"),
                     ("meeko", "meeko"), ("admet_ai", "admet_ai")]:
        if importlib.util.find_spec(mod) is None:
            missing.append(pkg)
    return missing


def run(stage: Stage) -> bool:
    cmd = [PY, str(SRC / stage.script)] + stage.args
    print(f"\n{'='*70}\n[{stage.n:>2}] {stage.script}   {stage.note}\n{'='*70}",
          flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    ok = r.returncode == 0
    print(f"[{stage.n:>2}] {'ok' if ok else 'FAILED'} in {time.time()-t:.0f}s",
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", nargs="*", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'#':>3}  {'status':<9} {'script':<28} needs")
        for s in STAGES:
            need = ", ".join(filter(None, [
                "network" if s.needs_net else "",
                "vina" if s.needs_vina else ""])) or "-"
            print(f"{s.n:>3}  {'done' if s.done else 'pending':<9} "
                  f"{s.script:<28} {need}")
        return

    missing = check_imports()
    if missing:
        print("missing packages: " + ", ".join(missing))
        print(f"  {PY} -m pip install " + " ".join(missing))
        return 1

    vina = have_vina()
    if not vina:
        print("note: AutoDock Vina not found, so the docking stage will be "
              "skipped.\n      Download it from "
              "https://github.com/ccsb-scripps/AutoDock-Vina/releases\n"
              f"      and put the binary in {ROOT}\n")

    selected = [s for s in STAGES
                if (args.only and s.n in args.only)
                or (not args.only and s.n >= args.start)]

    ran, skipped, failed = [], [], []
    for s in selected:
        if s.done and not args.force:
            skipped.append(s.n)
            print(f"[{s.n:>2}] {s.script:<28} already done, skipping")
            continue
        if s.needs_vina and not vina:
            skipped.append(s.n)
            print(f"[{s.n:>2}] {s.script:<28} needs AutoDock Vina, skipping")
            continue
        if run(s):
            ran.append(s.n)
        else:
            failed.append(s.n)
            if args.stop_on_error:
                break

    print(f"\n{'='*70}")
    print(f"ran {len(ran)}  skipped {len(skipped)}  failed {len(failed)}")
    if failed:
        print(f"failed stages: {failed}")
        return 1
    print("\nresults in results/, figures in figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
