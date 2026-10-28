"""
Step 6a — Build a screening library whose chemistry matches the training set.

ZINC's 2D tranches are named <W><P><R><B>.smi, where W is a molecular-weight
bin and P a logP bin, both lettered A..K. The published letter-to-range
mapping is easy to get wrong, and getting it wrong is expensive: a library
drawn from the wrong corner of the grid can be entirely outside the model's
applicability domain, which is only discovered after screening it.

So this script does not assume the mapping. It PROBES each tranche with a
small sample, measures the molecular weight and logP that tranche actually
contains, and then selects the tranches that overlap the training set's own
property distribution. The mapping is measured, not trusted.

  python src/06_get_library.py --probe            # measure the grid (once)
  python src/06_get_library.py --max 250000       # download matched tranches
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data" / "library"
PROC = ROOT / "data" / "processed"

MIRRORS = ["https://cache.docking.org/2D", "https://files.docking.org/2D"]
LETTERS = list("ABCDEFGHIJK")
SUFFIXES = ["AD", "AB"]          # anodyne; make-on-demand then in-stock
PROBE_N = 150                    # molecules sampled per tranche when probing
CONNECT_TIMEOUT, READ_TIMEOUT = 10, 120


def pick_mirror(explicit=None):
    if explicit:
        return explicit
    for base in MIRRORS:
        try:
            r = requests.get(f"{base}/HA/HAAB.smi", timeout=(CONNECT_TIMEOUT, 20),
                             stream=True)
            r.close()
            if r.status_code == 200:
                print(f"using mirror: {base}")
                return base
            print(f"  {base}: HTTP {r.status_code}")
        except Exception as exc:
            print(f"  {base}: unreachable ({type(exc).__name__})")
    raise SystemExit("No ZINC mirror reachable.")


def stream_smiles(base, code, suf, limit):
    """Yield (smiles, id) from one tranche, stopping after `limit`."""
    url = f"{base}/{code}/{code}{suf}.smi"
    try:
        r = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True)
        if r.status_code != 200:
            return
        r.encoding = r.encoding or "utf-8"
        n = 0
        for line in r.iter_lines(decode_unicode=True):
            if n >= limit:
                break
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", "ignore")
            parts = line.split()
            if len(parts) < 2 or parts[0].lower() == "smiles":
                continue
            yield parts[0], parts[1]
            n += 1
        r.close()
    except Exception:
        return


def probe(base) -> pd.DataFrame:
    """Measure the MW and logP each tranche actually holds."""
    rows = []
    codes = [f"{w}{p}" for w in LETTERS for p in LETTERS]
    for i, code in enumerate(codes, 1):
        mws, logps, n = [], [], 0
        for suf in SUFFIXES:
            for smi, _ in stream_smiles(base, code, suf, PROBE_N):
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                mws.append(Descriptors.MolWt(m))
                logps.append(Crippen.MolLogP(m))
                n += 1
            if n >= PROBE_N:
                break
        if n >= 10:
            rows.append({"tranche": code, "n": n,
                         "mw_med": float(np.median(mws)),
                         "mw_lo": float(np.percentile(mws, 5)),
                         "mw_hi": float(np.percentile(mws, 95)),
                         "logp_med": float(np.median(logps)),
                         "logp_lo": float(np.percentile(logps, 5)),
                         "logp_hi": float(np.percentile(logps, 95))})
            print(f"  [{i:>3}/{len(codes)}] {code}: n={n:<4} "
                  f"MW {rows[-1]['mw_med']:>6.1f}  logP {rows[-1]['logp_med']:>5.2f}",
                  flush=True)
        else:
            print(f"  [{i:>3}/{len(codes)}] {code}: empty/unavailable", flush=True)
    return pd.DataFrame(rows)


def select_tranches(grid: pd.DataFrame, target: str) -> list:
    """Keep tranches whose typical compound resembles the training chemistry."""
    src = PROC / f"{target}_curated_descriptors.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}; run 03_eda.py first")
    tr = pd.read_csv(src)
    mw_lo, mw_hi = tr["mw"].quantile([0.10, 0.90])
    lp_lo, lp_hi = tr["logp"].quantile([0.10, 0.90])
    print(f"\ntraining chemistry (10th-90th percentile):")
    print(f"  MW   {mw_lo:.0f} - {mw_hi:.0f}")
    print(f"  logP {lp_lo:.2f} - {lp_hi:.2f}")

    keep = grid[(grid["mw_med"].between(mw_lo, mw_hi)) &
                (grid["logp_med"].between(lp_lo, lp_hi))]
    if keep.empty:
        print("\n  no tranche centre falls inside that window; widening to "
              "any tranche that overlaps it at all")
        keep = grid[(grid["mw_hi"] >= mw_lo) & (grid["mw_lo"] <= mw_hi) &
                    (grid["logp_hi"] >= lp_lo) & (grid["logp_lo"] <= lp_hi)]
    print(f"\nmatched tranches: {len(keep)} of {len(grid)} probed")
    for _, r in keep.sort_values(["mw_med", "logp_med"]).iterrows():
        print(f"  {r['tranche']}  MW {r['mw_med']:>6.1f}  logP {r['logp_med']:>5.2f}")
    return keep["tranche"].tolist()


def download(base, tranches, max_compounds) -> Path:
    LIB.mkdir(parents=True, exist_ok=True)
    out = LIB / "zinc_matched.smi"
    per = max(1, max_compounds // max(1, len(tranches)))
    print(f"\nsampling up to {per:,} from each of {len(tranches)} tranches")
    seen, n, counts = set(), 0, {}

    def pull(code, cap, fh):
        nonlocal n
        taken = 0
        for suf in SUFFIXES:
            if taken >= cap:
                break
            for smi, zid in stream_smiles(base, code, suf, cap - taken):
                if zid in seen or n >= max_compounds:
                    continue
                if Chem.MolFromSmiles(smi) is None:
                    continue
                seen.add(zid)
                fh.write(f"{smi}\t{zid}\n")
                taken += 1
                n += 1
        return taken

    with out.open("w") as fh:
        for code in tranches:
            if n >= max_compounds:
                break
            counts[code] = pull(code, per, fh)
            print(f"  {code}: {counts[code]:,}  (total {n:,})", flush=True)
        exhausted = {c for c, v in counts.items() if v == 0}
        while n < max_compounds and len(exhausted) < len(tranches):
            live = [c for c in tranches if c not in exhausted]
            inc = max(1, (max_compounds - n) // len(live))
            print(f"\n  top-up: {max_compounds - n:,} short, +{inc:,} each "
                  f"from {len(live)} tranches", flush=True)
            progress = False
            for code in live:
                if n >= max_compounds:
                    break
                got = pull(code, inc, fh)
                if got:
                    counts[code] += got
                    progress = True
                else:
                    exhausted.add(code)
            if not progress:
                break

    ok = bad = 0
    with out.open() as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            p = line.split()
            if p and Chem.MolFromSmiles(p[0]) is not None:
                ok += 1
            else:
                bad += 1
    print(f"\nsanity check on the first {ok+bad} lines: {ok} parse, {bad} do not")
    print(f"{n:,} compounds -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="measure the tranche grid and stop")
    ap.add_argument("--max", type=int, default=250_000)
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--mirror", default=None)
    args = ap.parse_args()

    LIB.mkdir(parents=True, exist_ok=True)
    grid_path = LIB / "tranche_grid.csv"
    base = pick_mirror(args.mirror)

    if args.probe or not grid_path.exists():
        print(f"\nprobing the tranche grid ({PROBE_N} molecules each) ...")
        grid = probe(base)
        grid.to_csv(grid_path, index=False)
        print(f"\ngrid -> {grid_path}")
        if args.probe:
            return
    else:
        grid = pd.read_csv(grid_path)
        print(f"using cached tranche grid ({len(grid)} tranches)")

    tranches = select_tranches(grid, args.target)
    if not tranches:
        raise SystemExit("no tranches match the training chemistry")
    download(base, tranches, args.max)


if __name__ == "__main__":
    main()
