"""
Step 6b — Applicability domain calibration and virtual screen.

The step-5 diagnostics showed the model compresses its predictions and loses
accuracy on chemotypes unlike the training set. Screening 250k compounds
without accounting for that would produce a ranked list whose top entries are
mostly the ones the model understands least.

So the applicability domain threshold here is MEASURED, not assumed:

  1. Hold out scaffolds, predict them, and record each test compound's
     Tanimoto similarity to its nearest training neighbour.
  2. Bin by that similarity and compute RMSE per bin. This gives an empirical
     curve of error against similarity for THIS model on THIS target.
  3. Choose the threshold where binned RMSE first falls below a stated
     tolerance. Everything below it is reported as out-of-domain.

Then screen, filter for drug-likeness and PAINS, and pick a structurally
diverse shortlist with MaxMin so the output is not 1,000 analogues.
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LIB = ROOT / "data" / "library"
FIG = ROOT / "figures"
RESULTS = ROOT / "results"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = _load("04_baseline_model.py")
curate = _load("02_curate.py")

SEED = 42
RMSE_TOLERANCE = 0.85      # log units; the error we are willing to accept
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_pains = FilterCatalog(
    (lambda p: (p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS), p)[1])(
        FilterCatalogParams()))


def fps_of(smiles_list):
    out, keep = [], []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        out.append(_gen.GetFingerprint(m))
        keep.append(i)
    return out, np.array(keep)


def nearest_sim(query_fps, ref_fps, chunk=2000):
    """Max Tanimoto from each query to any reference compound."""
    out = np.empty(len(query_fps), dtype=np.float32)
    for i, q in enumerate(query_fps):
        out[i] = max(DataStructs.BulkTanimotoSimilarity(q, ref_fps))
        if (i + 1) % chunk == 0:
            print(f"    similarity {i+1:,}/{len(query_fps):,}", flush=True)
    return out


def calibrate_domain(df, X, y, repeats=5):
    """Empirical error-vs-similarity curve from held-out scaffolds."""
    smiles = df["std_smiles"].tolist()
    diag = _load("05_split_diagnostics.py")
    rows = []
    for s in range(repeats):
        tr, va, te = diag.shuffled_scaffold_split(smiles, seed=s)
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        tr_fps, _ = fps_of([smiles[i] for i in tr])
        te_fps, keep = fps_of([smiles[i] for i in te])
        sims = np.array([max(DataStructs.BulkTanimotoSimilarity(q, tr_fps))
                         for q in te_fps])
        for sim, obs, prd in zip(sims, y[te][keep], pred[keep]):
            rows.append({"sim": float(sim), "obs": float(obs), "pred": float(prd)})
        print(f"  calibration split {s+1}/{repeats} done", flush=True)

    cal = pd.DataFrame(rows)
    cal["err"] = cal["pred"] - cal["obs"]
    bins = np.arange(0.0, 1.01, 0.05)
    cal["bin"] = pd.cut(cal["sim"], bins, right=False)
    grp = cal.groupby("bin", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "sim_mid": g["sim"].mean(),
            "RMSE": float(np.sqrt(mean_squared_error(g["obs"], g["pred"]))),
            "bias": float(g["err"].median()),
        }), include_groups=False).reset_index()

    # Pick the LOWEST similarity bin from which the tolerance holds for that
    # bin and every bin above it. Taking the first bin that happens to meet
    # the tolerance would be fooled by one lucky low-similarity bin, which is
    # exactly the region where the estimate is noisiest.
    usable = grp[grp["n"] >= 20].sort_values("sim_mid").reset_index(drop=True)
    threshold = None
    for i in range(len(usable)):
        if (usable.loc[i:, "RMSE"] <= RMSE_TOLERANCE).all():
            threshold = float(usable.loc[i, "sim_mid"])
            break
    if threshold is None:
        # No region meets the tolerance. Fall back to the best-performing
        # populated bin and say so loudly rather than screening regardless.
        best = usable.loc[usable["RMSE"].idxmin()] if len(usable) else None
        threshold = float(best["sim_mid"]) if best is not None else 0.40
        print(f"\n  WARNING: no similarity region reaches RMSE <= "
              f"{RMSE_TOLERANCE}. Using the best bin ({threshold:.2f}, RMSE "
              f"{best['RMSE']:.2f}) and treating ALL predictions as "
              f"provisional." if best is not None else
              "\n  WARNING: calibration failed; defaulting to 0.40.")
    return cal, grp, threshold


def props(mol):
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotb": Descriptors.NumRotatableBonds(mol),
    }


def druglike(p):
    lipinski = sum([p["mw"] > 500, p["logp"] > 5, p["hba"] > 10, p["hbd"] > 5]) <= 1
    veber = p["rotb"] <= 10 and p["tpsa"] <= 140
    return lipinski and veber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tyk2ic50")
    ap.add_argument("--library", default=str(LIB / "zinc_matched.smi"))
    ap.add_argument("--top", type=int, default=2000, help="size of diverse shortlist")
    ap.add_argument("--calib-repeats", type=int, default=5)
    args = ap.parse_args()

    # ---- training set -------------------------------------------------
    df = pd.read_csv(PROC / f"{args.target}_curated.csv")
    X, keep = base.featurise(df["std_smiles"].tolist())
    df = df.iloc[keep].reset_index(drop=True)
    y = df["pActivity"].to_numpy()
    print(f"training set: {len(df):,} compounds")

    # ---- 1. calibrate the applicability domain -------------------------
    print(f"\ncalibrating applicability domain ({args.calib_repeats} scaffold splits) ...")
    cal, grp, threshold = calibrate_domain(df, X, y, args.calib_repeats)
    RESULTS.mkdir(parents=True, exist_ok=True)
    grp.to_csv(RESULTS / f"{args.target}_ad_calibration.csv", index=False)
    print(f"\n{'sim bin':>12} {'n':>6} {'RMSE':>7} {'bias':>7}")
    for _, r in grp.iterrows():
        print(f"{str(r['bin']):>12} {int(r['n']):>6} {r['RMSE']:>7.3f} {r['bias']:>7.3f}")
    print(f"\n  -> RMSE falls to <={RMSE_TOLERANCE} at Tanimoto ~{threshold:.2f}; "
          f"using that as the domain threshold.")

    # ---- 2. final model on everything ----------------------------------
    print("\nfitting final model on the full training set ...")
    final = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED)
    final.fit(X, y)
    train_fps, _ = fps_of(df["std_smiles"].tolist())

    # ---- 3. library ----------------------------------------------------
    lib_path = Path(args.library)
    if not lib_path.exists():
        raise SystemExit(f"missing library {lib_path}; run 06_get_library.py first")
    raw = pd.read_csv(lib_path, sep=r"\s+", engine="python", header=None,
                      names=["smiles", "id"], dtype=str)
    print(f"\nlibrary: {len(raw):,} raw entries")

    # Check the library parses before spending twenty minutes discovering it
    # does not. A file that looks fine can still be unusable - e.g. SMILES
    # written out as the repr of a bytes object.
    probe = raw["smiles"].head(50)
    n_ok = sum(Chem.MolFromSmiles(s) is not None for s in probe)
    if n_ok == 0:
        raise SystemExit(
            f"None of the first {len(probe)} library SMILES parse. First "
            f"entry is {probe.iloc[0]!r}.\nThe library file is malformed - "
            f"re-download it with 06_get_library.py.")
    if n_ok < len(probe) * 0.5:
        print(f"  WARNING: only {n_ok}/{len(probe)} sample SMILES parse; "
              f"the library may be partly corrupt")

    recs = []
    for n, row in enumerate(raw.itertuples(index=False), 1):
        if n % 25000 == 0:
            print(f"  standardising {n:,}/{len(raw):,} (kept {len(recs):,})", flush=True)
        res = curate.standardise(row.smiles)
        if res is None:
            continue
        smi, ikey, mol = res
        p = props(mol)
        if not druglike(p):
            continue
        if _pains.HasMatch(mol):
            continue
        recs.append({"id": row.id, "std_smiles": smi, "inchikey": ikey, **p})
    if not recs:
        raise SystemExit(
            "Every library compound was rejected during standardisation. "
            "That is a data problem, not a chemistry one - check the first "
            "few lines of the library file.")
    lib = pd.DataFrame(recs).drop_duplicates("inchikey").reset_index(drop=True)
    print(f"  after standardisation, drug-likeness and PAINS: {len(lib):,}")

    known = set(df["std_smiles"])
    lib = lib[~lib["std_smiles"].isin(known)].reset_index(drop=True)
    print(f"  after removing compounds already in training: {len(lib):,}")

    # ---- 4. predict + domain (in chunks) -------------------------------
    # Featurising 250k compounds at once needs ~4 GB for the matrix and about
    # the same again while it is being assembled. That is more than a laptop
    # will give up willingly, and the full matrix is never needed at once -
    # each chunk can be predicted and discarded.
    CHUNK = 20_000
    print(f"\npredicting and scoring domain in chunks of {CHUNK:,} ...")
    smiles_all = lib["std_smiles"].tolist()
    preds, sims, kept = [], [], []
    for start in range(0, len(smiles_all), CHUNK):
        part = smiles_all[start:start + CHUNK]
        Xc, kc = base.featurise(part)
        if len(kc) == 0:
            continue
        preds.append(final.predict(Xc.astype(np.float32)))
        part_fps, kf = fps_of([part[i] for i in kc])
        sims.append(np.array([max(DataStructs.BulkTanimotoSimilarity(q, train_fps))
                              for q in part_fps], dtype=np.float32))
        kept.extend((start + kc[kf]).tolist())
        del Xc
        print(f"  {min(start + CHUNK, len(smiles_all)):,}/{len(smiles_all):,}",
              flush=True)

    lib = lib.iloc[kept].reset_index(drop=True)
    lib["pred_pIC50"] = np.concatenate(preds)
    lib["nn_similarity"] = np.concatenate(sims)
    lib["in_domain"] = lib["nn_similarity"] >= threshold

    n_in = int(lib["in_domain"].sum())
    print(f"\n  in domain (Tanimoto >= {threshold:.2f}): {n_in:,} of {len(lib):,} "
          f"({n_in/len(lib)*100:.1f}%)")

    # ---- 5. diverse shortlist -----------------------------------------
    # If almost nothing is in domain, that is the finding, and silently
    # returning a shortlist of one is useless. Say so, then rank the whole
    # library by predicted potency with every compound flagged as an
    # extrapolation, so the downstream table cannot be mistaken for a
    # validated prediction.
    n_dom = int(lib["in_domain"].sum())
    extrapolating = n_dom < min(args.top, 50)
    if extrapolating:
        print(f"\n  *** Only {n_dom:,} compounds ({n_dom/len(lib)*100:.3f}%) fall "
              f"inside the validated domain (Tanimoto >= {threshold:.2f}).")
        print(f"  *** This library does not overlap the chemistry the model was "
              f"trained on.")
        print(f"  *** Proceeding by ranking on predicted potency alone. Every "
              f"compound below is an EXTRAPOLATION and the measured error at "
              f"this similarity is ~1-2 log units. Treat the shortlist as a "
              f"docking input only, never as a potency prediction.")
        pool = lib.sort_values("pred_pIC50", ascending=False)
    else:
        pool = lib[lib["in_domain"]].sort_values("pred_pIC50", ascending=False)
    pool = pool.head(min(len(pool), args.top * 5)).reset_index(drop=True)
    if len(pool) > args.top:
        pool_fps = [_gen.GetFingerprint(Chem.MolFromSmiles(s))
                    for s in pool["std_smiles"]]
        picker = MaxMinPicker()
        idx = list(picker.LazyBitVectorPick(pool_fps, len(pool_fps), args.top, seed=SEED))
        short = pool.iloc[idx].sort_values("pred_pIC50", ascending=False)
    else:
        short = pool
    short = short.reset_index(drop=True)
    short["extrapolation"] = extrapolating

    # Novelty. A compound sitting at very high similarity to a training
    # compound is a close analogue of known chemistry: safest for the model,
    # weakest as a discovery claim. Flag it rather than silently presenting
    # it as novel.
    # Novelty bands are anchored to the MEASURED domain threshold, not to
    # fixed numbers. With hardcoded edges a compound at 0.53 was labelled
    # "distant (out of domain)" while simultaneously passing a 0.52 domain
    # test - two parts of the same output contradicting each other.
    edges = [-0.01, threshold, threshold + 0.20, 0.95, 1.0, 1.01]
    labels = ["outside domain", "novel scaffold", "related analogue",
              "close analogue of known compound",
              "fingerprint-identical to a training compound"]
    for frame in (lib, short):
        # right=False so a compound exactly at the threshold counts as inside
        frame["novelty"] = pd.cut(frame["nn_similarity"], edges, labels=labels,
                                  right=False, include_lowest=True)
    n_novel = int((short["novelty"] == "novel scaffold").sum())
    n_close = int((short["novelty"] == "close analogue of known compound").sum())

    lib.to_csv(RESULTS / f"{args.target}_screen_all.csv", index=False)
    short.to_csv(RESULTS / f"{args.target}_shortlist.csv", index=False)

    # ---- figure --------------------------------------------------------
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(grp["sim_mid"], grp["RMSE"], "o-", color="#3b6ea5")
    ax[0].axhline(RMSE_TOLERANCE, ls="--", color="#c0392b", lw=1)
    ax[0].axvline(threshold, ls=":", color="k", lw=1)
    ax[0].set_xlabel("Tanimoto to nearest training compound")
    ax[0].set_ylabel("RMSE (log units)")
    ax[0].set_title("Measured error vs similarity")
    ax[1].hist(lib["nn_similarity"], bins=50, color="#7f8c8d")
    ax[1].axvline(threshold, ls=":", color="#c0392b", lw=1.5)
    ax[1].set_xlabel("nearest-neighbour similarity")
    ax[1].set_ylabel("library compounds")
    ax[1].set_title("Library coverage by the domain")
    ax[2].scatter(lib["nn_similarity"], lib["pred_pIC50"], s=2, alpha=0.15,
                  color="#7f8c8d")
    ax[2].scatter(short["nn_similarity"], short["pred_pIC50"], s=6, alpha=0.7,
                  color="#c0392b")
    ax[2].axvline(threshold, ls=":", color="k", lw=1)
    ax[2].set_xlabel("nearest-neighbour similarity")
    ax[2].set_ylabel("predicted pIC50")
    ax[2].set_title(f"Shortlist (red, n={len(short):,})")
    fig.tight_layout()
    fig.savefig(FIG / f"{args.target}_screen.png", dpi=150, bbox_inches="tight")

    if extrapolating:
        print(f"\n  REMINDER: this shortlist is out of the validated domain.")
    print(f"\nshortlist: {len(short):,} compounds")
    print(f"  predicted pIC50 {short['pred_pIC50'].min():.2f} - "
          f"{short['pred_pIC50'].max():.2f}")
    print(f"  similarity      {short['nn_similarity'].min():.2f} - "
          f"{short['nn_similarity'].max():.2f}")
    print(f"\n  novelty breakdown of the shortlist:")
    for label, count in short["novelty"].value_counts().items():
        print(f"    {str(label):<38} {count:>5,}")
    print(f"\n  -> {n_novel:,} genuinely novel scaffolds; {n_close:,} are close "
          f"analogues of known actives and carry little novelty claim.")
    print(f"\nwritten -> results/{args.target}_shortlist.csv")
    print(f"figure   -> figures/{args.target}_screen.png")
    print("\nNOTE: predicted potency is a ranking signal, not an affinity. The "
          "shortlist is the input to docking, not a result.")


if __name__ == "__main__":
    main()
