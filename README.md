# In-silico hit identification for TYK2 (JH2 pseudokinase domain)

A ligand-based screening pipeline for TYK2, built end to end: ChEMBL curation,
QSAR modelling with honest validation, a measured applicability domain, and
molecular docking with a redocking gate and a positive-control test.

The deliverable is a ranked list of 20 candidate compounds. The more useful
results are three findings about the method, two of which are negative.

---

## Findings

### 1. The model generalises to unseen scaffolds, modestly

Across ten shuffled Bemis–Murcko scaffold splits: **RMSE 0.714 ± 0.057** log
units, Spearman 0.678 ± 0.052. The optimism gap against a random split is
~0.08 log units.

Two results initially looked like failures and were not:

- A **greedy** scaffold split gave R² = −2.37. Assigning the largest scaffold
  groups to training leaves a test set of singleton scaffolds, which in ChEMBL
  are enriched for weak compounds — test mean pIC50 5.93 against training's
  7.20. R² is measured against the test set's own variance, so it collapses.
  The model was being graded on a different population.
- The **external Ki set** gave R² = −0.429 with Spearman 0.446. The median
  signed error was −1.129 log units: a constant offset, which is Cheng–Prusoff
  (IC50 = Ki(1 + [S]/Km)). Bias correction moves R² from −0.439 to 0.178 and
  leaves ranking untouched.

### 2. Applicability domain depends on chemotype, not physicochemical properties

Three libraries screened, one variable changed at a time:

| | A: ZINC, properties mismatched | B: ZINC, properties matched | C: kinase-focused |
|---|---|---|---|
| Compounds | 192,573 | 246,960 | 75,405 |
| MW median (training 410) | 437 | 410 | 421 |
| logP median (training 3.05) | 0.99 | 3.31 | 3.65 |
| Similarity, 99th pct | 0.343 | 0.354 | **0.710** |
| **In domain (≥ 0.52)** | **1** | **1** | **2,368** |

Matching molecular weight and lipophilicity almost exactly (A→B) changed
nothing. Changing chemotype (B→C) changed it by three orders of magnitude.
ECFP4 encodes substructure — hinge binders, privileged heterocycles — and those
cannot be obtained by filtering on bulk properties.

The 0.52 threshold is **measured, not assumed**: scaffolds were held out five
times, error binned by similarity to the nearest training compound, and the
threshold read off where RMSE falls below a stated 0.85 log unit tolerance.

### 3. Docking reproduced the crystal pose but could not rank actives

Redocking the co-crystallised ligand of PDB 6NZQ placed the top pose **1.94 Å**
from experiment with Vinardo scoring — the only pose under 6 Å.

The same setup then failed a positive-control test:

| Test | Result |
|---|---|
| 33 controls vs library, median score | −6.48 vs −6.32 |
| Mann–Whitney, controls better | p = 0.185 |
| Potent controls (pIC50 ≥ 8, n = 7) in top decile | **0 of 7** |
| Docking score vs true pIC50 (controls) | Spearman +0.32 (wrong direction) |
| Docking score vs QSAR prediction | Spearman +0.12 (disagreement) |

Pose prediction works; affinity ranking does not. Docking is therefore used
only to exclude compounds that cannot occupy the pocket, never to rank.

---

## Candidates, and their problem

20 compounds, ranked by predicted pIC50, all inside the validated domain.
Full table in `results/`.

**All twenty are JAK1 inhibitors.** That is both the strongest validation and
the central weakness.

It validates the model because nothing in the training data encodes kinase
phylogeny, yet given 55 kinases the model preferentially selected TYK2's
closest relatives — JAK1 enriched **15.7×**, JAK3 12.2×, JAK2 5.0×, nothing
else above 2.6×.

It is a weakness because the model has largely learned *JAK-family chemotype*
rather than *TYK2 activity*. TYK2 selectivity over JAK1/2/3 is the entire
clinical rationale for this target — pan-JAK inhibition carries anaemia and
infection risk. The model cannot see this, having been trained on TYK2 data
alone.

Note also that every candidate comes from ChEMBL and is already published.
This is a **repurposing hypothesis**, not a novel-compound claim.

---

## Reproducing

```bash
pip install -r requirements.txt

python src/01_fetch_chembl.py --target ALL
python src/02_curate.py --target tyk2ic50
python src/03_eda.py --target tyk2ic50
python src/04_baseline_model.py --target tyk2ic50 --external tyk2ki
python src/05_split_diagnostics.py --target tyk2ic50 --repeats 10
python src/06b_get_kinase_library.py
python src/07_screen.py --target tyk2ic50 --library data/library/kinase_focused.smi
python src/08_prepare_receptor.py
python src/09_dock.py --target tyk2ic50 --n 2000
```

Docking additionally needs [AutoDock Vina
1.2.7](https://github.com/ccsb-scripps/AutoDock-Vina/releases) in the project
root. Everything is open source; no commercial licence is required.

Bulk intermediates (screening libraries, full screen outputs, docking poses)
are gitignored — the scripts regenerate them.

---

## Three bugs worth knowing about

Each produced plausible-looking output and was caught only by checking against
ground truth.

| Bug | Symptom | Consequence if undetected |
|---|---|---|
| PDB files carry no bond orders | Aromatic rings read as saturated, amides as alcohols | Docking a molecule that does not exist, at a normal-looking affinity |
| Nearest-neighbour matching used as RMSD | Reported 2.35 Å for a 180° flip whose true RMSD was 10.6 Å | Every pose passes validation |
| Default Vina scoring on a pseudo-symmetric ligand | Ranked a flipped pose above the crystal one by 0.27 kcal/mol | A confident pose cluster, 10.5 Å wrong |

---

## Limitations

- Nothing has been tested experimentally.
- The candidates are probably not TYK2-selective (see above).
- The model compresses predictions: σ 0.48 predicted against 1.09 observed on
  the external set, so potent novel compounds are under-ranked.
- Only 3.1% of a kinase-focused library falls inside the applicability domain.
- The receptor is rigid and single-conformer; 13 incomplete residues were
  deleted (nearest 9.7 Å from the ligand, against 1.9–2.7 Å for pocket-lining
  residues).

## Next

1. Train JAK1/2/3 models and rank on selectivity margin rather than TYK2
   potency. This addresses the central weakness directly.
2. ADMET prediction on the final candidates.
3. Optional: Chemprop GNN benchmarked against the fingerprint baseline.

---

## Data

ChEMBL (CC BY-SA 3.0; derived datasets inherit share-alike) · ZINC20 (free for
research use) · PDB 6NZQ (public domain).
