# In-silico hit discovery for TYK2 (JH2 pseudokinase domain)

An end-to-end computational drug-discovery pipeline, built as a portfolio
project. Primary target TYK2 (CHEMBL3553); RORgamma-t (CHEMBL1163101) is
carried as a comparator.

## Rationale

TYK2 sits downstream of IFN-alpha, IL-12 and IL-23. Allosteric inhibitors that
bind the JH2 pseudokinase domain rather than the ATP site are clinically
validated (deucravacitinib), which means this project can be benchmarked
against reality instead of producing unfalsifiable hits.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `src/01_fetch_chembl.py` | `data/raw/<target>_chembl_raw.csv` |
| 2 | `src/02_curate.py` | `data/processed/<target>_curated.csv`, `results/<target>_attrition.csv` |
| 3 | `src/03_eda.py` | `figures/<target>_eda.png`, `results/<target>_activity_cliffs.csv` |
| 4 | baseline QSAR (RF / LightGBM, ECFP4, scaffold split) | pending |
| 5 | Chemprop D-MPNN comparison | pending |
| 6 | virtual screen + applicability domain | pending |
| 7 | docking into TYK2 JH2 + ADMET filters | pending |
| 8 | target dossier write-up | pending |

## Curation decisions (and why)

- **Element filter runs after fragment selection, not before.** Checking first
  discards every legitimate salt form; checking only afterwards lets a
  platinum diammine complex through as ammonia. Both filters plus a minimum
  heavy-atom count are needed.
- **Tautomers are canonicalised before deduplication.** Otherwise the same
  compound enters the dataset twice under two InChIKeys, and ends up on both
  sides of the train/test split.
- **Replicate measurements are kept, not deleted.** Spreads around 1 log unit
  between assays are normal. Dropping every compound with disagreement would
  preferentially remove the best-studied and most potent chemistry. Policy:
  median value, spread recorded, exclusion only above 2 log units.
- **Every filter writes its own attrition row**, so the methods section is a
  table rather than a claim.

## Running

```bash
pip install -r requirements.txt
python src/01_fetch_chembl.py --target ALL
python src/02_curate.py --target tyk2
python src/03_eda.py --target tyk2
```

Network note: `01_fetch_chembl.py` needs outbound access to `www.ebi.ac.uk`.
