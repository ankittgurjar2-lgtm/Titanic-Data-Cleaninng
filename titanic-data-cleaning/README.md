# Titanic Dataset — Data Acquisition, Cleaning & Preprocessing

Week 1 task for a data science internship: acquire a public dataset, clean it, and
preprocess it for modeling, with every decision documented and justified.

## Dataset

Raw (uncleaned) Titanic passenger manifest — 891 passengers, 11 original fields —
from Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
competition, originally compiled from public historical passenger records. Retrieved
via the [`seaborn-data`](https://github.com/mwaskom/seaborn-data) GitHub repository
(`raw/titanic.csv`), which mirrors the untouched Kaggle training file. A local copy is
included in `data/titanic_raw.csv` so the pipeline runs without a network call.

This is the *raw* version (not a pre-cleaned teaching copy), so it still contains its
original missing values, inconsistent free-text fields, and a few genuine anomalies —
e.g. a group of passengers with a recorded fare of $0.

## What the pipeline does

`titanic_cleaning_pipeline.py` runs the full process end-to-end:

1. **Exploration** — shape, dtypes, missing-value audit, descriptive stats, duplicate check
2. **Anomaly detection** — zero-fare passengers, shared ticket numbers (family groups, not duplicate rows)
3. **Feature engineering** — title extracted from name, deck / has_cabin from cabin, family_size from sibsp+parch
4. **Missing-value treatment** — column-specific: `embarked` (mode + historical cross-check), `age` (median grouped by title), `cabin` (re-expressed as `has_cabin` + `deck` rather than imputed)
5. **Outlier detection** — IQR method on `age` and `fare`, with the reasoning for keeping rather than deleting flagged points, plus a `log1p(fare)` transform for skew
6. **Preprocessing for modeling** — encoding, an 80/20 stratified train/test split, and feature scaling fit on the training split only (no leakage)

Full narrative, code, and figures are written up in
[`Week1_Data_Cleaning_Report.docx`](./Week1_Data_Cleaning_Report.docx).

## Results at a glance

| Metric | Before | After |
|---|---|---|
| Rows | 891 | 891 (none dropped) |
| Columns | 11 | 29 (after one-hot encoding) |
| Missing cells | 866 | 0 |

## Running it

```bash
pip install -r requirements.txt
python titanic_cleaning_pipeline.py
```

Outputs the cleaned dataset to `data/titanic_cleaned.csv` and six charts to `figures/`.

## Structure

```
titanic_cleaning_pipeline.py     # full pipeline, run top to bottom
data/titanic_raw.csv             # input (raw, as downloaded)
data/titanic_cleaned.csv         # output (cleaned + engineered)
figures/                         # generated charts
Week1_Data_Cleaning_Report.docx  # full written report
requirements.txt
```
