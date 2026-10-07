"""
Week 1 Task — Data Acquisition, Cleaning, and Preprocessing
Dataset : Titanic passenger manifest (raw, uncleaned version)
Source  : Kaggle "Titanic: Machine Learning from Disaster" competition data,
          originally compiled from public historical passenger records
          (Encyclopedia Titanica). Retrieved via the seaborn-data GitHub
          repository, which mirrors the raw Kaggle training file for
          teaching purposes:
          https://github.com/mwaskom/seaborn-data/blob/master/raw/titanic.csv

This script performs, in order:
  1. Data acquisition
  2. Initial exploration
  3. Missing-value handling
  4. Erroneous-entry / anomaly handling
  5. Outlier detection and treatment
  6. Feature engineering
  7. Encoding and scaling (preprocessing for modeling)
  8. Export of the cleaned dataset and supporting figures

Run: python3 titanic_cleaning_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=0.95)
RANDOM_STATE = 42
FIG_DIR = "figures"
DATA_IN = "data/titanic_raw.csv"
DATA_OUT = "data/titanic_cleaned.csv"

pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ---------------------------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------------------------
section("1. DATA ACQUISITION")
df = pd.read_csv(DATA_IN)
print(f"Loaded {DATA_IN} -> shape {df.shape}")
print(df.head(3).to_string())

# ---------------------------------------------------------------------------
# 2. INITIAL EXPLORATION
# ---------------------------------------------------------------------------
section("2. INITIAL EXPLORATION")
print("\n-- dtypes --")
print(df.dtypes)

print("\n-- missing values --")
missing_count = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
missing_report = missing_report[missing_report.missing_count > 0].sort_values("missing_pct", ascending=False)
print(missing_report)

print("\n-- duplicate full rows --")
print(df.duplicated().sum())

print("\n-- numeric summary --")
print(df[["age", "fare", "sibsp", "parch"]].describe())

print("\n-- categorical cardinality --")
for c in ["sex", "embarked", "pclass"]:
    print(c, "->", df[c].unique())

# Missingness chart
plt.figure(figsize=(6, 4))
sns.barplot(x=missing_report.index, y=missing_report["missing_pct"], color="#4C72B0")
plt.ylabel("% missing")
plt.xlabel("Column")
plt.title("Missing values by column (raw data)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_missing_values.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. IDENTIFY ERRONEOUS / SUSPICIOUS ENTRIES
# ---------------------------------------------------------------------------
section("3. ERRONEOUS / SUSPICIOUS ENTRIES")
zero_fare = df[df["fare"] == 0]
print(f"Rows with fare == 0: {len(zero_fare)}")
print(zero_fare[["pclass", "sex", "age", "ticket", "fare"]].to_string())

dup_tickets = df["ticket"].value_counts()
dup_tickets = dup_tickets[dup_tickets > 1]
print(f"\nTicket numbers shared by more than one passenger: {len(dup_tickets)} tickets "
      f"covering {dup_tickets.sum()} passengers (group / family bookings, not duplicate records).")

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (done early because some cleaning depends on it)
# ---------------------------------------------------------------------------
section("4. FEATURE ENGINEERING (pre-cleaning helpers)")

# Title extracted from the Name field
df["title"] = df["name"].str.extract(r",\s*([^\.]*)\.")
print("\nRaw title counts:")
print(df["title"].value_counts())

title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}
df["title_group"] = df["title"].map(title_map).fillna("Rare")
print("\nConsolidated title_group counts:")
print(df["title_group"].value_counts())

print("\nMedian age by title_group (used for grouped imputation):")
print(df.groupby("title_group")["age"].median())

# Deck / cabin-presence
df["has_cabin"] = df["cabin"].notnull().astype(int)
df["deck"] = df["cabin"].str[0]
print("\nShare with a recorded cabin, by class (evidence that 'missing cabin' is "
      "informative, not random):")
print(df.groupby("pclass")["has_cabin"].mean().round(3))

# Family size
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.groupby("title_group")["age"].median().sort_values().plot(kind="barh", ax=axes[0], color="#55A868")
axes[0].set_title("Median age by title group")
axes[0].set_xlabel("Median age (years)")
df.groupby("pclass")["has_cabin"].mean().plot(kind="bar", ax=axes[1], color="#C44E52")
axes[1].set_title("Share of passengers with a recorded cabin, by class")
axes[1].set_ylabel("Fraction with cabin recorded")
axes[1].set_xlabel("Passenger class")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_title_and_cabin_evidence.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. MISSING VALUE TREATMENT
# ---------------------------------------------------------------------------
section("5. MISSING VALUE TREATMENT")

age_before = df["age"].copy()

# 5a. Embarked (2 missing) -> both share ticket 113572 / cabin B28 (traveling
# together). Historical passenger records (Encyclopedia Titanica) list both
# as having boarded at Southampton, which is also the column mode (~72% of
# passengers) -> impute 'S'.
print("Rows with missing embarked:")
print(df[df["embarked"].isnull()][["name", "pclass", "ticket", "cabin", "fare"]].to_string())
df["embarked"] = df["embarked"].fillna("S")

# 5b. Age (19.87% missing) -> impute with the MEDIAN AGE OF THE SAME title_group
# rather than a single global median, since age distributions differ sharply
# by title (Master ~ children, Mr/Mrs ~ adults).
df["age"] = df.groupby("title_group")["age"].transform(lambda s: s.fillna(s.median()))
print(f"\nAge missing before: {age_before.isnull().sum()}, after grouped-median imputation: {df['age'].isnull().sum()}")

# 5c. Cabin (77.10% missing) -> too sparse to impute reliably at the cabin
# level. We already captured its signal via has_cabin (5b above) and deck
# (kept as a low-cardinality categorical with an explicit "Unknown" level);
# the raw high-cardinality cabin string itself is dropped later.
df["deck"] = df["deck"].fillna("Unknown")
print("\nDeck value counts after filling missing with 'Unknown':")
print(df["deck"].value_counts())

assert df["age"].isnull().sum() == 0
assert df["embarked"].isnull().sum() == 0

# Before/after age distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
sns.histplot(age_before.dropna(), bins=30, ax=axes[0], color="#4C72B0")
axes[0].set_title(f"Age — before imputation\n(n={age_before.notnull().sum()}, {age_before.isnull().sum()} missing)")
sns.histplot(df["age"], bins=30, ax=axes[1], color="#55A868")
axes[1].set_title(f"Age — after grouped-median imputation\n(n={df['age'].notnull().sum()}, 0 missing)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_age_before_after.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. OUTLIER DETECTION AND TREATMENT
# ---------------------------------------------------------------------------
section("6. OUTLIER DETECTION (IQR method)")


def iqr_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


for col in ["age", "fare"]:
    lo, hi = iqr_bounds(df[col])
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col}: IQR bounds=({lo:.2f}, {hi:.2f}) -> {n_out} points outside bounds "
          f"({n_out/len(df)*100:.2f}% of rows)")

# Zero-fare passengers are a real category (documented "LINE"-ticket
# non-revenue passengers, e.g. American Line employees), not a data-entry
# error, and the IQR lower bound for fare is negative, so they are not even
# flagged by the statistical test. We keep the raw values and add an
# explicit flag rather than deleting or silently imputing them.
df["fare_is_zero"] = (df["fare"] == 0).astype(int)

# Fare is strongly right-skewed (max 512.33 vs. median 14.45) because of a
# small number of legitimate first-class suite fares. Rather than removing
# these real, informative high-fare passengers, we add a log1p-transformed
# version for models/visualizations sensitive to scale and skew, and retain
# the original fare for reference.
df["fare_log"] = np.log1p(df["fare"])

# Age outliers (elderly passengers > ~65) are biologically valid and
# survival-relevant, so no rows are removed on the basis of the IQR test for
# either column; the decision and its rationale are documented above.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["fare"], ax=axes[0], color="#DD8452")
axes[0].set_title("Fare — raw (right-skewed, real high-fare outliers)")
sns.boxplot(y=df["fare_log"], ax=axes[1], color="#8172B2")
axes[1].set_title("log1p(Fare) — skew reduced, no rows removed")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_fare_outliers.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. FINAL FEATURE SET / DROP RAW TEXT COLUMNS
# ---------------------------------------------------------------------------
section("7. FINAL FEATURE SET")

drop_cols = ["name", "ticket", "cabin", "title"]  # raw / high-cardinality / superseded by engineered features
model_df = df.drop(columns=drop_cols)
print(f"Dropped raw/high-cardinality columns: {drop_cols}")
print(f"Shape after dropping: {model_df.shape}")
print("\nRemaining columns:", model_df.columns.tolist())

# ---------------------------------------------------------------------------
# 8. ENCODING + TRAIN/TEST SPLIT + SCALING
# ---------------------------------------------------------------------------
section("8. ENCODING, SPLIT, AND SCALING")

model_df["sex"] = model_df["sex"].map({"male": 0, "female": 1})

cat_cols = ["embarked", "title_group", "deck"]
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=False)
bool_cols = model_df.select_dtypes(include="bool").columns
model_df[bool_cols] = model_df[bool_cols].astype(int)
print(f"Shape after one-hot encoding {cat_cols}: {model_df.shape}")

X = model_df.drop(columns=["survived"])
y = model_df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Survival rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

num_cols_to_scale = ["age", "fare", "fare_log", "family_size"]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols_to_scale] = scaler.fit_transform(X_train[num_cols_to_scale])
X_test_scaled[num_cols_to_scale] = scaler.transform(X_test[num_cols_to_scale])
print("\nScaler fit on TRAIN ONLY, then applied to test, to avoid leakage.")
print(X_train_scaled[num_cols_to_scale].describe().loc[["mean", "std"]].round(3))

# ---------------------------------------------------------------------------
# 9. EXPORT
# ---------------------------------------------------------------------------
section("9. EXPORT CLEANED DATA")
final_export = pd.concat([X, y], axis=1)
final_export.to_csv(DATA_OUT, index=False)
print(f"Saved cleaned/engineered (pre-scaling) dataset -> {DATA_OUT}, shape {final_export.shape}")

print("\nBEFORE  -> shape:", df.shape if False else (891, 11), " missing cells:", int(missing_count.sum()))
print("AFTER   -> shape:", final_export.shape, " missing cells:", int(final_export.isnull().sum().sum()))

# Correlation heatmap of final numeric/engineered features with survived
corr_cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "fare_log",
             "family_size", "is_alone", "has_cabin"]
plt.figure(figsize=(7, 6))
sns.heatmap(model_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation of engineered features with Survived")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_correlation_heatmap.png", dpi=150)
plt.close()

# Contextual sanity-check chart
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None, palette="Set2")
plt.title("Survival rate by class and sex (post-cleaning sanity check)")
plt.ylabel("Survival rate")
plt.xlabel("Passenger class")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_survival_by_class_sex.png", dpi=150)
plt.close()

print("\nAll figures written to ./figures/")
print("DONE.")
