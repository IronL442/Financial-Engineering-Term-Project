import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

df = pd.read_csv(DATA_DIR / "nasdaq100_options_enriched.csv")
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── 1. Drop unnecessary identifier columns ────────────────────────────────────
DROP_COLS = ["contractSymbol", "expiration", "retrieval_datetime_utc"]
df = df.drop(columns=DROP_COLS)

# ── 2. Fill dividend NaN with 0 (NaN means company pays no dividend) ──────────
DIVIDEND_COLS = [
    "dividendRate", "dividendYield", "payoutRatio",
    "lastDividendValue", "fiveYearAvgDividendYield",
    "trailingAnnualDividendRate", "trailingAnnualDividendYield",
]
df[DIVIDEND_COLS] = df[DIVIDEND_COLS].fillna(0)

# ── 3. Fill remaining numeric NaN with column median ──────────────────────────
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ── 4. One-hot encode categorical columns ─────────────────────────────────────
CATEGORICAL_COLS = ["industry", "sector", "country"]

# Fill any NaN in categoricals before encoding
df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("Unknown")

df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=int)

# ── 5. Report ─────────────────────────────────────────────────────────────────
remaining_nan = df.isnull().sum().sum()
print(f"After preprocessing: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Remaining NaN: {remaining_nan}")

ohe_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)]
print(f"One-hot encoded columns added: {len(ohe_cols)}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
out = DATA_DIR / "nasdaq100_options_preprocessed.csv"
df.to_csv(out, index=False)
print(f"Saved → {out}")
