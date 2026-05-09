import pandas as pd
import yfinance as yf
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── 1. Load raw options data ──────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "nasdaq100_options_raw.csv")
print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")

# ── 2. Fetch .info for each ticker ───────────────────────────────────────────
# These are the company-level variables we want to add.
# All values are the same for every option row of the same company.

WANTED_KEYS = [
    # --- Risk ---
    "auditRisk", "boardRisk", "compensationRisk",
    "shareHolderRightsRisk", "overallRisk",

    # --- Financial ---
    "ebitda", "totalDebt", "totalRevenue", "revenueGrowth",
    "grossMargins", "operatingMargins", "profitMargins",
    "returnOnAssets", "returnOnEquity",
    "freeCashflow", "operatingCashflow", "earningsGrowth",
    "currentRatio", "quickRatio", "debtToEquity",
    "totalCash", "totalCashPerShare", "revenuePerShare",
    "bookValue", "priceToBook",
    "enterpriseValue", "enterpriseToRevenue", "enterpriseToEbitda",
    "forwardEps", "trailingEps", "pegRatio",
    "trailingPE", "forwardPE",

    # --- Dividend ---
    "dividendRate", "dividendYield", "payoutRatio",
    "lastDividendValue", "fiveYearAvgDividendYield",
    "trailingAnnualDividendRate", "trailingAnnualDividendYield",

    # --- Market ---
    "marketCap", "sharesOutstanding", "floatShares",
    "sharesShort", "shortRatio", "shortPercentOfFloat",
    "heldPercentInsiders", "heldPercentInstitutions",
    "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "fiftyDayAverage", "twoHundredDayAverage",
    "averageVolume", "averageVolume10days",

    # --- Company ---
    "fullTimeEmployees", "industry", "sector", "country",
]

tickers = df["ticker"].unique().tolist()
info_records = []

for i, ticker in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")
    try:
        info = yf.Ticker(ticker).info
        record = {"ticker": ticker}
        for key in WANTED_KEYS:
            record[key] = info.get(key, None)
        info_records.append(record)
    except Exception as e:
        print(f"  ⚠️  {ticker} failed: {e}")
        info_records.append({"ticker": ticker})  # empty row, will become NaN

    # Be polite to Yahoo Finance — avoid rate limiting
    time.sleep(0.3)

df_info = pd.DataFrame(info_records)
print(f"\nFetched info for {len(df_info)} tickers")
print(f"Info columns: {len(df_info.columns) - 1}")  # -1 for ticker column

# ── 3. Merge with options data ────────────────────────────────────────────────
df_enriched = df.merge(df_info, on="ticker", how="left")

print(f"\nEnriched dataset: {df_enriched.shape[0]} rows x {df_enriched.shape[1]} columns")
print(f"Columns: {df_enriched.columns.tolist()}")

# ── 4. Check missing values ───────────────────────────────────────────────────
missing = df_enriched.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"\nMissing values:")
    print(missing)

# ── 5. Save ───────────────────────────────────────────────────────────────────
output_path = DATA_DIR / "nasdaq100_options_enriched.csv"
df_enriched.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}")