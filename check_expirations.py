import yfinance as yf
import pandas as pd
from datetime import datetime

tickers = [
    'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALNY', 'AMAT', 'AMD',
    'AMGN', 'AMZN', 'APP', 'ARM', 'ASML', 'AVGO', 'AXON', 'BKNG', 'BKR', 'CCEP',
    'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX',
    'CTAS', 'CTSH', 'DASH', 'DDOG', 'DXCM', 'EA', 'EXC', 'FANG', 'FAST', 'FER',
    'FTNT', 'GEHC', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'INSM', 'INTC', 'INTU',
    'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'MAR', 'MCHP', 'MDLZ', 'MELI',
    'META', 'MNST', 'MPWR', 'MRVL', 'MSFT', 'MSTR', 'MU', 'NFLX', 'NVDA', 'NXPI',
    'ODFL', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PLTR', 'PYPL', 'QCOM',
    'REGN', 'ROP', 'ROST', 'SBUX', 'SHOP', 'SNDK', 'SNPS', 'STX', 'TMUS', 'TRI',
    'TSLA', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'WDC', 'WMT', 'XEL', 'ZS'
]

today = datetime.today()
records = []

for i, ticker in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] {ticker}...")
    try:
        t = yf.Ticker(ticker)
        expirations = t.options  # tuple of date strings 'YYYY-MM-DD'

        # Tau in days for each expiration
        taus = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            tau = (exp_date - today).days
            taus.append(tau)

        records.append({
            "ticker": ticker,
            "n_expirations": len(expirations),
            "min_tau_days": min(taus) if taus else None,
            "max_tau_days": max(taus) if taus else None,
            "expirations": list(expirations),
            "taus_days": taus,
        })

    except Exception as e:
        print(f"  ⚠️  {ticker} failed: {e}")
        records.append({"ticker": ticker, "n_expirations": 0})

df = pd.DataFrame(records)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Tickers processed: {len(df)}")
print(f"Avg expirations per ticker: {df['n_expirations'].mean():.1f}")
print(f"Min expirations: {df['n_expirations'].min()} ({df.loc[df['n_expirations'].idxmin(), 'ticker']})")
print(f"Max expirations: {df['n_expirations'].max()} ({df.loc[df['n_expirations'].idxmax(), 'ticker']})")
print(f"Avg max Tau: {df['max_tau_days'].mean():.0f} days")
print(f"Avg min Tau: {df['min_tau_days'].mean():.0f} days")

print("\nPer-ticker expiration count:")
print(df[['ticker', 'n_expirations', 'min_tau_days', 'max_tau_days']].to_string(index=False))

# ── All unique expiration dates across all tickers ────────────────────────────
all_exps = set()
for exps in df['expirations'].dropna():
    all_exps.update(exps)

all_exps = sorted(all_exps)
print(f"\nTotal unique expiration dates: {len(all_exps)}")
print("Dates:", all_exps)

# ── Save ──────────────────────────────────────────────────────────────────────
df[['ticker', 'n_expirations', 'min_tau_days', 'max_tau_days']].to_csv(
    "expiration_summary.csv", index=False
)
print("\nSaved → expiration_summary.csv")