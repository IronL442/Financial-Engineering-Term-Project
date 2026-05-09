"""
Collect Nasdaq-100 option-pricing data from Yahoo Finance.

This script creates a dataset similar to Table 2 in the paper:

S       = Price of the underlying asset
Strike  = Strike price of the option
Tau     = Residual time of the option, in years
Sigma   = Annualized volatility of returns of the underlying asset
Call    = 1 for call option, 0 for put option
Price   = Market price of the option

Design choice:
- The paper reports 73,154 options across 4,004 companies.
- That is roughly 18.27 option contracts per company.
- To get close to that scale while keeping useful Tau variation, this script collects:

    3 target expirations × (3 calls + 3 puts)
    = about 18 option rows per company

Target maturities:
- 30 days
- 180 days
- 720 days

Outputs:
- data/nasdaq100_options_model.csv  -> only Table 2 variables
- data/nasdaq100_options_raw.csv    -> includes identifiers for traceability
- data/nasdaq100_failures.csv       -> tickers that failed or had no usable options

Install:
pip install yfinance pandas numpy tqdm lxml requests
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm


def get_nasdaq100_tickers() -> list[str]:
    """
    Pull current Nasdaq-100 constituents from Wikipedia.

    Uses requests with a browser-like User-Agent to avoid HTTP 403 errors.
    Yahoo Finance uses '-' instead of '.' in tickers.
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    for table in tables:
        columns = [str(c).lower() for c in table.columns]

        if any("ticker" in c for c in columns) or any("symbol" in c for c in columns):
            for col in table.columns:
                col_lower = str(col).lower()

                if "ticker" in col_lower or "symbol" in col_lower:
                    tickers = (
                        table[col]
                        .astype(str)
                        .str.strip()
                        .str.replace(".", "-", regex=False)
                        .tolist()
                    )

                    return sorted(set(tickers))

    raise RuntimeError("Could not find Nasdaq-100 ticker table.")


def safe_float(x: Any) -> float | None:
    """
    Convert a value to float safely.
    """
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def annualized_volatility(ticker: str, lookback: str = "1y") -> float | None:
    """
    Compute annualized volatility of daily log returns.

    Sigma = std(daily log returns) * sqrt(252)
    """
    hist = yf.download(
        ticker,
        period=lookback,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if hist.empty:
        return None

    close = hist["Close"].dropna()

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    log_returns = np.log(close / close.shift(1)).dropna()

    if len(log_returns) < 30:
        return None

    return float(log_returns.std() * math.sqrt(252))


def get_underlying_price(ticker_obj: yf.Ticker) -> float | None:
    """
    Get current underlying stock price S.

    Prefer fast_info last_price, fallback to recent adjusted close.
    """
    try:
        price = safe_float(ticker_obj.fast_info.get("last_price"))

        if price is not None and price > 0:
            return price
    except Exception:
        pass

    try:
        hist = ticker_obj.history(period="5d", auto_adjust=True)

        if not hist.empty:
            price = safe_float(hist["Close"].dropna().iloc[-1])

            if price is not None and price > 0:
                return price
    except Exception:
        pass

    return None


def option_market_price(row: pd.Series) -> float | None:
    """
    Market price of the option.

    Yahoo provides bid, ask, and lastPrice.
    We use the bid-ask midpoint when possible.
    If bid/ask are unavailable, we fallback to lastPrice.
    """
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    last = safe_float(row.get("lastPrice"))

    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2

    if last is not None and last > 0:
        return last

    return None


def parse_target_tau_days(target_tau_days: str) -> list[int]:
    """
    Parse comma-separated target maturities.

    Example:
    "30,180,720" -> [30, 180, 720]
    """
    try:
        values = [int(x.strip()) for x in target_tau_days.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Target tau days must be comma-separated integers, e.g. 30,180,720"
        ) from e

    if not values:
        raise argparse.ArgumentTypeError("At least one target Tau day must be provided.")

    return values


def select_target_expirations(
    expirations: list[str],
    retrieval_dt: datetime,
    target_tau_days: list[int],
    min_tau_days: int = 14,
    max_tau_days: int | None = 900,
) -> list[str]:
    """
    Select expirations closest to fixed target maturities.

    This is better than selecting by index because it creates comparable Tau
    values across companies.

    Default targets:
    - 30 days: short maturity
    - 180 days: medium maturity
    - 720 days: long maturity

    With 3 calls and 3 puts per expiration, this gives:
    3 expirations × (3 calls + 3 puts) = about 18 rows per company.
    """
    valid_expirations: list[dict[str, Any]] = []

    for expiration in expirations:
        expiration_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days_to_expiration = (expiration_dt - retrieval_dt).days

        if days_to_expiration < min_tau_days:
            continue

        if max_tau_days is not None and days_to_expiration > max_tau_days:
            continue

        valid_expirations.append(
            {
                "expiration": expiration,
                "days_to_expiration": days_to_expiration,
            }
        )

    if not valid_expirations:
        return []

    selected: list[str] = []

    for target in target_tau_days:
        closest = min(
            valid_expirations,
            key=lambda x: abs(x["days_to_expiration"] - target),
        )

        selected.append(closest["expiration"])

    # Remove duplicates while preserving order.
    selected_unique: list[str] = []

    for expiration in selected:
        if expiration not in selected_unique:
            selected_unique.append(expiration)

    return selected_unique


def filter_options(
    option_df: pd.DataFrame,
    S: float,
    contracts_per_side: int = 3,
    require_liquidity: bool = True,
) -> pd.DataFrame:
    """
    Keep near-the-money options with usable prices.

    This function is called separately for calls and puts.

    contracts_per_side=3 means:
    - up to 3 calls per expiration
    - up to 3 puts per expiration

    With 3 expirations, that gives:
    3 expirations × (3 calls + 3 puts) = about 18 rows per company.
    """
    if option_df is None or option_df.empty:
        return pd.DataFrame()

    df = option_df.copy()

    required_cols = {"strike", "bid", "ask", "lastPrice"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        return pd.DataFrame()

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce")

    if "openInterest" in df.columns:
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
    else:
        df["openInterest"] = 0

    df = df.dropna(subset=["strike"])

    if df.empty:
        return df

    if require_liquidity:
        has_mid_price = (
            df["bid"].notna()
            & df["ask"].notna()
            & (df["bid"] > 0)
            & (df["ask"] > 0)
            & (df["ask"] >= df["bid"])
        )

        has_last_price = df["lastPrice"].notna() & (df["lastPrice"] > 0)

        # Do not require openInterest > 0 because Yahoo often reports missing/zero values.
        df = df[has_mid_price | has_last_price]

    if df.empty:
        return df

    df["distance_to_money"] = (df["strike"] - S).abs()

    return (
        df.sort_values("distance_to_money")
        .head(contracts_per_side)
        .drop(columns=["distance_to_money"], errors="ignore")
    )


def collect_options_for_ticker(
    ticker: str,
    vol_lookback: str,
    sleep_seconds: float,
    target_tau_days: list[int],
    contracts_per_side: int,
    min_tau_days: int,
    max_tau_days: int | None,
    require_liquidity: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Collect filtered option rows for one ticker.

    Returns:
    - list of raw rows
    - error/warning message, if any
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        retrieval_dt = datetime.now(timezone.utc)

        S = get_underlying_price(ticker_obj)
        Sigma = annualized_volatility(ticker, lookback=vol_lookback)

        if S is None:
            return [], "Missing underlying price S"

        if Sigma is None:
            return [], "Missing annualized volatility Sigma"

        all_expirations = list(ticker_obj.options)

        if not all_expirations:
            return [], "No option expirations found"

        selected_expirations = select_target_expirations(
            expirations=all_expirations,
            retrieval_dt=retrieval_dt,
            target_tau_days=target_tau_days,
            min_tau_days=min_tau_days,
            max_tau_days=max_tau_days,
        )

        if not selected_expirations:
            return [], "No valid expirations after Tau filtering"

        rows: list[dict[str, Any]] = []

        for expiration in selected_expirations:
            try:
                chain = ticker_obj.option_chain(expiration)
            except Exception as e:
                print(f"Warning: {ticker} {expiration} option chain failed: {e}")
                continue

            expiration_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # Residual time in years
            Tau = max((expiration_dt - retrieval_dt).days / 365.0, 0.0)

            for option_type, option_df in [
                ("call", chain.calls),
                ("put", chain.puts),
            ]:
                filtered_df = filter_options(
                    option_df=option_df,
                    S=S,
                    contracts_per_side=contracts_per_side,
                    require_liquidity=require_liquidity,
                )

                if filtered_df.empty:
                    continue

                for _, option in filtered_df.iterrows():
                    Strike = safe_float(option.get("strike"))
                    Price = option_market_price(option)

                    if Strike is None or Price is None:
                        continue

                    # Raw row includes identifiers for traceability.
                    # These identifiers are dropped from the final modeling dataset.
                    rows.append(
                        {
                            "ticker": ticker,
                            "contractSymbol": option.get("contractSymbol"),
                            "expiration": expiration,
                            "retrieval_datetime_utc": retrieval_dt.isoformat(),

                            # Table 2 variables
                            "S": S,
                            "Strike": Strike,
                            "Tau": Tau,
                            "Sigma": Sigma,
                            "Call": 1 if option_type == "call" else 0,
                            "Price": Price,
                        }
                    )

            time.sleep(sleep_seconds)

        if not rows:
            return [], "No usable option rows after filtering"

        return rows, None

    except Exception as e:
        return [], str(e)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", default="data")
    parser.add_argument("--vol-lookback", default="1y")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="Optional quick-test ticker limit")

    parser.add_argument(
        "--target-tau-days",
        type=parse_target_tau_days,
        default=[30, 180, 720],
        help="Comma-separated target maturities in days. Default: 30,180,720",
    )

    parser.add_argument(
        "--contracts-per-side",
        type=int,
        default=3,
        help="Number of near-the-money calls/puts to keep per expiration.",
    )

    parser.add_argument(
        "--min-tau-days",
        type=int,
        default=14,
        help="Minimum days to expiration.",
    )

    parser.add_argument(
        "--max-tau-days",
        type=int,
        default=900,
        help="Maximum days to expiration. Use a high value if you want LEAPS included.",
    )

    parser.add_argument(
        "--no-liquidity-filter",
        action="store_true",
        help="Disable bid/ask/lastPrice usability filter.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tickers = get_nasdaq100_tickers()

    if args.limit is not None:
        tickers = tickers[: args.limit]

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for ticker in tqdm(tickers, desc="Collecting Nasdaq-100 option data"):
        rows, error = collect_options_for_ticker(
            ticker=ticker,
            vol_lookback=args.vol_lookback,
            sleep_seconds=args.sleep,
            target_tau_days=args.target_tau_days,
            contracts_per_side=args.contracts_per_side,
            min_tau_days=args.min_tau_days,
            max_tau_days=args.max_tau_days,
            require_liquidity=not args.no_liquidity_filter,
        )

        all_rows.extend(rows)

        if error is not None:
            failures.append(
                {
                    "ticker": ticker,
                    "error": error,
                }
            )

    raw_df = pd.DataFrame(all_rows)

    table2_columns = ["S", "Strike", "Tau", "Sigma", "Call", "Price"]

    if raw_df.empty:
        model_df = pd.DataFrame(columns=table2_columns)
    else:
        model_df = raw_df[table2_columns].copy()

    failures_df = pd.DataFrame(failures)

    raw_path = outdir / "nasdaq100_options_raw.csv"
    model_path = outdir / "nasdaq100_options_model.csv"
    failures_path = outdir / "nasdaq100_failures.csv"

    raw_df.to_csv(raw_path, index=False)
    model_df.to_csv(model_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    print(f"Saved raw dataset: {raw_path}")
    print(f"Saved Table 2 modeling dataset: {model_path}")
    print(f"Saved failures: {failures_path}")
    print()
    print(f"Raw rows: {len(raw_df):,}")
    print(f"Model rows: {len(model_df):,}")
    print(f"Tickers with failures/warnings: {len(failures_df):,}")

    if not raw_df.empty:
        rows_per_ticker = raw_df.groupby("ticker").size()
        tau_summary = raw_df["Tau"].describe()

        print()
        print("Rows per ticker summary:")
        print(rows_per_ticker.describe())

        print()
        print("Tau summary:")
        print(tau_summary)

        print()
        print("Selected expirations per ticker:")
        print(raw_df.groupby("ticker")["expiration"].nunique().describe())

        print()
        print("Most common selected expirations:")
        print(raw_df["expiration"].value_counts().head(20))


if __name__ == "__main__":
    main()