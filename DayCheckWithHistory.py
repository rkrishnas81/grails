import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_build_trades(ticker: str, lookback_days: int) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    period_days = max(lookback_days + 40, 180)

    df = yf.download(
        ticker,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty or len(df) < 5:
        return pd.DataFrame()

    df = df.sort_index()

    # Handle MultiIndex columns (sometimes returned by yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in set(df.columns.get_level_values(1)):
            df = df.xs(ticker, axis=1, level=1)

    df = df[["Open", "Close"]].copy()
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Open", "Close"])

    # Keep lookback window + 1 for prev close + 1 for next open
    df = df.tail(lookback_days + 2)
    if len(df) < 5:
        return pd.DataFrame()

    close = df["Close"]
    open_ = df["Open"]
    prev_close = close.shift(1)

    negative_pct = ((close - prev_close) / prev_close) * 100
    next_open = open_.shift(-1)
    next_open_pct = ((next_open - close) / close) * 100

    # Trades can be evaluated only where both prev_close and next_open exist
    valid = df.index[1:-1]

    trades = pd.DataFrame({
        "Stock": ticker,
        "Date_of_Negative": valid.date,
        "Negative_%": negative_pct.loc[valid].astype(float),
        "Next_Day_Open_%": next_open_pct.loc[valid].astype(float),
    })

    # Keep only down days
    trades = trades[trades["Negative_%"] < 0].copy()
    trades = trades.sort_values("Date_of_Negative").reset_index(drop=True)

    # Row number
    trades.insert(0, "Row_No", range(1, len(trades) + 1))

    return trades


def find_best_winning_ranges(trades: pd.DataFrame,
                             min_trades: int = 8) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    neg = trades["Negative_%"].astype(float).to_numpy()
    ret = trades["Next_Day_Open_%"].astype(float).to_numpy()

    results = []
    seen_signatures = set()

    def add_result(rule_text: str, mask: np.ndarray):
        n = int(mask.sum())
        if n < min_trades:
            return

        sig = tuple(np.where(mask)[0].tolist())
        if sig in seen_signatures:
            return
        seen_signatures.add(sig)

        r = ret[mask]
        winrate = (r > 0).mean() * 100.0
        avg = r.mean()
        total = r.sum()

        score = (winrate / 100.0) * avg * np.log1p(n)

        results.append({
            "Rule": rule_text,
            "Trades": n,
            "WinRate_%": winrate,
            "AvgNextOpen_%": avg,
            "TotalNextOpen_%": total,
            "Score": score,
        })

    thresholds = np.arange(-0.25, -10.25, -0.25)
    for t in thresholds:
        add_result(f"Negative_% <= {t:.2f}", neg <= t)

    edges = np.arange(-10.0, 0.0, 0.25)
    for low in edges:
        for high in edges:
            if high <= low or high > -0.10:
                continue
            add_result(f"{low:.2f} < Negative_% <= {high:.2f}",
                       (neg > low) & (neg <= high))

    if not results:
        return pd.DataFrame()

    res = pd.DataFrame(results)
    res = res.sort_values(["Score", "TotalNextOpen_%"], ascending=False).reset_index(drop=True)

    for c in ["WinRate_%", "AvgNextOpen_%", "TotalNextOpen_%", "Score"]:
        res[c] = res[c].astype(float).round(3)

    return res


# --------- ADDED (ONLY FOR "MATCH TOP-3 RULES TODAY") ---------
def negative_pct_today(ticker: str) -> float | None:
    """
    Returns today's Negative_% (Close vs previous Close).
    If today is not a down day or data unavailable, returns None.
    """
    df = yf.download(
        ticker,
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty or len(df) < 2:
        return None

    df = df.sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in set(df.columns.get_level_values(1)):
            df = df.xs(ticker, axis=1, level=1)

    prev_close = float(df["Close"].iloc[-2])
    today_close = float(df["Close"].iloc[-1])

    neg_pct = ((today_close - prev_close) / prev_close) * 100
    return float(neg_pct) if neg_pct < 0 else None


def rule_matches(rule: str, neg_pct: float) -> bool:
    """
    Checks if neg_pct falls inside rule range.
    Handles:
      - 'Negative_% <= -2.00'
      - '-3.50 < Negative_% <= -2.00'
    """
    rule = rule.strip()

    # ---- THRESHOLD RULE FIRST ----
    # Example: "Negative_% <= -2.00"
    if rule.startswith("Negative_%") and "<=" in rule:
        _, high_str = rule.split("<=", 1)
        high = float(high_str.strip())
        return neg_pct <= high

    # ---- BAND RULE ----
    # Example: "-3.50 < Negative_% <= -2.00"
    if "<" in rule and "<=" in rule:
        left, right = rule.split("<", 1)
        low = float(left.strip())
        _, high_str = right.split("<=", 1)
        high = float(high_str.strip())
        return (neg_pct > low) and (neg_pct <= high)

    return False

# ------------------------------------------------------------


def main():
    tickers_input = input(
        "Enter stock ticker(s) (comma separated, ex: AAPL,NVDA,TSLA): "
    ).strip()

    lb = input("Enter number of days to look back [default 90]: ").strip()
    lookback_days = int(lb) if lb else 90

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not tickers:
        print("No valid tickers entered.")
        return

    # --------- ADDED (ONLY FOR "MATCH TOP-3 RULES TODAY") ---------
    stocks_matching_today = []
    # ------------------------------------------------------------

    for ticker in tickers:
        print("\n" + "=" * 25)
        print(f"STOCK: {ticker}")
        print("=" * 25)

        trades = fetch_and_build_trades(ticker, lookback_days)
        if trades.empty:
            print("No usable data / trades found.")
            continue

        print("\nOutput Table:")
        print(trades.round({"Negative_%": 2, "Next_Day_Open_%": 2}).to_string(index=False))

        print(f"\nTOTAL Sum of Next Day Open %: {trades['Next_Day_Open_%'].sum():.2f}%")
        print(f"TOTAL Sum of Negative %: {trades['Negative_%'].sum():.2f}%")

        best = find_best_winning_ranges(trades, min_trades=8)
        best = best[best["WinRate_%"] >= 75].reset_index(drop=True)


        if best.empty:
            print("\nNo rules with WinRate ≥ 75%.")
            continue

        print("\nTOP 3 RULES FOUND (WinRate ≥ 75%):")
        for rank, row in best.head(3).iterrows():

            print(f"\n#{rank + 1}")
            print(f"  Rule: {row['Rule']}")
            print(f"  Trades: {int(row['Trades'])}")
            print(f"  WinRate: {row['WinRate_%']:.2f}%")
            print(f"  Avg Next Open %: {row['AvgNextOpen_%']:.3f}%")
            print(f"  Total Next Open %: {row['TotalNextOpen_%']:.3f}%")

        # --------- ADDED (ONLY FOR "MATCH TOP-3 RULES TODAY") ---------
        today_neg = negative_pct_today(ticker)
        if today_neg is not None:
            for _, row in best.head(3).iterrows():
                if rule_matches(str(row["Rule"]), float(today_neg)):
                    stocks_matching_today.append(ticker)
                    break
        # ------------------------------------------------------------

    # --------- ADDED (ONLY FOR "MATCH TOP-3 RULES TODAY") ---------
    if stocks_matching_today:
        print("\nSTOCKS MATCHING TOP-3 RULES TODAY:")
        for s in sorted(set(stocks_matching_today)):
            print(s)
    else:
        print("\nNO STOCK MATCHED TOP-3 RULES TODAY.")
    # ------------------------------------------------------------


if __name__ == "__main__":
    main()
