from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


# =============================
# PATHS
# =============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "scanner_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================
# HELPERS
# =============================
def get_ticker_from_argv() -> str:
    # If user passed ticker in command line, use it
    if len(sys.argv) >= 2:
        t = sys.argv[1].upper().strip().replace(".", "-")
        if not t:
            raise SystemExit("❌ Empty ticker argument.")
        return t

    # Otherwise prompt user to type it
    ticker = input("📈 Please enter stock ticker (e.g. AAPL, TSLA): ").upper().strip().replace(".", "-")
    if not ticker:
        raise SystemExit("❌ No ticker entered.")
    return ticker


def roll_end_date(s: str | None) -> pd.Timestamp:
    d = pd.Timestamp.today().normalize() if not s else pd.to_datetime(s).normalize()
    if d.weekday() >= 5:
        d -= BDay(1)
    return d


def safe(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _extract_ohlcv(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance can return:
      - single-level columns: Open High Low Close Volume
      - multiindex columns: (TICKER, Open) ... when group_by="ticker"
    This returns a clean OHLCV df.
    """
    cols = ["Open", "High", "Low", "Close", "Volume"]

    if hist is None or hist.empty:
        raise SystemExit(f"❌ No data returned for {ticker}")

    # MultiIndex: (ticker, field)
    if isinstance(hist.columns, pd.MultiIndex):
        top = set(hist.columns.get_level_values(0))
        if ticker not in top:
            matches = [t for t in top if str(t).upper() == ticker.upper()]
            if not matches:
                raise SystemExit(f"❌ {ticker} not found in MultiIndex columns: {sorted(top)}")
            ticker = matches[0]

        df = hist[ticker]
        if not set(cols).issubset(df.columns):
            raise SystemExit(f"❌ {ticker} missing OHLCV. Got: {list(df.columns)}")
        return df[cols].dropna()

    # Single-level
    if set(cols).issubset(hist.columns):
        return hist[cols].dropna()

    raise SystemExit(f"❌ {ticker} missing OHLCV columns. Got: {list(hist.columns)}")


# =============================
# DOWNLOAD
# =============================
def download_history(ticker: str, end_date: pd.Timestamp) -> pd.DataFrame:
    hist = None
    for attempt in range(3):
        try:
            hist = yf.download(
                ticker,
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=False,   # IMPORTANT
                group_by="ticker",   # allows MultiIndex; we handle both
                threads=True,
            )
            break
        except Exception:
            time.sleep(0.6 * (2 ** attempt))

    df = _extract_ohlcv(hist, ticker)
    df.index = pd.to_datetime(df.index)
    df = df.loc[df.index <= end_date]
    if df.empty:
        raise SystemExit(f"❌ {ticker} has no rows after end_date filter.")
    return df


def download_spy_close(end_date: pd.Timestamp) -> pd.Series:
    hist = None
    for attempt in range(3):
        try:
            hist = yf.download(
                "SPY",
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=False,   # IMPORTANT
                group_by="ticker",
                threads=True,
            )
            break
        except Exception:
            time.sleep(0.6 * (2 ** attempt))

    df = _extract_ohlcv(hist, "SPY")
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index)
    close = close.loc[close.index <= end_date].dropna()
    if close.empty:
        raise SystemExit("❌ SPY close empty after end_date filter.")
    return close


# =============================
# FEATURES
# =============================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["PrevClose"] = d["Close"].shift(1)
    d = d.dropna(subset=["PrevClose"])

    d["DayRetPct"] = (d["Close"] / d["PrevClose"] - 1) * 100
    d["RangePct"] = ((d["High"] - d["Low"]) / d["PrevClose"]) * 100

    hl = (d["High"] - d["Low"]).replace(0, np.nan)
    d["ClosePos"] = ((d["Close"] - d["Low"]) / hl).clip(0, 1)

    d["Vol20"] = d["Volume"].rolling(20, min_periods=10).mean()
    d["VolRel"] = (d["Volume"] / d["Vol20"]).replace([np.inf, -np.inf], np.nan)

    d["Prior20High"] = d["High"].rolling(21).max().shift(1)
    d["DollarVol"] = d["Close"] * d["Volume"]

    d["SMA50"] = d["Close"].rolling(50, min_periods=50).mean()
    d["SMA50Slope"] = d["SMA50"].diff(5)

    d["RangePct10"] = d["RangePct"].rolling(10, min_periods=10).mean()
    d["RangePct20"] = d["RangePct"].rolling(20, min_periods=20).mean()

    return d


def add_relative_strength(d: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    out = d.copy()

    aligned_spy = spy_close.reindex(out.index).ffill()
    rs = (out["Close"] / aligned_spy).replace([np.inf, -np.inf], np.nan)

    out["RS"] = rs.astype(float)
    out["RS20High"] = out["RS"].rolling(20, min_periods=20).max()
    out["RS_SMA20"] = out["RS"].rolling(20, min_periods=20).mean()
    return out


# =============================
# SCORING
# =============================
def footprint_score(row: pd.Series) -> float:
    score = 0.0

    volrel = row.get("VolRel", np.nan)
    dayret = row.get("DayRetPct", np.nan)
    closepos = row.get("ClosePos", np.nan)
    rangepct = row.get("RangePct", np.nan)
    prior20h = row.get("Prior20High", np.nan)
    close = row.get("Close", np.nan)

    if np.isfinite(volrel) and volrel >= 1.35:
        score += 22
    if np.isfinite(dayret) and np.isfinite(volrel) and dayret > 0 and volrel >= 1.35:
        score += 16
    if np.isfinite(closepos) and closepos >= 0.72:
        score += 16
    if np.isfinite(rangepct) and np.isfinite(volrel) and rangepct <= 1.6 and volrel >= 1.35:
        score += 18
    if np.isfinite(prior20h) and np.isfinite(close) and np.isfinite(volrel) and close > prior20h and volrel >= 1.35:
        score += 14
    if np.isfinite(dayret) and np.isfinite(volrel) and dayret < -1 and volrel >= 1.6:
        score -= 22

    return float(max(0.0, min(100.0, score)))


def confirmation_bonus(d: pd.DataFrame, i: int) -> float:
    if i < 60:
        return 0.0

    r = d.iloc[i]
    bonus = 0.0

    sma50 = r.get("SMA50", np.nan)
    sma50s = r.get("SMA50Slope", np.nan)
    close = r.get("Close", np.nan)

    if np.isfinite(close) and np.isfinite(sma50) and np.isfinite(sma50s) and close > sma50 and sma50s > 0:
        bonus += 10

    rs = r.get("RS", np.nan)
    rs_sma20 = r.get("RS_SMA20", np.nan)
    rs20h = r.get("RS20High", np.nan)

    if np.isfinite(rs) and np.isfinite(rs_sma20) and rs > rs_sma20:
        bonus += 6
    if np.isfinite(rs) and np.isfinite(rs20h) and rs >= rs20h * 0.999:
        bonus += 6

    recent = d.iloc[max(0, i - 12): i]
    if len(recent) > 0:
        cond = (
            (recent["DayRetPct"] > 0) &
            (recent["VolRel"] >= 1.25) &
            (recent["ClosePos"] >= 0.60)
        )
        acc_days = int(cond.sum())
        if acc_days >= 2:
            bonus += 8
        if acc_days >= 3:
            bonus += 6

    r10 = r.get("RangePct10", np.nan)
    r20 = r.get("RangePct20", np.nan)
    if np.isfinite(r10) and np.isfinite(r20) and r10 < r20:
        bonus += 5

    return float(min(35.0, bonus))


# =============================
# MAIN
# =============================
def main() -> None:
    ticker = get_ticker_from_argv()
    end_date = roll_end_date(None)

    spy_close = download_spy_close(end_date)
    df = download_history(ticker, end_date)

    feats = add_features(df)
    feats = add_relative_strength(feats, spy_close)

    # Compute score for EVERY day
    rows: list[dict] = []
    for i in range(len(feats)):
        base = footprint_score(feats.iloc[i])
        bonus = confirmation_bonus(feats, i)
        score = float(min(100.0, base + bonus))

        rows.append({
            "Ticker": ticker,
            "Date": feats.index[i].strftime("%Y-%m-%d"),
            "Score": score,
            "BaseScore": base,
            "Bonus": bonus,
            "Close": safe(feats.iloc[i].get("Close", np.nan)),
            "Volume": safe(feats.iloc[i].get("Volume", np.nan)),
            "DollarVol": safe(feats.iloc[i].get("DollarVol", np.nan)),
            "RS": safe(feats.iloc[i].get("RS", np.nan)),
        })

    out = pd.DataFrame(rows)

    # Keep ONLY last 90 trading days
    out = out.tail(90).reset_index(drop=True)

    print("\nLAST 30 DAYS (EVERY DAY SCORE)")
    print("-" * 110)
    print(out.tail(30).to_string(index=False))

    path = os.path.join(OUTPUT_DIR, f"score_last90_{ticker}.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
