from __future__ import annotations

import os
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
def ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} (default {default}): ").strip()
    return s or default


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


def chunks(lst: list[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_tickers(s: str) -> list[str]:
    tickers = [t.strip().upper().replace(".", "-") for t in s.split(",")]
    return list(dict.fromkeys([t for t in tickers if t]))


def _extract_ohlcv(hist: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if hist is None or hist.empty:
        return None

    if isinstance(hist.columns, pd.MultiIndex):
        if ticker not in hist.columns.get_level_values(0):
            return None
        df = hist[ticker]
    else:
        df = hist

    if not set(cols).issubset(df.columns):
        return None

    return df[cols].dropna()


# =============================
# FEATURES
# =============================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["PrevClose"] = d["Close"].shift(1)
    d.dropna(inplace=True)

    d["DayRetPct"] = (d["Close"] / d["PrevClose"] - 1) * 100
    d["RangePct"] = ((d["High"] - d["Low"]) / d["PrevClose"]) * 100

    hl = (d["High"] - d["Low"]).replace(0, np.nan)
    d["ClosePos"] = ((d["Close"] - d["Low"]) / hl).clip(0, 1)

    d["Vol20"] = d["Volume"].rolling(20, min_periods=10).mean()
    d["VolRel"] = d["Volume"] / d["Vol20"]

    d["Prior20High"] = d["High"].rolling(21).max().shift(1)
    d["DollarVol"] = d["Close"] * d["Volume"]

    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA50Slope"] = d["SMA50"].diff(5)

    d["RangePct10"] = d["RangePct"].rolling(10).mean()
    d["RangePct20"] = d["RangePct"].rolling(20).mean()

    return d


# =============================
# SPY + RS
# =============================
def download_spy_close(end_date: pd.Timestamp) -> pd.Series:
    spy = yf.download(
        "SPY",
        period="1y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    df = _extract_ohlcv(spy, "SPY")
    if df is None or df.empty:
        raise SystemExit("❌ Failed to download SPY data.")

    close = df["Close"].loc[df.index <= end_date].dropna()
    if close.empty:
        raise SystemExit("❌ SPY close series empty.")

    return close


def add_relative_strength(d: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    out = d.copy()
    spy_aligned = spy_close.reindex(out.index).ffill()
    out["RS"] = out["Close"] / spy_aligned
    out["RS20High"] = out["RS"].rolling(20).max()
    out["RS_SMA20"] = out["RS"].rolling(20).mean()
    return out


# =============================
# SCORING
# =============================
def footprint_score(row: pd.Series):
    score = 0
    reasons = []

    if row["VolRel"] >= 1.35:
        score += 22
        reasons.append("Volume surge")

    if row["DayRetPct"] > 0 and row["VolRel"] >= 1.35:
        score += 16
        reasons.append("Up day + vol")

    if row["ClosePos"] >= 0.72:
        score += 16
        reasons.append("Close near high")

    if row["RangePct"] <= 1.6 and row["VolRel"] >= 1.35:
        score += 18
        reasons.append("Absorption")

    if row["Close"] > row["Prior20High"] and row["VolRel"] >= 1.35:
        score += 14
        reasons.append("Breakout + vol")

    if row["DayRetPct"] < -1 and row["VolRel"] >= 1.6:
        score -= 22
        reasons.append("Distribution risk")

    return max(0, min(100, score)), reasons


def confirmation_bonus(d: pd.DataFrame, idx: int):
    row = d.iloc[idx]
    bonus = 0
    reasons = []

    if row["Close"] > row["SMA50"] and row["SMA50Slope"] > 0:
        bonus += 10
        reasons.append("Above rising SMA50")

    if row["RS"] > row["RS_SMA20"]:
        bonus += 6
        reasons.append("RS improving")

    if row["RS"] >= row["RS20High"] * 0.999:
        bonus += 6
        reasons.append("RS near 20D high")

    return min(35, bonus), reasons


def score_on_last_bar(feats: pd.DataFrame, end_date: pd.Timestamp):
    feats = feats.loc[feats.index <= end_date]
    if feats.empty:
        return None

    idx = len(feats) - 1
    base, base_r = footprint_score(feats.iloc[idx])
    bonus, bonus_r = confirmation_bonus(feats, idx)

    return min(100, base + bonus), base_r + bonus_r, feats.iloc[idx]


# =============================
# DOWNLOAD MANY
# =============================
def download_many(tickers: list[str]):
    out = {}
    for batch in chunks(tickers, 60):
        hist = yf.download(
            " ".join(batch),
            period="1y",
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )

        for t in batch:
            df = _extract_ohlcv(hist, t)
            if df is not None and not df.empty:
                out[t] = df

        time.sleep(0.15)

    return out


# =============================
# MAIN
# =============================
def main():
    tickers = parse_tickers(input("Enter Tickers (comma separated): "))
    if not tickers:
        raise SystemExit("❌ No tickers entered.")

    end_date = roll_end_date(ask("Enter Date YYYY-MM-DD", ""))
    min_score = 60

    spy_close = download_spy_close(end_date)
    hist_map = download_many(tickers)

    results = []

    for t in tickers:
        df = hist_map.get(t)
        if df is None or len(df) < 80:
            continue

        feats = add_features(df)
        feats = add_relative_strength(feats, spy_close)

        out = score_on_last_bar(feats, end_date)
        if out is None:
            continue

        sc, reasons, row = out

        if sc < min_score:
            continue

        # ✅ LIQUIDITY FILTER (ONLY ADDITION)
        if safe(row["DollarVol"]) < 50_000_000:
            continue

        results.append({
            "Ticker": t,
            "BarDate": row.name.strftime("%Y-%m-%d"),
            "Score": sc,
            "Close": safe(row["Close"]),
            "DayRetPct": safe(row["DayRetPct"]),
            "VolRel": safe(row["VolRel"]),
            "DollarVol": safe(row["DollarVol"]),
            "RS": safe(row["RS"]),
            "Reasons": ", ".join(reasons),
        })

    if not results:
        print("No matches found.")
        return

    out_df = (
        pd.DataFrame(results)
        .sort_values(["Score", "VolRel", "DollarVol"], ascending=False)
        .reset_index(drop=True)
    )

    print(out_df)

    


if __name__ == "__main__":
    main()
