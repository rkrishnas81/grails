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
    if d.weekday() >= 5:  # weekend -> roll back to prior business day
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
    tickers = [t for t in tickers if t]
    return list(dict.fromkeys(tickers))  # de-dup preserve order


def _extract_ohlcv(hist: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if hist is None or hist.empty:
        return None

    if isinstance(hist.columns, pd.MultiIndex):
        top = set(hist.columns.get_level_values(0))
        if ticker not in top:
            matches = [t for t in top if str(t).upper() == ticker.upper()]
            if not matches:
                return None
            ticker = matches[0]
        df = hist[ticker]
        if not set(cols).issubset(df.columns):
            return None
        return df[cols].dropna()

    if set(cols).issubset(hist.columns):
        return hist[cols].dropna()

    return None


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


# =============================
# SPY + RS
# =============================
def download_spy_close(end_date: pd.Timestamp) -> pd.Series:
    spy = yf.download(
        "SPY",
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    df = _extract_ohlcv(spy, "SPY")
    if df is None or df.empty:
        raise SystemExit("❌ Failed to download SPY.")
    close = df["Close"].copy()
    close = close.loc[close.index <= end_date].dropna()
    if close.empty:
        raise SystemExit("❌ SPY close empty after end_date filter.")
    return close


def add_relative_strength(d: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    out = d.copy()
    aligned_spy = spy_close.reindex(out.index).ffill()
    out["RS"] = (out["Close"] / aligned_spy).replace([np.inf, -np.inf], np.nan).astype(float)
    out["RS20High"] = out["RS"].rolling(20, min_periods=20).max()
    out["RS_SMA20"] = out["RS"].rolling(20, min_periods=20).mean()
    return out


# =============================
# SCORING
# =============================
def footprint_score(row: pd.Series) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    volrel = row.get("VolRel", np.nan)
    dayret = row.get("DayRetPct", np.nan)
    closepos = row.get("ClosePos", np.nan)
    rangepct = row.get("RangePct", np.nan)
    prior20h = row.get("Prior20High", np.nan)
    close = row.get("Close", np.nan)

    if np.isfinite(volrel) and volrel >= 1.35:
        score += 22
        reasons.append("Volume surge")

    if np.isfinite(dayret) and np.isfinite(volrel) and dayret > 0 and volrel >= 1.35:
        score += 16
        reasons.append("Up day + vol")

    if np.isfinite(closepos) and closepos >= 0.72:
        score += 16
        reasons.append("Close near high")

    if np.isfinite(rangepct) and np.isfinite(volrel) and rangepct <= 1.6 and volrel >= 1.35:
        score += 18
        reasons.append("Absorption")

    if np.isfinite(prior20h) and np.isfinite(close) and np.isfinite(volrel) and close > prior20h and volrel >= 1.35:
        score += 14
        reasons.append("Breakout + vol")

    if np.isfinite(dayret) and np.isfinite(volrel) and dayret < -1 and volrel >= 1.6:
        score -= 22
        reasons.append("Distribution risk")

    score = float(max(0.0, min(100.0, score)))
    return score, reasons


def confirmation_bonus(d: pd.DataFrame, idx: int) -> tuple[float, list[str]]:
    if idx < 60:
        return 0.0, []

    row = d.iloc[idx]
    bonus = 0.0
    reasons: list[str] = []

    sma50 = row.get("SMA50", np.nan)
    sma50s = row.get("SMA50Slope", np.nan)
    close = row.get("Close", np.nan)

    if np.isfinite(close) and np.isfinite(sma50) and np.isfinite(sma50s) and close > sma50 and sma50s > 0:
        bonus += 10
        reasons.append("Above rising SMA50")

    rs = row.get("RS", np.nan)
    rs_sma20 = row.get("RS_SMA20", np.nan)
    rs20h = row.get("RS20High", np.nan)

    if np.isfinite(rs) and np.isfinite(rs_sma20) and rs > rs_sma20:
        bonus += 6
        reasons.append("RS improving")
    if np.isfinite(rs) and np.isfinite(rs20h) and rs >= rs20h * 0.999:
        bonus += 6
        reasons.append("RS near 20D high")

    recent = d.iloc[max(0, idx - 12): idx]
    if len(recent) > 0:
        cond = (
            (recent["DayRetPct"] > 0) &
            (recent["VolRel"] >= 1.25) &
            (recent["ClosePos"] >= 0.60)
        )
        acc_days = int(cond.sum())
        if acc_days >= 2:
            bonus += 8
            reasons.append(f"{acc_days} acc days (12D)")
        if acc_days >= 3:
            bonus += 6
            reasons.append("Repeat accumulation")

    r10 = row.get("RangePct10", np.nan)
    r20 = row.get("RangePct20", np.nan)
    if np.isfinite(r10) and np.isfinite(r20) and r10 < r20:
        bonus += 5
        reasons.append("Range contracting")

    bonus = float(min(35.0, bonus))
    return bonus, reasons


def score_on_last_bar(feats: pd.DataFrame, end_date: pd.Timestamp) -> tuple[float, list[str], pd.Series] | None:
    feats = feats.loc[feats.index <= end_date]
    if feats.empty:
        return None

    idx = len(feats) - 1
    base, base_r = footprint_score(feats.iloc[idx])
    bonus, bonus_r = confirmation_bonus(feats, idx)
    score = float(min(100.0, base + bonus))
    reasons = base_r + bonus_r
    return score, reasons, feats.iloc[idx]


# =============================
# DOWNLOAD MANY (FAST)
# =============================
def download_many(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for batch in chunks(tickers, 60):
        hist = None
        for attempt in range(3):
            try:
                hist = yf.download(
                    " ".join(batch),
                    period="1y",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    group_by="ticker",
                    threads=True,
                )
                break
            except Exception:
                time.sleep(0.6 * (2 ** attempt))

        if hist is None or hist.empty:
            continue

        for t in batch:
            df = _extract_ohlcv(hist, t)
            if df is not None and not df.empty:
                out[t] = df

        time.sleep(0.15)
    return out


# =============================
# MAIN
# =============================
def main() -> None:
    tickers_in = input("Enter Tickers (with Comma Separated): ").strip()
    tickers = parse_tickers(tickers_in)
    if not tickers:
        raise SystemExit("❌ No tickers entered.")

    end_date = roll_end_date(ask("Enter Date YYYY-MM-DD", ""))  # default today
    min_score = 50.0

    spy_close = download_spy_close(end_date)
    hist_map = download_many(tickers)

    results: list[dict] = []

    for t in tickers:
        df = hist_map.get(t)
        if df is None or df.empty:
            continue

        df = df.loc[df.index <= end_date]
        if len(df) < 80:
            continue

        feats = add_features(df)
        feats = add_relative_strength(feats, spy_close)

        out = score_on_last_bar(feats, end_date)
        if out is None:
            continue

        sc, reasons, row = out
        if sc < min_score:
            continue

        results.append({
            "Ticker": t,
            "BarDate": pd.Timestamp(row.name).strftime("%Y-%m-%d"),
            "Score": sc,
            "Close": safe(row.get("Close", np.nan)),
            "DayRetPct": safe(row.get("DayRetPct", np.nan)),
            "VolRel": safe(row.get("VolRel", np.nan)),
            "ClosePos": safe(row.get("ClosePos", np.nan)),
            "RangePct": safe(row.get("RangePct", np.nan)),
            "DollarVol": safe(row.get("DollarVol", np.nan)),
            "RS": safe(row.get("RS", np.nan)),
            "Reasons": ", ".join(reasons),
        })

    if not results:
        print(f"\nNo matches (Score >= {min_score:.0f}).")
        return

    out_df = (
        pd.DataFrame(results)
        .sort_values(["Score", "VolRel", "DollarVol"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    print(f"\nRESULTS (Score >= {min_score:.0f})")
    print("-" * 110)
    print(out_df.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, f"scan_tickers_{end_date.strftime('%Y-%m-%d')}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
