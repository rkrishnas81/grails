from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


# =============================
# PRINT SETTINGS (prevents "..." hiding BaseScore/Bonus)
# =============================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.colheader_justify", "center")


# =============================
# CONFIRMATION + BUY LIST SETTINGS
# =============================
USE_REGIME_CONFIRM = True          # QQQ market filter
USE_INTRADAY_60M_CONFIRM = True    # 60m "into close" confirmation

MIN_REGIME_SCORE = 60.0
MIN_INTRADAY_SCORE = 60.0

BUY_MIN_SCORE = 70.0
BUY_MIN_REGIME = 60.0              # ✅ CHANGED (was 70.0)
BUY_MIN_INTRADAY = 70.0
MAX_DAYRET_FOR_BUY = 6.0

PRINT_CONFIRM_TABLE = False
PRINT_BUY_LIST_TOP_N = 5
# =============================
# EARNINGS FILTER SETTINGS
# =============================
EARNINGS_EXCLUDE_DAYS = 5   # change this anytime (5 = next 5 days, 0 = today only, -1 = disable)



# =============================
# BASIC HELPERS
# =============================
def parse_tickers(s: str) -> List[str]:
    tickers = [t.strip().upper().replace(".", "-") for t in s.split(",")]
    out, seen = [], set()
    for t in tickers:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def subtract_tickers(universe: List[str], exclude: List[str]) -> List[str]:
    if not exclude:
        return universe
    ex = set(exclude)
    return [t for t in universe if t not in ex]


def roll_end_date(s: str | None) -> pd.Timestamp:
    d = pd.Timestamp.today().normalize() if not s else pd.to_datetime(s).normalize()
    if d.weekday() >= 5:
        d -= BDay(1)
    return d


# ✅ include the next trading day so shift(-1) has data for "yesterday"
def end_date_plus_one_trading_day(end_date: pd.Timestamp) -> pd.Timestamp:
    return (end_date + BDay(1)).normalize()


def safe(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


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


def _pct_str_to_float(v):
    if isinstance(v, str):
        v = v.strip().replace("%", "")
        return float(v) if v else np.nan
    try:
        return float(v)
    except Exception:
        return np.nan
# =============================
# EARNINGS DATE (NEXT) - cached
# =============================
_EARNINGS_CACHE: dict[str, str] = {}

def get_next_earnings_date_str(ticker: str) -> str:
    """
    Returns next earnings date as 'YYYY-MM-DD' (or '' if unavailable).
    Uses yfinance get_earnings_dates(limit=1).
    Cached to avoid repeated calls.
    """
    t = ticker.upper().replace(".", "-").strip()
    if not t:
        return ""

    if t in _EARNINGS_CACHE:
        return _EARNINGS_CACHE[t]

    try:
        edf = yf.Ticker(t).get_earnings_dates(limit=1)
        if edf is None or edf.empty:
            _EARNINGS_CACHE[t] = ""
            return ""

        # yfinance stores earnings datetime in the index
        dt = pd.to_datetime(edf.index[0])
        out = dt.strftime("%Y-%m-%d")
        _EARNINGS_CACHE[t] = out
        return out
    except Exception:
        _EARNINGS_CACHE[t] = ""
        return ""


# =============================
# ✅ SAME NET CALC, BUT FOR ANY SUBSET (NO LOGIC CHANGE)
# =============================
def _net_stats(df: pd.DataFrame) -> dict:
    """Returns Gain/Loss/Net using EXACT same rules as your current Net calc."""
    if df is None or df.empty:
        return {"gain": 0.0, "loss": 0.0, "net": 0.0}

    gain = float(
        np.where(
            (df["NextDayOpen%_num"] > 0) & (df["NextDayLow%_num"] < -1),
            -1.0,
            np.where(
                (df["NextDayOpen%_num"] > 0),
                df["NextDay%_num"],
                0.0
            )
        ).sum()
    )

    loss = float(
        df.loc[df["NextDayOpen%_num"] < 0, "NextDayOpen%_num"].sum(skipna=True)
    )

    net = gain - abs(loss)
    return {"gain": gain, "loss": loss, "net": net}


# =============================
# DOWNLOAD (DAILY)
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
                auto_adjust=False,  # IMPORTANT
                group_by="ticker",
                threads=True,
            )
            break
        except Exception:
            time.sleep(0.6 * (2 ** attempt))

    df = _extract_ohlcv(hist, ticker)
    if df is None or df.empty:
        raise SystemExit(f"❌ No data returned for {ticker}")

    df.index = pd.to_datetime(df.index)

    # keep one extra trading day beyond end_date
    end_plus = end_date_plus_one_trading_day(end_date)
    df = df.loc[df.index <= end_plus]

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
                auto_adjust=False,
                group_by="ticker",
                threads=True,
            )
            break
        except Exception:
            time.sleep(0.6 * (2 ** attempt))

    df = _extract_ohlcv(hist, "SPY")
    if df is None or df.empty:
        raise SystemExit("❌ Failed to download SPY data.")

    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index)

    end_plus = end_date_plus_one_trading_day(end_date)
    close = close.loc[close.index <= end_plus].dropna()

    if close.empty:
        raise SystemExit("❌ SPY close empty after end_date filter.")
    return close


def download_many(tickers: List[str], end_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    end_plus = end_date_plus_one_trading_day(end_date)

    for batch in chunks(tickers, 60):
        hist = yf.download(
            " ".join(batch),
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        for t in batch:
            df = _extract_ohlcv(hist, t)
            if df is None or df.empty:
                continue
            df.index = pd.to_datetime(df.index)
            df = df.loc[df.index <= end_plus]
            if not df.empty:
                out[t] = df
        time.sleep(0.15)
    return out


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
    """Match ByDate.py bonus logic exactly (no extra bonus rules)."""
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

    return float(min(35.0, bonus))


# =========================================================
# CONFIRMATION LAYER
#   ✅ FIX: IntradayScore must be calculated for the SAME bar_date
# =========================================================

# Small cache to reduce repeated intraday downloads
_INTRADAY_CACHE: dict[Tuple[str, str, str], pd.DataFrame] = {}


def _download_60m_window(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Download 60m bars for a specific historical window (works for past end_date).
    """
    start_s = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_s = pd.to_datetime(end).strftime("%Y-%m-%d")
    key = (ticker, start_s, end_s)
    if key in _INTRADAY_CACHE:
        return _INTRADAY_CACHE[key].copy()

    try:
        hist = yf.download(
            ticker,
            start=start_s,
            end=(pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="60m",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        _INTRADAY_CACHE[key] = pd.DataFrame()
        return pd.DataFrame()

    if hist is None or hist.empty:
        _INTRADAY_CACHE[key] = pd.DataFrame()
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(hist.columns, pd.MultiIndex):
        if ticker in hist.columns.get_level_values(0):
            df = hist[ticker].copy()
        else:
            first_key = hist.columns.get_level_values(0)[0]
            df = hist[first_key].copy()
    else:
        df = hist.copy()

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(set(df.columns)):
        _INTRADAY_CACHE[key] = pd.DataFrame()
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df.index = pd.to_datetime(df.index)

    _INTRADAY_CACHE[key] = df.copy()
    return df


def intraday_confirm_60m(ticker: str, target_date: pd.Timestamp) -> dict:
    """
    Intraday confirm for the SAME day as the daily signal (target_date).
    Uses last 3 60m bars on that date.
    """
    target_dt = pd.to_datetime(target_date).normalize()

    # Pull a small window around the target date so we always include that day’s bars
    start = target_dt - pd.Timedelta(days=5)
    end = target_dt + pd.Timedelta(days=1)

    idf = _download_60m_window(ticker, start=start, end=end)
    if idf.empty or len(idf) < 4:
        return {"ok": False, "score": 0.0, "reason": "no_60m_data_for_target_date"}

    day_df = idf[idf.index.date == target_dt.date()].copy()
    if len(day_df) < 4:
        return {"ok": False, "score": 0.0, "reason": f"not_enough_60m_bars_for_{target_dt.date()}"}

    closes = day_df["Close"].tail(3).astype(float).values
    try:
        slope = float(np.polyfit(np.arange(len(closes)), closes, 1)[0])
    except Exception:
        slope = 0.0

    last = day_df.iloc[-1]
    hi = float(last["High"])
    lo = float(last["Low"])
    cl = float(last["Close"])

    rng = (hi - lo) if (hi > lo) else np.nan
    close_pos = ((cl - lo) / rng) if np.isfinite(rng) and rng != 0 else 0.0

    v_last3 = float(day_df["Volume"].tail(3).astype(float).mean())
    v_med = float(day_df["Volume"].astype(float).median())
    v_ok = (v_med > 0) and (v_last3 >= 0.8 * v_med)

    score = 0.0
    if slope > 0:
        score += 40
    if close_pos >= 0.60:
        score += 35
    if v_ok:
        score += 25

    ok = score >= MIN_INTRADAY_SCORE
    return {"ok": ok, "score": float(score), "reason": f"date={target_dt.date()}, slope={slope:.5f}, closepos={close_pos:.2f}, v_ok={v_ok}"}


def regime_confirm_qqq(end_date: pd.Timestamp) -> dict:
    """
    RegimeScore is computed ONCE per run (as your original design).
    """
    try:
        qqq = yf.download(
            "QQQ",
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {"ok": True, "score": 50.0, "reason": "qqq_download_failed_default_ok"}

    if qqq is None or qqq.empty:
        return {"ok": True, "score": 50.0, "reason": "qqq_empty_default_ok"}

    # Flatten MultiIndex if present
    if isinstance(qqq.columns, pd.MultiIndex):
        if "QQQ" in qqq.columns.get_level_values(0):
            q = qqq["QQQ"].copy()
        else:
            first_key = qqq.columns.get_level_values(0)[0]
            q = qqq[first_key].copy()
    else:
        q = qqq.copy()

    need = {"Close", "High", "Low"}
    if not need.issubset(set(q.columns)):
        return {"ok": True, "score": 50.0, "reason": "qqq_missing_cols_default_ok"}

    q.index = pd.to_datetime(q.index)
    q = q.loc[q.index <= end_date].dropna(subset=["Close", "High", "Low"])
    if len(q) < 30:
        return {"ok": True, "score": 50.0, "reason": "qqq_short_history_default_ok"}

    close = q["Close"].astype(float)
    high = q["High"].astype(float)
    low = q["Low"].astype(float)

    sma20 = close.rolling(20).mean()
    mom5 = (close / close.shift(5) - 1) * 100

    prev = close.shift(1)
    rangepct = ((high - low) / prev) * 100
    r20 = rangepct.rolling(20).mean()

    last = close.index[-1]

    score = 0.0
    above20 = bool(close.loc[last] > sma20.loc[last]) if np.isfinite(sma20.loc[last]) else False
    mom_ok = bool(mom5.loc[last] > 0) if np.isfinite(mom5.loc[last]) else False

    if above20:
        score += 45
    if mom_ok:
        score += 35

    if np.isfinite(rangepct.loc[last]) and np.isfinite(r20.loc[last]) and r20.loc[last] > 0:
        if rangepct.loc[last] <= 1.5 * r20.loc[last]:
            score += 20

    ok = score >= MIN_REGIME_SCORE
    return {
        "ok": ok,
        "score": float(score),
        "reason": f"QQQ>20SMA={above20}, mom5={mom5.loc[last]:.2f}%, range={rangepct.loc[last]:.2f} vs avg20={r20.loc[last]:.2f}"
    }


# =============================
# HISTORY OUTPUT (per ticker): PRINT NOTHING unless Net (Gain - Loss) > 2
# =============================
def print_history_output(
    ticker: str,
    feats_in: pd.DataFrame,
    qqq_nextday_open_pct: pd.Series,
    end_date: pd.Timestamp,
    single_date_input: bool,
    regime_score_for_run: float,               # ✅ NEW (display-only)
    use_regime_confirm: bool,                 # ✅ NEW (display-only)
    use_intraday_60m_confirm: bool,           # ✅ NEW (display-only)
    min_day_thresh: float = 2.0,
    big_day_thresh: float = 6.0,
) -> None:

    feats = feats_in.copy()

    # runs for BOTH default date and typed date
    bar_dates = feats.index[feats.index <= end_date]
    if len(bar_dates) > 0:
        bar_date = bar_dates[-1]
        dayret = feats.loc[bar_date].get("DayRetPct", np.nan)
        if np.isfinite(dayret) and dayret > big_day_thresh:
            return
        if np.isfinite(dayret) and dayret < min_day_thresh:
            return

    feats["NextDay%"] = (feats["Close"].shift(-1) / feats["Close"] - 1) * 100
    feats["NextDayOpen%"] = (feats["Open"].shift(-1) / feats["Close"] - 1) * 100
    feats["NextDayHigh%"] = (feats["High"].shift(-1) / feats["Close"] - 1) * 100
    feats["NextDayLow%"] = (feats["Low"].shift(-1) / feats["Close"] - 1) * 100

    rows: list[dict] = []
    for i in range(len(feats)):
        base = footprint_score(feats.iloc[i])
        bonus = confirmation_bonus(feats, i)
        score = float(min(100.0, base + bonus))

        rows.append({
            "Ticker": ticker,
            "Signal Date": feats.index[i].strftime("%Y-%m-%d"),
            "SignalDay%": (f"{feats.iloc[i].get('DayRetPct', np.nan):.2f}%"
                           if np.isfinite(feats.iloc[i].get("DayRetPct", np.nan)) else ""),
            "Score": score,
            "NextDay%": (f"{feats.iloc[i].get('NextDay%', np.nan):.2f}%"
                         if np.isfinite(feats.iloc[i].get("NextDay%", np.nan)) else ""),
            "NextDayOpen%": (f"{feats.iloc[i].get('NextDayOpen%', np.nan):.2f}%"
                             if np.isfinite(feats.iloc[i].get("NextDayOpen%", np.nan)) else ""),
            "NextDayHigh%": (f"{feats.iloc[i].get('NextDayHigh%', np.nan):.2f}%"
                             if np.isfinite(feats.iloc[i].get("NextDayHigh%", np.nan)) else ""),
            "NextDayLow%": (f"{feats.iloc[i].get('NextDayLow%', np.nan):.2f}%"
                            if np.isfinite(feats.iloc[i].get("NextDayLow%", np.nan)) else ""),
            "BaseScore": base,
            "Bonus": bonus,
        })

    out = pd.DataFrame(rows)
    out = out.tail(90).reset_index(drop=True)
    out = out[out["Score"] >= 60].reset_index(drop=True)
    out.insert(0, "Row", out.index + 1)

    if out.empty:
        return

    # Exclude rule: consecutive signal days including bar_date
    bar_dates = feats.index[feats.index <= end_date]
    if len(bar_dates) >= 2:
        bar_date = bar_dates[-1]
        prev_date = bar_dates[-2]

        bar_str = pd.to_datetime(bar_date).strftime("%Y-%m-%d")
        prev_str = pd.to_datetime(prev_date).strftime("%Y-%m-%d")

        latest_signal_str = str(out["Signal Date"].max())

        bar_is_signal = (out["Signal Date"] == bar_str).any()
        prev_is_signal = (out["Signal Date"] == prev_str).any()

        if bar_is_signal and prev_is_signal and bar_str == latest_signal_str:
            return

    _tmp = out.copy()
    _tmp["NextDay%_num"] = _tmp["NextDay%"].apply(_pct_str_to_float)
    _tmp["NextDayOpen%_num"] = _tmp["NextDayOpen%"].apply(_pct_str_to_float)
    _tmp["NextDayLow%_num"] = _tmp["NextDayLow%"].apply(_pct_str_to_float)
    

    cond = ((_tmp["NextDayOpen%_num"] > 0) & (_tmp["NextDay%_num"] > 1))
    min_low_strong_up = float(_tmp.loc[cond, "NextDayLow%_num"].min()) if cond.any() else float("nan")
    min_low_strong_up_str = f"{min_low_strong_up:.2f}%" if np.isfinite(min_low_strong_up) else "NONE"

    sum_nextday_when_open_gt0 = float(
        np.where(
            (_tmp["NextDayOpen%_num"] > 0) & (_tmp["NextDayLow%_num"] < -1),
            -1.0,
            np.where(
                (_tmp["NextDayOpen%_num"] > 0),
                _tmp["NextDay%_num"],
                0.0
            )
        ).sum()
    ) if not _tmp.empty else 0.0

    sum_open_when_open_lt0 = float(
        _tmp.loc[_tmp["NextDayOpen%_num"] < 0, "NextDayOpen%_num"].sum(skipna=True)
    ) if not _tmp.empty else 0.0

    net = sum_nextday_when_open_gt0 - abs(sum_open_when_open_lt0)

    # Gate: PRINT NOTHING unless Net > 2
    if net <= 1.5:
        return

    # ===========================
    # ✅ NEW COLUMN (display-only): MeetsCriteria
    # Metric  Range
    # Score         72–82
    # IntradayScore ≥ 75
    # RegimeScore   ≥ 60   ✅ CHANGED (was 70)
    # DayRetPct     1–5%
    # ===========================
    # Compute intraday score PER signal date (display only; no effect on any logic/gates)
    rscore = float(regime_score_for_run) if use_regime_confirm else 75.0

    intraday_score_cache: dict[str, float] = {}
    intraday_scores: list[float] = []
    for sig_date_str in out["Signal Date"].astype(str).tolist():
        if use_intraday_60m_confirm:
            if sig_date_str in intraday_score_cache:
                iscore = intraday_score_cache[sig_date_str]
            else:
                intr = intraday_confirm_60m(ticker, pd.to_datetime(sig_date_str))
                iscore = float(intr["score"])
                intraday_score_cache[sig_date_str] = iscore
        else:
            iscore = 75.0
        intraday_scores.append(iscore)

    out["IntradayScore"] = intraday_scores
    out["RegimeScore"] = rscore

    out["MeetsCriteria"] = np.where(
        (out["Score"] >= 72.0) & (out["Score"] <= 82.0) &
        (out["IntradayScore"] >= 75.0) &
        (out["RegimeScore"] >= 60.0) &   # ✅ CHANGED (was 70.0)
        (out["SignalDay%"].apply(_pct_str_to_float) >= 1.0) &
        (out["SignalDay%"].apply(_pct_str_to_float) <= 5.0),
        "Yes",
        "No"
    )

    # Place MeetsCriteria right after Score (keep table readable)
    _col = out.pop("MeetsCriteria")
    _score_idx = list(out.columns).index("Score")
    out.insert(_score_idx + 1, "MeetsCriteria", _col)

    # ===========================
    # bucket summary
    current_signal_str = None
    if len(bar_dates) > 0:
        current_signal_str = pd.to_datetime(bar_dates[-1]).strftime("%Y-%m-%d")

    buckets = {
        "Total": _tmp,
        "Current": _tmp[_tmp["Signal Date"] == current_signal_str] if current_signal_str else _tmp.iloc[0:0],
        "60s": _tmp[(_tmp["Score"] >= 60) & (_tmp["Score"] < 70)],
        "70s": _tmp[(_tmp["Score"] >= 70) & (_tmp["Score"] < 80)],
        "80s": _tmp[(_tmp["Score"] >= 80) & (_tmp["Score"] < 90)],
        "90s": _tmp[_tmp["Score"] >= 90],
    }

    stats = {k: _net_stats(v) for k, v in buckets.items()}
    counts = {k: len(v) for k, v in buckets.items()}

    summary_df = pd.DataFrame(
        {
            f"Total({counts['Total']})":     [stats["Total"]["gain"],   stats["Total"]["loss"],   stats["Total"]["net"]],
            f"Current({counts['Current']})": [stats["Current"]["gain"], stats["Current"]["loss"], stats["Current"]["net"]],
            f"60s({counts['60s']})":         [stats["60s"]["gain"],     stats["60s"]["loss"],     stats["60s"]["net"]],
            f"70s({counts['70s']})":         [stats["70s"]["gain"],     stats["70s"]["loss"],     stats["70s"]["net"]],
            f"80s({counts['80s']})":         [stats["80s"]["gain"],     stats["80s"]["loss"],     stats["80s"]["net"]],
            f"90s({counts['90s']})":         [stats["90s"]["gain"],     stats["90s"]["loss"],     stats["90s"]["net"]],
        },
        index=[
            "Sum NextDay% (Open > 0, cap -1%)",
            "Sum Open% (Open < 0)",
            "Net (Gain - Loss)",
        ],
    )

    # ✅ FIX: applymap deprecated
    summary_df = summary_df.map(lambda x: f"{x:.2f}%")

    risk_open_val = float(_tmp["NextDayOpen%_num"].min(skipna=True)) if not _tmp.empty else float("nan")
    risk_open = f"{risk_open_val:.2f}%" if np.isfinite(risk_open_val) else "NONE"

    if np.isfinite(risk_open_val) and not _tmp.empty:
        _idx = _tmp["NextDayOpen%_num"].idxmin()
        risk_open_date = str(_tmp.loc[_idx, "Signal Date"])
        qqq_day_val = float(
            qqq_nextday_open_pct.reindex([pd.to_datetime(risk_open_date)], method="ffill").iloc[0]
        )
        qqq_day_str = f"{qqq_day_val:.2f}%" if np.isfinite(qqq_day_val) else "NONE"
    else:
        risk_open_date = "NONE"
        qqq_day_str = "NONE"

    print("\n" + "=" * 90)
    print(f"HISTORY OUTPUT FOR: {ticker}")
    print("=" * 90)

    print("\nLAST 90 DAYS (ONLY SCORE >= 60)")
    print("-" * 110)
    print(out.to_string(index=False))

    print("\n" + "=" * 61)
    print("SUMMARY".center(61))
    if len(bar_dates) > 0:
        bar_date = bar_dates[-1]
        i = int(feats.index.get_loc(bar_date))
        current_score = float(min(100.0, footprint_score(feats.iloc[i]) + confirmation_bonus(feats, i)))
        print(f"Current Score: {current_score:.1f}")

    print("\n" + "-" * 90)
    print(summary_df.to_string())
    print("-" * 90)

    print("\n" + "-" * 67 + "\n")
    print(f"  RISK (At Open)                             : {risk_open}")
    print(f"  DATE of RISK (At Open)                     : {risk_open_date}")
    print(f"  QQQ NextDayOpen% on that DATE              : {qqq_day_str}")
    print(f"  MIN NextDayLow% | Open>0 & NextDay%>1      : {min_low_strong_up_str}")
    print("\n" + "=" * 67)


# =============================
# MAIN: Scan table + History output ONLY for scan table tickers
# =============================
def main() -> None:
    tickers_raw = "A,AA,AAL,AAOI,AAON,AAPL,ABBV,ABVX,ACN,ADI,AEE,AEM,AEP,AER,AFL,AI,AIG,AKAM,ALK,ALLY,ALNY,AMAT,AMD,AMGN,AMP,AMT,ANF,AON,APG,APH,APLD,APO,APP,ARRY,ARW,ASTS,ATI,ATXS,AU,AVAV,AVGO,AVT,AVY,AXP,AXTI,AZO,B,BA,BABA,BAC,BAH,BBAI,BBY,BDX,BG,BIDU,BIIB,BILL,BKNG,BLK,BMNR,BMRN,BROS,BWA,C,CAH,CAMT,CARR,CAT,CB,CCL,CDNS,CF,CG,CHKP,CHRW,CI,CIEN,CL,CLS,CLSK,CMA,CMI,COF,COHR,COIN,COR,CORT,COST,CPT,CRDO,CRH,CRL,CRS,CSCO,CSIQ,CSX,CTSH,CUK,CVS,CVX,CWAN,DAL,DASH,DBX,DELL,DHR,DKNG,DKS,DLR,DOCU,DT,DUOL,ECL,ED,EL,ELAN,ELF,ELV,EMR,EOG,EPAM,EQIX,ERIC,ESTC,ETR,EXAS,EXE,EXPE,FANG,FCEL,FCX,FDS,FICO,FIG,FIS,FIVE,FIX,FLEX,FLNC,FLS,FLYW,FN,FNV,FOXA,FROG,FSLR,GE,GILD,GLW,GM,GOOG,GOOGL,GPC,GS,GTLB,HAL,HALO,HCA,HDB,HIG,HLT,HON,HOOD,HPE,HUM,HWM,IBKR,IBN,ICE,ILMN,INSM,INSP,INTC,INTU,IONQ,IONS,IOT,IQV,IR,IREN,IRM,ISRG,IT,IVZ,J,JBHT,JBL,JCI,JD,JKHY,JMIA,JPM,KEYS,KKR,KLAC,KLIC,KRMN,KTOS,KVYO,LC,LDOS,LEU,LITE,LLY,LMT,LOW,LPLA,LRCX,LSCC,LUV,LXEO,LYFT,MA,MAA,MBLY,MCD,MCHP,MDB,MDT,META,MIDD,MIR,MKSI,MMM,MMYT,MNDY,MORN,MP,MPWR,MRK,MS,MSCI,MSFT,MTB,MTN,MTSI,MTZ,MU,NBIS,NBIX,NDAQ,NEE,NEM,NET,NIO,NKE,NNE,NOC,NTES,NTNX,NTRA,NTRS,NUE,NVDA,NVO,NVS,NVT,NVTS,NXPI,NXT,OKLO,OKTA,OMC,ON,ONDS,ONON,ONTO,ORCL,OXY,PAAS,PATH,PAY,PBF,PCAR,PEGA,PEP,PFE,PG,PLD,PLNT,PLTR,PM,PNFP,PSTG,PSX,QBTS,RACE,RBLX,RDDT,RGLD,RIVN,RJF,RNG,ROK,ROKU,ROST,RTX,RUN,RY,SAP,SATS,SBUX,SCHW,SE,SEI,SFM,SGI,SHOP,SLB,SMCI,SN,SNAP,SNDK,SNX,SONY,SRE,STE,STLD,STM,STNE,STRL,STT,STX,SYK,SYNA,TAK,TEAM,TECH,TECK,TEL,TER,TJX,TMO,TROW,TSLA,TSM,TTD,TVTX,TXRH,U,UAL,UBER,UHS,UL,ULTA,UNH,UPS,USAR,UUUU,V,VRT,VSH,VTRS,VTYX,W,WBD,WDC,WFC,WING,WMT,WPM,WRBY,WSM,WST,WYNN,XOM,XP,ZEPP,ZETA,ZM,ZS,ZTS,SKY,ZIM,VSAT,AMZN"
    
    exclude_raw = input("Exclude tickers (optional, comma-separated): ").strip()
    date_raw = input("Enter Date YYYY-MM-DD (default ): ").strip()
    single_date_input = bool(date_raw)

    os.system("cls" if os.name == "nt" else "clear")

    tickers = parse_tickers(tickers_raw)
    exclude_tickers = parse_tickers(exclude_raw) if exclude_raw else []
    tickers = subtract_tickers(tickers, exclude_tickers)

    if not tickers:
        raise SystemExit("❌ No tickers left after exclusions.")


    end_date = roll_end_date(date_raw if date_raw else None)

    # Market regime once per run
    regime = {"ok": True, "score": 50.0, "reason": "regime_disabled"}
    if USE_REGIME_CONFIRM:
        regime = regime_confirm_qqq(end_date)

    MIN_SCORE = 60
    MIN_DOLLARVOL = 50_000_000

    spy_close = download_spy_close(end_date)

    # for QQQ line in history summary
    qqq_df = download_history("QQQ", end_date)
    qqq_nextday_open_pct = (qqq_df["Open"].shift(-1) / qqq_df["Close"] - 1) * 100

    hist_map = download_many(tickers, end_date)

    scan_rows: list[dict] = []
    feats_map: Dict[str, pd.DataFrame] = {}
    confirm_rows: list[dict] = []

    for t in tickers:
        df = hist_map.get(t)
        if df is None or len(df) < 80:
            continue

        feats = add_features(df)
        feats = add_relative_strength(feats, spy_close)
        feats_map[t] = feats

        bar_date = feats.index[feats.index <= end_date]
        if len(bar_date) == 0:
            continue
        bar_date = bar_date[-1]

        i = int(feats.index.get_loc(bar_date))
        base = footprint_score(feats.iloc[i])
        bonus = confirmation_bonus(feats, i)
        score = float(min(100.0, base + bonus))

        if score < MIN_SCORE:
            continue

        dollar_vol = safe(feats.iloc[i].get("DollarVol", np.nan))
        if np.isfinite(dollar_vol) and dollar_vol < MIN_DOLLARVOL:
            continue

        scan_rows.append({
            "Ticker": t,
            "BarDate": bar_date.strftime("%Y-%m-%d"),
            "SignalDay%": (f"{safe(feats.iloc[i].get('DayRetPct', np.nan)):.2f}%"
               if np.isfinite(safe(feats.iloc[i].get('DayRetPct', np.nan))) else ""),
            "Score": score,
            "BaseScore": base,
            "Bonus": bonus,
            "Close": safe(feats.iloc[i].get("Close", np.nan)),
            "DayRetPct": safe(feats.iloc[i].get("DayRetPct", np.nan)),
            "VolRel": safe(feats.iloc[i].get("VolRel", np.nan)),
            "DollarVol": dollar_vol,
            "RS": safe(feats.iloc[i].get("RS", np.nan)),
            "Reasons": "(see History output for full breakdown)",
        })

        # ✅ FIX: Intraday confirm must use SAME bar_date (signal day)
        intr = {"ok": True, "score": 50.0, "reason": "intraday_disabled"}
        if USE_INTRADAY_60M_CONFIRM:
            intr = intraday_confirm_60m(t, bar_date)

        dayret = safe(feats.iloc[i].get("DayRetPct", np.nan))
        rscore = float(regime["score"]) if USE_REGIME_CONFIRM else 75.0
        iscore = float(intr["score"]) if USE_INTRADAY_60M_CONFIRM else 75.0

        # ✅ NEW COLUMN (display-only): Yes/No criteria
        meets_criteria = (
            (score >= 72.0) and (score <= 82.0) and
            (iscore >= 75.0) and
            (rscore >= 60.0) and         # ✅ CHANGED (was 70.0)
            np.isfinite(dayret) and (dayret >= 1.0) and (dayret <= 5.0)
        )

        verdict = "WATCH"
        if (
            score >= BUY_MIN_SCORE and
            rscore >= BUY_MIN_REGIME and
            iscore >= BUY_MIN_INTRADAY and
            np.isfinite(dayret) and dayret <= MAX_DAYRET_FOR_BUY
        ):
            verdict = "BUY"

        if (USE_REGIME_CONFIRM and rscore < 60) or (USE_INTRADAY_60M_CONFIRM and iscore < 60):
            verdict = "SKIP"
        if np.isfinite(dayret) and dayret > 12:
            verdict = "SKIP"

        confirm_rows.append({
            "Ticker": t,
            "BarDate": bar_date.strftime("%Y-%m-%d"),
            "Score": round(score, 1),
            "DayRetPct": round(dayret, 2) if np.isfinite(dayret) else np.nan,
            "RegimeScore": round(rscore, 1),
            "IntradayScore": round(iscore, 1),
            "MeetsCriteria": "Yes" if meets_criteria else "No",  # ✅ NEW
            "Verdict": verdict,
            "Plan": "BUY signal close / sell next day"
        })

    if not scan_rows:
        print("No matches found.")
        return

    scan_df = (
        pd.DataFrame(scan_rows)
        .sort_values(["Score", "VolRel", "DollarVol"], ascending=False)
        .reset_index(drop=True)
    )
    scan_df["EarningsDate"] = scan_df["Ticker"].apply(get_next_earnings_date_str)

    # ✅ Exclude tickers with earnings in the next N days (controlled by EARNINGS_EXCLUDE_DAYS)
    if EARNINGS_EXCLUDE_DAYS >= 0:
        _e_ts = pd.to_datetime(scan_df["EarningsDate"], errors="coerce").dt.normalize()
        _ref = pd.to_datetime(end_date).normalize()
        _days = (_e_ts - _ref).dt.days

        scan_df = scan_df[
            _e_ts.isna() | ~((_days >= 0) & (_days <= EARNINGS_EXCLUDE_DAYS))
        ].reset_index(drop=True)


    # ✅ DO NOT CHANGE THIS 
    print(scan_df)
    # Print tickers from scan table (comma-separated, no spaces)
    # Print tickers from (Exclude input + scan table), comma-separated, no spaces
    combined = exclude_tickers + scan_df["Ticker"].astype(str).tolist()

    seen = set()
    combined_unique = []
    for t in combined:
        t = str(t).upper().replace(".", "-").strip()
        if t and t not in seen:
            combined_unique.append(t)
            seen.add(t)

    print("\n" + ",".join(combined_unique))



    # BUY LIST + CONFIRM TABLE
    if confirm_rows:
        cdf = pd.DataFrame(confirm_rows)

        buy_df = cdf[cdf["Verdict"] == "BUY"].copy()
        buy_df = buy_df.sort_values(["Score", "IntradayScore", "RegimeScore"], ascending=False).head(PRINT_BUY_LIST_TOP_N)

        print("\n" + "=" * 90)
        print(f"BUY LIST (top {PRINT_BUY_LIST_TOP_N}) — buy close / sell next day")
        print("=" * 90)
        if buy_df.empty:
            print("No BUY candidates today (based on thresholds).")
        else:
            print(buy_df[["Ticker", "BarDate", "Score", "DayRetPct", "IntradayScore", "RegimeScore", "Plan"]].to_string(index=False))

        if PRINT_CONFIRM_TABLE:
            print("\n" + "=" * 90)
            print("CONFIRMATION TABLE (full)")
            print("=" * 90)
            print(cdf.sort_values(["Verdict", "Score", "IntradayScore"], ascending=[True, False, False]).to_string(index=False))

        if USE_REGIME_CONFIRM:
            print("\nREGIME DETAILS:")
            print(f"  RegimeOK   : {regime['ok']}")
            print(f"  RegimeScore: {regime['score']:.1f}")
            print(f"  Reason     : {regime['reason']}")

        print("\nHistory Table Exclude Rules")
        print("1) Net (Gain - Loss) < 1.5%")
        print("2) If single date input AND SignalDay% > 12%")
        print("  Sum of NextDay% (Open > 0, cap -1%) means")
        print("  Open>0 and Low <-1  then -1")
        print("  Open>0 and Low >-1 then nextday ")
        print("  Sum of NextDayOpen% (Open < 0) means if Open<0 sum(Open)")
        print(f"  Earnings filter: exclude tickers with earnings in the next {EARNINGS_EXCLUDE_DAYS} day(s). (-1 = disabled)")
        print("4) Score Rules")
        print("    60+ ignore or watchlist, 70+ is good, 80+ is great, 90+ is rare and dangerous.")
        print("********7 Rules*********")
        print("1. Only Min 2% to Max 6% up on Signal Day (example 1%up, but gap down and solid came back still dont do)")
        print("2. Avoid Consecutive days Or often like repeated in 1 or 2 days")
        print("3. if open minus next Day Sell immediately")
        print("4. avoid If signal Day is ready for break out (except small candle rules).")
        print("5. avoid down day with Signal Days and less than 1% except same candle type of previous day (wait for next Day)")

        # History only for tickers in scan table (but print only if Net > 2, per print_history_output gate)
        for t in scan_df["Ticker"].tolist():
            feats = feats_map.get(t)
            if feats is None or feats.empty:
                continue

            print_history_output(
                t,
                feats,
                qqq_nextday_open_pct,
                end_date,
                single_date_input,
                float(regime["score"]),
                USE_REGIME_CONFIRM,
                USE_INTRADAY_60M_CONFIRM,
                min_day_thresh=2.0,
                big_day_thresh=6.0,
            )




if __name__ == "__main__":
    main()