import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# --------- PRECOMPUTED GRIDS (performance only; same rules) ----------
THRESHOLDS = np.arange(-0.25, -10.25, -0.25)
EDGES = np.arange(-10.0, 0.0, 0.25)  # includes -0.25, excludes 0.0
MAX_BAND_HIGH = -0.10  # same cutoff logic as your code
# --------------------------------------------------------------------


def _extract_ticker_frame(download_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Performance helper: safely extract a single ticker's OHLC frame from a
    batch yfinance download result (which may be MultiIndex or not).
    """
    if download_df is None or getattr(download_df, "empty", True):
        return pd.DataFrame()

    df = download_df

    # Common case for multi-ticker with group_by="ticker": columns are MultiIndex (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        # Try (ticker, field) layout first
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if ticker in set(lvl0):
            try:
                out = df[ticker].copy()
                return out
            except Exception:
                pass

        # Fallback to your original style: (field, ticker)
        if ticker in set(lvl1):
            try:
                out = df.xs(ticker, axis=1, level=1).copy()
                return out
            except Exception:
                pass

        return pd.DataFrame()

    # Single ticker: already flat columns like Open/Close/High/Low
    return df.copy()


def fetch_and_build_trades(ticker: str, lookback_days: int, _prefetched: pd.DataFrame | None = None) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    period_days = max(lookback_days + 40, 180)

    # PERFORMANCE: allow prefetched data (no behavior change; same math/output)
    if _prefetched is None:
        df = yf.download(
            ticker,
            period=f"{period_days}d",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
    else:
        df = _prefetched

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


def find_best_winning_ranges(trades: pd.DataFrame, min_trades: int = 8) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    # PERFORMANCE: convert once
    neg = trades["Negative_%"].to_numpy(dtype=float)
    ret = trades["Next_Day_Open_%"].to_numpy(dtype=float)

    results = []
    seen_signatures = set()

    def add_result(rule_text: str, mask: np.ndarray):
        n = int(mask.sum())
        if n < min_trades:
            return

        # PERFORMANCE: faster signature than tuple(np.where(...))
        sig = mask.tobytes()
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

    # Threshold rules (same set)
    for t in THRESHOLDS:
        add_result(f"Negative_% <= {t:.2f}", neg <= t)

    # Band rules (same set as your nested loops, but less Python overhead)
    # Original logic:
    # for low in edges:
    #   for high in edges:
    #     if high <= low or high > -0.10: continue
    #     add_result(...)
    for i, low in enumerate(EDGES):
        # high must be > low, and <= MAX_BAND_HIGH
        # since EDGES is increasing, start from i+1
        for high in EDGES[i + 1:]:
            if high > MAX_BAND_HIGH:
                break
            add_result(f"{low:.2f} < Negative_% <= {high:.2f}", (neg > low) & (neg <= high))

    if not results:
        return pd.DataFrame()

    res = pd.DataFrame(results)
    res = res.sort_values(["Score", "TotalNextOpen_%"], ascending=False).reset_index(drop=True)

    for c in ["WinRate_%", "AvgNextOpen_%", "TotalNextOpen_%", "Score"]:
        res[c] = res[c].astype(float).round(3)

    return res


# --------- ADDED (ONLY FOR "MATCH TOP-3 RULES TODAY") ---------
def negative_pct_today(ticker: str, _prefetched: pd.DataFrame | None = None) -> float | None:
    """
    Returns today's Negative_% (Close vs previous Close).
    If today is not a down day or data unavailable, returns None.
    PERFORMANCE: optionally uses prefetched data instead of re-downloading.
    """
    ticker = ticker.upper().strip()

    if _prefetched is None:
        df = yf.download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
    else:
        df = _prefetched

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
def get_next_earnings_date(ticker: str) -> str:
    """
    Best-effort next earnings date using yfinance.
    Returns YYYY-MM-DD or 'N/A'
    """
    try:
        tk = yf.Ticker(ticker)

        # ---- 1) Most reliable: earnings dates table (if available) ----
        # yfinance returns a DataFrame indexed by datetime for upcoming earnings.
        try:
            edf = tk.get_earnings_dates(limit=8)
            if edf is not None and not edf.empty:
                # Take the earliest upcoming earnings date in the table
                dt = pd.to_datetime(edf.index[0])
                return dt.date().isoformat()
        except Exception:
            pass

        # ---- 2) Fallback: calendar field (often empty) ----
        cal = tk.calendar
        if cal is not None and not cal.empty:
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"]

                # Could be Timestamp, list/Series, or range
                if isinstance(ed, (list, tuple, pd.Series)):
                    ed = ed[0]

                dt = pd.to_datetime(ed)
                return dt.date().isoformat()

    except Exception:
        pass

    return "N/A"

def earnings_is_soon(earnings_date_str: str, days: int) -> bool:
    """
    True if earnings date is within [0, days] days from today.
    Returns False for 'N/A' or unparseable dates (so those WILL show).
    """
    if not earnings_date_str or earnings_date_str == "N/A":
        return False

    try:
        ed = pd.to_datetime(earnings_date_str).date()
        today = date.today()
        delta = (ed - today).days
        return 0 <= delta <= days
    except Exception:
        return False


def main():
    MIN_TRADES_TABLE = 7
    MIN_EARNINGS_DAYS_AWAY = 6
    tickers_input ="DOCU,KMX,FAST,FROG,AMCR,BILL,AMKR,ENPH,LW,CSGP,CELH,INTC,SLB,RBRK,SEI,FIS,BWA,HUT,SU,LUV,CE,TEM,Z,TECK,LYB,PAAS,TSCO,BAC,EQT,AFRM,BROS,ETSY,BMRN,CG,BKR,ESTC,AA,FOUR,ALK,FTV,FCX,MP,SWKS,VSCO,ACMR,CARR,NKE,FLEX,FOXA,SOLS,ON,INSP,ADM,RBLX,AVT,SFM,MKC,KR,OMC,CAVA,DXCM,BBY,PSTG,OKLO,RKLB,URBN,LMND,IBKR,UBER,SYF,DAL,MCHP,AIG,CTSH,CVS,MRVL,CCI,HALO,OKE,ELF,HOOD,SKY,GM,CSCO,NDAQ,FLS,ROKU,NBIS,LSCC,IONS,EBAY,OKTA,SRE,BAH,SYNA,NEE,CRWV,PLNT,ZM,W,CF,WFC,CL,KTOS,TROW,TEAM,AKAM,KRMN,IRM,SGI,AAON,ANF,ETR,MNDY,Q,IR,TTMI,SBUX,EL,NOW,ASTS,GH,PNFP,MDT,EXAS,KKR,SCHW,AEE,CPT,ED,COP,EXE,CRDO,SATS,DDOG,SHOP,NVT,EOG,TWLO,NEM,DECK,TGT,BG,UAL,SN,AFL,UPS,WYNN,DUOL,ILMN,AEP,DELL,NXT,MRK,GLW,ABNB,ENTG,C,DLTR,XBI,PCAR,ZTS,A,BX,WMT,MAA,STT,APO,ATI,MSTR,PLTR,APH,PLD,QCOM,ANET,JCI,MTN,RDDT,AER,NBIX,HIG,ORCL,BE,DG,J,CAMT,XOM,VST,INSM,NTRS,GILD,NRG,TJX,DHI,IT,EMR,PSX,ARW,PG,PANW,MIDD,WDAY,ALB,STZ,COIN,MORN,FANG,ZS,SNOW,ICE,ALAB,PEP,AMT,SNX,DLR,RJF,MMM,LULU,NET,JKHY,MS,CVX,DASH,PM,NVDA,EPAM,IQV,CRL,ROST,TXRH,CRM,AVY,NUE,LDOS,HUM,VRT,RTX,CHRW,BIIB,PGR,STLD,FIVE,VLO,NTRA,FDS,DKS,AMD,ONTO,BDX,AMZN,MOD,TEL,DHR,UHS,FSLR,WSM,TXN,COF,HWM,ABBV,NXPI,CAH,COHR,JBHT,KEYS,LRCX,FNV,MTSI,EXPE,MTB,HON,MKSI,BA,WST,STE,AVAV,JBL,MTZ,CEG,LEU,WING,RGLD,ADBE,CIEN,UNH,AAPL,LOW,WDC,CDNS,ECL,CI,IBM,TER,CLS,HLT,ADI,GE,JPM,AMAT,GOOGL,GOOG,MCD,ALNY,CB,V,AVGO,ELV,AON,MDB,TSM,SYK,AXP,COR,CRS,AMGN,LPLA,HD,MU,CRWD,MSFT,STRL,CVNA,APP,TSLA,ROK,SPOT,STX,INTU,ISRG,HCA,FN,TMO,AMP,MA,LITE,MSCI,CMI,SNDK,LMT,META,ULTA,NOC,CAT,GEV,EQIX,GS,COST,BLK,LLY,MPWR,FIX,FICO,KLAC,AZO,BKNG"

    lb = input("Enter number of days to look back [default 90]: ").strip()
    lookback_days = int(lb) if lb else 90

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not tickers:
        print("No valid tickers entered.")
        return

    # PERFORMANCE: one download for all tickers instead of per-ticker (and instead of downloading twice per ticker)
    period_days = max(lookback_days + 40, 180)
    batch = yf.download(
        tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True
    )

    # Stores tuples like (ticker, rule_id, winrate, avg_next_open)
    stocks_matching_today = []

    for ticker in tickers:
        one = _extract_ticker_frame(batch, ticker)
        if one is None or one.empty:
            continue

        trades = fetch_and_build_trades(ticker, lookback_days, _prefetched=one)
        if trades.empty:
            continue

        best = find_best_winning_ranges(trades, min_trades=8)
        best = best[best["WinRate_%"] >= 75].reset_index(drop=True)
        best = best[best["AvgNextOpen_%"] >= 0.10].reset_index(drop=True)
        if best.empty:
            continue

        # PERFORMANCE: compute "today negative %" from same prefetched data
        today_neg = negative_pct_today(ticker, _prefetched=one)
        if today_neg is None:
            continue

        matched_info = None

        for idx, (_, row) in enumerate(best.head(3).iterrows(), start=1):
            if rule_matches(str(row["Rule"]), float(today_neg)):
                earnings_date = get_next_earnings_date(ticker)

                matched_info = (
                    ticker,
                    f"#{idx}",
                    int(row["Trades"]),
                    float(row["WinRate_%"]),
                    float(row["AvgNextOpen_%"]),
                    earnings_date,
                )
                break

        # ✅ if not matching today -> print nothing for this stock
        if matched_info is None:
            continue

        # ✅ matching today -> include in list and print details
        # matched_info = (ticker, rule_id, trades, winrate, avg_next_open)
        # matched_info = (ticker, rule_id, trades, winrate, avg_next_open, earnings_date)
        if (
            matched_info[2] >= MIN_TRADES_TABLE
            and not earnings_is_soon(matched_info[5], MIN_EARNINGS_DAYS_AWAY)
        ):
            stocks_matching_today.append(matched_info)



        print("\n" + "=" * 25)
        print(f"STOCK: {ticker}")
        print("=" * 25)

        print(f"\nTOTAL Sum of Next Day Open %: {trades['Next_Day_Open_%'].sum():.2f}%")
        print(f"TOTAL Sum of Negative %: {trades['Negative_%'].sum():.2f}%")

        print("\nTOP 3 RULES FOUND (WinRate ≥ 75%):")
        for rank, row in best.head(3).iterrows():
            print(f"\n#{rank + 1}")
            print(f"  Rule: {row['Rule']}")
            print(f"  Trades: {int(row['Trades'])}")
            print(f"  WinRate: {row['WinRate_%']:.2f}%")
            print(f"  Avg Next Open %: {row['AvgNextOpen_%']:.3f}%")
            print(f"  Total Next Open %: {row['TotalNextOpen_%']:.3f}%")

    if stocks_matching_today:
        print("\nSTOCKS MATCHING TOP-3 RULES TODAY:")
        print("Stock   Rule Id   Trades   WinRate   Avg Next Open %   Earnings Date")
        for stock, rule_id, trades, winrate, avg_next_open, earnings_date in sorted(stocks_matching_today, key=lambda x: x[0]):
            print(
                f"{stock:<7} {rule_id:<8} {trades:>6} "
                f"{winrate:>7.2f}% {avg_next_open:>14.3f}% {earnings_date:>14}"
            )
    else:
        print("\nNO STOCK MATCHED TOP-3 RULES TODAY.")


if __name__ == "__main__":
    main()
