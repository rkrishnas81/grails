from __future__ import annotations

import os
import time
from typing import List, Dict

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
# BUY LIST SETTINGS (kept: used for Verdict logic if you re-add it later)
# =============================
BUY_MIN_SCORE = 70.0
MAX_DAYRET_FOR_BUY = 6.0
PRINT_BUY_LIST_TOP_N = 5  # currently unused after cleanup, safe to keep or remove
MIN_SIGNAL_DAY_PCT = 2.5


# =============================
# EARNINGS FILTER SETTINGS
# =============================
EARNINGS_EXCLUDE_DAYS = 5   # (5 = next 5 days, 0 = today only, -1 = disable)

# =============================
# "First run of the day" skip list settings (NO DATE INPUT ONLY)
# =============================
FIRST_RUN_SKIP_MIN_DAYRET = 1.0
FIRST_RUN_SKIP_FILE_PREFIX = "skip_under1p_"
FIRST_RUN_SKIP_DIR = "daily_skips"

# =============================
# INTRADAY SETTINGS
# =============================
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "10d"   # Yahoo limits; 10d is usually safe
INTRADAY_BATCH_SIZE = 60  # keep your batching style
USE_LAST_COMPLETED_INTRADAY_BAR = True  # recommended for stability


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


def _now_pacific() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/Los_Angeles")


# =============================
# last_completed_intraday_bar_ts returns tz-aware Pacific
# =============================
def last_completed_intraday_bar_ts(interval: str = "5m") -> pd.Timestamp:
    """
    Returns tz-aware Pacific timestamp for the last COMPLETED intraday bar.
    """
    now = _now_pacific()
    if interval.endswith("m"):
        n = int(interval[:-1])
        floored = now.floor(f"{n}min")
        return floored - pd.Timedelta(minutes=n)
    return now


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

        dt = pd.to_datetime(edf.index[0])
        out = dt.strftime("%Y-%m-%d")
        _EARNINGS_CACHE[t] = out
        return out
    except Exception:
        _EARNINGS_CACHE[t] = ""
        return ""


# =============================
# DOWNLOAD (DAILY)
# =============================
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
# DOWNLOAD (INTRADAY 5m)
# =============================
def download_intraday_many(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Downloads intraday (5m) bars for the last N days (INTRADAY_PERIOD).
    Used ONLY when date input is blank AND it's a trading day (fixed in main()).
    """
    out: Dict[str, pd.DataFrame] = {}
    for batch in chunks(tickers, INTRADAY_BATCH_SIZE):
        hist = yf.download(
            " ".join(batch),
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
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
            out[t] = df
        time.sleep(0.15)
    return out


# =============================
# FIX: intraday_today_snapshot now uses *session_date* (end_date), not "now"
#      + remains timezone-safe
# =============================
def intraday_today_snapshot(
    intraday_df: pd.DataFrame,
    asof_ts_pacific: pd.Timestamp,
    session_date: pd.Timestamp,   # <-- NEW
) -> dict | None:
    """
    Build 'session day so far' candle using intraday bars up to asof_ts_pacific.
    Safely handles tz-aware vs tz-naive intraday indexes.
    Uses session_date (end_date) for day filtering, so weekends/holidays won't break.
    """
    if intraday_df is None or intraday_df.empty:
        return None

    idx = intraday_df.index
    session_day = pd.to_datetime(session_date).date()

    # If intraday index is tz-aware, convert Pacific as-of into that tz
    if getattr(idx, "tz", None) is not None:
        # Ensure input is Pacific tz-aware
        if asof_ts_pacific.tzinfo is None:
            asof_ts_pacific = asof_ts_pacific.tz_localize("America/Los_Angeles")

        asof_ts = asof_ts_pacific.tz_convert(idx.tz)
        d = intraday_df.loc[idx <= asof_ts]

        d_today = d[d.index.date == session_day]
    else:
        # tz-naive intraday index
        # Compare using naive local time equivalent of Pacific
        if asof_ts_pacific.tzinfo is not None:
            asof_ts = asof_ts_pacific.tz_convert("America/Los_Angeles").tz_localize(None)
        else:
            asof_ts = asof_ts_pacific

        d = intraday_df.loc[idx <= asof_ts]
        d_today = d[d.index.date == session_day]

    if d_today.empty:
        return None

    o = safe(d_today.iloc[0]["Open"])
    h = safe(d_today["High"].max())
    l = safe(d_today["Low"].min())
    c = safe(d_today.iloc[-1]["Close"])
    v = safe(d_today["Volume"].sum())

    if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c) and np.isfinite(v)):
        return None

    return {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v, "AsOfBarTime": d_today.index[-1]}


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

    # =============================
    # TC2000 EXACT FORMULA MATCH
    # Abs(100*((SUM(H9-L9 ... H18-L18) - C) / C))
    # =============================
    hl = d["High"] - d["Low"]
    sum_hl_9_18 = hl.shift(9).rolling(10).sum()
    d["TC2000_AbsRange"] = (100.0 * ((sum_hl_9_18 - d["Close"]) / d["Close"])).abs()

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


# =============================
# PRINT HELPER (kept: used)
# =============================
def print_scan_with_earnings_highlight(scan_df: pd.DataFrame, end_date: pd.Timestamp):
    if scan_df.empty:
        print(scan_df)
        return

    today = pd.to_datetime(end_date).normalize()
    tomorrow = (today + BDay(1)).normalize()
    red_set = {today.strftime("%Y-%m-%d"), tomorrow.strftime("%Y-%m-%d")}

    table = scan_df.to_string(index=False)
    lines = table.splitlines()

    print(lines[0])
    if len(lines) > 1:
        print(lines[1])

    header = lines[0]
    col_name = "EarningsDate"
    start = header.find(col_name)

    for line in lines[2:]:
        is_red = False
        if start != -1 and len(line) >= start + len(col_name):
            earnings_text = line[start:start + len(col_name)].strip()
            is_red = (earnings_text in red_set)

        if is_red:
            print(f"\033[91m{line}\033[0m")
        else:
            print(line)


# =============================
# MAIN
# =============================
def main() -> None:
    tickers_raw = "A,AA,AAL,AAOI,AAON,AAPL,ABBV,ADI,AEE,AEP,AER,AFL,AI,AIG,AKAM,ALK,ALLY,ALNY,AMAT,AMD,AMGN,AMP,AMT,ANF,AON,APG,APH,APLD,APO,APP,ARRY,ARW,ASTS,ATI,ATXS,AVAV,AVGO,AVT,AVY,AXP,AXTI,AZO,BA,BAC,BAH,BBY,BDX,BG,BIIB,BILL,BKNG,BLK,BMRN,BROS,BWA,C,CAH,CAMT,CARR,CAT,CB,CCL,CDNS,CF,CG,CHRW,CI,CIEN,CL,CLS,CLSK,CMA,CMI,COF,COHR,COIN,COR,CORT,COST,CPT,CRDO,CRL,CRS,CSCO,CSX,CTSH,CVX,CWAN,DAL,DASH,DBX,DELL,DHR,DKNG,DKS,DLR,DOCU,DT,DUOL,ECL,ED,EL,ELAN,ELF,ELV,EMR,EOG,EPAM,EQIX,ESTC,ETR,EXAS,EXE,EXPE,FANG,FCEL,FCX,FDS,FICO,FIG,FIS,FIVE,FIX,FLEX,FLNC,FLS,FLYW,FN,FNV,FOXA,FROG,FSLR,GE,GILD,GLW,GM,GOOG,GOOGL,GS,GTLB,HAL,HALO,HCA,HIG,HLT,HON,HOOD,HUM,HWM,IBKR,ICE,ILMN,INSM,INSP,INTC,INTU,IONQ,IONS,IOT,IQV,IR,IRM,ISRG,IT,IVZ,J,JBHT,JBL,JCI,JKHY,JPM,KEYS,KKR,KLAC,KRMN,KTOS,KVYO,LC,LDOS,LEU,LITE,LLY,LMT,LOW,LPLA,LRCX,LSCC,LUV,LXEO,LYFT,MA,MAA,MBLY,MCD,MCHP,MDB,MDT,META,MIDD,MIR,MKSI,MMM,MNDY,MORN,MP,MPWR,MRK,MS,MSCI,MSFT,MTB,MTN,MTSI,MTZ,MU,NBIS,NBIX,NDAQ,NEE,NEM,NET,NKE,NOC,NTNX,NTRA,NTRS,NUE,NVDA,NVT,NVTS,NXPI,NXT,OKLO,OKTA,OMC,ON,ONDS,ONTO,ORCL,OXY,PAAS,PATH,PAY,PBF,PCAR,PEGA,PEP,PFE,PG,PLD,PLNT,PLTR,PM,PNFP,PSTG,PSX,QBTS,RBLX,RDDT,RGLD,RIVN,RJF,RNG,ROK,ROKU,ROST,RTX,RUN,SATS,SBUX,SCHW,SEI,SFM,SGI,SLB,SMCI,SN,SNAP,SNDK,SNX,SRE,STE,STLD,STRL,STT,STX,SYK,SYNA,TEAM,TEL,TER,TJX,TMO,TROW,TSLA,TTD,TVTX,TXRH,U,UAL,UBER,UHS,ULTA,UNH,UPS,USAR,UUUU,V,VRT,VSH,VTRS,VTYX,W,WBD,WDC,WFC,WING,WMT,WRBY,WSM,WST,WYNN,XOM,ZM,ZS,ZTS,SKY,VSAT,AMZN,BKKT,HL,SEDG,LMND,CIFR,MGNI,MNTS,PL,RKLB,LUNR,FLY,RDW,MOS,FMC,JOBY,VFC,CRI,GAP,LULU,AEO,VSCO,URBN,OWL,CWH,CVNA,KMX,LCID,MOD,QS,AEVA,NPB,CELH,STZ,RIOT,WULF,MARA,HUT,FIGR,CE,HUN,METC,COMM,VIAV,UMAC,ANET,HPQ,OSS,QUBT,RGTI,RCAT,SOFI,UPST,M,KSS,GH,TGT,DLTR,SERV,SMR,PRGO,PCRX,AMPX,EOSE,TTMI,BE,OUST,LPTH,FLR,TIC,DECK,BIRK,SBET,SGHC,OSCR,DOCS,TEM,TXG,HIMS,PPTA,TWLO,GENI,SPOT,STUB,Z,DJT,TRIP,FUN,NCLH,REAL,CNK,PSKY,VSNT,ACHC,BRKR,RXST,TNDM,BBNX,ATEC,SOC,APA,RRC,OVV,MUR,CRK,AR,SM,LBRT,HLF,LW,BRBR,PCT,CSGP,DBRG,TWO,FRMI,COLD,HPP,PENN,CAVA,GEO,ENTG,ACMR,AEHR,AMKR,Q,ALAB,MRVL,AG,GTM,FRSH,COMP,PD,DDOG,SNOW,BTDR,MSTR,NOW,ASAN,SOUN,OS,RELY,CRWV,CORZ,RBRK,NTSK,FOUR,SOLS,KLAR,PGY,AFRM,AFRM,ENPH,AVTR,ALB,CC,OLN,ETSY,CART,CHWY,BBWI,GME,RHI,UPWK,CLF,CMCSA,RXO,VST,GEV,NRG,CEG,HE,CSIQ,MRNA,XBI,CRWD,RZLV,LAES,NUAI,LUMN,ASPI,POET,IMRX,ALT,SHLS,HTZ,TIGR,ACHR,VG,ABR,USAS,PAYO,SANA,DRIP,ULCC,NIO,JBLU,INDI,SG,BBAI,ASM,TROX,VZLA,NWL,MSOS,MNKD,BCRX,BW,BTG,COUR,SLDP,BULL,CRMD,AUR,RIG,PTEN,TLRY,NUVB,FSLY,DUST,TDOC,TSHA,MQ,CRGY,TSDD,ENVX,GDXD,NVAX,TMQ,TGB,UNIT,HIMZ,RXRX,LAR,UAA,ABCL,UA,NEXT,SLI,TE,OPEN,MSTX,MUD,TZA,BORR,BMNG,BMNU,UAMY,TMC,LAC,BFLY,PGEN,NVD,GRAB,AMDD,NB,XRPT,SOLT,CRCA,ABAT,TSM,,EXK,RKT,IREN,SHOP,TECK,UEC,PAGS,BMNR,NXE,NU,GFS,ZIM,ZETA,STNE,EBAY,CVE,QXO,XP,AES,VISN,FLG,BN,WT,BZ,LYB,AS,ABNB,CNQ,TXN,DD,NOV,SU,ALM,DXCM,CALY,BKR,SW,DHI,SYF,PINS,CRM,CNH,IBM,PYPL,CRBG,CNC,S,CMG,VLO,WY,FTV,QCOM,BX,TSCO,CCI,PGR,PRMB,MGY,SWKS,TOST,FAST,NEOG,CVS,JBS,PK,ACI,PANW,HOG,PR,ONON,ADBE,WDAY,ADM,DOC,DVN,GPK,HRL,CPRT,HD,OKE,AMCR,MKC,CPNG,HBAN,DG,AMH,COP,DOW,INVH,SIRI,XRAY,BAX,KR,EQT,CTRA,FLO,IP,CPB,CAG"

    exclude_raw = input("Exclude tickers (optional, comma-separated): ").strip()
    date_raw = input("Enter Date YYYY-MM-DD (default ): ").strip()

    max_signal_raw = input("Max Signal Day % (default 8): ").strip()
    if max_signal_raw == "":
        max_signal_day_pct = 8.0
    else:
        try:
            max_signal_day_pct = float(max_signal_raw)
        except ValueError:
            raise SystemExit("❌ Invalid Max Signal Day % input.")

    single_date_input = bool(date_raw)

    os.system("cls" if os.name == "nt" else "clear")

    tickers = parse_tickers(tickers_raw)
    exclude_tickers = parse_tickers(exclude_raw) if exclude_raw else []
    tickers = subtract_tickers(tickers, exclude_tickers)

    if not tickers:
        raise SystemExit("❌ No tickers left after exclusions.")

    end_date = roll_end_date(date_raw if date_raw else None)

    # =============================
    # FIX: only use intraday when date is blank AND end_date == "today" in Pacific.
    # If it's weekend/holiday, roll_end_date() will be earlier than today -> fall back to daily.
    # =============================
    now_pac_day = _now_pacific().normalize().tz_localize(None)  # naive date in Pacific
    end_day = pd.to_datetime(end_date).normalize()
    use_intraday_today = (not single_date_input) and (end_day == now_pac_day)

    # =============================
    # load today's skip file ONLY when date input is blank (keep existing behavior)
    # NOTE: skip file uses end_date (rolled) as its "today" label, which is fine.
    # =============================
    skip_file_path = None
    is_first_run_today = False
    skip_today_set: set[str] = set()
    if not single_date_input:
        try:
            os.makedirs(FIRST_RUN_SKIP_DIR, exist_ok=True)
        except Exception:
            pass

        today_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        skip_file_path = os.path.join(FIRST_RUN_SKIP_DIR, f"{FIRST_RUN_SKIP_FILE_PREFIX}{today_str}.txt")
        is_first_run_today = not os.path.exists(skip_file_path)

        if os.path.exists(skip_file_path):
            try:
                with open(skip_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        tt = line.strip().upper().replace(".", "-")
                        if tt:
                            skip_today_set.add(tt)
            except Exception:
                skip_today_set = set()

        if skip_today_set:
            tickers = [tt for tt in tickers if tt not in skip_today_set]

        if not tickers:
            raise SystemExit("❌ No tickers left after exclusions + today skip file.")
    # =============================

    MIN_SCORE = 60
    MIN_DOLLARVOL = 50_000_000

    spy_close = download_spy_close(end_date)
    hist_map = download_many(tickers, end_date)

    intraday_map: Dict[str, pd.DataFrame] = {}
    asof_ts = None
    if use_intraday_today:
        intraday_map = download_intraday_many(tickers)
        asof_ts = last_completed_intraday_bar_ts(INTRADAY_INTERVAL) if USE_LAST_COMPLETED_INTRADAY_BAR else _now_pacific()

    scan_rows: list[dict] = []
    newly_skipped_today: set[str] = set()

    for t in tickers:
        df = hist_map.get(t)
        if df is None or len(df) < 80:
            continue

        feats = add_features(df)
        feats = add_relative_strength(feats, spy_close)

        bar_date = feats.index[feats.index <= end_date]
        if len(bar_date) == 0:
            continue
        bar_date = bar_date[-1]
        i = int(feats.index.get_loc(bar_date))

        row_for_score = feats.iloc[i].copy()

        if use_intraday_today:
            intr = intraday_map.get(t)
            try:
                snap = intraday_today_snapshot(intr, asof_ts, end_date) if (intr is not None and asof_ts is not None) else None
            except Exception:
                continue

            if snap is None:
                continue

            prev_close = safe(row_for_score.get("PrevClose", np.nan))
            if not np.isfinite(prev_close) or prev_close == 0:
                continue

            # overwrite OHLCV from intraday
            row_for_score["Open"] = snap["Open"]
            row_for_score["High"] = snap["High"]
            row_for_score["Low"] = snap["Low"]
            row_for_score["Close"] = snap["Close"]
            row_for_score["Volume"] = snap["Volume"]

            c = snap["Close"]
            h = snap["High"]
            l = snap["Low"]
            v = snap["Volume"]

            row_for_score["DayRetPct"] = (c / prev_close - 1) * 100
            row_for_score["RangePct"] = ((h - l) / prev_close) * 100
            hl = (h - l)
            row_for_score["ClosePos"] = 0.5 if hl == 0 else float(np.clip((c - l) / hl, 0, 1))

            row_for_score["DollarVol"] = c * v

            vol20 = safe(row_for_score.get("Vol20", np.nan))
            row_for_score["VolRel"] = np.nan if (not np.isfinite(vol20) or vol20 == 0) else (v / vol20)

        # first run skip rule (NO DATE INPUT ONLY)
        if not single_date_input and is_first_run_today:
            _dayret_tmp = safe(row_for_score.get("DayRetPct", np.nan))
            if np.isfinite(_dayret_tmp) and _dayret_tmp < FIRST_RUN_SKIP_MIN_DAYRET:
                newly_skipped_today.add(t)
                continue

        base = footprint_score(row_for_score)
        bonus = confirmation_bonus(feats, i)
        score = float(min(100.0, base + bonus))

        dayret = safe(row_for_score.get("DayRetPct", np.nan))

        # NextDay% (backtest only; keep logic)
        nextday_pct = np.nan
        if i + 1 < len(feats):
            c0 = safe(feats.iloc[i].get("Close", np.nan))
            c1 = safe(feats.iloc[i + 1].get("Close", np.nan))
            if np.isfinite(c0) and np.isfinite(c1) and c0 != 0:
                nextday_pct = (c1 / c0 - 1) * 100

        if score < MIN_SCORE:
            continue

        dollar_vol = safe(row_for_score.get("DollarVol", np.nan))
        if np.isfinite(dollar_vol) and dollar_vol < MIN_DOLLARVOL:
            continue
        if np.isfinite(dayret) and dayret > max_signal_day_pct:
            continue
        if not (np.isfinite(dayret) and dayret >= MIN_SIGNAL_DAY_PCT):
            continue

        scan_rows.append({
            "Ticker": t,
            "BarDate": bar_date.strftime("%Y-%m-%d"),
            "SignalDay%": (f"{dayret:.2f}%" if np.isfinite(dayret) else ""),
            "NextDay%": (f"{nextday_pct:.2f}%" if np.isfinite(nextday_pct) else ""),
            "Score": score,
            "BaseScore": base,
            "Bonus": bonus,
            "Close": safe(row_for_score.get("Close", np.nan)),
            "TC2000_AbsRange": round(safe(row_for_score.get("TC2000_AbsRange", np.nan)), 2),
        })

    # write today's skip file at end of first run (NO DATE INPUT ONLY)
    if not single_date_input and is_first_run_today and skip_file_path:
        if newly_skipped_today:
            try:
                tmp_path = skip_file_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for t in sorted(newly_skipped_today):
                        f.write(t + "\n")
                try:
                    os.replace(tmp_path, skip_file_path)
                except Exception:
                    with open(skip_file_path, "w", encoding="utf-8") as f:
                        for t in sorted(newly_skipped_today):
                            f.write(t + "\n")
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            try:
                with open(skip_file_path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass

    if not scan_rows:
        print("No matches found.")
        return

    scan_df = (
        pd.DataFrame(scan_rows)
        .sort_values(["Score"], ascending=False)
        .reset_index(drop=True)
    )
    scan_df["EarningsDate"] = scan_df["Ticker"].apply(get_next_earnings_date_str)

    # FILTER: TC2000_AbsRange < 80 (ONLY when date input is blank AND intraday mode is active)
    # If it's weekend/holiday, we are in daily mode and should not apply this intraday-only filter.
    if (not single_date_input) and use_intraday_today:
        scan_df = scan_df[
            scan_df["TC2000_AbsRange"].notna() &
            (scan_df["TC2000_AbsRange"] < 80)
        ].reset_index(drop=True)

    # Exclude tickers with earnings in the next N days (BUT KEEP TODAY)
    if EARNINGS_EXCLUDE_DAYS >= 0:
        _e_ts = pd.to_datetime(scan_df["EarningsDate"], errors="coerce").dt.normalize()
        _ref = pd.to_datetime(end_date).normalize()
        _days = (_e_ts - _ref).dt.days

        scan_df = scan_df[
            _e_ts.isna() | ~((_days >= 1) & (_days <= EARNINGS_EXCLUDE_DAYS))
        ].reset_index(drop=True)

    print_scan_with_earnings_highlight(scan_df, end_date)

    combined = exclude_tickers + scan_df["Ticker"].astype(str).tolist()
    seen = set()
    combined_unique = []
    for t in combined:
        t = str(t).upper().replace(".", "-").strip()
        if t and t not in seen:
            combined_unique.append(t)
            seen.add(t)
    print("\n" + ",".join(combined_unique))


if __name__ == "__main__":
    main()