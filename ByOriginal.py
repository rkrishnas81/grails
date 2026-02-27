from __future__ import annotations

import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


# =============================
# PRINT SETTINGS
# =============================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.colheader_justify", "center")


# =============================
# CORE THRESHOLDS (ONLY FILTERS LEFT)
# =============================
MIN_SCORE = 60
MIN_DOLLARVOL = 50_000_000
MIN_SIGNAL_DAY_PCT = 2.5
MAX_SIGNAL_DAY_PCT_DEFAULT = 8.0

NEXTDAY_MIN_UP_PCT = 2.0
NEXTDAY_MIN_DOWN_PCT = -2.0

# =============================
# VOLUME BOOST (matches your example script behavior)
# Boosts ONLY the BarDate row being scored (not historical days)
# =============================
VOLUME_BOOST_PCT = 0.0  # set 0 to disable


# =============================
# ANSI COLORS (terminal)
# =============================
RED = "\033[91m"   # bright red
RESET = "\033[0m"


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


def roll_to_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.to_datetime(d).normalize()
    if d.weekday() >= 5:
        d -= BDay(1)
    return d


def roll_end_date(s: str | None) -> pd.Timestamp:
    d = pd.Timestamp.today().normalize() if not s else pd.to_datetime(s).normalize()
    return roll_to_trading_day(d)


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


# =============================
# EARNINGS (for red highlighting)
# =============================
def get_earnings_dates(ticker: str, limit: int = 12) -> pd.DatetimeIndex:
    """
    Earnings dates as normalized (date-only) timestamps.
    """
    try:
        tk = yf.Ticker(ticker)
        edf = tk.get_earnings_dates(limit=limit)  # DataFrame indexed by earnings datetime
        if edf is None or edf.empty:
            return pd.DatetimeIndex([])
        idx = pd.to_datetime(edf.index).tz_localize(None).normalize()
        return pd.DatetimeIndex(sorted(idx.unique()))
    except Exception:
        return pd.DatetimeIndex([])


def is_red_earn_match(bar_date: pd.Timestamp, earnings_dates: pd.DatetimeIndex) -> bool:
    """
    True if BarDate is earnings date OR next trading day is earnings date.
    """
    if earnings_dates is None or len(earnings_dates) == 0:
        return False
    bd = pd.to_datetime(bar_date).normalize()
    next_bd = (bd + BDay(1)).normalize()
    return (bd in earnings_dates) or (next_bd in earnings_dates)


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

    hl2 = d["High"] - d["Low"]
    sum_hl_9_18 = hl2.shift(9).rolling(10).sum()
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
# VOLUME BOOST (BarDate row only)
# =============================
def apply_volume_boost(row: pd.Series, boost_pct: float) -> pd.Series:
    if boost_pct is None or boost_pct == 0:
        return row

    v_raw = safe(row.get("Volume", np.nan))
    if not np.isfinite(v_raw):
        return row

    v = v_raw * (1.0 + boost_pct / 100.0)
    row["Volume"] = v

    close = safe(row.get("Close", np.nan))
    if np.isfinite(close):
        row["DollarVol"] = close * v

    vol20 = safe(row.get("Vol20", np.nan))
    row["VolRel"] = np.nan if (not np.isfinite(vol20) or vol20 == 0) else (v / vol20)

    return row


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
# MAIN
# =============================
def main() -> None:
    tickers_raw = "A,AA,AAL,AAOI,AAON,AAPL,ABBV,ADI,AEE,AEP,AER,AFL,AI,AIG,AKAM,ALK,ALLY,ALNY,AMAT,AMD,AMGN,AMP,AMT,ANF,AON,APG,APH,APLD,APO,APP,ARRY,ARW,ASTS,ATI,AVAV,AVGO,AVT,AVY,AXP,AXTI,AZO,BA,BAC,BAH,BBY,BDX,BG,BIIB,BILL,BKNG,BLK,BMRN,BROS,BWA,C,CAH,CAMT,CARR,CAT,CB,CCL,CDNS,CF,CG,CHRW,CI,CIEN,CL,CLS,CLSK,CMI,COF,COHR,COIN,COR,CORT,COST,CPT,CRDO,CRL,CRS,CSCO,CSX,CTSH,CVX,CWAN,DAL,DASH,DBX,DELL,DHR,DKNG,DKS,DLR,DOCU,DT,DUOL,ECL,ED,EL,ELAN,ELF,ELV,EMR,EOG,EPAM,EQIX,ESTC,ETR,EXAS,EXE,EXPE,FANG,FCEL,FCX,FDS,FICO,FIG,FIS,FIVE,FIX,FLEX,FLNC,FLS,FLYW,FN,FNV,FOXA,FROG,FSLR,GE,GILD,GLW,GM,GOOG,GOOGL,GS,GTLB,HAL,HALO,HCA,HIG,HLT,HON,HOOD,HUM,HWM,IBKR,ICE,ILMN,INSM,INSP,INTC,INTU,IONQ,IONS,IOT,IQV,IR,IRM,ISRG,IT,IVZ,J,JBHT,JBL,JCI,JKHY,JPM,KEYS,KKR,KLAC,KRMN,KTOS,KVYO,LC,LDOS,LEU,LITE,LLY,LMT,LOW,LPLA,LRCX,LSCC,LUV,LXEO,LYFT,MA,MAA,MBLY,MCD,MCHP,MDB,MDT,META,MIDD,MIR,MKSI,MMM,MNDY,MORN,MP,MPWR,MRK,MS,MSCI,MSFT,MTB,MTN,MTSI,MTZ,MU,NBIS,NBIX,NDAQ,NEE,NEM,NET,NKE,NOC,NTNX,NTRA,NTRS,NUE,NVDA,NVT,NVTS,NXPI,NXT,OKLO,OKTA,OMC,ON,ONDS,ONTO,ORCL,OXY,PAAS,PATH,PAY,PBF,PCAR,PEGA,PEP,PFE,PG,PLD,PLNT,PLTR,PM,PNFP,PSTG,PSX,QBTS,RBLX,RDDT,RGLD,RIVN,RJF,RNG,ROK,ROKU,ROST,RTX,RUN,SATS,SBUX,SCHW,SEI,SFM,SGI,SLB,SMCI,SN,SNAP,SNDK,SNX,SRE,STE,STLD,STRL,STT,STX,SYK,SYNA,TEAM,TEL,TER,TJX,TMO,TROW,TSLA,TTD,TVTX,TXRH,U,UAL,UBER,UHS,ULTA,UNH,UPS,USAR,UUUU,V,VRT,VSH,VTRS,VTYX,W,WBD,WDC,WFC,WING,WMT,WRBY,WSM,WST,WYNN,XOM,ZM,ZS,ZTS,SKY,VSAT,AMZN,BKKT,HL,SEDG,LMND,CIFR,MGNI,MNTS,PL,RKLB,LUNR,FLY,RDW,MOS,FMC,JOBY,VFC,CRI,GAP,LULU,AEO,VSCO,URBN,OWL,CWH,CVNA,KMX,LCID,MOD,QS,AEVA,NPB,CELH,STZ,RIOT,WULF,MARA,HUT,FIGR,CE,HUN,METC,VIAV,UMAC,ANET,HPQ,OSS,QUBT,RGTI,RCAT,SOFI,UPST,M,KSS,GH,TGT,DLTR,SERV,SMR,PRGO,PCRX,AMPX,EOSE,TTMI,BE,OUST,LPTH,FLR,TIC,DECK,BIRK,SBET,SGHC,OSCR,DOCS,TEM,TXG,HIMS,PPTA,TWLO,GENI,SPOT,STUB,Z,DJT,TRIP,FUN,NCLH,REAL,CNK,PSKY,VSNT,ACHC,BRKR,RXST,TNDM,BBNX,ATEC,SOC,APA,RRC,OVV,MUR,CRK,AR,SM,LBRT,HLF,LW,BRBR,PCT,CSGP,DBRG,TWO,FRMI,COLD,HPP,PENN,CAVA,GEO,ENTG,ACMR,AEHR,AMKR,Q,ALAB,MRVL,AG,GTM,FRSH,COMP,PD,DDOG,SNOW,BTDR,MSTR,NOW,ASAN,SOUN,OS,RELY,CRWV,CORZ,RBRK,NTSK,FOUR,SOLS,KLAR,PGY,AFRM,AFRM,ENPH,AVTR,ALB,CC,OLN,ETSY,CART,CHWY,BBWI,GME,RHI,UPWK,CLF,CMCSA,RXO,VST,GEV,NRG,CEG,HE,CSIQ,MRNA,XBI,CRWD,RZLV,LAES,NUAI,LUMN,ASPI,POET,IMRX,ALT,SHLS,HTZ,TIGR,ACHR,VG,ABR,USAS,PAYO,SANA,DRIP,ULCC,NIO,JBLU,INDI,SG,BBAI,ASM,TROX,VZLA,NWL,MSOS,MNKD,BCRX,BW,BTG,COUR,SLDP,BULL,CRMD,AUR,RIG,PTEN,TLRY,NUVB,FSLY,DUST,TDOC,TSHA,MQ,CRGY,TSDD,ENVX,GDXD,NVAX,TMQ,TGB,UNIT,HIMZ,RXRX,LAR,UAA,ABCL,UA,NEXT,SLI,TE,OPEN,MSTX,MUD,TZA,BORR,BMNG,BMNU,UAMY,TMC,LAC,BFLY,PGEN,NVD,GRAB,NB,XRPT,SOLT,CRCA,ABAT,TSM,,EXK,RKT,IREN,SHOP,TECK,UEC,PAGS,BMNR,NXE,NU,GFS,ZIM,ZETA,STNE,EBAY,CVE,QXO,XP,AES,VISN,FLG,BN,WT,BZ,LYB,AS,ABNB,CNQ,TXN,DD,NOV,SU,ALM,DXCM,CALY,BKR,SW,DHI,SYF,PINS,CRM,CNH,IBM,PYPL,CRBG,CNC,S,CMG,VLO,WY,FTV,QCOM,BX,TSCO,CCI,PGR,PRMB,MGY,SWKS,TOST,FAST,NEOG,CVS,JBS,PK,ACI,PANW,HOG,PR,ONON,ADBE,WDAY,ADM,DOC,DVN,GPK,HRL,CPRT,HD,OKE,AMCR,MKC,CPNG,HBAN,DG,AMH,COP,DOW,INVH,SIRI,XRAY,BAX,KR,EQT,CTRA,FLO,IP,CPB,CAG"

    start_raw = input("StartDate YYYY-MM-DD (or 02/23/2026): ").strip()
    end_raw = input("EndDate   YYYY-MM-DD (or 02/26/2026): ").strip()

    max_signal_raw = input("Max Signal Day % (default 8): ").strip()
    if max_signal_raw == "":
        max_signal_day_pct = MAX_SIGNAL_DAY_PCT_DEFAULT
    else:
        try:
            max_signal_day_pct = float(max_signal_raw)
        except ValueError:
            raise SystemExit("❌ Invalid Max Signal Day % input.")

    os.system("cls" if os.name == "nt" else "clear")

    tickers = parse_tickers(tickers_raw)

    if end_raw:
        end_date = roll_to_trading_day(pd.to_datetime(end_raw))
    else:
        end_date = roll_end_date(None)

    if start_raw:
        start_date = roll_to_trading_day(pd.to_datetime(start_raw))
    else:
        start_date = end_date

    if start_date > end_date:
        raise SystemExit("❌ StartDate must be <= EndDate.")

    # Build earnings map once (so we can color rows later)
    earnings_map: dict[str, pd.DatetimeIndex] = {}
    for t in tickers:
        earnings_map[t] = get_earnings_dates(t, limit=12)

    spy_close = download_spy_close(end_date)
    hist_map = download_many(tickers, end_date)

    scan_rows: list[dict] = []

    for t in tickers:
        df = hist_map.get(t)
        if df is None or len(df) < 80:
            continue

        feats = add_features(df)
        feats = add_relative_strength(feats, spy_close)

        scan_dates = feats.index[(feats.index >= start_date) & (feats.index <= end_date)]
        if len(scan_dates) == 0:
            continue

        for bar_date in scan_dates:
            i = int(feats.index.get_loc(bar_date))
            if i + 1 >= len(feats):
                continue

            row_for_score = apply_volume_boost(feats.iloc[i].copy(), VOLUME_BOOST_PCT)

            base = footprint_score(row_for_score)
            bonus = confirmation_bonus(feats, i)
            score = float(min(100.0, base + bonus))

            dayret = safe(row_for_score.get("DayRetPct", np.nan))

            c0 = safe(feats.iloc[i].get("Close", np.nan))
            c1 = safe(feats.iloc[i + 1].get("Close", np.nan))
            nextday_pct = np.nan
            if np.isfinite(c0) and np.isfinite(c1) and c0 != 0:
                nextday_pct = (c1 / c0 - 1) * 100

            if not (np.isfinite(nextday_pct) and (nextday_pct >= NEXTDAY_MIN_UP_PCT or nextday_pct <= NEXTDAY_MIN_DOWN_PCT)):
                continue

            if score < MIN_SCORE:
                continue

            dollar_vol = safe(row_for_score.get("DollarVol", np.nan))
            if np.isfinite(dollar_vol) and dollar_vol < MIN_DOLLARVOL:
                continue

            if np.isfinite(dayret) and dayret > max_signal_day_pct:
                continue

            if not (np.isfinite(dayret) and dayret >= MIN_SIGNAL_DAY_PCT):
                continue

            red_earn = is_red_earn_match(bar_date, earnings_map.get(t))

            scan_rows.append({
                "Ticker": t,
                "BarDate": bar_date.strftime("%Y-%m-%d"),
                "SignalDay%": f"{dayret:.2f}%",
                "NextDay%": f"{nextday_pct:.2f}%",
                "Score": score,
                "BaseScore": base,
                "Bonus": bonus,
                "Close": safe(feats.iloc[i].get("Close", np.nan)),
                "TC2000_AbsRange": round(safe(feats.iloc[i].get("TC2000_AbsRange", np.nan)), 2),
                "RedEarnings": bool(red_earn),  # <-- used only for coloring
            })

    if not scan_rows:
        print("No matches found.")
        return

    scan_df = (
        pd.DataFrame(scan_rows)
        .sort_values(["BarDate", "Score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Color rows red if BarDate == EarningsDate OR BarDate+1BDay == EarningsDate
    display_df = scan_df.drop(columns=["RedEarnings"], errors="ignore")

    header = "  ".join(display_df.columns)
    print(header)
    print("-" * len(header))

    for idx, r in display_df.iterrows():
        row_text = "  ".join(str(r[c]) for c in display_df.columns)
        if bool(scan_df.loc[idx, "RedEarnings"]):
            print(f"{RED}{row_text}{RESET}")
        else:
            print(row_text)


if __name__ == "__main__":
    main()