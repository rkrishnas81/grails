import yfinance as yf
import pandas as pd
from datetime import datetime

# -------------------------------------------------
# CONFIG — LOOKBACK 1 YEAR ONLY
# -------------------------------------------------

START_DATE = (datetime.today() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")


def get_data(ticker: str) -> pd.DataFrame:
    print(f"\nDownloading last 1 year of data for {ticker} ...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)

    if df.empty:
        print(f"No data for {ticker}")
        return df

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    return df


def add_tc2000_columns(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"]
    O = df["Open"]
    L = df["Low"]
    V = df["Volume"]

    df["C1"] = C.shift(1)
    df["C2"] = C.shift(2)
    df["O2"] = O.shift(2)
    df["V1"] = V.shift(1)

    # 1-day % return (close → next close)
    df["ret_1d"] = (C.shift(-1) / C - 1.0) * 100.0

    # Next-day open gap vs today's close (%)
    df["open_gap_1d"] = (df["Open"].shift(-1) / C - 1.0) * 100.0

    # SignalDayUp: today's close vs yesterday close (%)
    df["signal_change"] = (C / df["C1"] - 1.0) * 100.0

    return df


def apply_tc2000_condition(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"]
    L = df["Low"]
    C1 = df["C1"]
    C2 = df["C2"]
    O2 = df["O2"]
    V1 = df["V1"]

    cond = (
        (C > 50) &
        (V1 > 1_000_000) &
        # 4% above the low
        (100 * (C / L - 1) >= 4) &
        (100 * (C1 / C2 - 1) < -2) &
        (C2 > O2)
    )

    return df.loc[cond].copy()


def backtest_ticker(ticker: str):
    print("\n" + "=" * 70)
    print(f"BACKTEST FOR {ticker}")
    print("=" * 70)

    df = get_data(ticker)
    if df.empty:
        return

    df = add_tc2000_columns(df)
    signals = apply_tc2000_condition(df)

    if signals.empty:
        print(f"\nNo signals found for {ticker}.")
        return

    # P/L of next-close on $10,000
    signals["pl_10k_1d"] = 10000 * (signals["ret_1d"] / 100.0)

    signals["signal_date"] = signals.index
    signals = signals.sort_values("signal_date", ascending=False)

    # Filter to SignalDayUp > 0 (i.e. signal_change > 0)
    filtered = signals[signals["signal_change"] > 0].copy()

    print(f"\nTotal rows with SignalDayUp > 0 in last 1 year: {len(filtered)}")

    # Build table (all rows)
    table = filtered[[
        "signal_date",
        "Close",
        "signal_change",
        "open_gap_1d",
        "ret_1d",
        "pl_10k_1d"
    ]].copy()

    table = table.rename(
        columns={
            "signal_date": "Signal Date",
            "signal_change": "SignalDayUp",
            "open_gap_1d": "OpenGap",
            "ret_1d": "NextDay",
            "pl_10k_1d": "PL_1d_10000"
        }
    )

    # Formatting for console + CSV
    if not table.empty:
        table["Signal Date"] = pd.to_datetime(table["Signal Date"]).dt.strftime("%Y-%m-%d")
        table["Close"] = table["Close"].map(lambda x: f"{x:.2f}")
        table["SignalDayUp"] = table["SignalDayUp"].map(lambda x: f"{x:.2f}%")
        table["OpenGap"] = table["OpenGap"].map(lambda x: f"{x:.2f}%")
        table["NextDay"] = table["NextDay"].map(lambda x: f"{x:.2f}%")
        table["PL_1d_10000"] = table["PL_1d_10000"].map(lambda x: f"{x:,.2f}")

    def print_table(tdf: pd.DataFrame):
        headers = ["Signal Date", "Close", "SignalDayUp", "OpenGap", "NextDay", "PL_1d_10000"]
        widths = [12, 8, 12, 10, 10, 14]

        header_line = " | ".join([h.ljust(w) for h, w in zip(headers, widths)])
        print("\n" + header_line)
        print("-" * (sum(widths) + 3 * (len(widths) - 1)))

        for _, row in tdf.iterrows():
            line = " | ".join([
                str(row["Signal Date"]).ljust(widths[0]),
                str(row["Close"]).rjust(widths[1]),
                str(row["SignalDayUp"]).rjust(widths[2]),
                str(row["OpenGap"]).rjust(widths[3]),
                str(row["NextDay"]).rjust(widths[4]),
                str(row["PL_1d_10000"]).rjust(widths[5]),
            ])
            print(line)

    # Print table
    print(f"\nSignals for {ticker} (last 1 year, SignalDayUp > 0):")
    if not table.empty:
        print_table(table)
    else:
        print("No rows found.")

    # Summary (all filtered rows)
    print("\nSummary (last 1 year):")
    if filtered.empty:
        print("  Number days : 0")
        print("  Profit days : 0")
        print("  Loss days   : 0")
        print("  Win rate %  : 0.00")
        print("  OpenGap > 0 days : 0")
    else:
        total = len(filtered)
        profit = (filtered["pl_10k_1d"] > 0).sum()
        loss = (filtered["pl_10k_1d"] < 0).sum()
        win_rate = (profit / total) * 100
        open_gap_positive = (filtered["open_gap_1d"] > 0).sum()

        print(f"  Number days : {total}")
        print(f"  Profit days : {profit}")
        print(f"  Loss days   : {loss}")
        print(f"  Win rate %  : {win_rate:.2f}")
        print(f"  OpenGap > 0 days : {open_gap_positive}")

    # Reminders
    print("\nREMINDERS:")
    print("Reminder: This analysis is limited to the last 1 year of data.")
    print("Reminder: All results are based only on days where SignalDayUp > 0.")
    print("Reminder: Past performance is not indicative of future results.")
    print("Reminder: This strategy filters only stocks above $50 with volume above 1M.")

    # Save CSV (formatted table)
    out_file = f"{ticker}_tc2000_signals_last_year.csv"
    table.to_csv(out_file, index=False)
    print(f'\nSaved to "{out_file}".')


def main():
    raw = input("Enter tickers comma separated (e.g. MP, UAL, AAPL) [default MP]: ").strip()

    if not raw:
        tickers = ["MP"]
    else:
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    print(f"\nTickers to process: {', '.join(tickers)}")

    for ticker in tickers:
        backtest_ticker(ticker)


if __name__ == "__main__":
    main()
