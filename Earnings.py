import yfinance as yf

def get_next_earnings_date(ticker):
    stock = yf.Ticker(ticker)
    
    # Get earnings dates (yfinance returns a DataFrame)
    earnings_dates = stock.get_earnings_dates(limit=1)
    
    if earnings_dates is None or earnings_dates.empty:
        return "No earnings date available"
    
    # The index contains the earnings datetime
    next_earnings_date = earnings_dates.index[0]
    return next_earnings_date

if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g. AAPL): ").upper()
    result = get_next_earnings_date(ticker)
    print(f"Next earnings date for {ticker}: {result}")
