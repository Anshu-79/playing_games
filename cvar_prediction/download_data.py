import yfinance as yf

def get_nifty_20_year_data():
    # Nifty 50 symbol on Yahoo Finance is ^NSEI
    nifty = yf.Ticker("^NSEI")

    # Fetch 20 years of data
    df = nifty.history(period="20y")

    # Clean up: Reset index to make 'Date' a column
    df.reset_index(inplace=True)

    return df

# Usage
df = get_nifty_20_year_data()
df.to_csv("data/raw/nifty.csv", index=False)
