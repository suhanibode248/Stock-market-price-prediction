import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import argparse
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_URL = "sqlite:///quantsense.db"

def fetch_yfinance_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical OHLCV data from Yahoo Finance."""
    logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        logging.warning(f"No data fetched for {symbol}.")
        return df
        
    df.reset_index(inplace=True)
    # Rename columns to match database schema
    df = df.rename(columns={
        "Date": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    df["symbol"] = symbol
    df = df[["time", "symbol", "open", "high", "low", "close", "volume"]]
    return df

def ingest_data(symbol: str, start_date: str, end_date: str):
    df = fetch_yfinance_data(symbol, start_date, end_date)
    if not df.empty:
        engine = create_engine(DB_URL)
        try:
            # For MVP, we'll append. If duplicate keys are an issue later, we can add unique constraints
            df.to_sql('stock_prices', engine, if_exists='append', index=False)
            logging.info(f"Successfully ingested {len(df)} rows for {symbol}.")
        except Exception as e:
            logging.error(f"Failed to ingest data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest historical stock data from Yahoo Finance.")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"], help="List of ticker symbols")
    parser.add_argument("--days", type=int, default=365, help="Number of days to fetch")
    args = parser.parse_args()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    for symbol in args.symbols:
        ingest_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
