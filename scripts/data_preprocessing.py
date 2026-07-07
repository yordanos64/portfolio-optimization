import os
import yfinance as yf
import pandas as pd

def challenge_data_extractor():
    # 1. Ensure structural directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2015-01-01'
    end_date = '2026-06-30'
    
    print(f"🚀 [Task 1] Fetching {tickers} from YFinance API...")
    
    # Download data with group_by='ticker' to easily isolate close prices
    raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    # Save the completely untouched raw multi-index dataframe
    raw_data.to_csv('data/raw/raw_prices.csv')
    print("✅ Raw API data stored in: data/raw/raw_prices.csv")
    
    # 2. Extract Close prices for each ticker safely across modern yfinance structures
    cleaned_df = pd.DataFrame()
    for ticker in tickers:
        if ticker in raw_data.columns.levels[0]:
            # Modern yfinance utilizes 'Close' for adjusted prices by default
            cleaned_df[ticker] = raw_data[ticker]['Close']
    
    # 3. Clean missing values (handling weekends/market holidays)
    cleaned_df.ffill(inplace=True)
    cleaned_df.bfill(inplace=True)
    
    # 4. Save the final clean version for modeling
    cleaned_df.to_csv('data/processed/cleaned_prices.csv')
    print("✅ Preprocessed clean data stored in: data/processed/cleaned_prices.csv")
    
    print("\n📊 First few rows of your clean dataset:")
    print(cleaned_df.head())

if __name__ == "__main__":
    challenge_data_extractor()
