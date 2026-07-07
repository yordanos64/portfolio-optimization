import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def run_adf_test(series, name):
    """Performs the Augmented Dickey-Fuller test to determine stationarity."""
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"  * {name} -> ADF Stat: {result[0]:.4f} | p-value: {p_value:.4e}")
    print(f"    Conclusion: {'Stationary (Pass)' if p_value <= 0.05 else 'Non-Stationary (Fail)'}")

def calculate_portfolio_risks(df, risk_free_rate=0.04):
    """Calculates daily returns, 95% Value at Risk, and Annualized Sharpe Ratio."""
    returns = df.pct_change().dropna()
    summary = {}
    
    for asset in df.columns:
        asset_rets = returns[asset]
        # 95% Historical Value at Risk
        var_95 = np.percentile(asset_rets, 5)
        # Annualized Sharpe Ratio calculation
        annualized_mean = asset_rets.mean() * 252
        annualized_vol = asset_rets.std() * np.sqrt(252)
        sharpe = (annualized_mean - risk_free_rate) / annualized_vol
        
        summary[asset] = {
            "95% Value at Risk (VaR)": f"{var_95 * 100:.2f}%",
            "Annualized Sharpe Ratio": round(sharpe, 4)
        }
    return pd.DataFrame(summary)

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/cleaned_prices.csv', index_col='Date', parse_dates=True)
    
    print("\n🧐 1. Stationarity Analysis (Augmented Dickey-Fuller Test)")
    print("-" * 60)
    print("[Raw Closing Prices]")
    for asset in df.columns:
        run_adf_test(df[asset], f"{asset} Raw Price")
        
    print("\n[Daily Percentage Returns]")
    returns = df.pct_change().dropna()
    for asset in returns.columns:
        run_adf_test(returns[asset], f"{asset} Return")
        
    print("\n🛡️ 2. Portfolio Management Risk Metrics Dashboard")
    print("-" * 60)
    risk_metrics = calculate_portfolio_risks(df)
    print(risk_metrics.to_string())
