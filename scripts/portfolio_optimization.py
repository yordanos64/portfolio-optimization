import os
import pandas as pd
import numpy as np

def run_portfolio_optimization(num_portfolios=10000, risk_free_rate=0.04):
    print("⏳ [Task 3] Ingesting processed asset prices...")
    # Load historical datasets from Task 1
    df = pd.read_csv('data/processed/cleaned_prices.csv', index_col='Date', parse_dates=True)
    
    # Isolate daily percentage return trends
    returns = df.pct_change().dropna()
    tickers = list(df.columns)
    num_assets = len(tickers)
    
    # Annualize expected returns and covariance matrix (252 trading sessions/year)
    annual_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Matrix configurations to capture simulation iterations
    results = np.zeros((3 + num_assets, num_portfolios))
    
    print(f"🎲 Simulating {num_portfolios} random allocation weight combinations...")
    np.random.seed(42) # Locked seed for reproducible statistical configurations
    
    for i in range(num_portfolios):
        # Generate random weight arrays that sum up to exactly 1.0 (100%)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate expected annualized performance metrics
        p_return = np.dot(weights, annual_returns)
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        p_sharpe = (p_return - risk_free_rate) / p_volatility
        
        # Store results matrix rows
        results[0, i] = p_return
        results[1, i] = p_volatility
        results[2, i] = p_sharpe
        
        for j in range(num_assets):
            results[3 + j, i] = weights[j]
            
    # Package arrays into a structured dataframe
    columns = ['Return', 'Volatility', 'Sharpe'] + [f'Weight_{t}' for t in tickers]
    sim_df = pd.DataFrame(results.T, columns=columns)
    
    # Isolate structural boundaries (Maximum Sharpe Ratio & Minimum Volatility profiles)
    max_sharpe = sim_df.loc[sim_df['Sharpe'].idxmax()]
    min_vol = sim_df.loc[sim_df['Volatility'].idxmin()]
    
    print("\n🎯 --- Task 3 Portfolio Optimization Dashboard ---")
    print("\n📈 [MAXIMUM SHARPE RATIO PORTFOLIO]")
    print(f"  - Annualized Return: {max_sharpe['Return']*100:.2f}%")
    print(f"  - Annualized Volatility: {max_sharpe['Volatility']*100:.2f}%")
    print(f"  - Max Sharpe Metric: {max_sharpe['Sharpe']:.4f}")
    print("  - Optimal Allocations:")
    for t in tickers:
        print(f"    * {t}: {max_sharpe[f'Weight_{t}']*100:.2f}%")
        
    print("\n🛡️ [MINIMUM VOLATILITY PORTFOLIO]")
    print(f"  - Annualized Return: {min_vol['Return']*100:.2f}%")
    print(f"  - Annualized Volatility: {min_vol['Volatility']*100:.2f}%")
    print(f"  - Sharpe Metric: {min_vol['Sharpe']:.4f}")
    print("  - Optimal Allocations:")
    for t in tickers:
        print(f"    * {t}: {min_vol[f'Weight_{t}']*100:.2f}%")
        
    # Save the tracking matrices for Task 4 consumption
    sim_df.to_csv('data/processed/simulated_portfolios.csv', index=False)
    print("\n✅ Simulation matrices exported to: data/processed/simulated_portfolios.csv")

if __name__ == "__main__":
    run_portfolio_optimization()
