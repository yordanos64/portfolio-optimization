import os
import pandas as pd
import numpy as np

def run_portfolio_optimization(num_portfolios=10000, risk_free_rate=0.04):
    print("⏳ Loading cleaned asset return tracks...")
    # Load processed prices
    df = pd.read_csv('data/processed/cleaned_prices.csv', index_col='Date', parse_dates=True)
    
    # Calculate daily percentage returns
    returns = df.pct_change().dropna()
    tickers = list(df.columns)
    num_assets = len(tickers)
    
    # Calculate annualized metrics (252 trading days per year)
    annual_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Arrays to store simulation arrays
    results = np.zeros((3 + num_assets, num_portfolios))
    weights_record = []
    
    print(f"🎲 Simulating {num_portfolios} random weight combinations...")
    np.random.seed(42) # Ensure consistent outputs
    
    for i in range(num_portfolios):
        # Generate random weights that sum exactly to 1.0 (100%)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Calculate expected annualized portfolio return
        p_return = np.dot(weights, annual_returns)
        
        # Calculate expected annualized portfolio variance/volatility
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate resulting Sharpe Ratio
        p_sharpe = (p_return - risk_free_rate) / p_volatility
        
        # Store basic data points
        results[0, i] = p_return
        results[1, i] = p_volatility
        results[2, i] = p_sharpe
        
        # Store individual asset weight slices
        for j in range(num_assets):
            results[3 + j, i] = weights[j]
            
    # Convert array matrix to structured DataFrame
    columns = ['Return', 'Volatility', 'Sharpe'] + [f'Weight_{t}' for t in tickers]
    sim_df = pd.DataFrame(results.T, columns=columns)
    
    # Extract structural portfolio configurations
    max_sharpe_portfolio = sim_df.loc[sim_df['Sharpe'].idxmax()]
    min_vol_portfolio = sim_df.loc[sim_df['Volatility'].idxmin()]
    
    print("\n🎯 --- Portfolio Optimization Results ---")
    print("\n[MAX SHARPE RATIO PORTFOLIO - Optimal Return/Risk balance]")
    print(f" Annualized Return: {max_sharpe_portfolio['Return']*100:.2f}%")
    print(f" Annualized Volatility (Risk): {max_sharpe_portfolio['Volatility']*100:.2f}%")
    print(f" Optimized Sharpe Ratio: {max_sharpe_portfolio['Sharpe']:.4f}")
    print(" Allocation Allocation Weights:")
    for t in tickers:
        print(f"  * {t}: {max_sharpe_portfolio[f'Weight_{t}']*100:.2f}%")
        
    print("\n[MINIMUM VOLATILITY PORTFOLIO - Safest Capital Allocation]")
    print(f" Annualized Return: {min_vol_portfolio['Return']*100:.2f}%")
    print(f" Annualized Volatility (Risk): {min_vol_portfolio['Volatility']*100:.2f}%")
    print(f" Optimized Sharpe Ratio: {min_vol_portfolio['Sharpe']:.4f}")
    print(" Allocation Allocation Weights:")
    for t in tickers:
        print(f"  * {t}: {min_vol_portfolio[f'Weight_{t}']*100:.2f}%")
        
    # Export optimization tables to results directory
    os.makedirs('data/processed', exist_ok=True)
    sim_df.to_csv('data/processed/simulated_portfolios.csv', index=False)
    print("\n✅ Simulation weights exported to: data/processed/simulated_portfolios.csv")

if __name__ == "__main__":
    run_portfolio_optimization()
