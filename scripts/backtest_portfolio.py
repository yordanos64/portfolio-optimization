import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_portfolio_backtest():
    print("⏳ [Task 4] Loading preprocessed financial price histories...")
    # Load cleaned asset data
    df = pd.read_csv('data/processed/cleaned_prices.csv', index_col='Date', parse_dates=True)
    
    # Calculate daily percentage returns
    daily_returns = df.pct_change().dropna()
    
    # Aligning weights to dataframe alphabetical order: BND, SPY, TSLA
    # Calculated from your Task 3 dashboard: 91.66% BND, 8.33% SPY, 0.01% TSLA
    tickers = sorted(list(df.columns))
    print(f"Aligning allocation parameters to order: {tickers}")
    
    # Weights array maps exactly to [BND, SPY, TSLA]
    opt_weights = np.array([0.9166, 0.0833, 0.0001])
    
    print("📈 Running transactional backtesting simulation loops...")
    # Vectorized matrix product yields the strategy return vector
    opt_returns = daily_returns[tickers].dot(opt_weights)
    bench_returns = daily_returns['SPY'] # 100% SPY Market Benchmark
    
    # Compute cumulative growth tracking curves (assuming a $1 starting base)
    opt_cumulative = (1 + opt_returns).cumprod()
    bench_cumulative = (1 + bench_returns).cumprod()
    
    # Metrics Calculator Function
    def calculate_performance_metrics(returns_series, cumulative_series):
        # Total Return percentage
        total_return = (cumulative_series.iloc[-1] - 1) * 100
        
        # Annualized Volatility percentage (252 trading days)
        annualized_vol = returns_series.std() * np.sqrt(252) * 100
        
        # Maximum Drawdown calculation
        rolling_peaks = cumulative_series.cummax()
        drawdowns = (cumulative_series - rolling_peaks) / rolling_peaks
        max_drawdown = drawdowns.min() * 100
        
        return total_return, annualized_vol, max_drawdown

    opt_ret, opt_vol, opt_dd = calculate_performance_metrics(opt_returns, opt_cumulative)
    bench_ret, bench_vol, bench_dd = calculate_performance_metrics(bench_returns, bench_cumulative)
    
    print("\n🏁 --- Task 4 Financial Strategy Backtest Report ---")
    print(f"🔵 OPTIMIZED PORTFOLIO (91.66% BND / 8.33% SPY / 0.01% TSLA):")
    print(f"  - Cumulative Total Strategy Return : {opt_ret:.2f}%")
    print(f"  - Annualized Strategy Volatility   : {opt_vol:.2f}%")
    print(f"  - Maximum Peak-to-Trough Drawdown  : {opt_dd:.2f}%")
    
    print(f"\n🟠 BENCHMARK MARKET PORTFOLIO (100% SPY):")
    print(f"  - Cumulative Total Strategy Return : {bench_ret:.2f}%")
    print(f"  - Annualized Strategy Volatility   : {bench_vol:.2f}%")
    print(f"  - Maximum Peak-to-Trough Drawdown  : {bench_dd:.2f}%")
    
    # 4. Generate and Export Strategy Comparison Graph
    plt.figure(figsize=(11, 5.5))
    plt.plot(opt_cumulative, label='Optimized Defensive Strategy Portfolio (Low Risk)', color='royalblue', lw=1.5)
    plt.plot(bench_cumulative, label='Market Benchmark Portfolio (100% SPY)', color='darkorange', lw=1.5)
    plt.title('GMF Investments - Historical Strategy Backtesting Vector (2015 - 2026)')
    plt.xlabel('Trading Timeline')
    plt.ylabel('Growth Value of an Initial $1 Investment')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs('docs', exist_ok=True)
    graph_path = 'docs/backtest_equity_curve.png'
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Strategy performance graph successfully exported to: {graph_path}")
    plt.close()

if __name__ == "__main__":
    run_portfolio_backtest()
