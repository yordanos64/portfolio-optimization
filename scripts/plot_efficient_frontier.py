import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_frontier_plot():
    if not os.path.exists('data/processed/simulated_portfolios.csv'):
        print("❌ Simulation data missing! Please run portfolio_optimization.py first.")
        return
        
    df = pd.read_csv('data/processed/simulated_portfolios.csv')
    
    plt.figure(figsize=(10, 6))
    # Plot all simulated metrics colored by Sharpe
    sc = plt.scatter(df['Volatility']*100, df['Return']*100, c=df['Sharpe'], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(sc, label='Sharpe Ratio')
    
    # Highlight Max Sharpe point
    max_sharpe = df.loc[df['Sharpe'].idxmax()]
    plt.scatter(max_sharpe['Volatility']*100, max_sharpe['Return']*100, color='red', marker='*', s=200, label='Max Sharpe Ratio')
    
    # Highlight Min Volatility point
    min_vol = df.loc[df['Volatility'].idxmin()]
    plt.scatter(min_vol['Volatility']*100, min_vol['Return']*100, color='blue', marker='*', s=200, label='Minimum Volatility')
    
    plt.title('Efficient Frontier Simulation — TSLA / BND / SPY')
    plt.xlabel('Annualized Volatility (Risk %) ')
    plt.ylabel('Expected Annualized Return (%)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the chart image
    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/efficient_frontier.png', dpi=300, bbox_inches='tight')
    print("📊 Efficient Frontier visual chart successfully generated and saved to: docs/efficient_frontier.png")
    plt.close()

if __name__ == "__main__":
    generate_frontier_plot()
