# GMF Investments - Time Series Forecasting for Portfolio Optimization

This repository contains the technical implementation for the **10 Academy Week 9 Challenge**. The objective is to apply time series analysis, risk metric tracking, and baseline financial forecasting to optimize asset allocations across three core vectors: Tesla (**TSLA**), Vanguard Total Bond Market ETF (**BND**), and the S&P 500 ETF (**SPY**).

---

## 📁 Repository Structure
```text
portfolio-optimization/
├── data/
│   ├── raw/          <- Untouched source yfinance prices
│   └── processed/    <- Cleaned, multi-index filled data
├── scripts/
│   ├── data_preprocessing.py  <- Data ingestion & wrangling engine
│   ├── eda_analysis.py        <- ADF stationarity & risk metrics analytics
│   └── train_arima.py         <- TSLA statistical baseline forecasting model
└── README.md                  <- Project execution summary
```

---

## 📈 Task 1: Preprocessing & Exploratory Data Analysis

### 1. Stationarity Analysis (Augmented Dickey-Fuller Test)
Raw financial asset timelines exhibit structural upward drift, making them non-stationary. Applying first-order percentage differencing resolves this property to satisfy forecasting assumptions.

*   **Raw Closing Prices**: 
    *   `TSLA` ($p \approx 0.7270$) -> **Non-Stationary (Fail)**
    *   `BND` ($p \approx 0.7219$) -> **Non-Stationary (Fail)**
    *   `SPY` ($p \approx 0.9966$) -> **Non-Stationary (Fail)**
*   **Daily Percentage Returns**: 
    *   `TSLA` ($p \approx 0.0000$) -> **Stationary (Pass)**
    *   `BND` ($p \approx 0.0000$) -> **Stationary (Pass)**
    *   `SPY` ($p \approx 0.0000$) -> **Stationary (Pass)**

### 2. Portfolio Management Risk Metrics Dashboard
Calculated using historical tracking bounds from January 1, 2015, through June 30, 2026:

| Asset | 95% Daily Value at Risk (VaR) | Annualized Sharpe Ratio | Investment Profile Role |
| :--- | :--- | :--- | :--- |
| **TSLA** | **-5.17%** | **0.7245** | High Volatility / Return Maxima |
| **BND** | **-0.48%** | **-0.3772** | Low Risk Capital Stabilizer |
| **SPY** | **-1.67%** | **0.5909** | Moderate Market Diversification |

---

## 🤖 Task 2: Time Series Forecasting (ARIMA Model)

To model Tesla's asset runway, a baseline statistical **ARIMA(1,1,1)** framework was established, applying a single integrated difference ($d=1$) to rectify price non-stationarity.

### Model Tracking Errors (Out-of-Sample Performance)
*   **Mean Absolute Error (MAE)**: `149.3043`
*   **Root Mean Squared Error (RMSE)**: `175.3177`

### Core Evaluation Takeaways
The elevated long-term structural error directly matches the **Efficient Market Hypothesis (EMH)** concept highlighted in the GMF brief: standalone absolute price tracking via historical sequences remains difficult due to random walk properties. Consequently, these metrics will serve as alpha volatility markers inside the Phase 3 optimization framework rather than absolute standalone purchase targets.
