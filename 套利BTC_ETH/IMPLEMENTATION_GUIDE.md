# ETH/BTC Statistical Arbitrage Strategy - Implementation Guide

## 🎯 Overview

This is an **institutional-grade mean reversion arbitrage system** for ETH/BTC pairs trading, implementing rigorous quantitative methodologies from top-tier hedge funds (Two Sigma, WorldQuant, Citadel).

## 📊 Key Features

### ✅ Implemented Improvements Over Basic Framework

1. **Advanced Cointegration Testing**
   - Engle-Granger two-step methodology
   - Rolling cointegration stability checks
   - Johansen test support
   - Comprehensive stationarity analysis

2. **Dynamic Hedge Ratio Calculation**
   - **Kalman Filter** (default) - Adapts to regime changes in real-time
   - Rolling OLS (fallback) - Traditional approach
   - Continuous beta monitoring and adjustment

3. **Sophisticated Signal Generation**
   - Z-score based entry/exit with adaptive thresholds
   - Half-life regime detection (filters non-tradeable periods)
   - Volatility-adjusted signal thresholds
   - Multiple position sizing methods (Fixed, Kelly, Volatility-based)

4. **Realistic Backtesting**
   - Transaction costs (maker/taker fees)
   - **Funding rate simulation** (critical for perpetual futures)
   - Slippage modeling
   - Mark-to-market P&L tracking

5. **Professional Risk Management**
   - Maximum drawdown monitoring
   - Position size limits
   - Stop-loss at 4σ
   - Regime-based trading filters

6. **Comprehensive Analytics**
   - Sharpe, Sortino, Calmar ratios
   - Win rate, profit factor, expectancy
   - Detailed trade-by-trade analysis
   - Multiple visualization dashboards

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional, for live trading)
set BINANCE_API_KEY=your_api_key
set BINANCE_API_SECRET=your_api_secret
```

### Run the Strategy

```bash
python main.py
```

This will execute the complete workflow:
1. Load ETH/BTC historical data
2. Test cointegration
3. Calculate dynamic hedge ratios
4. Generate trading signals
5. Run backtest simulation
6. Generate performance reports and visualizations

### Output Files

All results are saved to `D:\strategy\套利BTC_ETH\output\`:
- `cointegration_analysis.png` - Cointegration test results
- `performance_dashboard.png` - Portfolio performance metrics
- `trade_analysis.png` - Trade-by-trade breakdown
- `strategy_analysis.png` - Spread, Z-score, beta evolution
- `trades.csv` - Detailed trade log
- `portfolio_values.csv` - Time series of portfolio value
- `signals.csv` - All trading signals with metadata
- `performance_metrics.csv` - Summary statistics

## 📁 Project Structure

```
套利BTC_ETH/
├── config.py                    # Strategy configuration
├── data_loader.py               # Data loading and preprocessing
├── cointegration.py             # Cointegration testing
├── hedge_ratio.py               # Dynamic beta calculation
├── signal_generator.py          # Trading signal generation
├── backtester.py                # Backtesting engine
├── performance_analytics.py     # Performance analysis
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── output/                      # Results directory
```

## ⚙️ Configuration

Edit `config.py` to customize strategy parameters:

### Key Parameters

```python
# Hedge Ratio Method
USE_KALMAN = True  # True: Kalman Filter, False: Rolling OLS
ROLLING_WINDOW = 720  # 30 days for OLS

# Signal Generation
ZSCORE_ENTRY = 2.0   # Entry threshold (2 standard deviations)
ZSCORE_EXIT = 0.5    # Exit threshold
ZSCORE_STOP = 4.0    # Stop loss threshold

# Risk Management
INITIAL_CAPITAL = 100000.0  # Starting capital (USD)
MAX_POSITION_SIZE = 1.0     # 100% of capital
MAX_LEVERAGE = 3.0          # Maximum leverage

# Transaction Costs
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.0004  # 0.04%
SLIPPAGE_BPS = 5.0  # 5 basis points
```

## 🔬 Strategy Logic

### Entry Conditions
- **Long Spread** (Long ETH, Short BTC): When Z-score < -2.0
  - ETH is relatively cheap vs BTC
- **Short Spread** (Short ETH, Long BTC): When Z-score > +2.0
  - ETH is relatively expensive vs BTC

### Exit Conditions
- **Take Profit**: Z-score returns to ±0.5 (mean reversion complete)
- **Stop Loss**: |Z-score| > 4.0 (structural breakdown)
- **Regime Filter**: Half-life outside acceptable range [2h, 240h]

### Position Sizing
The strategy uses **beta-hedged portfolios**:
- If Long Spread: Buy $X of ETH, Sell $β×X of BTC
- If Short Spread: Sell $X of ETH, Buy $β×X of BTC

Where β is dynamically calculated via Kalman Filter or Rolling OLS.

## 📈 Performance Metrics

The system calculates:
- **Return Metrics**: Total return, CAGR
- **Risk Metrics**: Volatility, Max Drawdown, Sharpe, Sortino, Calmar
- **Trade Metrics**: Win rate, Profit factor, Expectancy, Avg holding period
- **Cost Analysis**: Transaction costs, Funding costs breakdown

## 🎓 Theoretical Foundation

### Why This Works

1. **Cointegration**: ETH and BTC have a long-term equilibrium relationship
2. **Mean Reversion**: Short-term deviations from equilibrium are temporary
3. **Beta Hedging**: Neutralizes directional market risk
4. **Statistical Edge**: Exploits temporary mispricings

### Critical Assumptions

⚠️ **Strategy Viability Depends On:**
- Cointegration relationship remains stable
- Half-life is reasonable (not too fast/slow)
- Transaction costs < expected profit per trade
- Funding rates don't erode returns

## 🛠️ Advanced Usage

### Custom Analysis

```python
from main import *
from config import config

# Load data
loader = DataLoader(config.DATA_DIR, config.ASSET_Y, config.ASSET_X, config.TIMEFRAME)
df = loader.prepare_full_dataset()

# Test cointegration
analyzer = CointegrationAnalyzer()
results = analyzer.engle_granger_two_step(df['log_eth'], df['log_btc'])

# Compare hedge ratio methods
hedge_calc = HedgeRatioCalculator()
comparison = hedge_calc.compare_methods(df['log_eth'], df['log_btc'])
```

### Parameter Optimization

Modify `config.py` and re-run `main.py` to test different parameter combinations:
- Entry/exit thresholds
- Window sizes
- Position sizing methods
- Risk limits

## ⚠️ Important Warnings

### Before Live Trading

1. **Verify Cointegration**: Run full cointegration tests on recent data
2. **Check Half-Life**: Ensure mean reversion speed is appropriate
3. **Simulate Funding**: Use real historical funding rates, not simulated
4. **Test Execution**: Practice with small sizes first
5. **Monitor Beta**: Watch for regime changes that invalidate the relationship

### Known Risks

- **Regime Change**: ETH/BTC relationship can break down (e.g., ETH 2.0 upgrade)
- **Funding Drag**: High funding rates can kill profitability
- **Execution Risk**: Slippage on large orders
- **Liquidity Risk**: Market impact during volatile periods
- **Model Risk**: Historical cointegration ≠ future cointegration

## 📚 References

This implementation follows methodologies from:
- Engle & Granger (1987) - Cointegration theory
- Ornstein-Uhlenbeck Process - Mean reversion modeling
- Kalman (1960) - Optimal state estimation
- Kelly Criterion - Position sizing

## 🔄 Next Steps

1. **Run Backtest**: Execute `main.py` to validate on historical data
2. **Analyze Results**: Review performance metrics and visualizations
3. **Optimize Parameters**: Tune thresholds based on backtest results
4. **Paper Trade**: Test on live data without real money
5. **Go Live**: Start with small position sizes

## 📞 Support

For questions or issues:
- Review the original `readme.md` for theoretical background
- Check output visualizations for diagnostic information
- Examine trade logs in `output/trades.csv`

---

**Disclaimer**: This is a research and educational tool. Past performance does not guarantee future results. Always test thoroughly before risking real capital.
