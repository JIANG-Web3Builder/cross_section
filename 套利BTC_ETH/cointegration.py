"""
Cointegration testing module - Foundation of pairs trading strategy
This module validates whether ETH and BTC have a long-term equilibrium relationship
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class CointegrationAnalyzer:
    """
    Performs rigorous cointegration testing using Engle-Granger methodology
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = {}
        
    def test_stationarity(self, series: pd.Series, name: str = "Series") -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity
        H0: Series has unit root (non-stationary)
        H1: Series is stationary
        """
        result = adfuller(series.dropna(), autolag='AIC')
        
        output = {
            'test_statistic': result[0],
            'p_value': result[1],
            'n_lags': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < self.significance_level
        }
        
        print(f"\n{'='*60}")
        print(f"ADF Test Results for {name}")
        print(f"{'='*60}")
        print(f"Test Statistic:        {output['test_statistic']:.6f}")
        print(f"P-value:               {output['p_value']:.6f}")
        print(f"Number of Lags:        {output['n_lags']}")
        print(f"Number of Observations: {output['n_obs']}")
        print(f"\nCritical Values:")
        for key, value in output['critical_values'].items():
            print(f"  {key}: {value:.4f}")
        
        if output['is_stationary']:
            print(f"\n✓ REJECT H0: {name} is STATIONARY (p={output['p_value']:.4f})")
        else:
            print(f"\n✗ FAIL TO REJECT H0: {name} is NON-STATIONARY (p={output['p_value']:.4f})")
        
        return output
    
    def engle_granger_two_step(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Engle-Granger Two-Step Cointegration Test
        
        Step 1: OLS regression y = α + β*x + ε
        Step 2: ADF test on residuals ε
        
        Returns:
            Dictionary with regression results and cointegration test
        """
        print(f"\n{'='*60}")
        print("ENGLE-GRANGER TWO-STEP COINTEGRATION TEST")
        print(f"{'='*60}")
        
        # Step 1: OLS Regression
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        alpha = model.params[0]
        beta = model.params[1]
        residuals = model.resid
        
        print(f"\nStep 1: OLS Regression")
        print(f"  α (intercept): {alpha:.6f}")
        print(f"  β (hedge ratio): {beta:.6f}")
        print(f"  R-squared: {model.rsquared:.6f}")
        print(f"  Adj R-squared: {model.rsquared_adj:.6f}")
        
        # Step 2: Test residuals for stationarity
        print(f"\nStep 2: Testing Residuals for Stationarity")
        residual_test = self.test_stationarity(residuals, "Residuals")
        
        # Cointegration conclusion
        is_cointegrated = residual_test['is_stationary']
        
        print(f"\n{'='*60}")
        if is_cointegrated:
            print("✓✓✓ COINTEGRATION DETECTED ✓✓✓")
            print(f"ETH and BTC have a long-term equilibrium relationship")
            print(f"Hedge Ratio (β): {beta:.6f}")
            print(f"Strategy is VIABLE for pairs trading")
        else:
            print("✗✗✗ NO COINTEGRATION ✗✗✗")
            print(f"ETH and BTC do NOT have a stable long-term relationship")
            print(f"Strategy is NOT RECOMMENDED")
        print(f"{'='*60}\n")
        
        return {
            'alpha': alpha,
            'beta': beta,
            'residuals': residuals,
            'r_squared': model.rsquared,
            'model': model,
            'residual_test': residual_test,
            'is_cointegrated': is_cointegrated
        }
    
    def johansen_test(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Johansen cointegration test (alternative method)
        More powerful for multiple time series
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        data = pd.DataFrame({'y': y, 'x': x}).dropna()
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        print(f"\n{'='*60}")
        print("JOHANSEN COINTEGRATION TEST")
        print(f"{'='*60}")
        print(f"Trace Statistic: {result.lr1[0]:.4f}")
        print(f"Critical Value (95%): {result.cvt[0, 1]:.4f}")
        
        is_cointegrated = result.lr1[0] > result.cvt[0, 1]
        
        if is_cointegrated:
            print("✓ Cointegration detected (Johansen test)")
        else:
            print("✗ No cointegration (Johansen test)")
        
        return {
            'trace_stat': result.lr1[0],
            'critical_value_95': result.cvt[0, 1],
            'is_cointegrated': is_cointegrated
        }
    
    def rolling_cointegration_test(self, y: pd.Series, x: pd.Series, 
                                   window: int = 720) -> pd.DataFrame:
        """
        Rolling cointegration test to check stability over time
        
        Args:
            window: Rolling window size (e.g., 720 hours = 30 days)
        """
        print(f"\nRunning rolling cointegration test (window={window})...")
        
        results = []
        
        for i in range(window, len(y)):
            y_window = y.iloc[i-window:i]
            x_window = x.iloc[i-window:i]
            
            # Quick OLS and ADF test
            X = sm.add_constant(x_window)
            model = sm.OLS(y_window, X).fit()
            residuals = model.resid
            
            try:
                adf_result = adfuller(residuals, autolag='AIC')
                p_value = adf_result[1]
                beta = model.params[1]
            except:
                p_value = np.nan
                beta = np.nan
            
            results.append({
                'timestamp': y.index[i],
                'p_value': p_value,
                'beta': beta,
                'is_cointegrated': p_value < self.significance_level if not np.isnan(p_value) else False
            })
        
        df_results = pd.DataFrame(results).set_index('timestamp')
        
        cointegration_rate = df_results['is_cointegrated'].mean()
        print(f"Cointegration stability: {cointegration_rate*100:.2f}% of rolling windows")
        
        return df_results
    
    def plot_cointegration_analysis(self, y: pd.Series, x: pd.Series, 
                                   results: Dict, output_path: str = None):
        """
        Visualize cointegration analysis results
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Cointegration Analysis: ETH vs BTC', fontsize=16, fontweight='bold')
        
        # 1. Price series (log scale)
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        ax1.plot(y.index, np.exp(y), label='ETH', color='blue', alpha=0.7)
        ax1_twin.plot(x.index, np.exp(x), label='BTC', color='orange', alpha=0.7)
        ax1.set_ylabel('ETH Price (USD)', color='blue')
        ax1_twin.set_ylabel('BTC Price (USD)', color='orange')
        ax1.set_title('Price Series')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Log price series
        ax2 = axes[0, 1]
        ax2.plot(y.index, y, label='log(ETH)', alpha=0.7)
        ax2.plot(x.index, x, label='log(BTC)', alpha=0.7)
        ax2.set_title('Log-Transformed Prices')
        ax2.set_ylabel('Log Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot with regression line
        ax3 = axes[1, 0]
        ax3.scatter(x, y, alpha=0.3, s=1)
        
        # Add regression line
        beta = results['beta']
        alpha = results['alpha']
        x_line = np.array([x.min(), x.max()])
        y_line = alpha + beta * x_line
        ax3.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'y = {alpha:.4f} + {beta:.4f}x\nR² = {results["r_squared"]:.4f}')
        ax3.set_xlabel('log(BTC)')
        ax3.set_ylabel('log(ETH)')
        ax3.set_title('Cointegration Relationship')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals over time
        ax4 = axes[1, 1]
        residuals = results['residuals']
        ax4.plot(residuals.index, residuals, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax4.axhline(y=residuals.std(), color='orange', linestyle='--', alpha=0.5)
        ax4.axhline(y=-residuals.std(), color='orange', linestyle='--', alpha=0.5)
        ax4.set_title('Spread (Residuals) Over Time')
        ax4.set_ylabel('Residual')
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals distribution
        ax5 = axes[2, 0]
        ax5.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_title('Residuals Distribution')
        ax5.set_xlabel('Residual Value')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. ACF of residuals
        ax6 = axes[2, 1]
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals.dropna(), lags=50, ax=ax6, alpha=0.05)
        ax6.set_title('Autocorrelation of Residuals')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        
        return fig
