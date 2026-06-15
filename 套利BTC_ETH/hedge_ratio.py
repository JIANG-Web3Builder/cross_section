"""
Dynamic Hedge Ratio Calculation Module
Implements both Rolling OLS and Kalman Filter approaches
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Optional
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')

class HedgeRatioCalculator:
    """
    Calculate dynamic hedge ratio (beta) between ETH and BTC
    Professional implementation with both Rolling OLS and Kalman Filter
    """
    
    def __init__(self, method: str = 'kalman', window: int = 720):
        """
        Args:
            method: 'rolling_ols' or 'kalman'
            window: Lookback window for rolling OLS (hours)
        """
        self.method = method
        self.window = window
        
    def rolling_ols(self, y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Rolling OLS regression to calculate dynamic beta
        
        Returns:
            beta, alpha, residuals (spread)
        """
        print(f"Calculating hedge ratio using Rolling OLS (window={self.window})...")
        
        betas = []
        alphas = []
        spreads = []
        timestamps = []
        
        for t in range(self.window, len(y)):
            # Get window data
            y_window = y.iloc[t-self.window:t]
            x_window = x.iloc[t-self.window:t]
            
            # OLS regression
            X = sm.add_constant(x_window)
            model = sm.OLS(y_window, X).fit()
            
            alpha = model.params[0]
            beta = model.params[1]
            
            # Calculate current spread (out-of-sample)
            current_spread = y.iloc[t] - (alpha + beta * x.iloc[t])
            
            betas.append(beta)
            alphas.append(alpha)
            spreads.append(current_spread)
            timestamps.append(y.index[t])
        
        # Convert to Series
        beta_series = pd.Series(betas, index=timestamps, name='beta')
        alpha_series = pd.Series(alphas, index=timestamps, name='alpha')
        spread_series = pd.Series(spreads, index=timestamps, name='spread')
        
        print(f"Beta range: {beta_series.min():.4f} to {beta_series.max():.4f}")
        print(f"Beta mean: {beta_series.mean():.4f} ± {beta_series.std():.4f}")
        
        return beta_series, alpha_series, spread_series
    
    def kalman_filter(self, y: pd.Series, x: pd.Series, 
                     transition_cov: float = 0.0001,
                     observation_cov: float = 1.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Kalman Filter for dynamic beta estimation
        
        This is superior to Rolling OLS because:
        1. Adapts faster to regime changes
        2. Provides optimal estimates under uncertainty
        3. No lag from fixed window
        
        State Space Model:
        - State: [alpha, beta] (hidden parameters)
        - Observation: y_t = alpha + beta * x_t + noise
        
        Args:
            transition_cov: Process noise (how much beta can change per step)
            observation_cov: Measurement noise
        """
        print(f"Calculating hedge ratio using Kalman Filter...")
        
        # Prepare data
        observations = y.values.reshape(-1, 1)
        n = len(y)
        
        # Initialize state means and covariances
        state_means = np.zeros((n, 2))
        state_means[0] = [0, 1]  # Initial guess [alpha=0, beta=1]
        
        # Simple online Kalman filter implementation
        P = np.eye(2)  # State covariance
        Q = transition_cov * np.eye(2)  # Process noise
        R = observation_cov  # Measurement noise
        
        for t in range(1, n):
            # Prediction step
            state_pred = state_means[t-1]
            P_pred = P + Q
            
            # Observation matrix for current time
            H = np.array([[1, x.iloc[t]]])
            
            # Update step
            y_pred = H @ state_pred
            residual = y.iloc[t] - y_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T / S  # Kalman gain
            
            state_means[t] = state_pred + (K * residual).flatten()
            P = (np.eye(2) - K @ H) @ P_pred
        
        # Extract alpha and beta
        alphas = state_means[:, 0]
        betas = state_means[:, 1]
        
        # Calculate spread
        spreads = y.values - (alphas + betas * x.values)
        
        # Convert to Series
        alpha_series = pd.Series(alphas, index=y.index, name='alpha')
        beta_series = pd.Series(betas, index=y.index, name='beta')
        spread_series = pd.Series(spreads, index=y.index, name='spread')
        
        print(f"Beta range: {beta_series.min():.4f} to {beta_series.max():.4f}")
        print(f"Beta mean: {beta_series.mean():.4f} ± {beta_series.std():.4f}")
        print(f"Beta final value: {beta_series.iloc[-1]:.4f}")
        
        return beta_series, alpha_series, spread_series
    
    def calculate(self, y: pd.Series, x: pd.Series, **kwargs) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Main interface to calculate hedge ratio
        
        Returns:
            beta, alpha, spread
        """
        if self.method == 'rolling_ols':
            return self.rolling_ols(y, x)
        elif self.method == 'kalman':
            transition_cov = kwargs.get('transition_cov', 0.0001)
            observation_cov = kwargs.get('observation_cov', 1.0)
            return self.kalman_filter(y, x, transition_cov, observation_cov)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compare_methods(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        Compare Rolling OLS vs Kalman Filter
        """
        print("\n" + "="*60)
        print("COMPARING HEDGE RATIO METHODS")
        print("="*60)
        
        # Rolling OLS
        print("\n1. Rolling OLS:")
        beta_ols, alpha_ols, spread_ols = self.rolling_ols(y, x)
        
        # Kalman Filter
        print("\n2. Kalman Filter:")
        beta_kf, alpha_kf, spread_kf = self.kalman_filter(y, x)
        
        # Align indices (OLS starts later due to window)
        common_index = beta_ols.index.intersection(beta_kf.index)
        
        comparison = pd.DataFrame({
            'beta_ols': beta_ols.loc[common_index],
            'beta_kalman': beta_kf.loc[common_index],
            'spread_ols': spread_ols.loc[common_index],
            'spread_kalman': spread_kf.loc[common_index]
        })
        
        # Calculate differences
        comparison['beta_diff'] = comparison['beta_kalman'] - comparison['beta_ols']
        comparison['spread_diff'] = comparison['spread_kalman'] - comparison['spread_ols']
        
        print("\n" + "="*60)
        print("COMPARISON STATISTICS")
        print("="*60)
        print(f"Beta correlation: {comparison['beta_ols'].corr(comparison['beta_kalman']):.4f}")
        print(f"Spread correlation: {comparison['spread_ols'].corr(comparison['spread_kalman']):.4f}")
        print(f"Mean beta difference: {comparison['beta_diff'].mean():.6f}")
        print(f"Std beta difference: {comparison['beta_diff'].std():.6f}")
        
        return comparison


class SpreadAnalyzer:
    """
    Analyze spread characteristics for mean reversion trading
    """
    
    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process
        
        The OU process: dX_t = θ(μ - X_t)dt + σdW_t
        Half-life = ln(2) / θ
        
        Returns:
            Half-life in hours (same unit as data frequency)
        """
        # Lag the spread
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Remove NaN
        df = pd.DataFrame({'spread': spread, 'spread_lag': spread_lag, 'spread_diff': spread_diff}).dropna()
        
        # Regression: Δspread_t = λ * spread_{t-1} + ε
        # where λ = -θ
        X = sm.add_constant(df['spread_lag'])
        model = sm.OLS(df['spread_diff'], X).fit()
        
        lambda_param = model.params[1]
        
        # Half-life calculation
        if lambda_param < 0:
            half_life = -np.log(2) / lambda_param
        else:
            half_life = np.inf  # No mean reversion
        
        return half_life
    
    @staticmethod
    def rolling_half_life(spread: pd.Series, window: int = 720) -> pd.Series:
        """
        Calculate rolling half-life to monitor regime changes
        """
        half_lives = []
        timestamps = []
        
        for t in range(window, len(spread)):
            spread_window = spread.iloc[t-window:t]
            hl = SpreadAnalyzer.calculate_half_life(spread_window)
            half_lives.append(hl)
            timestamps.append(spread.index[t])
        
        return pd.Series(half_lives, index=timestamps, name='half_life')
    
    @staticmethod
    def hurst_exponent(spread: pd.Series, max_lag: int = 100) -> float:
        """
        Calculate Hurst exponent to measure mean reversion strength
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Calculate standard deviation of differences
            std = np.std(np.subtract(spread[lag:].values, spread[:-lag].values))
            tau.append(std)
        
        # Linear regression on log-log plot
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        
        X = sm.add_constant(log_lags)
        model = sm.OLS(log_tau, X).fit()
        hurst = model.params[1]
        
        return hurst
