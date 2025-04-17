# utils/time_series_utils.py
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

def extract_seasonality(time_series, freq='W', periods=None):
    """
    Extract seasonality component from a time series.
    
    Args:
        time_series: Pandas Series with DatetimeIndex
        freq: Frequency of seasonality ('W' for weekly, 'M' for monthly)
        periods: Number of periods for the seasonal component
        
    Returns:
        Dictionary containing seasonal indices and detected seasonality strength
    """
    # Handle missing values
    time_series = time_series.fillna(time_series.mean())
    
    # Determine periods if not specified
    if periods is None:
        if freq == 'W':
            periods = 7  # Weekly seasonality
        elif freq == 'M':
            periods = 12  # Monthly seasonality
        else:
            periods = 7  # Default
    
    # Require at least 2 * periods data points
    if len(time_series) < 2 * periods:
        return {
            "seasonality_detected": False,
            "seasonal_indices": {},
            "strength": 0.0
        }
    
    try:
        # Apply seasonal decomposition
        decomposition = seasonal_decompose(
            time_series, 
            model='additive', 
            period=periods,
            extrapolate_trend='freq'
        )
        
        seasonal = decomposition.seasonal
        trend = decomposition.trend
        resid = decomposition.resid
        
        # Calculate strength of seasonality
        var_seasonal = np.var(seasonal)
        var_resid = np.var(resid)
        var_total = var_seasonal + var_resid
        
        strength = var_seasonal / var_total if var_total > 0 else 0
        
        # Create seasonal indices
        if freq == 'W':
            # For weekly seasonality, map to days of week
            seasonal_indices = {}
            unique_indices = np.unique(seasonal.index.dayofweek)
            for day in unique_indices:
                day_values = seasonal[seasonal.index.dayofweek == day]
                seasonal_indices[int(day)] = float(day_values.mean())
        else:
            # For other frequencies, just return the unique seasonal components
            seasonal_indices = {i: float(val) for i, val in enumerate(seasonal.unique())}
        
        return {
            "seasonality_detected": strength > 0.1,  # Threshold for significant seasonality
            "seasonal_indices": seasonal_indices,
            "strength": float(strength)
        }
        
    except Exception as e:
        print(f"Error extracting seasonality: {e}")
        return {
            "seasonality_detected": False,
            "seasonal_indices": {},
            "strength": 0.0
        }

def detect_anomalies(time_series, window=10, sigma=3.0):
    """
    Detect anomalies in a time series using rolling statistics.
    
    Args:
        time_series: Pandas Series
        window: Size of the rolling window
        sigma: Number of standard deviations to use as threshold
        
    Returns:
        Dictionary containing detected anomalies and their indices
    """
    if len(time_series) < window + 1:
        return {
            "anomalies_detected": False,
            "anomaly_indices": [],
            "anomaly_values": []
        }
    
    # Calculate rolling mean and standard deviation
    rolling_mean = time_series.rolling(window=window, center=True).mean()
    rolling_std = time_series.rolling(window=window, center=True).std()
    
    # For the first and last (window//2) points, use the overall mean and std
    overall_mean = time_series.mean()
    overall_std = time_series.std()
    
    # Fill NaN values in rolling calculations
    rolling_mean = rolling_mean.fillna(overall_mean)
    rolling_std = rolling_std.fillna(overall_std)
    
    # Identify anomalies
    anomalies = abs(time_series - rolling_mean) > (sigma * rolling_std)
    anomaly_indices = list(anomalies[anomalies].index)
    anomaly_values = list(time_series[anomalies])
    
    return {
        "anomalies_detected": len(anomaly_indices) > 0,
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values
    }