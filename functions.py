import pandas as pd
import numpy as np

def safe_float(value):
    """Safely convert to float, return None if fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_current_price_and_date(recent_hist, data, info):
    """Get current price and last date from available sources."""
    current_price = None
    last_date = None
    
    if isinstance(recent_hist, pd.DataFrame) and not recent_hist.empty:
        current_price = safe_float(recent_hist['Close'].iloc[-1])
        last_date = recent_hist.index[-1]
    
    if current_price is None and isinstance(data, pd.DataFrame) and not data.empty:
        current_price = safe_float(data['Close'].iloc[-1])
        last_date = data.index[-1]
    
    if current_price is None:
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    
    return current_price, last_date


def humanize_number(x):
    """Convert large numbers to human-readable format."""
    try:
        x = float(x)
    except Exception:
        return "N/A"
    if pd.isna(x):
        return "N/A"
    abs_x = abs(x)
    if abs_x >= 1e12:
        return f"{x/1e12:.2f}T"
    if abs_x >= 1e9:
        return f"{x/1e9:.2f}B"
    if abs_x >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs_x >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.0f}"

def get_historical_values(data, column='Close', periods=[30, 90, 180, 365]):
    """Get historical values for specified periods (in days) for any column."""
    values = []
    if isinstance(data, pd.DataFrame) and not data.empty and column in data.columns and len(data) > 1:
        # Clean the data first - remove NaN values
        clean_data = data[column].dropna()
        if clean_data.empty:
            return values
            
        for period in periods:
            if len(clean_data) >= period:
                val = clean_data.iloc[-period]
                values.append(safe_float(val))
            elif len(clean_data) > 0:
                # If we don't have enough data, use the oldest available value
                val = clean_data.iloc[0]
                values.append(safe_float(val))
            else:
                values.append(None)
    return values