import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator, DayLocator, WeekdayLocator
from typing import Dict, Optional
import requests
import time
from IPython.display import Markdown, display
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Global cache for stock data
stocks = {}

# Global ticker object
ticker = None

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

# Utility functions
def get_ticker(symbol: str):
    """Get or create a yfinance Ticker object."""
    global ticker
    if ticker is not None and hasattr(ticker, 'ticker') and ticker.ticker == symbol:
        return ticker
    ticker = yf.Ticker(symbol)
    return ticker

def fetch_stock_data(symbol: str, period: str = '1y'):
    """Fetch stock data, using cache if available."""
    global stocks
    cache_key = f"{symbol}_{period}"
    if cache_key in stocks:
        return stocks[cache_key]
    ticker = get_ticker(symbol)
    data = ticker.history(period=period)
    if not data.empty:
        stocks[cache_key] = data
    return data

def setup_date_formatting(ax, period):
    """Set up appropriate date formatting for x-axis based on period."""
    
    if period in ['1d', '5d']:
        # For very short periods, show hours/minutes
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(DayLocator())
    elif period in ['1wk', '1mo']:
        # For weeks/months, show days
        ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(DayLocator(interval=7))  # Weekly ticks
    elif period in ['3mo', '6mo']:
        # For 3-6 months, show weeks/months
        ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(MonthLocator(interval=1))
    elif period in ['1y', '2y']:
        # For 1-2 years, show months
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(MonthLocator(interval=3))  # Quarterly ticks
    elif period in ['5y', '10y', 'max']:
        # For longer periods, show years
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.xaxis.set_major_locator(YearLocator())
    else:
        # Default fallback
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

def make_cnn_api_request(api_url: str, timeout: int = 10) -> Optional[requests.Response]:
    """Make a request to CNN Business API with retry logic and multiple headers."""
    headers_list = [
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.cnn.com/",
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.cnn.com/",
        },
        {"User-Agent": "curl/7.88.1", "Accept": "*/*"},
    ]

    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    resp = None
    for h in headers_list:
        try:
            resp = session.get(api_url, headers=h, timeout=timeout)
            if resp.status_code == 200:
                break
            time.sleep(0.5)
        except requests.exceptions.RequestException:
            resp = None
            time.sleep(0.5)

    return resp

def display_structured_data_as_markdown(data, title: str, symbol: str, field_mappings: Dict[str, str] = None):
    """Display structured data (list of dicts) as formatted markdown."""
    md_lines = []

    try:
        # Normalize pandas structures to list/dict
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_dict(orient='records') if isinstance(data, pd.DataFrame) else data.tolist()

        # If a single item was returned as a dict, wrap it
        if isinstance(data, dict):
            data = [data]

        # Header
        if not data:
            md_lines.append(f"**{title}:** Not available for {symbol}")
        else:
            md_lines.append(f"### {title} for {symbol}")
            md_lines.append('')

            # Display items
            for i, item in enumerate(data, 1):
                try:
                    if isinstance(item, dict):
                        # Use field mappings if provided, otherwise try common fields
                        if field_mappings:
                            primary = item.get(field_mappings.get('primary', 'name'), 'Unknown')
                            secondary = item.get(field_mappings.get('secondary', 'title'), 'Unknown')
                        else:
                            # Default mappings
                            primary = (item.get('name') or item.get('fullName') or item.get('personName') or
                                     item.get('person_name') or item.get('symbol') or 'Unknown')
                            secondary = (item.get('title') or item.get('position') or item.get('role') or
                                       item.get('value') or item.get('shares') or 'Unknown')

                        details = f"**{secondary}** — {primary}"

                        # Add extra fields if present
                        extras = []
                        extra_fields = ['since', 'startDate', 'appointed', 'age', 'years', 'pctHeld', 'value']
                        for field in extra_fields:
                            if field in item and item[field] is not None:
                                if field in ['pctHeld']:
                                    extras.append(f"{item[field]:.2f}%")
                                elif field in ['value']:
                                    extras.append(f"${item[field]:,.0f}")
                                else:
                                    extras.append(f"{field.replace('pctHeld', '% held').replace('startDate', 'since')}: {item[field]}")

                        if extras:
                            details += " (" + ", ".join(extras) + ")"

                        md_lines.append(f"{i}. {details}")
                    else:
                        # Fallback for other types
                        md_lines.append(f"{i}. {item}")
                except Exception:
                    md_lines.append(f"{i}. Unknown item")

        display(Markdown("\n\n".join(md_lines)))
    except Exception as e:
        display(Markdown(f"**Error displaying {title}:** {e}"))

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate common technical indicators for stock data."""
    df = data.copy()

    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)

    # Stochastic Oscillator
    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()

    # On-Balance Volume
    df['OBV'] = (df['Volume'] * ((df['Close'] > df['Close'].shift(1)).astype(int) * 2 - 1)).cumsum()

    return df