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

def display_balance_sheet_html(ticker_obj):
    """
    Display a formatted balance sheet as HTML table with assets, liabilities, and equity.
    
    Parameters:
    -----------
    ticker_obj : yfinance.Ticker
        The yfinance ticker object with balance_sheet data
    """
    from IPython.display import HTML
    
    # Define balance sheet groups
    assets_items = [
        "SECTION: Current Assets",
        "Cash And Cash Equivalents",
        "Other Short Term Investments",
        "Cash Cash Equivalents And Short Term Investments",
        "Receivables",
        "Accounts Receivable",
        "Inventory",
        "Prepaid Assets",
        "Other Current Assets",
        "TOTAL: Current Assets",
        "",
        "SECTION: Non-Current Assets",
        "Net PPE",
        "Gross PPE",
        "Accumulated Depreciation",
        "Goodwill And Other Intangible Assets",
        "Goodwill",
        "Other Intangible Assets",
        "Investments And Advances",
        "Long Term Equity Investment",
        "Other Non Current Assets",
        "TOTAL: Total Non Current Assets",
        "",
        "TOTAL: Total Assets"
    ]

    liabilities_items = [
        "SECTION: Current Liabilities",
        "Payables And Accrued Expenses",
        "Accounts Payable",
        "Current Accrued Expenses",
        "Income Tax Payable",
        "Current Debt",
        "Current Debt And Capital Lease Obligation",
        "Other Current Borrowings",
        "Line Of Credit",
        "TOTAL: Current Liabilities",
        "",
        "SECTION: Non-Current Liabilities",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Non Current Deferred Liabilities",
        "Other Non Current Liabilities",
        "TOTAL: Total Non Current Liabilities Net Minority Interest",
        "",
        "TOTAL: Total Liabilities Net Minority Interest"
    ]

    equity_items = [
        "SECTION: Equity",
        "Stockholders Equity",
        "Common Stock Equity",
        "Common Stock",
        "Capital Stock",
        "Additional Paid In Capital",
        "Retained Earnings",
        "Treasury Stock",
        "Other Equity Adjustments",
        "Minority Interest",
        "TOTAL: Total Equity Gross Minority Interest"
    ]

    df = ticker_obj.balance_sheet.copy()

    # Build HTML table
    html = """
    <style>
        .balance-sheet {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        .balance-sheet th {
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #34495e;
        }
        .balance-sheet td {
            padding: 8px 12px;
            border: 1px solid #ddd;
        }
        .section-header {
            font-weight: bold;
            background-color: #ecf0f1;
            font-size: 15px;
        }
        .total-row {
            font-weight: bold;
            background-color: #f8f9fa;
            border-top: 2px solid #2c3e50;
        }
        .indent {
            padding-left: 30px;
        }
        .amount {
            text-align: right;
        }
        .empty-row {
            height: 10px;
            background-color: white;
            border: none;
        }
    </style>

    <table class="balance-sheet">
        <tr>
            <th colspan="2" style="text-align: center;">ASSETS</th>
            <th colspan="2" style="text-align: center;">LIABILITIES & EQUITY</th>
        </tr>
    """

    # Helper function to format value
    def format_value(val):
        if val == "" or val is None:
            return ""
        try:
            return f"${float(val):,.0f}"
        except:
            return ""

    # Helper function to get value from df
    def get_value(item):
        if item in df.index:
            return df.loc[item].values[0]
        return ""

    # Build rows
    max_rows = max(len(assets_items), len(liabilities_items) + len(equity_items) + 2)

    asset_idx = 0
    liab_idx = 0
    equity_idx = 0
    in_equity = False

    for i in range(max_rows):
        html += "    <tr>"
        
        # Left side (Assets)
        if asset_idx < len(assets_items):
            item = assets_items[asset_idx]
            if item == "":
                html += '<td colspan="2" class="empty-row"></td>'
            elif item.startswith("TOTAL:"):
                actual_item = item.replace("TOTAL: ", "")
                val = get_value(actual_item)
                html += f'<td class="total-row">{actual_item}</td><td class="total-row amount">{format_value(val)}</td>'
            elif item.startswith("SECTION:"):
                section_name = item.replace("SECTION: ", "")
                html += f'<td colspan="2" class="section-header">{section_name}</td>'
            else:
                val = get_value(item)
                if val != "":
                    html += f'<td class="indent">{item}</td><td class="amount">{format_value(val)}</td>'
                else:
                    html += f'<td colspan="2"></td>'
            asset_idx += 1
        else:
            html += '<td colspan="2"></td>'
        
        # Right side (Liabilities & Equity)
        if not in_equity and liab_idx < len(liabilities_items):
            item = liabilities_items[liab_idx]
            if item == "":
                html += '<td colspan="2" class="empty-row"></td>'
            elif item.startswith("TOTAL:"):
                actual_item = item.replace("TOTAL: ", "")
                val = get_value(actual_item)
                html += f'<td class="total-row">{actual_item}</td><td class="total-row amount">{format_value(val)}</td>'
            elif item.startswith("SECTION:"):
                section_name = item.replace("SECTION: ", "")
                html += f'<td colspan="2" class="section-header">{section_name}</td>'
            else:
                val = get_value(item)
                if val != "":
                    html += f'<td class="indent">{item}</td><td class="amount">{format_value(val)}</td>'
                else:
                    html += f'<td colspan="2"></td>'
            liab_idx += 1
            if liab_idx >= len(liabilities_items):
                in_equity = True
        elif in_equity and equity_idx < len(equity_items):
            item = equity_items[equity_idx]
            if item == "":
                html += '<td colspan="2" class="empty-row"></td>'
            elif item.startswith("TOTAL:"):
                actual_item = item.replace("TOTAL: ", "")
                val = get_value(actual_item)
                html += f'<td class="total-row">{actual_item}</td><td class="total-row amount">{format_value(val)}</td>'
            elif item.startswith("SECTION:"):
                section_name = item.replace("SECTION: ", "")
                html += f'<td colspan="2" class="section-header">{section_name}</td>'
            else:
                val = get_value(item)
                if val != "":
                    html += f'<td class="indent">{item}</td><td class="amount">{format_value(val)}</td>'
                else:
                    html += f'<td colspan="2"></td>'
            equity_idx += 1
        else:
            html += '<td colspan="2"></td>'
        
        html += "</tr>\n"

    html += "</table>"
    
    return HTML(html)

def plot_analyst_recommendations(ticker_obj, selected_stock: str):
    """
    Plot analyst recommendations summary as a bar chart.
    
    Parameters:
    -----------
    ticker_obj : yfinance.Ticker
        The yfinance ticker object
    selected_stock : str
        Stock symbol for display in chart title
    """
    import matplotlib.pyplot as plt
    
    try:
        rec = ticker_obj.recommendations_summary
    except (NameError, AttributeError):
        display(Markdown('**`ticker` not defined. Run STOCK SELECTION cell first.**'))
        return

    if rec is None or (isinstance(rec, pd.DataFrame) and rec.empty):
        display(Markdown('**No recommendations_summary available.**'))
        return

    try:
        if isinstance(rec, pd.DataFrame):
            latest = rec.iloc[-1]
            
            # Column name mappings
            col_map = {
                'strongBuy': 'Strong Buy', 'strong_buy': 'Strong Buy', 'Strong Buy': 'Strong Buy',
                'buy': 'Buy', 'Buy': 'Buy', 
                'hold': 'Hold', 'Hold': 'Hold',
                'sell': 'Sell', 'Sell': 'Sell', 
                'strongSell': 'Strong Sell', 'strong_sell': 'Strong Sell', 'Strong Sell': 'Strong Sell'
            }
            
            # Extract counts
            counts = {
                col_map[col]: latest[col] 
                for col in rec.columns 
                if col in col_map and pd.notna(latest[col]) and latest[col] > 0
            }
            
            if not counts:
                display(Markdown('**No valid recommendation data in recent period.**'))
                return
            
            # Create series in proper order
            order = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
            s = pd.Series(counts).reindex([p for p in order if p in counts])
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color mapping
            color_map = {
                'Strong Buy': '#00aa00', 
                'Buy': '#66cc66', 
                'Hold': '#ffaa00', 
                'Sell': '#ff6666', 
                'Strong Sell': '#cc0000'
            }
            colors = [color_map.get(label, '#7f7f7f') for label in s.index]
            
            # Bar chart
            bars = ax.bar(s.index, s.values, color=colors, edgecolor='black', alpha=0.85, width=0.6)
            
            # Title with period if available
            period = f" (Period: {latest['period']})" if 'period' in rec.columns else ""
            ax.set_title(f'Current Analyst Recommendations — {selected_stock}{period}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Analysts', fontsize=12)
            ax.set_xlabel('Recommendation', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Annotations
            for bar in bars:
                if (h := bar.get_height()) > 0:
                    ax.annotate(f'{int(h)}', 
                              xy=(bar.get_x() + bar.get_width() / 2, h), 
                              xytext=(0, 3),
                              textcoords='offset points', 
                              ha='center', va='bottom', 
                              fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        else:
            display(Markdown(f'**Unexpected type: {type(rec)}**'))
    except Exception as e:
        display(Markdown(f'**Error:** {e}'))
        import traceback
        print(traceback.format_exc())


def create_mini_chart(values, current_value, color='#3b82f6'):
    """
    Create a base64-encoded mini chart (small bar chart).
    
    Parameters:
    -----------
    values : list
        List of historical values to plot (last 4 will be used)
    current_value : float
        Current value for reference
    color : str
        Color for the bars (default: blue)
    
    Returns:
    --------
    str or None
        Base64-encoded PNG image data URI, or None if values are invalid
    """
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt
    
    if not values or all(x is None for x in values):
        return None
    
    # Keep only last 4 values
    values = [None] * (4 - len(values[-4:])) + values[-4:]
    
    fig, ax = plt.subplots(figsize=(1.2, 0.7))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    bars = ax.bar(range(4), [x or 0 for x in values], color=color, width=0.7)
    
    for bar in bars:
        bar.set_capstyle('round')
        bar.set_joinstyle('round')
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=50, bbox_inches='tight', transparent=True)
    plt.close(fig)
    
    buffer.seek(0)
    return f'data:image/png;base64,{base64.b64encode(buffer.read()).decode()}'


def create_card_html(name, value, change=None, chart=None, subtitle=None):
    """
    Create an HTML card for metric display.
    
    Parameters:
    -----------
    name : str
        Metric name/label
    value : str
        Metric value to display
    change : float, optional
        Percentage change for color-coding (-1 to 1 or percentage)
    chart : str, optional
        Base64-encoded chart image URI
    subtitle : str, optional
        Additional subtitle text
    
    Returns:
    --------
    str
        HTML string for the card
    """
    # Format change indicator
    change_html = ''
    if change is not None:
        color = "#10b981" if change > 0 else "#ef4444" if change < 0 else "#6b7280"
        arrow = "▲" if change > 0 else "▼" if change < 0 else "●"
        change_html = f'<div style="color: {color}; font-size: 12px; margin-top: 4px;">{arrow} {abs(change):.1f}%</div>'
    
    # Format subtitle
    subtitle_html = f'<div style="color: #9ca3af; font-size: 11px; margin-top: 2px;">{subtitle}</div>' if subtitle else ''
    
    # Style template
    card_style = 'background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'
    
    # Content template
    content = f'<div style="color: #6b7280; font-size: 13px; font-weight: 500; margin-bottom: 8px;">{name}</div><div style="color: #111827; font-size: 24px; font-weight: 700;">{value}</div>{subtitle_html}{change_html}'
    
    # If chart present, display side-by-side
    if chart:
        return f'<div style="{card_style} display: flex; justify-content: space-between; align-items: center; gap: 12px;"><div style="flex: 1;">{content}</div><div style="flex-shrink: 0;"><img src="{chart}" style="width: 70px; height: 40px;"></div></div>'
    else:
        return f'<div style="{card_style}">{content}</div>'


def create_performance_card_html(name, value):
    """
    Create an HTML card for performance/risk metrics.
    
    Parameters:
    -----------
    name : str
        Metric name/label
    value : float or None
        Metric value (None for N/A)
    
    Returns:
    --------
    str
        HTML string for the card
    """
    if value is None:
        return create_card_html(name, 'N/A')
    
    # Determine color and symbol
    color = "#10b981" if value > 0 else "#ef4444" if value < 0 else "#6b7280"
    symbol = "▲" if value > 0 else "▼" if value < 0 else "●"
    
    # Format value - check if it should be percentage or decimal
    if any(k in name for k in ['%', 'Return', 'Volatility', 'Drawdown', 'VaR']):
        formatted_value = f'{symbol} {abs(value):.1f}%'
    else:
        formatted_value = f'{symbol} {abs(value):.2f}'
    
    card_style = 'background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'
    return f'<div style="{card_style}"><div style="color: #6b7280; font-size: 13px; font-weight: 500; margin-bottom: 8px;">{name}</div><div style="color: {color}; font-size: 24px; font-weight: 700;">{formatted_value}</div></div>'


def display_stock_metrics(selected_stock, data, ticker_obj, recent_history, info_dict):
    """
    Display comprehensive stock metrics dashboard with cards and mini charts.
    
    Parameters:
    -----------
    selected_stock : str
        Stock ticker symbol
    data : pd.DataFrame
        Historical price data with technical indicators
    ticker_obj : yfinance.Ticker
        yfinance ticker object
    recent_history : pd.DataFrame
        Recent price history (e.g., 5 days)
    info_dict : dict
        Stock info dictionary from yfinance
    
    Returns:
    --------
    None (displays HTML output)
    """
    from IPython.display import HTML, display
    
    # ===== Extract historical values for mini charts =====
    quarterly_data = {}
    if isinstance(data, pd.DataFrame) and not data.empty and isinstance(data.index, pd.DatetimeIndex):
        qd = data.resample('Q').last()
        qd = qd.iloc[-5:-1] if len(qd) >= 5 else qd
        quarterly_data = {c: qd[c].tolist() for c in ['Close', 'RSI', 'Volume'] if c in qd.columns}
    
    hp = quarterly_data.get('Close', [])  # historical prices
    hr = quarterly_data.get('RSI', [])    # historical RSI
    hv = quarterly_data.get('Volume', []) # historical volume
    
    # ===== Extract current values =====
    cp, ld = get_current_price_and_date(recent_history, data, info_dict)
    
    # Day-to-day price change
    pc1 = None
    if isinstance(recent_history, pd.DataFrame) and len(recent_history) >= 2:
        p0 = safe_float(recent_history['Close'].iloc[-2])
        p1 = safe_float(recent_history['Close'].iloc[-1])
        pc1 = (p1 - p0) / p0 * 100 if p0 else None
    
    # ===== Dividend Yield =====
    dy = info_dict.get('dividendYield')
    dyp = (dy * 100 if dy and dy < 0.5 else dy if dy and dy < 20 else None) if dy else None
    
    # ===== Main metrics =====
    m = {
        k: info_dict.get(v)
        for k, v in [
            ('market_cap', 'marketCap'),
            ('pe_ratio', 'trailingPE'),
            ('eps', 'trailingEps'),
            ('pb_ratio', 'priceToBook'),
            ('week_52_high', 'fiftyTwoWeekHigh'),
            ('week_52_low', 'fiftyTwoWeekLow'),
            ('volume', 'volume'),
            ('beta', 'beta'),
            ('ebitda', 'ebitda'),
            ('book_value', 'bookValue'),
            ('shares_outstanding', 'sharesOutstanding'),
            ('profit_margin', 'profitMargins'),
            ('operating_margin', 'operatingMargins'),
            ('return_on_assets', 'returnOnAssets'),
            ('return_on_equity', 'returnOnEquity'),
        ]
    }
    
    m.update({
        'dividend_yield': dyp,
        'company_name': info_dict.get('longName') or info_dict.get('shortName') or selected_stock,
        'sector': info_dict.get('sector', 'N/A'),
        'industry': info_dict.get('industry', 'N/A'),
    })
    
    # ===== Technical indicators =====
    tech = {}
    if isinstance(data, pd.DataFrame):
        for ind, col in [('rsi', 'RSI'), ('macd', 'MACD'), ('stochastic_k', '%K'), ('volume', 'Volume')]:
            if col in data.columns:
                s = data[col].dropna()
                if not s.empty:
                    tech[ind] = safe_float(s.iloc[-1])
        
        if all(c in data.columns for c in ['Close', 'Upper_Band', 'Lower_Band']):
            cv = safe_float(data['Close'].iloc[-1])
            u = safe_float(data['Upper_Band'].iloc[-1])
            l = safe_float(data['Lower_Band'].iloc[-1])
            if cv and u and l and u != l:
                tech['bollinger_b'] = (cv - l) / (u - l)
    
    # ===== Performance metrics (30d, 90d, 6m, 1y) =====
    pf = {}
    if isinstance(data, pd.DataFrame) and not data.empty and 'Close' in data.columns:
        cl = data['Close']
        for n, d in [('30d', 30), ('90d', 90), ('6m', 126), ('1y', 252)]:
            if len(data) >= d:
                pf[n] = ((cl.iloc[-1] - cl.iloc[-d]) / cl.iloc[-d]) * 100
            elif len(data) > 126 and n == '1y':
                pf[n] = ((cl.iloc[-1] - cl.iloc[0]) / cl.iloc[0]) * 100
            else:
                pf[n] = None
    
    # ===== Risk metrics =====
    rk = {k: None for k in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'annualized_volatility', 'annualized_return', 'max_drawdown', 'var_95']}
    
    if isinstance(data, pd.DataFrame) and not data.empty and 'Close' in data.columns and len(data) >= 30:
        rt = data['Close'].pct_change().dropna()
        if not rt.empty:
            tr = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
            yr = len(data) / 252
            
            rk['annualized_return'] = ((1 + tr) ** (1 / yr) - 1) * 100 if yr > 0 else None
            rk['annualized_volatility'] = rt.std() * np.sqrt(252) * 100
            
            if rk['annualized_volatility'] and rk['annualized_volatility'] > 0:
                rk['sharpe_ratio'] = (rk['annualized_return'] or 0) / rk['annualized_volatility']
            
            cm = (1 + rt).cumprod()
            dd = (cm - cm.expanding().max()) / cm.expanding().max()
            rk['max_drawdown'] = dd.min() * 100
            
            dr = rt[rt < 0]
            if len(dr) > 0:
                ds = dr.std() * np.sqrt(252)
                if ds > 0:
                    rk['sortino_ratio'] = (rk['annualized_return'] or 0) / (ds * 100)
            
            if rk['max_drawdown'] and rk['max_drawdown'] < 0 and rk['annualized_return']:
                rk['calmar_ratio'] = rk['annualized_return'] / abs(rk['max_drawdown'])
            
            rk['var_95'] = abs(rt.quantile(0.05)) * 100
    
    # ===== Quarterly earnings data =====
    qe = getattr(ticker_obj, 'earnings_dates', None)
    qeps = qe['EPS'].dropna().tail(4).tolist() if qe is not None and not qe.empty and 'EPS' in qe.columns else []
    
    # ===== Year-over-year changes =====
    chg = {}
    if m['market_cap'] and hp and cp and cp != 0:
        m1y = (data['Close'].iloc[-250] if isinstance(data, pd.DataFrame) and not data.empty and len(data) >= 250 else hp[0]) * (m['market_cap'] / cp)
        if m1y and m1y != 0:
            chg['market_cap'] = ((m['market_cap'] - m1y) / m1y) * 100
    
    if m['eps'] and qeps and qeps[0] != 0:
        chg['eps'] = ((m['eps'] - qeps[0]) / qeps[0]) * 100
    
    # ===== Mini charts =====
    cht = {}
    if hp and cp:
        cht['price'] = create_mini_chart(hp, cp)
    if m['market_cap'] and hp and cp and cp != 0:
        cht['market_cap'] = create_mini_chart([p * (m['market_cap'] / cp) for p in hp], m['market_cap'])
    if m['pe_ratio'] and qeps:
        qpe = [cp / e for e in qeps if e and e != 0]
        if qpe:
            cht['pe'] = create_mini_chart(qpe, m['pe_ratio'])
    if m['eps'] and qeps:
        cht['eps'] = create_mini_chart(qeps, m['eps'])
    if m['pb_ratio'] and hp and m['book_value'] and m['book_value'] != 0:
        cht['pb'] = create_mini_chart([p / m['book_value'] for p in hp], m['pb_ratio'])
    if tech.get('rsi') and hr:
        cht['rsi'] = create_mini_chart(hr, tech['rsi'])
    if tech.get('volume') and hv:
        cht['volume'] = create_mini_chart(hv, tech['volume'])
    
    # ===== Build HTML =====
    html_parts = [
        f'''<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px;">
            <h2 style="color: #111827; margin-bottom: 8px;">{m['company_name']} ({selected_stock})</h2>
            <div style="color: #6b7280; font-size: 14px; margin-bottom: 24px;">{m['sector']} • {m['industry']}</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px;">'''
    ]
    
    # Add metric cards
    metrics = [
        ('Current Price', f'${cp:.2f}' if cp else 'N/A', pc1, cht.get('price'), None),
        ('Market Cap', f'${humanize_number(m["market_cap"])}' if m['market_cap'] else 'N/A', chg.get('market_cap'), cht.get('market_cap'), None),
        ('Shares Outstanding', humanize_number(m['shares_outstanding']) if m['shares_outstanding'] else 'N/A', None, None, None),
        ('P/E Ratio', f'{m["pe_ratio"]:.2f}' if m['pe_ratio'] else 'N/A', None, cht.get('pe'), None),
        ('EPS', f'${m["eps"]:.2f}' if m['eps'] else 'N/A', chg.get('eps'), cht.get('eps'), None),
        ('EBITDA', f'${humanize_number(m["ebitda"])}' if m['ebitda'] else 'N/A', None, None, None),
        ('Dividend Yield', f'{m["dividend_yield"]:.2f}%' if m['dividend_yield'] else 'N/A', None, None, None),
        ('P/B Ratio', f'{m["pb_ratio"]:.2f}' if m['pb_ratio'] else 'N/A', None, cht.get('pb'), None),
        ('52W High', f'${m["week_52_high"]:.2f}' if m['week_52_high'] else 'N/A', None, None, None),
        ('52W Low', f'${m["week_52_low"]:.2f}' if m['week_52_low'] else 'N/A', None, None, None),
        ('Volume', humanize_number(m['volume'] or tech.get('volume')) if (m['volume'] or tech.get('volume')) else 'N/A', None, cht.get('volume'), None),
        ('RSI', f'{tech["rsi"]:.1f}' if tech.get('rsi') else 'N/A', None, cht.get('rsi'), None),
        ('MACD', f'{tech["macd"]:.3f}' if tech.get('macd') is not None else 'N/A', None, None, None),
        ('Bollinger %B', f'{tech["bollinger_b"]:.2f}' if tech.get('bollinger_b') is not None else 'N/A', None, None, None),
        ('Stochastic %K', f'{tech["stochastic_k"]:.1f}' if tech.get('stochastic_k') is not None else 'N/A', None, None, None),
        ('Beta', f'{m["beta"]:.2f}' if m['beta'] else 'N/A', None, None, '(vs S&P 500)'),
    ]
    
    for name, value, change, chart, subtitle in metrics:
        html_parts.append(create_card_html(name, value, change, chart, subtitle))
    
    html_parts.append('</div><h3 style="color: #111827; margin-top: 24px; margin-bottom: 16px;">Profitability Metrics</h3><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px;">')
    
    for name, key in [('Profit Margin', 'profit_margin'), ('Operating Margin', 'operating_margin'), ('Return on Assets', 'return_on_assets'), ('Return on Equity', 'return_on_equity')]:
        val = m.get(key)
        if val is not None:
            val_pct = val * 100
            html_parts.append(create_card_html(name, f'{val_pct:.2f}%'))
        else:
            html_parts.append(create_card_html(name, 'N/A'))
    
    html_parts.append('</div><h3 style="color: #111827; margin-top: 24px; margin-bottom: 16px;">Risk Metrics</h3><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px;">')
    
    for name, key in [('Sharpe Ratio', 'sharpe_ratio'), ('Sortino Ratio', 'sortino_ratio'), ('Calmar Ratio', 'calmar_ratio'), ('Annualized Volatility', 'annualized_volatility'), ('Annualized Return', 'annualized_return'), ('Max Drawdown', 'max_drawdown'), ('VaR (95%)', 'var_95')]:
        html_parts.append(create_performance_card_html(name, rk.get(key)))
    
    html_parts.append('</div><h3 style="color: #111827; margin-top: 24px; margin-bottom: 16px;">Performance</h3><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px;">')
    
    for name, key in [('30-Day Return', '30d'), ('90-Day Return', '90d'), ('6-Month Return', '6m'), ('1-Year Return', '1y')]:
        html_parts.append(create_performance_card_html(name, pf.get(key)))
    
    html_parts.append('</div></div>')
    
    display(HTML(''.join(html_parts)))


def calculate_capm_metrics(selected_stock: str, weeks_to_fetch: int = 150, capm_period: str = '4y') -> Dict:
    """
    Calculate CAPM (Capital Asset Pricing Model) metrics for a stock.
    
    Parameters:
    -----------
    selected_stock : str
        Stock ticker symbol
    weeks_to_fetch : int
        Number of weeks of data to use for beta calculation (default: 150)
    capm_period : str
        Period to fetch data for (default: '4y')
    
    Returns:
    --------
    dict
        Dictionary containing CAPM metrics:
        - beta: Beta coefficient (systematic risk)
        - alpha: Jensen's alpha (excess return)
        - risk_free_rate: Risk-free rate (%)
        - market_return_annual: Annual market return (%)
        - stock_return_annual: Annual stock return (%)
        - expected_return_capm: Expected return from CAPM (%)
        - r_squared: R-squared value
        - correlation: Correlation with market
        - aligned_data: DataFrame with aligned returns for visualization
    """
    # Fetch stock data for CAPM period
    stock_ticker = yf.Ticker(selected_stock)
    stock_data_capm = stock_ticker.history(period=capm_period)
    
    # Fetch market data (S&P 500 as market proxy)
    market_ticker = yf.Ticker("^GSPC")
    market_data = market_ticker.history(period=capm_period)
    
    # Get risk-free rate (10-year Treasury yield)
    treasury = yf.Ticker("^TNX")
    treasury_data = treasury.history(period="5d")
    risk_free_rate = treasury_data['Close'].iloc[-1] / 100 if not treasury_data.empty else 0.04
    
    # Merge price data
    merged_prices = pd.DataFrame({
        'Stock': stock_data_capm['Close'],
        'Market': market_data['Close']
    }).dropna()
    
    # Resample to weekly data (end of week - Friday)
    weekly_prices = merged_prices.resample('W-FRI').last().dropna()
    
    # Limit to exactly weeks_to_fetch (or available data if less)
    if len(weekly_prices) > weeks_to_fetch:
        weekly_prices = weekly_prices.iloc[-weeks_to_fetch:]
    
    # Calculate weekly returns
    stock_returns = weekly_prices['Stock'].pct_change().dropna()
    market_returns = weekly_prices['Market'].pct_change().dropna()
    
    # Create aligned returns dataframe
    aligned_data = pd.DataFrame({
        'Stock': stock_returns,
        'Market': market_returns
    }).dropna()
    
    if len(aligned_data) < 20:
        raise ValueError(f"Insufficient data for CAPM calculation. Only {len(aligned_data)} weekly returns available. Need at least 20.")
    
    # Calculate Beta using linear regression (more robust)
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(aligned_data['Market'], aligned_data['Stock'])
    beta = slope
    
    # Calculate returns (annualized using 52 weeks)
    market_return_annual = aligned_data['Market'].mean() * 52
    stock_return_annual = aligned_data['Stock'].mean() * 52
    
    # Calculate expected return and alpha
    expected_return_capm = risk_free_rate + beta * (market_return_annual - risk_free_rate)
    alpha = stock_return_annual - expected_return_capm
    
    # Calculate R-squared
    correlation = aligned_data['Stock'].corr(aligned_data['Market'])
    r_squared = correlation ** 2
    
    return {
        'beta': beta,
        'alpha': alpha,
        'risk_free_rate': risk_free_rate,
        'market_return_annual': market_return_annual,
        'stock_return_annual': stock_return_annual,
        'expected_return_capm': expected_return_capm,
        'r_squared': r_squared,
        'correlation': correlation,
        'aligned_data': aligned_data,
        'weeks_used': len(aligned_data)
    }


def display_capm_analysis(selected_stock: str, capm_metrics: Dict) -> None:
    """
    Display CAPM analysis metrics and interpretation.
    
    Parameters:
    -----------
    selected_stock : str
        Stock ticker symbol (for display)
    capm_metrics : dict
        Dictionary from calculate_capm_metrics()
    """
    beta = capm_metrics['beta']
    alpha = capm_metrics['alpha']
    risk_free_rate = capm_metrics['risk_free_rate']
    market_return_annual = capm_metrics['market_return_annual']
    stock_return_annual = capm_metrics['stock_return_annual']
    expected_return_capm = capm_metrics['expected_return_capm']
    r_squared = capm_metrics['r_squared']
    correlation = capm_metrics['correlation']
    weeks_used = capm_metrics['weeks_used']
    
    print(f"\n{'='*60}")
    print(f"CAPM ANALYSIS FOR {selected_stock}")
    print(f"{'='*60}")
    print(f"Risk-Free Rate (10Y Treasury):    {risk_free_rate*100:.2f}%")
    print(f"Market Return (Annualized):        {market_return_annual*100:.2f}%")
    print(f"Stock Return (Annualized):         {stock_return_annual*100:.2f}%")
    print(f"\nBeta:                              {beta:.3f}")
    print(f"Alpha (Jensen's):                  {alpha*100:.2f}%")
    print(f"Expected Return (CAPM):            {expected_return_capm*100:.2f}%")
    print(f"R-squared:                         {r_squared:.3f}")
    print(f"Correlation with Market:           {correlation:.3f}")
    print(f"Weeks of Data Used:                {weeks_used}")
    print(f"{'='*60}\n")
    
    print("INTERPRETATION:")
    if beta > 1:
        print(f"• Beta > 1: Stock is MORE volatile than the market ({beta:.2f}x market risk)")
    elif beta < 1:
        print(f"• Beta < 1: Stock is LESS volatile than the market ({beta:.2f}x market risk)")
    else:
        print(f"• Beta ≈ 1: Stock moves in line with the market")
    
    if alpha > 0:
        print(f"• Positive Alpha: Stock outperformed expectations by {alpha*100:.2f}%")
    else:
        print(f"• Negative Alpha: Stock underperformed expectations by {abs(alpha)*100:.2f}%")
    
    print(f"• R²={r_squared:.1%}: {r_squared*100:.1f}% of stock movement explained by market")


def plot_capm_analysis(selected_stock: str, capm_metrics: Dict) -> None:
    """
    Plot CAPM analysis visualizations (scatter plot and Security Market Line).
    
    Parameters:
    -----------
    selected_stock : str
        Stock ticker symbol (for display)
    capm_metrics : dict
        Dictionary from calculate_capm_metrics()
    """
    import matplotlib.pyplot as plt
    
    beta = capm_metrics['beta']
    alpha = capm_metrics['alpha']
    risk_free_rate = capm_metrics['risk_free_rate']
    market_return_annual = capm_metrics['market_return_annual']
    stock_return_annual = capm_metrics['stock_return_annual']
    expected_return_capm = capm_metrics['expected_return_capm']
    r_squared = capm_metrics['r_squared']
    aligned_data = capm_metrics['aligned_data']
    weeks_used = capm_metrics['weeks_used']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Stock vs Market Returns Scatter
    ax1 = axes[0]
    ax1.scatter(aligned_data['Market']*100, aligned_data['Stock']*100, alpha=0.5, s=20, color='blue')
    z = np.polyfit(aligned_data['Market'], aligned_data['Stock'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(aligned_data['Market'].min(), aligned_data['Market'].max(), 100)
    ax1.plot(x_line*100, p(x_line)*100, "r--", linewidth=2, label=f'Beta = {beta:.3f}')
    ax1.set_xlabel('Market Return (%)', fontsize=12)
    ax1.set_ylabel(f'{selected_stock} Return (%)', fontsize=12)
    ax1.set_title(f'{selected_stock} vs Market (S&P 500) Returns - {weeks_used} Weeks', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Security Market Line (SML)
    ax2 = axes[1]
    beta_range = np.linspace(-0.5, 2.5, 100)
    expected_returns = risk_free_rate + beta_range * (market_return_annual - risk_free_rate)
    ax2.plot(beta_range, expected_returns*100, 'b-', linewidth=2, label='Security Market Line (SML)')
    
    # Plot risk-free rate
    ax2.scatter(0, risk_free_rate*100, color='green', s=100, zorder=5, label=f'Risk-Free Rate ({risk_free_rate*100:.1f}%)')
    
    # Plot market
    ax2.scatter(1, market_return_annual*100, color='orange', s=100, zorder=5, label=f'Market (β=1)')
    
    # Plot the stock
    ax2.scatter(beta, stock_return_annual*100, color='red', s=150, marker='D', zorder=5, 
                label=f'{selected_stock} (β={beta:.2f})')
    
    # Plot expected return for the stock
    ax2.scatter(beta, expected_return_capm*100, color='purple', s=100, marker='x', zorder=5,
                label=f'Expected Return ({expected_return_capm*100:.1f}%)')
    
    # Add alpha visualization
    if abs(alpha) > 0.001:
        ax2.arrow(beta, expected_return_capm*100, 0, alpha*100, 
                  head_width=0.05, head_length=abs(alpha)*50, fc='green' if alpha > 0 else 'red',
                  ec='green' if alpha > 0 else 'red', linewidth=2, alpha=0.7)
        ax2.text(beta + 0.1, (expected_return_capm + alpha/2)*100, 
                 f'α={alpha*100:.1f}%', fontsize=10, fontweight='bold',
                 color='green' if alpha > 0 else 'red')
    
    ax2.set_xlabel('Beta (Systematic Risk)', fontsize=12)
    ax2.set_ylabel('Expected Return (%)', fontsize=12)
    ax2.set_title('Security Market Line (SML) - CAPM', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def format_holdings_markdown(holders_df: pd.DataFrame, shares_outstanding: Optional[int], title: str, top_n: int = 10) -> str:
    """
    Format holdings data as markdown with shares and percentages.
    
    Parameters:
    -----------
    holders_df : pd.DataFrame
        DataFrame with holdings data (columns: 'Holder' and 'Shares')
    shares_outstanding : int, optional
        Total shares outstanding for percentage calculation
    title : str
        Title for the holdings section
    top_n : int
        Number of top holdings to display (default: 10)
    
    Returns:
    --------
    str
        Formatted markdown string
    """
    if holders_df is None or holders_df.empty:
        return f"**{title}:** Not available"
    
    lines = [f"### {title}", ""]
    
    for i, row in enumerate(holders_df.head(top_n).itertuples(index=False), 1):
        try:
            # Extract holder name and shares from row
            holder = getattr(row, 'Holder', list(row)[0] if row else 'Unknown')
            shares = getattr(row, 'Shares', list(row)[1] if len(list(row)) > 1 else 0)
            
            if shares:
                shares = float(shares)
                shares_str = f"{humanize_number(shares)} shares"
                if shares_outstanding and shares_outstanding > 0:
                    shares_str += f" ({(shares/shares_outstanding)*100:.2f}%)"
            else:
                shares_str = 'N/A'
            
            lines.append(f"{i}. **{holder}** — {shares_str}")
        except:
            lines.append(f"{i}. **Unknown** — N/A")
    
    return '\n\n'.join(lines)


def display_company_holdings(selected_stock: str, ticker_obj) -> None:
    """
    Display company officers, institutional holdings, and mutual fund holdings.
    
    Parameters:
    -----------
    selected_stock : str
        Stock ticker symbol
    ticker_obj : yfinance.Ticker
        yfinance ticker object
    """
    info = ticker_obj.info or {}
    shares_outstanding = info.get('sharesOutstanding')
    
    # --- Company Officers ---
    try:
        officers = info.get('companyOfficers')
        display_structured_data_as_markdown(officers, "Company Officers", selected_stock)
    except Exception as e:
        display(Markdown(f"**Error displaying company officers:** {e}"))
    
    # --- Institutional Holdings ---
    try:
        holders = ticker_obj.institutional_holders
        if holders is None or holders.empty:
            raise ValueError("No institutional holdings data")
        
        held_pct = info.get('heldPercentInstitutions')
        header = f"### Top Institutional Holdings for {selected_stock}\n\n"
        header += f"**Percent held by Institutions:** {held_pct*100:.2f}%" if held_pct else "**Percent held by Institutions:** N/A"
        
        holdings_md = format_holdings_markdown(holders, shares_outstanding, "", 10)
        # Extract just the holdings list (skip the title)
        holdings_list = holdings_md.split('\n\n', 2)[-1] if '\n\n' in holdings_md else holdings_md
        display(Markdown(header + '\n\n' + holdings_list))
    except:
        display(Markdown(f"**Institutional Holdings:** Not available for {selected_stock}"))
    
    # --- Mutual Fund Holdings ---
    try:
        mf_holdings = ticker_obj.mutualfund_holders
        if mf_holdings is not None and not mf_holdings.empty:
            display(Markdown(format_holdings_markdown(mf_holdings, shares_outstanding, "Top Mutual Fund Holdings")))
    except Exception as e:
        display(Markdown(f"**Mutual Fund Holdings:** Not available for {selected_stock}"))


# =====================================================
# World Stock Market Map Functions
# =====================================================

def fetch_single_stock(country, ticker):
    """Fetch 5-day history and calculate daily change for a stock ticker."""
    import sys, io
    try:
        old_stderr, sys.stderr = sys.stderr, io.StringIO()
        hist = yf.Ticker(ticker).history(period='5d')
        sys.stderr = old_stderr
        if len(hist) >= 2:
            latest, prev = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
            return {'country': country, 'ticker': ticker, 'daily_change': ((latest - prev) / prev) * 100, 'close': latest}
    except:
        sys.stderr = old_stderr
    return None


def get_daily_changes(country_indices):
    """Fetch daily changes for multiple stock indices using concurrent requests."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single_stock, c, t): c for c, t in country_indices.items()}
        data = [f.result() for f in as_completed(futures) if f.result()]
    return pd.DataFrame(data)


def get_color(chg):
    """Return color code based on daily change percentage."""
    if chg <= -2:
        return '#8B0000'
    elif chg <= -1:
        return '#FF0000'
    elif chg < 0:
        return '#FF8C00'
    elif chg <= 1:
        return '#FFFF00'
    elif chg <= 2:
        return '#90EE90'
    else:
        return '#006400'


def create_world_map(df):
    """Create folium world map with stock market data overlay.
    
    Args:
        df: DataFrame with columns ['country', 'ticker', 'daily_change', 'close']
    
    Returns:
        folium.Map object
    """
    import folium
    
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron', 
                   zoom_control=False, scrollWheelZoom=False, doubleClickZoom=False, boxZoom=False, keyboard=False)
    
    world_geo = requests.get('https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json').json()
    
    folium.Choropleth(geo_data=world_geo, name='Stock Index Daily Change', data=df, columns=['country', 'daily_change'],
                      key_on='feature.properties.name', fill_color='RdYlGn', fill_opacity=0.8, line_opacity=0.3,
                      legend_name='Daily Change (%)', nan_fill_color='lightgray', nan_fill_opacity=0.4).add_to(m)
    
    for _, row in df.iterrows():
        for feature in world_geo['features']:
            if feature['properties']['name'] == row['country']:
                popup = f"<div style='font-family: Arial; width: 200px;'><b style='font-size: 14px;'>{row['country']}</b><br><hr style='margin: 5px 0;'><b>Index:</b> {row['ticker']}<br><b>Daily Change:</b> <span style='color: {'green' if row['daily_change'] >= 0 else 'red'}; font-size: 16px; font-weight: bold;'>{row['daily_change']:+.2f}%</span><br><b>Close:</b> {row['close']:.2f}</div>"
                color = get_color(row['daily_change'])
                folium.GeoJson({'type': 'FeatureCollection', 'features': [feature]},
                              style_function=lambda x, c=color: {'fillColor': c, 'color': 'black', 'weight': 1, 'fillOpacity': 0.8},
                              popup=folium.Popup(popup, max_width=250), name=row['country']).add_to(m)
                break
    
    folium.LayerControl().add_to(m)
    return m


def save_world_map(world_map, output_dir='assets'):
    """Save world map to HTML file outside notebooks folder.
    
    Args:
        world_map: folium.Map object
        output_dir: Directory to save the map (relative to project root)
    
    Returns:
        Path to saved file
    """
    import os
    from pathlib import Path
    
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    save_path = project_root / output_dir
    save_path.mkdir(exist_ok=True)
    
    filename = save_path / 'stock_index_world_map.html'
    world_map.save(str(filename))
    
    return str(filename)