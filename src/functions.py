import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator, DayLocator
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
    for df in [recent_hist, data]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            return safe_float(df['Close'].iloc[-1]), df.index[-1]
    return info.get('currentPrice') or info.get('regularMarketPrice'), None


def humanize_number(x):
    """Convert large numbers to human-readable format."""
    try:
        x = float(x)
    except Exception:
        return "N/A"
    if pd.isna(x):
        return "N/A"
    abs_x = abs(x)
    scales = [(1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')]
    for scale, suffix in scales:
        if abs_x >= scale:
            return f"{x/scale:.2f}{suffix}"
    return f"{x:.0f}"

def get_historical_values(data, column='Close', periods=[30, 90, 180, 365]):
    """Get historical values for specified periods (in days) for any column."""
    if not (isinstance(data, pd.DataFrame) and not data.empty and column in data.columns and len(data) > 1):
        return []
    clean_data = data[column].dropna()
    if clean_data.empty:
        return []
    return [safe_float(clean_data.iloc[-p if len(clean_data) >= p else 0]) for p in periods]

# Utility functions
def get_ticker(symbol: str):
    """Get or create a yfinance Ticker object."""
    global ticker
    if ticker is not None:
        return ticker
    ticker = yf.Ticker(symbol)
    return ticker

def fetch_stock_data(symbol: str, period: str = '1y'):
    """Fetch stock data, using cache if available."""
    cache_key = f"{symbol}_{period}"
    if cache_key not in stocks:
        data = get_ticker(symbol).history(period=period)
        if not data.empty:
            stocks[cache_key] = data
    return stocks.get(cache_key, pd.DataFrame())

def setup_date_formatting(ax, period):
    """Set up appropriate date formatting for x-axis based on period."""
    formatters = {
        ('1d', '5d'): (DateFormatter('%H:%M'), DayLocator()),
        ('1wk', '1mo'): (DateFormatter('%b %d'), DayLocator(interval=7)),
        ('3mo', '6mo'): (DateFormatter('%b %d'), MonthLocator(interval=1)),
        ('1y', '2y'): (DateFormatter('%b %Y'), MonthLocator(interval=3)),
        ('5y', '10y', 'max'): (DateFormatter('%Y'), YearLocator()),
    }
    default = (DateFormatter('%b %Y'), MonthLocator(interval=3))
    
    for periods, (fmt, locator) in formatters.items():
        if period in periods:
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_major_locator(locator)
            return
    ax.xaxis.set_major_formatter(default[0])

def make_cnn_api_request(api_url: str, timeout: int = 10) -> Optional[requests.Response]:
    """Make a request to CNN Business API with retry logic and multiple headers."""
    headers_list = [
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36", "Accept": "application/json", "Referer": "https://www.cnn.com/"},
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept": "application/json", "Referer": "https://www.cnn.com/"},
        {"User-Agent": "curl/7.88.1", "Accept": "*/*"},
    ]
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])))
    
    for h in headers_list:
        try:
            resp = session.get(api_url, headers=h, timeout=timeout)
            if resp.status_code == 200:
                return resp
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return None

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
                        extra_fields = {'pctHeld': '{:.2f}%', 'value': '${:,.0f}'}
                        for field in ['since', 'startDate', 'appointed', 'age', 'years', 'pctHeld', 'value']:
                            if field in item and item[field] is not None:
                                if field in extra_fields:
                                    extras.append(extra_fields[field].format(item[field]))
                                else:
                                    label = field.replace('startDate', 'since')
                                    extras.append(f"{label}: {item[field]}")

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
    close = df['Close']
    
    # Moving averages & RSI
    df['MA_20'], df['MA_50'] = close.rolling(20).mean(), close.rolling(50).mean()
    delta = close.diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    ema_12, ema_26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df['MACD'] = macd
    df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = macd - df['MACD_Signal']
    
    # Bollinger Bands
    sma_20, std_20 = close.rolling(20).mean(), close.rolling(20).std()
    df['Upper_Band'], df['Lower_Band'] = sma_20 + 2*std_20, sma_20 - 2*std_20
    
    # Stochastic Oscillator
    high_14, low_14 = df['High'].rolling(14).max(), df['Low'].rolling(14).min()
    df['%K'] = 100 * ((close - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(3).mean()
    
    # On-Balance Volume
    df['OBV'] = (df['Volume'] * ((close > close.shift(1)).astype(int) * 2 - 1)).cumsum()
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
    """Create an HTML card for metric display."""
    change_html = ''
    if change is not None:
        color = "#10b981" if change > 0 else "#ef4444" if change < 0 else "#6b7280"
        arrow = "▲" if change > 0 else "▼" if change < 0 else "●"
        change_html = f'<div style="color: {color}; font-size: 12px; margin-top: 4px;">{arrow} {abs(change):.1f}%</div>'
    
    subtitle_html = f'<div style="color: #9ca3af; font-size: 11px; margin-top: 2px;">{subtitle}</div>' if subtitle else ''
    card_style = 'background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'
    content = f'<div style="color: #6b7280; font-size: 13px; font-weight: 500; margin-bottom: 8px;">{name}</div><div style="color: #111827; font-size: 24px; font-weight: 700;">{value}</div>{subtitle_html}{change_html}'
    
    if chart:
        return f'<div style="{card_style} display: flex; justify-content: space-between; align-items: center; gap: 12px;"><div style="flex: 1;">{content}</div><div style="flex-shrink: 0;"><img src="{chart}" style="width: 70px; height: 40px;"></div></div>'
    return f'<div style="{card_style}">{content}</div>'


def create_performance_card_html(name, value):
    """Create an HTML card for performance/risk metrics."""
    if value is None:
        return create_card_html(name, 'N/A')
    
    color = "#10b981" if value > 0 else "#ef4444" if value < 0 else "#6b7280"
    symbol = "▲" if value > 0 else "▼" if value < 0 else "●"
    formatted_value = f'{symbol} {abs(value):.1f}%' if any(k in name for k in ['%', 'Return', 'Volatility', 'Drawdown', 'VaR']) else f'{symbol} {abs(value):.2f}'
    
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
    """Format holdings data as markdown with shares and percentages."""
    if holders_df is None or holders_df.empty:
        return f"**{title}:** Not available"
    
    lines = [f"### {title}", ""]
    for i, row in enumerate(holders_df.head(top_n).itertuples(index=False), 1):
        try:
            holder = getattr(row, 'Holder', list(row)[0] if row else 'Unknown')
            shares = float(getattr(row, 'Shares', list(row)[1] if len(list(row)) > 1 else 0) or 0)
            shares_str = f"{humanize_number(shares)} shares"
            if shares_outstanding and shares > 0:
                shares_str += f" ({(shares/shares_outstanding)*100:.2f}%)"
            lines.append(f"{i}. **{holder}** — {shares_str if shares else 'N/A'}")
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
    """Fetch 2-day history and calculate daily change for a stock ticker (optimized for speed)."""
    try:
        hist = yf.Ticker(ticker).history(period='2d', timeout=10)
        if len(hist) >= 2:
            latest, prev = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
            return {'country': country, 'ticker': ticker, 'daily_change': ((latest - prev) / prev) * 100, 'close': latest}
    except Exception:
        pass
    return None


# Global cache for world map data (expires after 1 hour)
_world_map_cache = {'data': None, 'timestamp': None}

def get_daily_changes(country_indices, use_cache=True):
    """Fetch daily changes for multiple stock indices using concurrent requests (optimized).
    
    Args:
        country_indices: Dict of country names to ticker symbols
        use_cache: If True, use cached data if available (< 1 hour old)
    
    Returns:
        DataFrame with country, ticker, daily_change, close columns
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Check cache first
    if use_cache and _world_map_cache['data'] is not None:
        elapsed = time.time() - _world_map_cache['timestamp']
        if elapsed < 3600:  # Cache valid for 1 hour
            return _world_map_cache['data'].copy()
    
    # Increased max_workers from 10 to 20 for better parallelization
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_single_stock, c, t): c for c, t in country_indices.items()}
        data = [f.result() for f in as_completed(futures) if f.result()]
    
    df = pd.DataFrame(data)
    
    # Update cache
    _world_map_cache['data'] = df
    _world_map_cache['timestamp'] = time.time()
    
    return df


def get_color(chg):
    """Return color code based on daily change percentage."""
    colors = [(-float('inf'), '#8B0000'), (-2, '#FF0000'), (-1, '#FF8C00'), (0, '#FFFF00'), (1, '#90EE90')]
    return next((c for t, c in reversed(colors) if chg > t), '#006400')


def create_world_map(df):
    """Create folium world map with stock market data overlay (optimized).
    
    Args:
        df: DataFrame with columns ['country', 'ticker', 'daily_change', 'close']
    
    Returns:
        folium.Map object
    """
    import folium
    
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron', 
                   zoom_control=False, scrollWheelZoom=False, doubleClickZoom=False, boxZoom=False, keyboard=False)
    
    # Fetch GeoJSON with timeout and better error handling
    try:
        session = requests.Session()
        response = session.get('https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json', timeout=10)
        world_geo = response.json()
    except Exception as e:
        print(f"Warning: Could not fetch GeoJSON: {e}. Map may be incomplete.")
        return m
    
    folium.Choropleth(geo_data=world_geo, name='Stock Index Daily Change', data=df, columns=['country', 'daily_change'],
                      key_on='feature.properties.name', fill_color='RdYlGn', fill_opacity=0.8, line_opacity=0.3,
                      legend_name='Daily Change (%)', nan_fill_color='lightgray', nan_fill_opacity=0.4).add_to(m)
    
    # Pre-build lookup dict for O(1) access instead of O(n) iteration per country
    country_lookup = {feature['properties']['name']: feature for feature in world_geo.get('features', [])}
    
    for _, row in df.iterrows():
        feature = country_lookup.get(row['country'])
        if feature:
            popup = f"<div style='font-family: Arial; width: 200px;'><b style='font-size: 14px;'>{row['country']}</b><br><hr style='margin: 5px 0;'><b>Index:</b> {row['ticker']}<br><b>Daily Change:</b> <span style='color: {'green' if row['daily_change'] >= 0 else 'red'}; font-size: 16px; font-weight: bold;'>{row['daily_change']:+.2f}%</span><br><b>Close:</b> {row['close']:.2f}</div>"
            color = get_color(row['daily_change'])
            folium.GeoJson({'type': 'FeatureCollection', 'features': [feature]},
                          style_function=lambda x, c=color: {'fillColor': c, 'color': 'black', 'weight': 1, 'fillOpacity': 0.8},
                          popup=folium.Popup(popup, max_width=250), name=row['country']).add_to(m)
    
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


# =============================================================================
# FRED/Economic Data Functions
# =============================================================================

def fetch_fred_series(fred, series_id, start_date=None, use_cache=True):
    """Fetch FRED series data with aggressive caching.
    
    Args:
        fred: FRED API client instance
        series_id: FRED series ID string
        start_date: Optional start date for filtering (YYYY-MM-DD format)
        use_cache: Use in-memory cache to avoid repeated API calls
    
    Returns:
        pandas.Series with filtered data, or None if fetch fails
    """
    import pandas as pd
    
    # In-memory cache for series data
    if not hasattr(fetch_fred_series, '_cache'):
        fetch_fred_series._cache = {}
    
    cache_key = f"{series_id}_{start_date}"
    
    if use_cache and cache_key in fetch_fred_series._cache:
        return fetch_fred_series._cache[cache_key].copy()
    
    try:
        data = fred.get_series(series_id)
        if start_date:
            data = data.loc[start_date:]
        
        if use_cache:
            fetch_fred_series._cache[cache_key] = data.copy()
        
        return data
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return None


def add_recession_shading(ax, fred, start_date=None, use_cache=True):
    """Add grey shading to plot for US recession periods with caching.
    
    Args:
        ax: matplotlib axes object
        fred: FRED API client instance
        start_date: Optional start date for filtering recession data
        use_cache: Use cached recession data to avoid repeated API calls
    """
    # Cache recession data globally since it's used frequently
    if not hasattr(add_recession_shading, '_recession_cache'):
        add_recession_shading._recession_cache = {}
    
    try:
        cache_key = f"USREC_{start_date}"
        
        if use_cache and cache_key in add_recession_shading._recession_cache:
            starts, ends = add_recession_shading._recession_cache[cache_key]
        else:
            recession = fred.get_series('USREC')
            if start_date:
                recession = recession.loc[start_date:]
            
            # Vectorized approach for better performance
            in_recession = (recession == 1).astype(int)
            changes = in_recession.diff()
            
            starts = recession.index[changes == 1]
            ends = recession.index[changes == -1]
            
            # Handle edge cases
            if len(recession) > 0 and recession.iloc[0] == 1:
                starts = pd.DatetimeIndex([recession.index[0]]).union(starts)
            if len(recession) > 0 and recession.iloc[-1] == 1:
                ends = ends.union(pd.DatetimeIndex([recession.index[-1]]))
            
            if use_cache:
                add_recession_shading._recession_cache[cache_key] = (starts, ends)
        
        # Draw shading
        for start, end in zip(starts, ends):
            ax.axvspan(start, end, color='grey', alpha=0.3, zorder=0)
            
    except Exception as e:
        print(f"Could not add recession shading: {e}")


def plot_fred_series(fred, series_id, start_date=None, ax=None, title=None, 
                     ylabel=None, add_recessions=True, **plot_kwargs):
    """Plot a single FRED series with optional recession shading.
    
    Args:
        fred: FRED API client instance
        series_id: FRED series ID string
        start_date: Optional start date for filtering
        ax: Optional matplotlib axes (creates new figure if None)
        title: Optional plot title (fetches from FRED if None)
        ylabel: Optional y-axis label (fetches from FRED if None)
        add_recessions: Whether to add recession shading
        **plot_kwargs: Additional arguments passed to plot()
    
    Returns:
        matplotlib axes object
    """
    import matplotlib.pyplot as plt
    
    # Fetch data
    data = fetch_fred_series(fred, series_id, start_date)
    if data is None:
        return None
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get series info for labels if not provided
    if title is None or ylabel is None:
        try:
            info = fred.get_series_info(series_id)
            if title is None:
                title = info.get('title', series_id)
            if ylabel is None:
                ylabel = info.get('units', 'Value')
        except:
            title = title or series_id
            ylabel = ylabel or 'Value'
    
    # Plot
    ax.plot(data.index, data.values, **plot_kwargs)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # Add recession shading
    if add_recessions:
        add_recession_shading(ax, fred, start_date)
    
    return ax


def calculate_yoy_change(data, periods=12):
    """Calculate year-over-year percentage change.
    
    Args:
        data: pandas Series
        periods: Number of periods for change calculation (default 12 for monthly data)
    
    Returns:
        pandas Series with percentage changes
    """
    return data.pct_change(periods) * 100


def plot_inflation_rate(fred, series_id, start_date=None, ax=None, title=None, 
                        add_recessions=True, **plot_kwargs):
    """Plot year-over-year inflation rate for a price index series.
    
    Args:
        fred: FRED API client instance
        series_id: FRED series ID for price index (e.g., 'CPIAUCSL')
        start_date: Optional start date for filtering
        ax: Optional matplotlib axes
        title: Optional plot title
        add_recessions: Whether to add recession shading
        **plot_kwargs: Additional arguments passed to plot()
    
    Returns:
        matplotlib axes object
    """
    import matplotlib.pyplot as plt
    
    # Fetch data
    data = fetch_fred_series(fred, series_id, start_date)
    if data is None:
        return None
    
    # Calculate inflation rate
    inflation_rate = calculate_yoy_change(data, periods=12)
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get title if not provided
    if title is None:
        try:
            info = fred.get_series_info(series_id)
            base_title = info.get('title', series_id)
            title = f'{base_title} - YoY % Change'
        except:
            title = f'{series_id} - YoY % Change'
    
    # Plot
    ax.plot(inflation_rate.index, inflation_rate.values, **plot_kwargs)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Year-over-Year % Change')
    ax.grid(True, alpha=0.3)
    
    # Add recession shading
    if add_recessions:
        add_recession_shading(ax, fred, start_date)
    
    return ax


def fetch_multiple_series_concurrent(fred, series_dict, start_date=None, max_workers=20):
    """Fetch multiple FRED series concurrently with aggressive parallelization.
    
    Args:
        fred: FRED API client instance
        series_dict: Dict mapping series IDs to display names
        start_date: Optional start date for filtering
        max_workers: Maximum number of concurrent workers (default 20 for speed)
    
    Returns:
        Dict mapping display names to pandas Series (None for failed fetches)
    """
    import concurrent.futures
    
    def fetch_single(series_id):
        try:
            data = fred.get_series(series_id)
            if start_date:
                data = data.loc[start_date:]
            return data
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return None
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all at once for maximum parallelization
        future_to_series = {executor.submit(fetch_single, sid): (sid, name) 
                           for sid, name in series_dict.items()}
        
        for future in concurrent.futures.as_completed(future_to_series):
            sid, name = future_to_series[future]
            results[name] = future.result()
    
    return results


def plot_sector_inflation_grid(fred, sector_series_dict, start_date=None, 
                               figsize=(15, 15), add_recessions=True):
    """Plot multiple sector inflation rates in a grid layout.
    
    Args:
        fred: FRED API client instance
        sector_series_dict: Dict mapping series IDs to sector names
        start_date: Optional start date for filtering
        figsize: Figure size tuple
        add_recessions: Whether to add recession shading
    
    Returns:
        matplotlib figure and axes array
    """
    import matplotlib.pyplot as plt
    import math
    
    # Fetch all series concurrently
    results = fetch_multiple_series_concurrent(fred, sector_series_dict, start_date)
    
    # Calculate grid dimensions
    n_series = len(sector_series_dict)
    n_cols = 3
    n_rows = math.ceil(n_series / n_cols)
    
    # Create grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Plot each series
    for i, (series_id, name) in enumerate(sector_series_dict.items()):
        data = results.get(name)
        ax = axes[i]
        
        if data is not None:
            # Calculate YoY change
            rate = calculate_yoy_change(data, periods=12)
            ax.plot(rate.index, rate.values)
            ax.set_title(name)
            ax.set_xlabel('Date')
            ax.set_ylabel('YoY % Change')
            ax.grid(True, alpha=0.3)
            
            if add_recessions:
                add_recession_shading(ax, fred, start_date)
        else:
            ax.text(0.5, 0.5, f'{name}\nData Not Available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(name)
    
    # Hide unused subplots
    for i in range(n_series, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def clear_fred_cache():
    """Clear all FRED data caches to force fresh API calls."""
    if hasattr(fetch_fred_series, "_cache"):
        fetch_fred_series._cache.clear()
    if hasattr(add_recession_shading, "_recession_cache"):
        add_recession_shading._recession_cache.clear()
    print("FRED caches cleared")


def prefetch_common_series(fred, start_date=None):
    """Pre-fetch commonly used series to warm up the cache.
    
    Args:
        fred: FRED API client instance
        start_date: Optional start date for filtering
    
    Returns:
        Number of series successfully cached
    """
    import matplotlib.pyplot as plt
    
    common_series = {
        "USREC": "Recession Indicator",
        "PAYEMS": "Employment",
        "JTSHIR": "Job Hires",
        "EXPGSC1": "Exports",
        "IMPGSC1": "Imports",
        "CPIAUCSL": "CPI",
    }
    
    print("Warming up cache with common series...")
    results = fetch_multiple_series_concurrent(fred, common_series, start_date, max_workers=20)
    
    # Store in cache
    if not hasattr(fetch_fred_series, "_cache"):
        fetch_fred_series._cache = {}
    
    for series_id, name in common_series.items():
        if results.get(name) is not None:
            cache_key = f"{series_id}_{start_date}"
            fetch_fred_series._cache[cache_key] = results[name].copy()
    
    # Pre-cache recession data
    if results.get("Recession Indicator") is not None:
        if not hasattr(add_recession_shading, "_recession_cache"):
            add_recession_shading._recession_cache = {}
        fig, ax = plt.subplots()
        add_recession_shading(ax, fred, start_date)
        plt.close(fig)
    
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"Cached {success_count}/{len(common_series)} series")
    return success_count

