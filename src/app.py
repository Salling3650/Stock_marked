"""Stock Analysis Dashboard - Comprehensive financial analysis tool"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import base64
import matplotlib.pyplot as plt
from io import BytesIO
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Analysis Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.metric-card{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}.positive{color:#10b981}.negative{color:#ef4444}.neutral{color:#6b7280}.section-header{font-size:24px;font-weight:700;margin-top:24px;margin-bottom:16px;color:#111827}</style>""", unsafe_allow_html=True)

def fetch_stock_data(ticker, period='2y'):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period), stock
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def calculate_technical_indicators(data):
    df = data.copy()
    df['MA_20'], df['MA_50'] = df['Close'].rolling(20).mean(), df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    exp12, exp26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'], df['Signal'] = exp12 - exp26, (exp12 - exp26).ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'], df['BB_Lower'] = df['BB_Middle'] + (bb_std * 2), df['BB_Middle'] - (bb_std * 2)
    low_14, high_14 = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(3).mean()
    return df

def calculate_risk_metrics(data):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 30: return {}
    total_return, years = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1, len(data) / 252
    annualized_return = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else None
    annualized_volatility = returns.std() * np.sqrt(252) * 100
    cumulative = (1 + returns).cumprod()
    max_drawdown = ((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min() * 100
    downside_std = returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': annualized_return / annualized_volatility if annualized_volatility > 0 else None,
        'Sortino Ratio': annualized_return / (downside_std * 100) if downside_std > 0 else None,
        'Calmar Ratio': annualized_return / abs(max_drawdown) if max_drawdown < 0 else None,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': abs(returns.quantile(0.05)) * 100
    }

def calculate_altman_z_score(ticker_obj, market_cap):
    try:
        bs, inc = ticker_obj.balance_sheet, ticker_obj.income_stmt
        if bs.empty or inc.empty: return None, None
        latest_bs, latest_inc = bs.iloc[:, 0], inc.iloc[:, 0]
        current_assets = latest_bs.get('Current Assets', latest_bs.get('Total Current Assets'))
        current_liabilities = latest_bs.get('Current Liabilities', latest_bs.get('Total Current Liabilities'))
        total_assets, total_liabilities = latest_bs.get('Total Assets'), latest_bs.get('Total Liabilities Net Minority Interest', latest_bs.get('Total Liabilities'))
        retained_earnings, ebit, revenue = latest_bs.get('Retained Earnings'), latest_inc.get('EBIT', latest_inc.get('Operating Income')), latest_inc.get('Total Revenue')
        if all(x is not None for x in [current_assets, current_liabilities, total_assets, total_liabilities, retained_earnings, ebit, revenue, market_cap]):
            wc = current_assets - current_liabilities
            z = 1.2*(wc/total_assets) + 1.4*(retained_earnings/total_assets) + 3.3*(ebit/total_assets) + 0.6*(market_cap/total_liabilities) + (revenue/total_assets)
            return z, "Safe Zone" if z > 2.99 else "Grey Zone" if z >= 1.81 else "Distress Zone"
    except: pass
    return None, None

def calculate_capm(stock_ticker, selected_stock, weeks=150):
    try:
        stock_data, market_data = stock_ticker.history(period='max'), yf.Ticker("^GSPC").history(period='max')
        treasury_data = yf.Ticker("^TNX").history(period="5d")
        risk_free_rate = treasury_data['Close'].iloc[-1] / 100 if not treasury_data.empty else 0.04
        for df in [stock_data, market_data]:
            if 'DatetimeIndex' not in str(type(df.index)): df.index = pd.to_datetime(df.index)
        merged = pd.DataFrame({'Stock': stock_data['Close'], 'Market': market_data['Close']}).dropna()
        if 'DatetimeIndex' not in str(type(merged.index)): merged.index = pd.to_datetime(merged.index)
        weekly = merged.resample('W-FRI').last().dropna().tail(weeks)
        aligned = pd.DataFrame({'Stock': weekly['Stock'].pct_change(), 'Market': weekly['Market'].pct_change()}).dropna()
        slope, _, r_value, _, _ = stats.linregress(aligned['Market'], aligned['Stock'])
        beta = slope
        market_return_annual, stock_return_annual = aligned['Market'].mean() * 52, aligned['Stock'].mean() * 52
        expected_return_capm = risk_free_rate + beta * (market_return_annual - risk_free_rate)
        up_market, down_market = aligned[aligned['Market'] > 0], aligned[aligned['Market'] <= 0]
        beta_up = up_market['Stock'].cov(up_market['Market']) / up_market['Market'].var() if len(up_market) > 5 and up_market['Market'].var() != 0 else beta
        beta_down = down_market['Stock'].cov(down_market['Market']) / down_market['Market'].var() if len(down_market) > 5 and down_market['Market'].var() != 0 else beta
        return {'beta': beta, 'alpha': stock_return_annual - expected_return_capm, 'expected_return': expected_return_capm,
                'actual_return': stock_return_annual, 'risk_free_rate': risk_free_rate, 'market_return': market_return_annual,
                'r_squared': r_value ** 2, 'aligned_data': aligned, 'beta_up': beta_up, 'beta_down': beta_down,
                'beta_asymmetry': abs(beta_up - beta_down), 'up_market': up_market, 'down_market': down_market}
    except Exception as e:
        st.error(f"CAPM error: {e}")
        return None

def humanize_number(num):
    if num is None: return "N/A"
    abs_num = abs(num)
    if abs_num >= 1e12: return f"${num/1e12:.2f}T"
    elif abs_num >= 1e9: return f"${num/1e9:.2f}B"
    elif abs_num >= 1e6: return f"${num/1e6:.2f}M"
    elif abs_num >= 1e3: return f"${num/1e3:.2f}K"
    else: return f"${num:.2f}"

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('metadata', {}).get('error') is None and data.get('data'):
                item = data['data'][0]
                score = int(item['value'])
                rating = item['value_classification'].lower()
                # Get historical
                hist_resp = requests.get("https://api.alternative.me/fng/?limit=0", timeout=10)
                hist_df = None
                if hist_resp.status_code == 200:
                    hist_data = hist_resp.json()
                    if hist_data.get('data'):
                        hist_df = pd.DataFrame([{"timestamp": pd.to_datetime(int(d['timestamp']), unit='s'), "score": int(d['value'])}
                                               for d in hist_data['data'] if d.get('value')])
                return {'score': score, 'rating': rating, 'historical': hist_df}
    except: pass
    return None

def ch(v,c,cl='#3b82f6'):
    if not v or all(x is None for x in v):return None
    v=[None]*(4-len(v[-4:]))+v[-4:];f,a=plt.subplots(figsize=(1.2,.7));f.patch.set_alpha(0);a.set_facecolor('none');b=a.bar(range(4),[x or 0 for x in v],color=cl,width=.7)
    for x in b:x.set_capstyle('round');x.set_joinstyle('round')
    a.axis('off');plt.tight_layout(pad=0);u=BytesIO();plt.savefig(u,format='png',dpi=50,bbox_inches='tight',transparent=True);plt.close(f);u.seek(0);return f'data:image/png;base64,{base64.b64encode(u.read()).decode()}'

def cd(n,v,c=None,h=None,s=None):
    if c is not None:
        color = "#10b981" if c > 0 else "#ef4444" if c < 0 else "#6b7280"
        symbol = "▲" if c > 0 else "▼" if c < 0 else "●"
        ch_div = f'<div style="color: {color}; font-size: 12px; margin-top: 4px;">{symbol} {abs(c):.1f}%</div>'
    else:
        ch_div = ''
    sb = f'<div style="color: #9ca3af; font-size: 11px; margin-top: 2px;">{s}</div>' if s else ''
    st = 'background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'
    ct = f'<div style="color: #6b7280; font-size: 13px; font-weight: 500; margin-bottom: 8px;">{n}</div><div style="color: #111827; font-size: 24px; font-weight: 700;">{v}</div>{sb}{ch_div}'
    if h:
        return f'<div style="{st} display: flex; justify-content: space-between; align-items: center; gap: 12px;"><div style="flex: 1;">{ct}</div><div style="flex-shrink: 0;"><img src="{h}" style="width: 70px; height: 40px;"></div></div>'
    else:
        return f'<div style="{st}">{ct}</div>'

def pc(n,c):
    if c is None:
        return cd(n,'N/A')
    cl = '#10b981' if c > 0 else '#ef4444' if c < 0 else '#6b7280'
    sy = '▲' if c > 0 else '▼' if c < 0 else '●'
    if any(k in n for k in ['%', 'Return', 'Volatility', 'Drawdown', 'VaR']):
        val = f'{sy} {abs(c):.1f}%'
    else:
        val = f'{sy} {abs(c):.2f}'
    return f'<div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);"><div style="color: #6b7280; font-size: 13px; font-weight: 500; margin-bottom: 8px;">{n}</div><div style="color: {cl}; font-size: 24px; font-weight: 700;">{val}</div></div>'

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def get_current_price_and_date(rh, data, info):
    if hasattr(rh, 'columns') and not rh.empty:
        cp = rh['Close'].iloc[-1]
        ld = rh.index[-1].strftime('%Y-%m-%d')
    elif hasattr(data, 'columns') and not data.empty:
        cp = data['Close'].iloc[-1]
        ld = data.index[-1].strftime('%Y-%m-%d')
    else:
        cp = info.get('currentPrice') or info.get('regularMarketPrice')
        ld = datetime.now().strftime('%Y-%m-%d')
    return cp, ld

def main():
    st.title("📈 Stock Analysis Dashboard")
    st.markdown("*Comprehensive financial analysis and visualization tool*")
    
    with st.sidebar:
        st.header("Settings")
        ticker_input = st.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, MSFT, GOOGL").upper()
        period = st.selectbox("Select Time Period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=4)
        st.subheader("Analysis Options")
        show_capm = st.checkbox("Show CAPM & Beta Analysis", value=False)
        show_financials = st.checkbox("Show Financial Statements", value=False)
        show_holdings = st.checkbox("Show Institutional Holdings", value=False)
        show_news = st.checkbox("Show News & Sentiment", value=False)
        fetch_button = st.button("🔄 Fetch Data", type="primary", use_container_width=True)
    
    if 'ticker' not in st.session_state or fetch_button:
        st.session_state.ticker, st.session_state.period = ticker_input, period
        st.session_state.data, st.session_state.stock_obj = None, None
    
    if st.session_state.data is None or fetch_button:
        with st.spinner(f"Fetching data for {ticker_input}..."):
            data, stock_obj = fetch_stock_data(ticker_input, period)
            if data is not None and not data.empty:
                data = calculate_technical_indicators(data)
                st.session_state.data, st.session_state.stock_obj = data, stock_obj
                st.success(f"✅ Successfully loaded data for {ticker_input}")
            else:
                st.error(f"❌ Could not fetch data for {ticker_input}")
                return
    
    data, stock_obj = st.session_state.data, st.session_state.stock_obj
    if data is None or data.empty:
        st.info("👈 Enter a ticker and click 'Fetch Data'")
        return
    
    info = stock_obj.info if hasattr(stock_obj, 'info') else {}
    # Display key metrics with visual cards and mini charts
    selected_stock = ticker_input
    data = st.session_state.data
    stock_obj = st.session_state.stock_obj

    if not selected_stock:
        st.error("No stock selected.")
    else:
        t = stock_obj
        rh = t.history(period='5d')
        if data is None or (data is not None and data.empty):
            data = t.history(period='2y')
            if hasattr(data, 'columns') and not data.empty:
                data = calculate_technical_indicators(data)
        i = getattr(t, 'info', {}) or {}
        qe = getattr(t, 'earnings_dates', None)
        qeps = qe['EPS'].dropna().tail(4).tolist() if qe is not None and not qe.empty and 'EPS' in qe.columns else []
        qv = {}
        if hasattr(data, 'columns') and not data.empty and 'DatetimeIndex' in str(type(data.index)):
            qd = data.resample('Q').last()
            qd = qd.iloc[-5:-1] if len(qd) >= 5 else qd
            qv = {c: qd[c].tolist() for c in ['Close', 'RSI', 'Volume'] if c in qd.columns}
        hp, hr, hv = qv.get('Close', []), qv.get('RSI', []), qv.get('Volume', [])
        cp, ld = get_current_price_and_date(rh, data, i)
        pc1 = None
        if hasattr(rh, 'columns') and len(rh) >= 2:
            p0, p1 = safe_float(rh['Close'].iloc[-2]), safe_float(rh['Close'].iloc[-1])
            pc1 = (p1 - p0) / p0 * 100 if p0 else None
        dy = i.get('dividendYield')
        dyp = (dy * 100 if dy and dy < 0.5 else dy if dy and dy < 20 else None) if dy else None
        m = {k: i.get(v) for k, v in [('market_cap', 'marketCap'), ('pe_ratio', 'trailingPE'), ('eps', 'trailingEps'), ('pb_ratio', 'priceToBook'), ('week_52_high', 'fiftyTwoWeekHigh'), ('week_52_low', 'fiftyTwoWeekLow'), ('volume', 'volume'), ('beta', 'beta'), ('ebitda', 'ebitda'), ('book_value', 'bookValue'), ('shares_outstanding', 'sharesOutstanding'), ('profit_margin', 'profitMargins'), ('operating_margin', 'operatingMargins'), ('return_on_assets', 'returnOnAssets'), ('return_on_equity', 'returnOnEquity')]}
        m.update({'dividend_yield': dyp, 'company_name': i.get('longName') or i.get('shortName') or selected_stock, 'sector': i.get('sector', 'N/A'), 'industry': i.get('industry', 'N/A')})
        tech = {}
        if hasattr(data, 'columns'):
            for ind, col in [('rsi', 'RSI'), ('macd', 'MACD'), ('stochastic_k', '%K'), ('volume', 'Volume')]:
                if col in data.columns and not (s := data[col].dropna()).empty:
                    tech[ind] = safe_float(s.iloc[-1])
            if all(c in data.columns for c in ['Close', 'Upper_Band', 'Lower_Band']):
                cv, u, l = safe_float(data['Close'].iloc[-1]), safe_float(data['Upper_Band'].iloc[-1]), safe_float(data['Lower_Band'].iloc[-1])
                tech['bollinger_b'] = (cv - l) / (u - l) if cv and u and l and u != l else None
        pf = {}
        if hasattr(data, 'columns') and not data.empty and 'Close' in data.columns:
            cl = data['Close']
            for n, d in [('30d', 30), ('90d', 90), ('6m', 126), ('1y', 252)]:
                pf[n] = ((cl.iloc[-1] - cl.iloc[-d]) / cl.iloc[-d]) * 100 if len(data) >= d else ((cl.iloc[-1] - cl.iloc[0]) / cl.iloc[0]) * 100 if n == '1y' and len(data) > 126 else None
        rk = {k: None for k in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'annualized_volatility', 'annualized_return', 'max_drawdown', 'var_95']}
        if hasattr(data, 'columns') and not data.empty and 'Close' in data.columns and len(data) >= 30:
            rt = data['Close'].pct_change().dropna()
            if not rt.empty:
                tr = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
                yr = len(data) / 252
                rk['annualized_return'] = ((1 + tr) ** (1 / yr) - 1) * 100 if yr > 0 else None
                rk['annualized_volatility'] = rt.std() * np.sqrt(252) * 100
                rk['sharpe_ratio'] = (rk['annualized_return'] or 0) / rk['annualized_volatility'] if rk['annualized_volatility'] and rk['annualized_volatility'] > 0 else None
                cm = (1 + rt).cumprod()
                dd = (cm - cm.expanding().max()) / cm.expanding().max()
                rk['max_drawdown'] = dd.min() * 100
                dr = rt[rt < 0]
                rk['sortino_ratio'] = (rk['annualized_return'] or 0) / ((ds := dr.std() * np.sqrt(252)) * 100) if len(dr) > 0 and (ds := dr.std() * np.sqrt(252)) > 0 else None
                rk['calmar_ratio'] = rk['annualized_return'] / abs(rk['max_drawdown']) if rk['max_drawdown'] and rk['max_drawdown'] < 0 and rk['annualized_return'] else None
                rk['var_95'] = abs(rt.quantile(0.05)) * 100
        chg = {}
        if m['market_cap'] and hp and cp and cp != 0:
            m1y = (data['Close'].iloc[-250] if hasattr(data, 'columns') and not data.empty and len(data) >= 250 else hp[0]) * (m['market_cap'] / cp)
            chg['market_cap'] = ((m['market_cap'] - m1y) / m1y) * 100 if m1y and m1y != 0 else None
        if m['eps'] and qeps and qeps[0] != 0:
            chg['eps'] = ((m['eps'] - qeps[0]) / qeps[0]) * 100
        cht = {}
        if hp and cp:
            cht['price'] = ch(hp, cp)
        if m['market_cap'] and hp and cp and cp != 0:
            cht['market_cap'] = ch([p * (m['market_cap'] / cp) for p in hp], m['market_cap'])
        if m['pe_ratio'] and qeps and (qpe := [cp / e for e in qeps if e and e != 0]):
            cht['pe'] = ch(qpe, m['pe_ratio'])
        if m['eps'] and qeps:
            cht['eps'] = ch(qeps, m['eps'])
        if m['pb_ratio'] and hp and m['book_value'] and m['book_value'] != 0:
            cht['pb'] = ch([p / m['book_value'] for p in hp], m['pb_ratio'])
        if tech.get('rsi') and hr:
            cht['rsi'] = ch(hr, tech['rsi'])
        if tech.get('volume') and hv:
            cht['volume'] = ch(hv, tech['volume'])
        h = [f'''<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px;"><h2 style="color: #111827; margin-bottom: 8px;">{m['company_name']} ({selected_stock})</h2><div style="color: #6b7280; font-size: 14px; margin-bottom: 24px;">{m['sector']} • {m['industry']}</div><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px;">''']
        for n, v, c, ch_img, s in [('Current Price', f'${cp:.2f}' if cp else 'N/A', pc1, cht.get('price'), None), ('Market Cap', f'${humanize_number(m["market_cap"])}' if m['market_cap'] else 'N/A', chg.get('market_cap'), cht.get('market_cap'), None), ('Shares Outstanding', humanize_number(m['shares_outstanding']) if m['shares_outstanding'] else 'N/A', None, None, None), ('P/E Ratio', f'{m["pe_ratio"]:.2f}' if m['pe_ratio'] else 'N/A', None, cht.get('pe'), None), ('EPS', f'${m["eps"]:.2f}' if m['eps'] else 'N/A', chg.get('eps'), cht.get('eps'), None), ('EBITDA', f'${humanize_number(m["ebitda"])}' if m['ebitda'] else 'N/A', None, None, None), ('Dividend Yield', f'{m["dividend_yield"]:.2f}%' if m['dividend_yield'] else 'N/A', None, None, None), ('P/B Ratio', f'{m["pb_ratio"]:.2f}' if m['pb_ratio'] else 'N/A', None, cht.get('pb'), None), ('52W High', f'${m["week_52_high"]:.2f}' if m['week_52_high'] else 'N/A', None, None, None), ('52W Low', f'${m["week_52_low"]:.2f}' if m['week_52_low'] else 'N/A', None, None, None), ('Volume', humanize_number(m['volume'] or tech.get('volume')) if (m['volume'] or tech.get('volume')) else 'N/A', None, cht.get('volume'), None), ('RSI', f'{tech["rsi"]:.1f}' if tech.get('rsi') else 'N/A', None, cht.get('rsi'), None), ('MACD', f'{tech["macd"]:.3f}' if tech.get('macd') is not None else 'N/A', None, None, None), ('Bollinger %B', f'{tech["bollinger_b"]:.2f}' if tech.get('bollinger_b') is not None else 'N/A', None, None, None), ('Stochastic %K', f'{tech["stochastic_k"]:.1f}' if tech.get('stochastic_k') is not None else 'N/A', None, None, None), ('Beta', f'{m["beta"]:.2f}' if m['beta'] else 'N/A', None, None, '(vs S&P 500)')]:
            h.append(cd(n, v, c, ch_img, s))
        h.append('</div><h3 style="color: #111827; margin-top: 24px; margin-bottom: 16px;">Profitability Metrics</h3><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px;">')
        for n, k in [('Profit Margin', 'profit_margin'), ('Operating Margin', 'operating_margin'), ('Return on Assets', 'return_on_assets'), ('Return on Equity', 'return_on_equity')]:
            val = m.get(k)
            if val is not None:
                val_pct = val * 100
                h.append(cd(n, f'{val_pct:.2f}%'))
            else:
                h.append(cd(n, 'N/A'))
        h.append('</div><h3 style="color: #111827; margin-top: 24px; margin-bottom: 16px;">Risk Metrics</h3><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px;">')
        zs = None
        zi = None
        try:
            bs = getattr(t, 'balance_sheet', None)
            inc = getattr(t, 'income_stmt', None)
            lb = bs.iloc[:, 0] if bs is not None and not bs.empty else {}
            li = inc.iloc[:, 0] if inc is not None and not inc.empty else {}
            ca = lb.get('Current Assets', lb.get('Total Current Assets'))
            cl = lb.get('Current Liabilities', lb.get('Total Current Liabilities'))
            ta = lb.get('Total Assets')
            tl = lb.get('Total Liabilities Net Minority Interest', lb.get('Total Liabilities'))
            re = lb.get('Retained Earnings')
            eb = li.get('EBIT', li.get('Operating Income'))
            rv = li.get('Total Revenue')
            zs = 1.2 * (ca - cl) / ta + 1.4 * re / ta + 3.3 * eb / ta + 0.6 * m['market_cap'] / tl + rv / ta if all(x and ta and tl for x in [ca, cl, re, eb, rv, m['market_cap']]) else None
            zi = 'Safe Zone' if zs and zs > 2.99 else 'Grey Zone' if zs and zs >= 1.81 else 'Distress Zone' if zs else None
        except:
            pass
        for n, k in [('Sharpe Ratio', 'sharpe_ratio'), ('Sortino Ratio', 'sortino_ratio'), ('Calmar Ratio', 'calmar_ratio'), ('Annualized Volatility', 'annualized_volatility'), ('Annualized Return', 'annualized_return'), ('Max Drawdown', 'max_drawdown'), ('VaR (95%)', 'var_95')]:
            h.append(pc(n, rk.get(k)))
        h.append(cd('Altman Z-Score', f'{zs:.2f}' if zs else 'N/A', None, None, zi))
        h.append('</div><h3 style="color: #111827; margin-top: 24px; margin-bottom: 16px;">Performance</h3><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px;">')
        for n, k in [('30-Day Return', '30d'), ('90-Day Return', '90d'), ('6-Month Return', '6m'), ('1-Year Return', '1y')]:
            h.append(pc(n, pf.get(k)))
        h.append('</div></div>')
        st.markdown(''.join(h), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Stock Price & Technical Dashboard")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25],
                       subplot_titles=('Price, MAs & Bollinger Bands', 'Volume', 'RSI'))
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                 name=ticker_input, increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    for ma, color in [('MA_20', '#2962FF'), ('MA_50', '#FF6D00')]:
        fig.add_trace(go.Scatter(x=data.index, y=data[ma], mode='lines', name=f'{ma[3:]}-day MA', line=dict(color=color, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', opacity=0.5), row=1, col=1)
    colors = ['#26a69a' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ef5350' for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume']/1e6, name='Volume (M)', marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.update_layout(height=900, hovermode='x unified', template='plotly_white', xaxis_rangeslider_visible=False, showlegend=True)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (M)", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Indicators Dashboard (6 charts in 2 rows)
    st.markdown("### Advanced Technical Indicators")
    
    # Calculate additional indicators
    cp = data['Close']
    dr = cp.pct_change()
    cum_return = ((1 + dr).cumprod() - 1) * 100
    
    # Cumulative drawdown
    cum = (1 + dr).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()) * 100
    
    # OBV
    obv = (np.sign(cp.diff()) * data['Volume']).fillna(0).cumsum() / 1e6
    
    # Rolling Sharpe (60-day)
    rolling_sharpe = (dr.rolling(60).mean() / dr.rolling(60).std()) * np.sqrt(252)
    
    # Create 6-chart grid
    fig2 = make_subplots(
        rows=2, cols=3,
        subplot_titles=('MACD', 'Max Drawdown', 'Stochastic Oscillator',
                       'Daily Returns Distribution', 'On-Balance Volume', 'Rolling Sharpe (60d)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. MACD
    fig2.add_trace(go.Scatter(
        x=data.index, y=data['MACD'],
        mode='lines', name='MACD',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    fig2.add_trace(go.Scatter(
        x=data.index, y=data['Signal'],
        mode='lines', name='Signal',
        line=dict(color='red', width=2)
    ), row=1, col=1)
    
    fig2.add_trace(go.Bar(
        x=data.index, y=data['MACD_Hist'],
        name='Histogram', marker_color='gray',
        opacity=0.5
    ), row=1, col=1)
    
    fig2.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    
    # 2. Max Drawdown
    fig2.add_trace(go.Scatter(
        x=data.index, y=dd,
        mode='lines', name='Drawdown',
        fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=2)
    ), row=1, col=2)
    
    # 3. Stochastic
    fig2.add_trace(go.Scatter(
        x=data.index, y=data['%K'],
        mode='lines', name='%K',
        line=dict(color='blue', width=2)
    ), row=1, col=3)
    
    fig2.add_trace(go.Scatter(
        x=data.index, y=data['%D'],
        mode='lines', name='%D',
        line=dict(color='red', width=2)
    ), row=1, col=3)
    
    fig2.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=1, col=3)
    fig2.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=1, col=3)
    
    # 4. Daily Returns Histogram
    fig2.add_trace(go.Histogram(
        x=dr.dropna() * 100,
        nbinsx=50, name='Returns',
        marker_color='purple', opacity=0.7
    ), row=2, col=1)
    
    # 5. OBV
    fig2.add_trace(go.Scatter(
        x=data.index, y=obv,
        mode='lines', name='OBV',
        line=dict(color='orange', width=2)
    ), row=2, col=2)
    
    # 6. Rolling Sharpe
    fig2.add_trace(go.Scatter(
        x=data.index, y=rolling_sharpe,
        mode='lines', name='Sharpe',
        line=dict(color='green', width=2)
    ), row=2, col=3)
    
    fig2.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=3)
    
    # Update layout
    fig2.update_layout(
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    fig2.update_yaxes(title_text="Value", row=1, col=1)
    fig2.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
    fig2.update_yaxes(title_text="Value", row=1, col=3, range=[0, 100])
    fig2.update_yaxes(title_text="Frequency", row=2, col=1)
    fig2.update_yaxes(title_text="OBV (M)", row=2, col=2)
    fig2.update_yaxes(title_text="Sharpe", row=2, col=3)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### 🔧 Current Technical Indicator Values")
    cs=st.columns(4)
    rsi,macd,sig,sk=data['RSI'].iloc[-1],data['MACD'].iloc[-1],data['Signal'].iloc[-1],data['%K'].iloc[-1]
    cs[0].metric("RSI (14)",f"{rsi:.1f}","Overbought"if rsi>70 else"Oversold"if rsi<30 else"Neutral")
    cs[1].metric("MACD",f"{macd:.3f}","Bullish"if macd>sig else"Bearish")
    cs[2].metric("Stoch%K",f"{sk:.1f}","Overbought"if sk>80 else"Oversold"if sk<20 else"Neutral")
    bbp=((data['Close'].iloc[-1]-data['BB_Lower'].iloc[-1])/(data['BB_Upper'].iloc[-1]-data['BB_Lower'].iloc[-1]))
    cs[3].metric("BB%B",f"{bbp:.2f}","Upper"if bbp>0.8 else"Lower"if bbp<0.2 else"Mid")
    
    st.markdown("---\n## ⚠️ Risk Metrics")
    rm=calculate_risk_metrics(data)
    cs=st.columns(4)
    for c,(l,k,u)in zip(cs,[("Ann.Return","Annualized Return","%"),("Ann.Vol","Annualized Volatility","%"),("Sharpe","Sharpe Ratio",""),("Sortino","Sortino Ratio","")]):
        v=rm.get(k);c.metric(l,f"{v:.2f}{u}"if v is not None else"N/A")
    cs=st.columns(3)
    for c,(l,k,u)in zip(cs,[("MaxDD","Max Drawdown","%"),("VaR95","VaR (95%)","%"),("Calmar","Calmar Ratio","")]):
        v=rm.get(k);c.metric(l,f"{v:.2f}{u}"if v is not None else"N/A")
    
    z,zi=calculate_altman_z_score(stock_obj,info.get('marketCap'))
    if z is not None:
        st.markdown("### Altman Z-Score")
        c1,c2=st.columns([1,2])
        c1.metric("Z-Score",f"{z:.2f}",zi)
        c2.info(f"**{zi}** - Safe>2.99, Grey 1.81-2.99, Distress<1.81")
    
    if show_capm:
        st.markdown("---\n## 📈 CAPM Analysis")
        with st.spinner("Calculating..."):cr=calculate_capm(stock_obj,ticker_input)
        if cr:
            cs=st.columns(4)
            cs[0].metric("Beta",f"{cr['beta']:.3f}");cs[0].caption(('More'if cr['beta']>1 else'Less'if cr['beta']<1 else'Similar')+' volatile')
            cs[1].metric("Alpha",f"{cr['alpha']*100:.2f}%");cs[1].caption(('Outperforming'if cr['alpha']>0 else'Underperforming')+' expectations')
            cs[2].metric("Exp.Return",f"{cr['expected_return']*100:.2f}%");cs[2].caption("Per CAPM")
            cs[3].metric("R²",f"{cr['r_squared']:.3f}");cs[3].caption(f"{cr['r_squared']*100:.1f}% explained")
            st.markdown("### Stock vs Market Returns")
            al=cr['aligned_data']
            fc=go.Figure()
            fc.add_trace(go.Scatter(x=al['Market']*100,y=al['Stock']*100,mode='markers',name='Weekly',marker=dict(size=5,color='blue',opacity=0.5)))
            xl=np.linspace(al['Market'].min(),al['Market'].max(),100)
            fc.add_trace(go.Scatter(x=xl*100,y=cr['beta']*xl*100,mode='lines',name=f'β={cr["beta"]:.3f}',line=dict(color='red',width=2,dash='dash')))
            fc.update_layout(title=f'{ticker_input} vs S&P500',xaxis_title='Mkt%',yaxis_title=f'{ticker_input}%',hovermode='closest',template='plotly_white',height=500)
            fc.add_hline(y=0,line_dash="dash",line_color="gray",opacity=0.5);fc.add_vline(x=0,line_dash="dash",line_color="gray",opacity=0.5)
            st.plotly_chart(fc,use_container_width=True)
            st.markdown("### Nonlinear Beta")
            c1,c2,c3=st.columns(3)
            c1.metric("β Up",f"{cr['beta_up']:.3f}");c1.caption(f"{len(cr['up_market'])} wks")
            c2.metric("β Down",f"{cr['beta_down']:.3f}");c2.caption(f"{len(cr['down_market'])} wks")
            ap=(cr['beta_asymmetry']/cr['beta']*100)if cr['beta']!=0 else 0
            c3.metric("β Asymm",f"{cr['beta_asymmetry']:.3f}");c3.caption(f"{ap:.1f}% dev")
            if cr['beta_up']>cr['beta_down']+0.15:st.success("🚀 GROWTH AMPLIFIER")
            elif cr['beta_down']>cr['beta_up']+0.15:st.warning("⚠️ DEFENSIVE AMPLIFIER")
            else:st.info("⚖️ SYMMETRIC")
            fd=go.Figure()
            up,dn=cr['up_market'],cr['down_market']
            fd.add_trace(go.Scatter(x=up['Market']*100,y=up['Stock']*100,mode='markers',name=f'Up β={cr["beta_up"]:.2f}',marker=dict(size=5,color='green',opacity=0.6)))
            fd.add_trace(go.Scatter(x=dn['Market']*100,y=dn['Stock']*100,mode='markers',name=f'Down β={cr["beta_down"]:.2f}',marker=dict(size=5,color='red',opacity=0.6)))
            if len(up)>1:
                z=np.polyfit(up['Market'],up['Stock'],1);xl=np.linspace(up['Market'].min(),up['Market'].max(),50)
                fd.add_trace(go.Scatter(x=xl*100,y=np.poly1d(z)(xl)*100,mode='lines',name='Up Trend',line=dict(color='green',width=2,dash='dash')))
            if len(dn)>1:
                z=np.polyfit(dn['Market'],dn['Stock'],1);xl=np.linspace(dn['Market'].min(),dn['Market'].max(),50)
                fd.add_trace(go.Scatter(x=xl*100,y=np.poly1d(z)(xl)*100,mode='lines',name='Down Trend',line=dict(color='red',width=2,dash='dash')))
            fd.update_layout(title='Market Regime',xaxis_title='Mkt%',yaxis_title=f'{ticker_input}%',template='plotly_white',height=500)
            fd.add_hline(y=0,line_dash="dash",line_color="gray",opacity=0.5);fd.add_vline(x=0,line_dash="dash",line_color="gray",opacity=0.5)
            st.plotly_chart(fd,use_container_width=True)
    
    st.markdown("---\n## 📰 Latest News")
    try:
        nw=getattr(stock_obj,'news',None)
        if not nw:st.info("📭 No news")
        else:
            st.write(f"*{len(nw)} articles*");nc=0
            for i,a in enumerate(nw[:10],1):
                try:
                    ct=a.get('content',{})if isinstance(a,dict)else{}
                    pr=(ct.get('provider',{})if isinstance(ct,dict)else{}).get('displayName','Unknown')
                    sm=ct.get('summary')or ct.get('description')or ct.get('title')or'No summary'
                    pd=ct.get('pubDate')or ct.get('displayTime')or a.get('pubDate')or a.get('date')or'Unknown'
                    if pd!='Unknown':
                        try:pd=datetime.fromisoformat(str(pd).replace('Z','+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                        except:pass
                    sms=(sm[:100]+'...')if isinstance(sm,str)and len(sm)>100 else sm
                    st.markdown(f"**{i}. {pr}**");st.caption(sms);st.caption(f"*{pd}*");nc+=1
                except:pass
            if nc==0:st.info("📭 No articles")
    except:st.info("📭 News unavailable")
    
    st.markdown("## 🏢 Institutional Holdings")
    c1,c2=st.columns(2)
    with c1:
        try:
            ih=stock_obj.institutional_holders
            if ih is not None and not ih.empty:
                st.markdown("### Top Institutional")
                for _,r in ih.head(10).iterrows():
                    h,s=r.get('Holder','Unknown'),r.get('Shares',0);p=None
                    for cl in['% Out','pctHeld','Percent Held','% Held','Pct Held']:
                        if cl in r.index and r[cl]is not None:p=r[cl];break
                    st.markdown(f"**{h}**");st.caption(f"{humanize_number(s)} shares"+(f" ({p if p>1 else p*100:.2f}%)"if p else""))
            else:st.info("No data")
        except:st.info("No data")
    with c2:
        try:
            mh=stock_obj.mutualfund_holders
            if mh is not None and not mh.empty:
                st.markdown("### Top Mutual Funds")
                for _,r in mh.head(10).iterrows():
                    h,s=r.get('Holder','Unknown'),r.get('Shares',0);p=None
                    for cl in['% Out','pctHeld','Percent Held','% Held','Pct Held']:
                        if cl in r.index and r[cl]is not None:p=r[cl];break
                    st.markdown(f"**{h}**");st.caption(f"{humanize_number(s)} shares"+(f" ({p if p>1 else p*100:.2f}%)"if p else""))
            else:st.info("No data")
        except:st.info("No data")
    
    if show_financials:st.markdown("---\n## 📊 Financial Statements")
    t1,t2,t3=st.tabs(["Income Statement","Balance Sheet","Cash Flow"])
    with t1:
        try:
            inc=stock_obj.income_stmt
            if inc is not None and not inc.empty:st.dataframe(inc.head(20),use_container_width=True)
            else:st.info("No data")
        except:st.info("No data")
    with t2:
        try:
            bal=stock_obj.balance_sheet
            if bal is not None and not bal.empty:st.dataframe(bal.head(20),use_container_width=True)
            else:st.info("No data")
        except:st.info("No data")
    with t3:
        try:
            cf=stock_obj.cashflow
            if cf is not None and not cf.empty:st.dataframe(cf.head(20),use_container_width=True)
            else:st.info("No data")
        except:st.info("No data")
    
    st.markdown("---\n## 🎯 Analyst Coverage")
    c1,c2=st.columns(2)
    with c1:
        try:
            rc=stock_obj.recommendations_summary
            if rc is not None and not rc.empty:
                st.markdown("### Current Recommendations");lt=rc.iloc[-1]
                rd={'Strong Buy':lt.get('strongBuy',0),'Buy':lt.get('buy',0),'Hold':lt.get('hold',0),'Sell':lt.get('sell',0),'Strong Sell':lt.get('strongSell',0)}
                rd={k:v for k,v in rd.items()if v>0}
                if rd:
                    fr=go.Figure(data=[go.Bar(x=list(rd.keys()),y=list(rd.values()),marker_color=['#00aa00','#66cc66','#ffaa00','#ff6666','#cc0000'][:len(rd)])])
                    fr.update_layout(title="Recommendations",yaxis_title="# Analysts",template='plotly_white',height=300)
                    st.plotly_chart(fr,use_container_width=True)
                else:st.info("No data")
            else:st.info("No data")
        except:st.info("No data")
    with c2:
        try:
            ug=stock_obj.upgrades_downgrades
            if ug is not None and not ug.empty:
                st.markdown("### Recent Upgrades/Downgrades")
                for _,r in ug.head(5).iterrows():st.markdown(f"**{r.get('Firm','Unknown')}**");st.caption(f"{r.get('Action','Unknown')} → {r.get('ToGrade','N/A')}")
            else:st.info("No data")
        except:st.info("No data")
    
    st.markdown("---\n## 😱 Market Sentiment - Fear & Greed Index")
    fg=get_fear_greed_index()
    if fg and fg.get('score')is not None:
        c1,c2=st.columns([1,2])
        with c1:
            sc,rt=fg['score'],fg['rating']
            st.metric("Score",f"{sc:.1f}",rt)
            if sc<=25:st.error("🔴 Extreme Fear - Buy opp")
            elif sc<=45:st.warning("🟠 Fear - Cautious")
            elif sc<=55:st.info("🟡 Neutral")
            elif sc<=75:st.success("🟢 Greed - Optimistic")
            else:st.error("🔴 Extreme Greed - Sell signal")
        with c2:
            hd=fg.get('historical')
            if hd is not None and not hd.empty:
                ff=go.Figure()
                ff.add_trace(go.Scatter(x=hd['timestamp'],y=hd['score'],mode='lines',name='F&G',line=dict(color='#1e3a8a',width=2),fill='tozeroy',fillcolor='rgba(30,58,138,0.2)'))
                ff.add_hline(y=25,line_dash="dash",line_color="red",opacity=0.5)
                ff.add_hline(y=75,line_dash="dash",line_color="green",opacity=0.5)
                ff.update_layout(title="Fear & Greed - History",xaxis_title="Date",yaxis_title="Score",template='plotly_white',height=300,yaxis=dict(range=[0,100]))
                st.plotly_chart(ff,use_container_width=True)
    else:st.info("Fear & Greed data unavailable")
    
    st.markdown("---")
    st.markdown("""<div style='text-align:center;color:#6b7280;padding:20px;'><p>📊 Stock Analysis Dashboard | Yahoo Finance</p><p style='font-size:12px;'>⚠️ Educational purposes only. Not financial advice.</p></div>""",unsafe_allow_html=True)

if __name__ == "__main__":
    main()
