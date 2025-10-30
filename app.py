"""
Stock Analysis Dashboard
A comprehensive financial analysis tool built with Streamlit
"""

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
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    .neutral { color: #6b7280; }
    .section-header {
        font-size: 24px;
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 16px;
        color: #111827;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def fetch_stock_data(ticker, period='2y'):
    """Fetch stock data (no caching for ticker object)"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data, stock
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

@st.cache_data(ttl=300)
def get_cached_history(ticker, period='2y'):
    """Fetch only price history with caching"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    except Exception as e:
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    return df

def calculate_risk_metrics(data):
    """Calculate risk metrics"""
    returns = data['Close'].pct_change().dropna()
    
    if len(returns) < 30:
        return {}
    
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
    years = len(data) / 252
    
    annualized_return = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else None
    annualized_volatility = returns.std() * np.sqrt(252) * 100
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else None
    
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative - cumulative.expanding().max()) / cumulative.expanding().max()
    max_drawdown = drawdown.min() * 100
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annualized_return / (downside_std * 100) if downside_std > 0 else None
    
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else None
    var_95 = abs(returns.quantile(0.05)) * 100
    
    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95
    }

def calculate_altman_z_score(ticker_obj, market_cap):
    """Calculate Altman Z-Score"""
    try:
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt
        
        if bs.empty or inc.empty:
            return None, None
        
        latest_bs = bs.iloc[:, 0]
        latest_inc = inc.iloc[:, 0]
        
        current_assets = latest_bs.get('Current Assets', latest_bs.get('Total Current Assets'))
        current_liabilities = latest_bs.get('Current Liabilities', latest_bs.get('Total Current Liabilities'))
        total_assets = latest_bs.get('Total Assets')
        total_liabilities = latest_bs.get('Total Liabilities Net Minority Interest', latest_bs.get('Total Liabilities'))
        retained_earnings = latest_bs.get('Retained Earnings')
        ebit = latest_inc.get('EBIT', latest_inc.get('Operating Income'))
        revenue = latest_inc.get('Total Revenue')
        
        if all(x is not None for x in [current_assets, current_liabilities, total_assets, 
                                        total_liabilities, retained_earnings, ebit, revenue, market_cap]):
            working_capital = current_assets - current_liabilities
            X1 = working_capital / total_assets
            X2 = retained_earnings / total_assets
            X3 = ebit / total_assets
            X4 = market_cap / total_liabilities
            X5 = revenue / total_assets
            
            z_score = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + X5
            
            if z_score > 2.99:
                interpretation = "Safe Zone"
            elif z_score >= 1.81:
                interpretation = "Grey Zone"
            else:
                interpretation = "Distress Zone"
            
            return z_score, interpretation
    except:
        pass
    
    return None, None

def calculate_capm(stock_ticker, selected_stock, weeks=150):
    """Calculate CAPM metrics"""
    try:
        # Fetch data
        stock_data = stock_ticker.history(period='max')
        market_ticker = yf.Ticker("^GSPC")
        market_data = market_ticker.history(period='max')
        
        # Get risk-free rate
        treasury = yf.Ticker("^TNX")
        treasury_data = treasury.history(period="5d")
        risk_free_rate = treasury_data['Close'].iloc[-1] / 100 if not treasury_data.empty else 0.04
        
        # Ensure datetime index
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)
        
        # Merge and resample to weekly
        merged = pd.DataFrame({
            'Stock': stock_data['Close'],
            'Market': market_data['Close']
        }).dropna()
        
        # Ensure merged has datetime index
        if not isinstance(merged.index, pd.DatetimeIndex):
            merged.index = pd.to_datetime(merged.index)
        
        weekly = merged.resample('W-FRI').last().dropna().tail(weeks)
        
        # Calculate returns
        stock_returns = weekly['Stock'].pct_change().dropna()
        market_returns = weekly['Market'].pct_change().dropna()
        
        aligned = pd.DataFrame({
            'Stock': stock_returns,
            'Market': market_returns
        }).dropna()
        
        # Calculate beta using regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned['Market'], aligned['Stock']
        )
        beta = slope
        
        # Calculate metrics
        market_return_annual = aligned['Market'].mean() * 52
        stock_return_annual = aligned['Stock'].mean() * 52
        expected_return_capm = risk_free_rate + beta * (market_return_annual - risk_free_rate)
        alpha = stock_return_annual - expected_return_capm
        r_squared = r_value ** 2
        
        # Nonlinear beta (market regime analysis)
        up_market = aligned[aligned['Market'] > 0]
        down_market = aligned[aligned['Market'] <= 0]
        
        beta_up = (up_market['Stock'].cov(up_market['Market']) / up_market['Market'].var() 
                   if len(up_market) > 5 and up_market['Market'].var() != 0 else beta)
        beta_down = (down_market['Stock'].cov(down_market['Market']) / down_market['Market'].var() 
                     if len(down_market) > 5 and down_market['Market'].var() != 0 else beta)
        beta_asymmetry = abs(beta_up - beta_down)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'expected_return': expected_return_capm,
            'actual_return': stock_return_annual,
            'risk_free_rate': risk_free_rate,
            'market_return': market_return_annual,
            'r_squared': r_squared,
            'aligned_data': aligned,
            'beta_up': beta_up,
            'beta_down': beta_down,
            'beta_asymmetry': beta_asymmetry,
            'up_market': up_market,
            'down_market': down_market
        }
    except Exception as e:
        st.error(f"CAPM calculation error: {e}")
        return None

def humanize_number(num):
    """Convert number to human-readable format"""
    if num is None:
        return "N/A"
    
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch CNN Fear & Greed Index"""
    try:
        session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.cnn.com/",
        }
        
        resp = session.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", 
                          headers=headers, timeout=8)
        
        if resp.status_code == 200:
            data = resp.json()
            fg_data = data.get("fear_and_greed", {})
            score = fg_data.get('score')
            rating = fg_data.get('rating')
            
            # Historical data
            historical = data.get("fear_and_greed_historical", {}).get("data", [])
            hist_df = pd.DataFrame([
                {"timestamp": pd.to_datetime(item.get("x"), unit='ms', utc=True), 
                 "score": item.get("y")}
                for item in historical if item.get("x") and item.get("y") is not None
            ])
            
            return {'score': score, 'rating': rating, 'historical': hist_df}
    except:
        pass
    
    return None

# Main App
def main():
    st.title("📈 Stock Analysis Dashboard")
    st.markdown("*Comprehensive financial analysis and visualization tool*")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Stock selector
        ticker_input = st.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        # Time period selector
        period = st.selectbox(
            "Select Time Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            index=4,
            help="Historical data period to analyze"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        show_capm = st.checkbox("Show CAPM & Beta Analysis", value=False)
        show_financials = st.checkbox("Show Financial Statements", value=False)
        show_holdings = st.checkbox("Show Institutional Holdings", value=False)
        show_news = st.checkbox("Show News & Sentiment", value=False)
        
        # Fetch data button
        fetch_button = st.button("🔄 Fetch Data", type="primary", use_container_width=True)
    
    # Initialize session state
    if 'ticker' not in st.session_state or fetch_button:
        st.session_state.ticker = ticker_input
        st.session_state.period = period
        st.session_state.data = None
        st.session_state.stock_obj = None
    
    # Fetch data
    if st.session_state.data is None or fetch_button:
        with st.spinner(f"Fetching data for {ticker_input}..."):
            data, stock_obj = fetch_stock_data(ticker_input, period)
            
            if data is not None and not data.empty:
                data = calculate_technical_indicators(data)
                st.session_state.data = data
                st.session_state.stock_obj = stock_obj
                st.success(f"✅ Successfully loaded data for {ticker_input}")
            else:
                st.error(f"❌ Could not fetch data for {ticker_input}. Please check the ticker symbol.")
                return
    
    data = st.session_state.data
    stock_obj = st.session_state.stock_obj
    
    if data is None or data.empty:
        st.info("👈 Enter a ticker symbol and click 'Fetch Data' to begin")
        return
    
    # Get stock info
    info = stock_obj.info if hasattr(stock_obj, 'info') else {}
    
    # Header with company info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### {info.get('longName', ticker_input)} ({ticker_input})")
        st.markdown(f"*{info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}*")
    
    with col2:
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) >= 2 else current_price
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price else 0
        
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f}%"
        )
    
    with col3:
        market_cap = info.get('marketCap')
        if market_cap:
            st.metric("Market Cap", humanize_number(market_cap))
    
    # Key Metrics Row
    st.markdown("---")
    cols = st.columns(6)
    
    metrics = [
        ("P/E Ratio", info.get('trailingPE'), ".2f"),
        ("EPS", info.get('trailingEps'), ".2f"),
        ("Beta", info.get('beta'), ".2f"),
        ("52W High", info.get('fiftyTwoWeekHigh'), ".2f"),
        ("52W Low", info.get('fiftyTwoWeekLow'), ".2f"),
        ("Volume", info.get('volume'), ",")
    ]
    
    for col, (label, value, fmt) in zip(cols, metrics):
        if value is not None:
            if fmt == ",":
                col.metric(label, f"{value:,}")
            else:
                col.metric(label, f"{value:{fmt}}")
        else:
            col.metric(label, "N/A")
    
    # Interactive Price Chart
    st.markdown("---")
    st.markdown("## 📊 Stock Price & Technical Dashboard")
    
    # Main price chart with volume and RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price, Moving Averages & Bollinger Bands', 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker_input,
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MA_20'],
        mode='lines', name='20-day MA',
        line=dict(color='#2962FF', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MA_50'],
        mode='lines', name='50-day MA',
        line=dict(color='#FF6D00', width=2)
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Upper'],
        mode='lines', name='BB Upper',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Lower'],
        mode='lines', name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)',
        opacity=0.5
    ), row=1, col=1)
    
    # Volume
    colors = ['#26a69a' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ef5350' 
              for i in range(len(data))]
    
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume']/1e6,
        name='Volume (M)', marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        mode='lines', name='RSI',
        line=dict(color='purple', width=2)
    ), row=3, col=1)
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    
    fig.update_layout(
        height=900,
        hovermode='x unified',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
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
    
    # Technical Indicators Summary (Always shown)
    st.markdown("### 🔧 Current Technical Indicator Values")
    
    cols = st.columns(4)
    
    with cols[0]:
        rsi = data['RSI'].iloc[-1]
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
    
    with cols[1]:
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
        macd_signal = "Bullish" if macd > signal else "Bearish"
        st.metric("MACD", f"{macd:.3f}", macd_signal)
    
    with cols[2]:
        stoch_k = data['%K'].iloc[-1]
        stoch_status = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
        st.metric("Stochastic %K", f"{stoch_k:.1f}", stoch_status)
    
    with cols[3]:
        bb_position = ((data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / 
                      (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]))
        bb_status = "Upper Band" if bb_position > 0.8 else "Lower Band" if bb_position < 0.2 else "Middle"
        st.metric("Bollinger Band %B", f"{bb_position:.2f}", bb_status)
    
    # Risk Metrics (Always shown)
    st.markdown("---")
    st.markdown("## ⚠️ Risk Metrics")
    
    risk_metrics = calculate_risk_metrics(data)
    
    cols = st.columns(4)
    risk_items = [
        ("Annualized Return", "Annualized Return", "%"),
        ("Annualized Volatility", "Annualized Volatility", "%"),
        ("Sharpe Ratio", "Sharpe Ratio", ""),
        ("Sortino Ratio", "Sortino Ratio", ""),
    ]
    
    for col, (label, key, unit) in zip(cols, risk_items):
        value = risk_metrics.get(key)
        if value is not None:
            col.metric(label, f"{value:.2f}{unit}")
        else:
            col.metric(label, "N/A")
    
    cols = st.columns(3)
    more_risk = [
        ("Max Drawdown", "Max Drawdown", "%"),
        ("VaR (95%)", "VaR (95%)", "%"),
        ("Calmar Ratio", "Calmar Ratio", ""),
    ]
    
    for col, (label, key, unit) in zip(cols, more_risk):
        value = risk_metrics.get(key)
        if value is not None:
            col.metric(label, f"{value:.2f}{unit}")
        else:
            col.metric(label, "N/A")
    
    # Altman Z-Score
    z_score, z_interpretation = calculate_altman_z_score(stock_obj, info.get('marketCap'))
    
    if z_score is not None:
        st.markdown("### Altman Z-Score")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Z-Score", f"{z_score:.2f}", z_interpretation)
        with col2:
            st.info(f"""
            **Interpretation:** {z_interpretation}
            - **Safe Zone** (Z > 2.99): Low bankruptcy risk
            - **Grey Zone** (1.81 ≤ Z ≤ 2.99): Moderate risk
            - **Distress Zone** (Z < 1.81): High bankruptcy risk
            """)
    
    # CAPM Analysis
    if show_capm:
        st.markdown("---")
        st.markdown("## 📈 CAPM Analysis")
        
        with st.spinner("Calculating CAPM metrics..."):
            capm_results = calculate_capm(stock_obj, ticker_input)
        
        if capm_results:
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Beta", f"{capm_results['beta']:.3f}")
                beta_interp = "More volatile" if capm_results['beta'] > 1 else "Less volatile" if capm_results['beta'] < 1 else "Similar"
                st.caption(f"{beta_interp} than market")
            
            with cols[1]:
                st.metric("Alpha", f"{capm_results['alpha']*100:.2f}%")
                alpha_interp = "Outperforming" if capm_results['alpha'] > 0 else "Underperforming"
                st.caption(f"{alpha_interp} expectations")
            
            with cols[2]:
                st.metric("Expected Return", f"{capm_results['expected_return']*100:.2f}%")
                st.caption("Per CAPM model")
            
            with cols[3]:
                st.metric("R-Squared", f"{capm_results['r_squared']:.3f}")
                st.caption(f"{capm_results['r_squared']*100:.1f}% explained by market")
            
            # CAPM Scatter Plot
            st.markdown("### Stock vs Market Returns")
            
            aligned = capm_results['aligned_data']
            
            fig_capm = go.Figure()
            
            # Scatter plot
            fig_capm.add_trace(go.Scatter(
                x=aligned['Market']*100,
                y=aligned['Stock']*100,
                mode='markers',
                name='Weekly Returns',
                marker=dict(size=5, color='blue', opacity=0.5)
            ))
            
            # Regression line
            x_line = np.linspace(aligned['Market'].min(), aligned['Market'].max(), 100)
            y_line = capm_results['beta'] * x_line * 100
            
            fig_capm.add_trace(go.Scatter(
                x=x_line*100,
                y=y_line,
                mode='lines',
                name=f'Beta = {capm_results["beta"]:.3f}',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_capm.update_layout(
                title=f'{ticker_input} vs S&P 500 Returns',
                xaxis_title='Market Return (%)',
                yaxis_title=f'{ticker_input} Return (%)',
                hovermode='closest',
                template='plotly_white',
                height=500
            )
            
            fig_capm.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_capm.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_capm, use_container_width=True)
            
            # Nonlinear Beta Analysis
            st.markdown("### Nonlinear Beta (Market Regime Analysis)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Beta (Up Markets)", f"{capm_results['beta_up']:.3f}")
                st.caption(f"{len(capm_results['up_market'])} weeks")
            with col2:
                st.metric("Beta (Down Markets)", f"{capm_results['beta_down']:.3f}")
                st.caption(f"{len(capm_results['down_market'])} weeks")
            with col3:
                asymmetry_pct = (capm_results['beta_asymmetry'] / capm_results['beta'] * 100) if capm_results['beta'] != 0 else 0
                st.metric("Beta Asymmetry", f"{capm_results['beta_asymmetry']:.3f}")
                st.caption(f"{asymmetry_pct:.1f}% deviation")
            
            # Interpretation
            if capm_results['beta_up'] > capm_results['beta_down'] + 0.15:
                st.success("🚀 **GROWTH AMPLIFIER:** Stock gains MORE in up-markets than loses in down-markets. Positive asymmetry favors long-term holders.")
            elif capm_results['beta_down'] > capm_results['beta_up'] + 0.15:
                st.warning("⚠️ **DEFENSIVE AMPLIFIER:** Stock loses MORE in down-markets than gains in up-markets. Higher downside risk.")
            else:
                st.info("⚖️ **SYMMETRIC:** Stock behavior is similar in both market conditions.")
            
            # Dual-beta scatter plot
            fig_dual = go.Figure()
            
            up = capm_results['up_market']
            down = capm_results['down_market']
            
            fig_dual.add_trace(go.Scatter(
                x=up['Market']*100,
                y=up['Stock']*100,
                mode='markers',
                name=f'Up-Market (β={capm_results["beta_up"]:.2f})',
                marker=dict(size=5, color='green', opacity=0.6)
            ))
            
            fig_dual.add_trace(go.Scatter(
                x=down['Market']*100,
                y=down['Stock']*100,
                mode='markers',
                name=f'Down-Market (β={capm_results["beta_down"]:.2f})',
                marker=dict(size=5, color='red', opacity=0.6)
            ))
            
            # Regression lines
            if len(up) > 1:
                z = np.polyfit(up['Market'], up['Stock'], 1)
                x_line = np.linspace(up['Market'].min(), up['Market'].max(), 50)
                y_line = np.poly1d(z)(x_line)
                fig_dual.add_trace(go.Scatter(
                    x=x_line*100, y=y_line*100,
                    mode='lines', name='Up-Market Trend',
                    line=dict(color='green', width=2, dash='dash')
                ))
            
            if len(down) > 1:
                z = np.polyfit(down['Market'], down['Stock'], 1)
                x_line = np.linspace(down['Market'].min(), down['Market'].max(), 50)
                y_line = np.poly1d(z)(x_line)
                fig_dual.add_trace(go.Scatter(
                    x=x_line*100, y=y_line*100,
                    mode='lines', name='Down-Market Trend',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig_dual.update_layout(
                title='Nonlinear Beta: Market Regime Analysis',
                xaxis_title='Market Return (%)',
                yaxis_title=f'{ticker_input} Return (%)',
                hovermode='closest',
                template='plotly_white',
                height=500
            )
            
            fig_dual.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_dual.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_dual, use_container_width=True)
    
    # News & Holdings (Optional sections)
    if show_news:
        # News & Information
        st.markdown("---")
        st.markdown("## 📰 Latest News")
    
    try:
        # Force fresh fetch of news
        news = []
        try:
            # Try to get news directly from ticker
            ticker_news = stock_obj.get_news() if hasattr(stock_obj, 'get_news') else stock_obj.news
            if ticker_news:
                news = ticker_news
        except:
            # Fallback: try news attribute
            if hasattr(stock_obj, 'news'):
                news = stock_obj.news
        
        if news and len(news) > 0:
            news_count = 0
            for article in news[:15]:  # Try more articles to find valid ones
                # Extract title from various possible fields
                title = None
                for title_field in ['title', 'headline', 'summary']:
                    if article.get(title_field):
                        title = str(article[title_field])
                        if len(title) > 10:  # Ensure it's a real title
                            break
                
                # Extract link
                link = None
                for link_field in ['link', 'url', 'canonical_url', 'guid']:
                    if article.get(link_field):
                        link = str(article[link_field])
                        break
                
                # Extract publisher/source
                publisher = None
                if article.get('publisher'):
                    publisher = str(article['publisher'])
                elif article.get('source'):
                    src = article['source']
                    if isinstance(src, dict):
                        publisher = src.get('name') or src.get('id')
                    else:
                        publisher = str(src)
                
                # Extract date
                pub_date = None
                for date_field in ['providerPublishTime', 'publish_time', 'published', 'pubDate']:
                    if article.get(date_field):
                        try:
                            date_val = article[date_field]
                            if isinstance(date_val, (int, float)):
                                pub_date = datetime.fromtimestamp(date_val).strftime('%b %d, %Y')
                            else:
                                pub_date = pd.to_datetime(date_val).strftime('%b %d, %Y')
                            break
                        except:
                            continue
                
                # Only show if we have a valid title
                if title and len(title) > 10:
                    display_link = link if link else '#'
                    display_publisher = publisher if publisher else 'Unknown Source'
                    display_date = f" • {pub_date}" if pub_date else ""
                    
                    st.markdown(f"""
                    **[{title}]({display_link})**  
                    *{display_publisher}*{display_date}
                    """)
                    
                    news_count += 1
                    if news_count >= 5:
                        break
            
            if news_count == 0:
                st.info("📭 No recent news articles available for this stock")
        else:
            st.info("📭 No recent news articles available for this stock")
    except Exception as e:
        st.info(f"📭 News data temporarily unavailable")
    
    # Institutional & Mutual Fund Holdings
    st.markdown("## 🏢 Institutional Holdings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            inst_holders = stock_obj.institutional_holders
            if inst_holders is not None and not inst_holders.empty:
                st.markdown("### Top Institutional Holders")
                for idx, row in inst_holders.head(5).iterrows():
                    holder = row.get('Holder', 'Unknown')
                    shares = row.get('Shares', 0)
                    pct_held = row.get('% Out', 0)
                    st.markdown(f"**{holder}**")
                    st.caption(f"{humanize_number(shares)} shares ({pct_held*100:.2f}%)")
            else:
                st.info("No institutional holdings data available")
        except:
            st.info("No institutional holdings data available")
    
    with col2:
        try:
            mutual_holders = stock_obj.mutualfund_holders
            if mutual_holders is not None and not mutual_holders.empty:
                st.markdown("### Top Mutual Fund Holders")
                for idx, row in mutual_holders.head(5).iterrows():
                    holder = row.get('Holder', 'Unknown')
                    shares = row.get('Shares', 0)
                    pct_held = row.get('% Out', 0)
                    st.markdown(f"**{holder}**")
                    st.caption(f"{humanize_number(shares)} shares ({pct_held*100:.2f}%)")
            else:
                st.info("No mutual fund holdings data available")
        except:
            st.info("No mutual fund holdings data available")
    
    # Financial Statements
    if show_financials:
        st.markdown("---")
        st.markdown("## 📊 Financial Statements")
    
    tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    
    with tab1:
        try:
            inc_stmt = stock_obj.income_stmt
            if inc_stmt is not None and not inc_stmt.empty:
                st.dataframe(inc_stmt.head(20), use_container_width=True)
            else:
                st.info("No income statement data available")
        except:
            st.info("No income statement data available")
    
    with tab2:
        try:
            balance = stock_obj.balance_sheet
            if balance is not None and not balance.empty:
                st.dataframe(balance.head(20), use_container_width=True)
            else:
                st.info("No balance sheet data available")
        except:
            st.info("No balance sheet data available")
    
    with tab3:
        try:
            cashflow = stock_obj.cashflow
            if cashflow is not None and not cashflow.empty:
                st.dataframe(cashflow.head(20), use_container_width=True)
            else:
                st.info("No cash flow data available")
        except:
            st.info("No cash flow data available")
        
        # Analyst Recommendations (in financials section)
        st.markdown("---")
        st.markdown("## 🎯 Analyst Coverage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            recommendations = stock_obj.recommendations_summary
            if recommendations is not None and not recommendations.empty:
                st.markdown("### Current Recommendations")
                latest = recommendations.iloc[-1]
                
                rec_data = {
                    'Strong Buy': latest.get('strongBuy', 0),
                    'Buy': latest.get('buy', 0),
                    'Hold': latest.get('hold', 0),
                    'Sell': latest.get('sell', 0),
                    'Strong Sell': latest.get('strongSell', 0)
                }
                
                rec_data = {k: v for k, v in rec_data.items() if v > 0}
                
                if rec_data:
                    fig_rec = go.Figure(data=[
                        go.Bar(
                            x=list(rec_data.keys()),
                            y=list(rec_data.values()),
                            marker_color=['#00aa00', '#66cc66', '#ffaa00', '#ff6666', '#cc0000'][:len(rec_data)]
                        )
                    ])
                    
                    fig_rec.update_layout(
                        title="Analyst Recommendations",
                        yaxis_title="Number of Analysts",
                        template='plotly_white',
                        height=300
                    )
                    
                    st.plotly_chart(fig_rec, use_container_width=True)
                else:
                    st.info("No recommendation data")
            else:
                st.info("No recommendations available")
        except:
            st.info("No recommendations available")
    
    with col2:
        try:
            upgrades = stock_obj.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                st.markdown("### Recent Upgrades/Downgrades")
                for idx, row in upgrades.head(5).iterrows():
                    firm = row.get('Firm', 'Unknown')
                    action = row.get('Action', 'Unknown')
                    grade = row.get('ToGrade', 'N/A')
                    st.markdown(f"**{firm}**")
                    st.caption(f"{action} → {grade}")
            else:
                st.info("No upgrades/downgrades data")
        except:
            st.info("No upgrades/downgrades data")
    
    # Market Sentiment - Fear & Greed Index
    st.markdown("---")
    st.markdown("## 😱 Market Sentiment - Fear & Greed Index")
    
    fg_data = get_fear_greed_index()
    
    if fg_data and fg_data.get('score') is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            score = fg_data['score']
            rating = fg_data['rating']
            
            st.metric("Current Score", f"{score:.1f}", rating)
            
            if score <= 25:
                st.error("🔴 Extreme Fear - Possible buying opportunity")
            elif score <= 45:
                st.warning("🟠 Fear - Market is cautious")
            elif score <= 55:
                st.info("🟡 Neutral - Balanced market")
            elif score <= 75:
                st.success("🟢 Greed - Market is optimistic")
            else:
                st.error("🔴 Extreme Greed - Possible selling signal")
        
        with col2:
            hist_df = fg_data.get('historical')
            if hist_df is not None and not hist_df.empty:
                fig_fg = go.Figure()
                
                fig_fg.add_trace(go.Scatter(
                    x=hist_df['timestamp'],
                    y=hist_df['score'],
                    mode='lines',
                    name='Fear & Greed Score',
                    line=dict(color='#1e3a8a', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(30, 58, 138, 0.2)'
                ))
                
                # Reference lines
                fig_fg.add_hline(y=25, line_dash="dash", line_color="red", opacity=0.5)
                fig_fg.add_hline(y=75, line_dash="dash", line_color="green", opacity=0.5)
                
                fig_fg.update_layout(
                    title="Fear & Greed Index - Historical Trend",
                    xaxis_title="Date",
                    yaxis_title="Score (0-100)",
                    template='plotly_white',
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_fg, use_container_width=True)
    else:
        st.info("Fear & Greed Index data not available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p>📊 Stock Analysis Dashboard | Data provided by Yahoo Finance</p>
        <p style='font-size: 12px;'>⚠️ This tool is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
