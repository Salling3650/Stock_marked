# 📈 Stock Analysis Dashboard

A comprehensive, interactive web-based stock analysis tool built with Streamlit and Python.

## 🚀 Features

### 📊 Interactive Visualizations
- **Candlestick Charts** - Professional price visualization with OHLC data
- **Technical Indicators** - Moving averages, Bollinger Bands, RSI, MACD
- **Volume Analysis** - Color-coded volume bars
- **Real-time Updates** - Live data from Yahoo Finance

### 📈 Financial Metrics
- **Price Analysis** - Current price, 52-week high/low, price changes
- **Valuation Metrics** - P/E ratio, EPS, market cap, P/B ratio
- **Performance Metrics** - Returns over various time periods

### ⚠️ Risk Analysis
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside risk measurement
- **Max Drawdown** - Maximum peak-to-trough decline
- **VaR (95%)** - Value at Risk calculation
- **Altman Z-Score** - Bankruptcy prediction model

### 📐 CAPM Analysis
- **Beta Calculation** - Systematic risk measurement (150 weeks of data)
- **Alpha (Jensen's)** - Excess returns over expected
- **Security Market Line** - Visual CAPM representation
- **R-Squared** - Market correlation strength

### 🔧 Technical Indicators
- **RSI (14)** - Overbought/oversold conditions
- **MACD** - Trend following momentum
- **Stochastic Oscillator** - Price momentum
- **Bollinger Bands** - Volatility and price levels

## 🛠️ Installation

### Local Setup

1. **Clone or download this repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
The app will automatically open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud (Free!)

### Option 1: Deploy from GitHub

1. **Push your code to GitHub**
```bash
git add .
git commit -m "Add Streamlit dashboard"
git push origin main
```

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

3. **Click "New app"**

4. **Connect your GitHub repository**
   - Repository: `your-username/Stock_marked`
   - Branch: `main`
   - Main file path: `app.py`

5. **Click "Deploy"** 🚀

Your app will be live at: `https://your-app-name.streamlit.app`

### Option 2: Quick Deploy Button

Add this to your GitHub README:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
```

## 📱 Usage

### Basic Usage

1. **Enter a ticker symbol** in the sidebar (e.g., AAPL, MSFT, GOOGL)
2. **Select time period** for historical analysis
3. **Choose analysis options**:
   - Technical Indicators
   - CAPM Analysis
   - Risk Metrics
4. **Click "Fetch Data"** to load the analysis

### Advanced Features

- **Hover over charts** to see detailed data points
- **Zoom and pan** on interactive charts
- **Compare different time periods** using the period selector
- **Export data** by right-clicking charts

## 🎯 Use Cases

- **Individual Investors** - Research stocks before investing
- **Students** - Learn financial analysis and CAPM
- **Analysts** - Quick fundamental and technical analysis
- **Traders** - Technical indicator monitoring
- **Portfolio Managers** - Risk assessment and diversification

## 🔐 Data & Privacy

- All data fetched from **Yahoo Finance API** (free & public)
- **No personal data stored** - everything runs in your session
- **No login required** - completely open access
- Data is cached for 5 minutes to improve performance

## 🛡️ Disclaimer

**⚠️ Important:** This tool is for educational and informational purposes only. It is NOT financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 🐛 Troubleshooting

### "Could not fetch data" error
- Check if the ticker symbol is correct
- Verify internet connection
- Some tickers may not have complete historical data

### Charts not loading
- Refresh the page
- Try a different time period
- Clear browser cache

### Slow performance
- Reduce the time period
- Close other browser tabs
- Check internet speed

## 🔄 Updates & Maintenance

The app automatically fetches the latest data each time you run it. No manual updates needed!

## 📊 Technologies Used

- **Streamlit** - Web framework
- **yfinance** - Yahoo Finance API wrapper
- **Plotly** - Interactive charts
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **SciPy** - Statistical analysis

## 🤝 Contributing

Feel free to fork this project and add your own features:
- Additional technical indicators
- More chart types
- News integration
- Portfolio tracking
- Backtesting capabilities

## 📄 License

This project is open source and available for educational use.

## 🌟 Support

If you find this helpful, consider:
- ⭐ Starring the GitHub repository
- 🐛 Reporting bugs and issues
- 💡 Suggesting new features
- 🔄 Sharing with others

## 📞 Contact

Created by: Your Name
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)

---

**Made with ❤️ and Python**
