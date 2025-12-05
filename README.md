<div align="center">

# Comprehensive Stock Analysis Dashboard

### *Professional-grade stock market analysis powered by Python*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![yfinance](https://img.shields.io/badge/Data-yfinance-red.svg)](https://pypi.org/project/yfinance/)

*A comprehensive Jupyter Notebook for in-depth stock market analysis, combining technical indicators, fundamental data, options analytics, and market intelligence in one powerful tool.*

[Features](#features) • [Quick Start](#quick-start) • [Analysis Sections](#analysis-sections) • [Examples](#stock-examples)

---

</div>

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Salling3650/Stock_marked.git
cd Stock_marked

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Running the Streamlit Dashboard

```bash
streamlit run src/app.py
```

#### Using Jupyter Notebooks

1. Navigate to the `notebooks/` folder
2. Open any notebook (e.g., `01_stock_analysis.ipynb`)
3. Run the setup cells to import dependencies
4. Edit the stock symbol and run analysis sections

---

## Project Structure

```
Stock_marked/
├── src/                         # Source code
│   ├── app.py                   # Streamlit dashboard
│   └── functions.py             # Core utility functions
├── notebooks/                   # Jupyter notebooks
│   ├── 01_stock_analysis.ipynb
│   ├── 02_portfolio_analysis.ipynb
│   ├── 03_cointegration_analysis.ipynb
│   └── 04_economy.ipynb
├── data/                        # Data files (gitignored)
├── assets/                      # Static files and outputs
├── config/                      # Configuration files (gitignored)
├── docs/                        # Documentation
└── tests/                       # Unit tests
```

---

## Features

<table>
<tr>
<td width="50%">

### Technical Analysis
- Real-time price charts
- 20+ technical indicators
- Risk metrics (Sharpe, Sortino, VaR)
- Bollinger Bands & MACD
- Volume analysis

</td>
<td width="50%">

### Fundamental Data
- Company financials
- Earnings reports
- Balance sheets
- Institutional holdings
- Insider transactions

</td>
</tr>
<tr>
<td width="50%">

### Options Analytics
- Full option chains
- IV term structure
- Put/Call ratios
- Open interest analysis
- Strike price distribution

</td>
<td width="50%">

### Market Intelligence
- Fear & Greed Index
- Global market map
- Competitor analysis
- Analyst ratings
- News feed

</td>
</tr>
</table>

---

## Analysis Sections

## Analysis Sections

### Technical Analysis Dashboard
> *Complete technical analysis with 11 interactive charts*

| Indicator | Description |
|-----------|-------------|
| **Price Chart** | 20-day & 50-day moving averages with Bollinger Bands |
| **Volume** | Trading volume (millions) color-coded by direction |
| **RSI** | Relative Strength Index (70/30 levels) |
| **MACD** | Moving Average Convergence Divergence |
| **Stochastic** | %K and %D oscillators (80/20 levels) |
| **Volatility** | Annualized volatility percentage |
| **Returns** | Cumulative returns & distribution histogram |
| **Drawdown** | Peak-to-trough decline analysis |
| **OBV** | On-Balance Volume indicator |
| **Sharpe** | 60-day rolling risk-adjusted returns |

### Key Metrics Dashboard
> *Visual cards with mini trend charts*

```
✓ Current Price & Daily Change    ✓ Market Cap & Shares Outstanding
✓ P/E Ratio & EPS                 ✓ 52-Week High/Low
✓ Beta & Volatility               ✓ Dividend Yield
✓ RSI & MACD                      ✓ Bollinger %B
```

### Company Information
> *Deep dive into company fundamentals*

- **Competitors**: Top 10 competitors from CNN Business API
- **Institutional Holdings**: Major institutional positions with ownership %
- **Mutual Funds**: Top mutual fund holdings
- **Officers**: Executive team roster
- **News Feed**: Latest articles with AI summaries
- **Insider Activity**: Recent buying/selling transactions

### Financial Statements
> *Quarterly and annual financial data with trend analysis*

- **Income Statements**: Revenue, Net Income, Profit Margins
- **Balance Sheets**: Assets, Debt, Debt-to-Asset ratios
- **Detailed Breakdown**: Complete line-by-line financials
- **Insider Roster**: Current insider holdings

### Analyst Coverage
> *Professional analyst insights and price targets*

- **Price Targets**: Historical overlays with distributions
- **Upgrades/Downgrades**: Recent rating changes

### Market Intelligence
> *Global market sentiment and insights*

- **Fear & Greed Index**: CNN's market sentiment indicator
- **Historical Trends**: Sentiment evolution over time
- **AI Insights**: Generated analysis from CNN Business
- **Global Markets**: Interactive world map of daily index performance

### Options Analysis
> *Comprehensive options chain analytics*

| Feature | Details |
|---------|---------|
| **Option Chains** | Calls & puts for nearest expiration |
| **Put/Call Ratio** | Open interest ratios across expirations |
| **Expected Move** | IV-based price movement predictions |
| **OI Distribution** | Visual charts with median/mean strikes |
| **IV Term Structure** | Implied volatility across time |
| **Strike Analysis** | Detailed OI, volume, and IV by strike |

---

## Stock Examples

<table>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/AAPL-Apple-999999?style=for-the-badge&logo=apple" alt="Apple"/>
<br><code>user_input = 'AAPL'</code>
</td>
<td align="center">
<img src="https://img.shields.io/badge/GOOGL-Google-4285F4?style=for-the-badge&logo=google" alt="Google"/>
<br><code>user_input = 'GOOGL'</code>
</td>
<td align="center">
<img src="https://img.shields.io/badge/TSLA-Tesla-CC0000?style=for-the-badge&logo=tesla" alt="Tesla"/>
<br><code>user_input = 'TSLA'</code>
</td>
</tr>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/NVDA-NVIDIA-76B900?style=for-the-badge&logo=nvidia" alt="NVIDIA"/>
<br><code>user_input = 'NVDA'</code>
</td>
<td align="center">
<img src="https://img.shields.io/badge/MSFT-Microsoft-00A4EF?style=for-the-badge&logo=microsoft" alt="Microsoft"/>
<br><code>user_input = 'MSFT'</code>
</td>
<td align="center">
<img src="https://img.shields.io/badge/SPY-S&P_500-FFD700?style=for-the-badge" alt="S&P 500"/>
<br><code>user_input = 'SPY'</code>
</td>
</tr>
</table>

---

## Dependencies

```python
yfinance      # Real-time stock data
pandas        # Data manipulation
numpy         # Numerical operations
matplotlib    # Visualization
scipy         # Statistical analysis
statsmodels   # Time series analysis
requests      # API calls
folium        # Interactive maps
```

**Installation:**
```bash
pip install yfinance pandas numpy matplotlib scipy statsmodels requests folium
```

---

## Project Structure

```
Stock_marked/
│
├── Stock analysis.ipynb           # Main analysis notebook
├── functions.py                   # Helper functions and utilities
├── README.md                      # Documentation (you are here!)
├── stock_index_world_map.html     # Generated world map
└── fredapi.txt                    # API configurations
```

---

## Usage Tips

| Tip | Description |
|-----|-------------|
| **Sequential Run** | Run cells sequentially for first-time use |
| **Independent Sections** | Each section can be run separately after setup |
| **Time Periods** | Adjust `period` variable: `'1mo'`, `'3mo'`, `'1y'`, `'5y'` |
| **Navigation** | Use Table of Contents for quick section access |
| **Multiple Stocks** | Change `user_input` and re-run selection cell |

---

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is open source and available under the **MIT License**.

```
MIT License - feel free to use this project for personal or commercial purposes
```

---

## Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

---
