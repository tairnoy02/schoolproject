# 📊 Trading Analytics Dashboard

An interactive **Streamlit** web app for screening and analyzing **S&P 100** stocks using technical indicators and trading strategies — powered by real-time data from Yahoo Finance.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Candlestick Charts** | OHLC price visualization with SMA overlays (20 / 50 / 100) |
| **Moving Average Comparison** | Side-by-side view of SMA, WMA, EMA, and Hull MA for windows 20 & 50 |
| **ADX Indicator** | Average Directional Index with DI+ / DI− for trend strength analysis |
| **Bollinger Bands** | Upper & lower bands (2σ) with Z-Score chart for overbought/oversold detection |
| **Trading Strategies** | Three back-tested strategies with buy/sell signals and profit tracking |

### 📈 Strategies

1. **SMA Crossover** — Buy when price crosses above SMA20, sell when it drops below.
2. **HH/LL Breakout** — Buy at 20-day lowest low, sell at 20-day highest high.
3. **Mean Reversion (Z-Score)** — Buy when Z-Score < −2 (oversold), sell when price reverts to the mean.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/tairnoy02/schoolproject.git
cd schoolproject

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run dashboard.py
```

The app will open at **http://localhost:8501**.

---

## 📂 Project Structure

```
schoolproject/
├── dashboard.py              # Streamlit dashboard application
├── trading_project (9).py    # Original analysis notebook (Colab export)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — Interactive web UI
- **[yfinance](https://github.com/ranaroussi/yfinance)** — Real-time stock data from Yahoo Finance
- **[Plotly](https://plotly.com/python/)** — Interactive charts and visualizations
- **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)** — Data processing and numerical computation

---

## 📊 Technical Indicators Explained

| Indicator | Formula / Method |
|---|---|
| **SMA** | Simple average of the last *n* High prices |
| **WMA** | Weighted average giving more weight to recent prices |
| **EMA** | Exponential smoothing: `α·price + (1−α)·prev_EMA` |
| **HMA** | Hull MA: `WMA(2·WMA(n/2) − WMA(n), √n)` — reduces lag |
| **ADX** | Measures trend strength (0–100) using DI+ and DI− |
| **Bollinger Bands** | SMA ± 2× standard deviation |
| **Z-Score** | `(Price − SMA) / σ` — standard deviations from mean |

---

## 🌐 Deployment

This app can be deployed for free on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Select the repository, branch (`main`), and main file (`dashboard.py`)
4. Click **Deploy**

---

## 📝 License

This project is for educational purposes as part of a school project.
