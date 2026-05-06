"""
Trading Analytics Dashboard — Streamlit
Visualizes S&P 100 stock screening, moving averages, ADX, Bollinger Bands,
and three trading strategies from the trading_project notebook.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────── Page Config ────────────────────────────
st.set_page_config(page_title="Trading Dashboard", page_icon="📈", layout="wide")

# ──────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg:#0a0e17;--card:rgba(17,24,39,.7);--accent:#6366f1;--accent2:#22d3ee;--green:#10b981;--red:#ef4444;--text:#e2e8f0;--muted:#94a3b8}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:#0a0e17!important;color:var(--text);font-family:'Inter',sans-serif}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1629 0%,#0a0e17 100%)!important;border-right:1px solid rgba(99,102,241,.15)}
[data-testid="stSidebar"] .stMarkdown h1{background:linear-gradient(135deg,#6366f1,#22d3ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.5rem}
div.stMetric{background:var(--card);border:1px solid rgba(99,102,241,.2);border-radius:12px;padding:16px;backdrop-filter:blur(10px)}
div.stMetric label{color:var(--muted)!important;font-weight:500}
div.stMetric [data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700}
.stSelectbox>div>div,.stMultiSelect>div>div{background:var(--card)!important;border-color:rgba(99,102,241,.3)!important;color:var(--text)!important}
.stTabs [data-baseweb="tab"]{color:var(--muted);font-weight:500}
.stTabs [aria-selected="true"]{color:var(--accent2)!important;border-bottom-color:var(--accent2)!important}
.hero{text-align:center;padding:20px 0 10px}
.hero h1{font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,#6366f1 0%,#a855f7 50%,#22d3ee 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.hero p{color:var(--muted);font-size:1rem}
.card{background:var(--card);border:1px solid rgba(99,102,241,.15);border-radius:14px;padding:20px;margin-bottom:16px;backdrop-filter:blur(12px)}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────── S&P 100 ───────────────────────────────
SP100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "ANET","APA","APD","AVGO","AXP","BA","BAC","BIIB","BK","BKNG",
    "BLK","BMY","BSX","C","CAT","CHTR","CL","CMCSA","COF","COP",
    "COST","CRM","CSCO","CVS","CVX","DHR","DIS","DOW","DUK","EMR",
    "EXC","F","FDX","GD","GE","GILD","GM","GOOG","GOOGL","GS",
    "HON","IBM","INTC","JNJ","JPM","KO","LIN","LLY","LMT","LOW",
    "MA","MAR","MCD","MDLZ","MDT","MET","MMM","MO","MRK","MS",
    "MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP","PFE","PG","PM",
    "PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO",
    "TMUS","TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WBA",
    "WFC","WMT","XOM"
]

# ──────────────────── Core analysis functions ────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock(ticker, period="1y"):
    dat = yf.Ticker(ticker)
    df = dat.history(period=period).reset_index()
    df = df[['Date','Open','High','Low','Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def bunch_mean(df, bunch, bunch_size, idx):
    col = f"SMA{bunch_size}"
    if col not in df.columns:
        df[col] = np.nan
    df.loc[idx, col] = bunch['High'].mean()

def weighted_bunch_mean(df, bunch, bunch_size, idx):
    col = f"WMA{bunch_size}"
    if col not in df.columns:
        df[col] = np.nan
    wm, ws = 0.0, 0.0
    for i, (_, row) in enumerate(bunch.iterrows()):
        w = i + 1
        wm += w * row['High']
        ws += w
    df.loc[idx, col] = wm / ws

def iter_fn(df, bunch_size, func):
    for i in range(len(df)):
        end = i + bunch_size
        if end > len(df):
            break
        bunch = df.iloc[i:end]
        func(df, bunch, bunch_size, end - 1)
    return df

def exponential_moving_average(df, n):
    alpha = 2 / (n + 1)
    ema = []
    for i, p in enumerate(df["Close"]):
        ema.append(p if i == 0 else alpha * p + (1 - alpha) * ema[-1])
    df[f"EMA_{n}"] = ema
    return df

def calc_wma_col(df, size):
    col = f"WMA{size}"
    if col not in df.columns:
        df[col] = np.nan
    for i in range(size - 1, len(df)):
        bunch = df.iloc[i + 1 - size:i + 1]
        wm, ws = 0.0, 0.0
        for j, (_, row) in enumerate(bunch.iterrows()):
            w = j + 1; wm += w * row['High']; ws += w
        df.loc[i, col] = wm / ws
    return col

def hull_moving_average(df, n):
    half = max(int(n / 2), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    col_full = calc_wma_col(df, n)
    col_half = calc_wma_col(df, half)
    df["HMA_temp"] = 2 * df[col_half].astype(float) - df[col_full].astype(float)
    out = f"HMA{n}"
    df[out] = np.nan
    for i in range(sqrt_n - 1, len(df)):
        bunch = df["HMA_temp"].iloc[i + 1 - sqrt_n:i + 1]
        if bunch.isna().any():
            continue
        wm, ws = 0.0, 0.0
        for j, val in enumerate(bunch):
            w = j + 1; wm += w * float(val); ws += w
        df.loc[i, out] = wm / ws
    return df

def highest_high(df, w):
    df[f'HH{w}'] = df['High'].rolling(window=w).max()
    return df

def lowest_low(df, w):
    df[f'LL{w}'] = df['Low'].rolling(window=w).min()
    return df

def calculate_adx(df, n=14):
    df = df.copy()
    df['H_diff'] = df['High'].diff()
    df['L_diff'] = -df['Low'].diff()
    df['DM+'] = np.where((df['H_diff'] > df['L_diff']) & (df['H_diff'] > 0), df['H_diff'], 0)
    df['DM-'] = np.where((df['L_diff'] > df['H_diff']) & (df['L_diff'] > 0), df['L_diff'], 0)
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['TR_s'] = df['TR'].ewm(alpha=1/n, adjust=False).mean()
    df['DM+_s'] = df['DM+'].ewm(alpha=1/n, adjust=False).mean()
    df['DM-_s'] = df['DM-'].ewm(alpha=1/n, adjust=False).mean()
    df['DI+'] = 100 * df['DM+_s'] / df['TR_s']
    df['DI-'] = 100 * df['DM-_s'] / df['TR_s']
    df['DX'] = 100 * ((df['DI+'] - df['DI-']).abs() / (df['DI+'] + df['DI-']))
    df['ADX'] = df['DX'].ewm(alpha=1/n, adjust=False).mean()
    return df

def add_bollinger(df, window=20):
    if f'SMA{window}' not in df.columns:
        df = iter_fn(df, window, bunch_mean)
    df['std_dev'] = df['High'].rolling(window=window).std()
    df['Upper_Band'] = df[f'SMA{window}'].astype(float) + 2 * df['std_dev']
    df['Lower_Band'] = df[f'SMA{window}'].astype(float) - 2 * df['std_dev']
    df['Z_Score'] = (df['High'] - df[f'SMA{window}'].astype(float)) / df['std_dev']
    return df

def sma_filter(df):
    mask = pd.Series(True, index=df.index)
    for s in [20, 50, 100]:
        col = f'SMA{s}'
        if col in df.columns:
            mask = mask & (df['High'] > df[col].astype(float))
    df['above_avg'] = mask
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def full_analysis(ticker, period="1y"):
    df = fetch_stock(ticker, period)
    n_rows = len(df)
    # Only compute indicators when we have enough data
    for s in [20, 50, 100]:
        if n_rows >= s:
            df = iter_fn(df, s, bunch_mean)
    sma_filter(df)
    for s in [20, 50, 100]:
        if n_rows >= s:
            df = iter_fn(df, s, weighted_bunch_mean)
    for s in [20, 50]:
        if n_rows >= s:
            df = exponential_moving_average(df, s)
    for s in [20, 50]:
        if n_rows >= s:
            df = hull_moving_average(df, s)
    df = highest_high(df, 20)
    df = lowest_low(df, 20)
    df = calculate_adx(df)
    df = add_bollinger(df)
    return df

# ─────────────────── Strategy functions ──────────────────────────────

def strategy_sma(df):
    eps = 0.001
    sub = df.tail(100).copy().reset_index(drop=True)
    rows, hold, bp, cp = [], 0, 0.0, 0.0
    for i in range(len(sub)):
        p, s = sub.loc[i,'High'], float(sub.loc[i,'SMA20'])
        if hold == 0:
            f = 1 if p > s + eps else 0
        else:
            f = 0 if p < s + eps else 1
        if hold == 0 and f == 1: bp = p; profit = 0.0
        elif f == 1: profit = p - bp
        elif hold == 1 and f == 0: profit = p - bp; cp = profit
        else: profit = cp
        rows.append({'Day':i+1,'Price':round(p,3),'SMA20':round(s,3),'Hold':hold,'Signal':f,'Profit':round(profit,3)})
        hold = f
    return pd.DataFrame(rows).set_index('Day')

def strategy_hh_ll(df):
    sub = df.tail(100).copy().reset_index(drop=True)
    sub['HH_prev'] = sub['HH20'].shift(1)
    sub['LL_prev'] = sub['LL20'].shift(1)
    rows, hold, bp, cp = [], 0, 0.0, 0.0
    for i in range(1, len(sub)):
        if pd.isna(sub.loc[i,'LL_prev']): continue
        p = sub.loc[i,'High']; hh = sub.loc[i,'HH_prev']; ll = sub.loc[i,'LL_prev']
        if hold == 0: f = 1 if p < ll else 0
        else: f = 0 if p > hh else 1
        if hold == 0 and f == 1: bp = p; profit = 0.0
        elif f == 1: profit = p - bp
        elif hold == 1 and f == 0: profit = p - bp; cp = profit
        else: profit = cp
        rows.append({'Day':i+1,'Price':round(p,3),'LL':round(ll,3),'HH':round(hh,3),'Hold':hold,'Signal':f,'Profit':round(profit,3)})
        hold = f
    return pd.DataFrame(rows).set_index('Day')

def strategy_mean_rev(df):
    sub = df.tail(100).copy().reset_index(drop=True)
    rows, hold, bp, cp = [], 0, 0.0, 0.0
    for i in range(len(sub)):
        p = sub.loc[i,'High']; sma = float(sub.loc[i,'SMA20']); z = float(sub.loc[i,'Z_Score']) if not pd.isna(sub.loc[i,'Z_Score']) else 0
        if hold == 0: f = 1 if z < -2 else 0
        else: f = 0 if p >= sma else 1
        if hold == 0 and f == 1: bp = p; profit = 0.0
        elif f == 1: profit = p - bp
        elif hold == 1 and f == 0: profit = p - bp; cp = profit
        else: profit = cp
        rows.append({'Day':i+1,'Price':round(p,3),'SMA20':round(sma,3),'Z':round(z,3),'Hold':hold,'Signal':f,'Profit':round(profit,3)})
        hold = f
    return pd.DataFrame(rows).set_index('Day')

# ─────────────────── Chart helpers ───────────────────────────────────
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(99,102,241,.1)"),
    yaxis=dict(gridcolor="rgba(99,102,241,.1)"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def styled(fig):
    fig.update_layout(**DARK)
    return fig

# ═══════════════════════════ SIDEBAR ═════════════════════════════════
with st.sidebar:
    st.markdown("# 📈 Trading Lab")
    st.markdown("---")
    tickers = st.multiselect("**Stock Tickers**", SP100, default=["AAPL"])
    period = st.selectbox("**Period**", ["3mo","6mo","1y","2y","5y"], index=2)
    st.markdown("---")
    st.caption("Data from Yahoo Finance · Built with Streamlit")

# ═══════════════════════════ MAIN ════════════════════════════════════
st.markdown('<div class="hero"><h1>📊 Trading Analytics Dashboard</h1><p>S&P 100 · Moving Averages · Strategies · Technical Indicators</p></div>', unsafe_allow_html=True)

if not tickers:
    st.warning("Please select at least one stock.")
    st.stop()

stock_tabs = st.tabs(tickers)
for stock_idx, ticker in enumerate(tickers):
    with stock_tabs[stock_idx]:
        with st.spinner(f"⏳ Crunching {ticker} data…"):
            df = full_analysis(ticker, period)

        # ───────────── KPI Row ──────────────
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", f"${latest['Close']:.2f}", f"{latest['Close']-prev['Close']:.2f}")
        c2.metric("High", f"${latest['High']:.2f}")
        c3.metric("Low", f"${latest['Low']:.2f}")
        c4.metric("ADX (14)", f"{latest['ADX']:.1f}")
        z_val = latest['Z_Score'] if not pd.isna(latest['Z_Score']) else 0
        c5.metric("Z-Score", f"{z_val:.2f}")

        st.markdown("")

        # ───────────── Tabs ──────────────
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🕯️ Price & SMAs", "📐 MA Comparison", "📉 ADX", "🎯 Bollinger", "💰 Strategies", "⚖️ Strategy Comparison"])

        # ── Tab 1: Price + SMAs ──
        with tab1:
            fig = go.Figure()
            fig.add_candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC")
            for s, c in [(20,"#6366f1"),(50,"#22d3ee"),(100,"#f59e0b")]:
                col = f'SMA{s}'
                if col in df.columns:
                    fig.add_scatter(x=df['Date'], y=df[col], mode='lines', name=f'SMA {s}', line=dict(color=c, width=2))
            above = df['High'].where(df.get('above_avg', pd.Series(False, index=df.index)), np.nan)
            fig.add_scatter(x=df['Date'], y=above, mode='lines', line=dict(color='#10b981', width=3, dash='dot'), name='Above all SMAs')
            fig.update_layout(title=f"{ticker} — Price & Simple Moving Averages", xaxis_rangeslider_visible=False)
            st.plotly_chart(styled(fig), use_container_width=True)

        # ── Tab 2: MA Comparison 20 vs 50 ──
        with tab2:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Window 20", "Window 50"), horizontal_spacing=0.06)
            colors = {"SMA":"#10b981","WMA":"#6366f1","EMA_":"#ef4444","HMA":"#22d3ee"}
            for col_idx, win in enumerate([20, 50], 1):
                fig.add_scatter(x=df['Date'], y=df['High'], mode='lines', name='Price', line=dict(color='rgba(148,163,184,0.35)', width=1), row=1, col=col_idx, showlegend=(col_idx==1))
                for prefix, clr in colors.items():
                    c = f"{prefix}{win}" if prefix != "EMA_" else f"EMA_{win}"
                    if c in df.columns:
                        fig.add_scatter(x=df['Date'], y=df[c], mode='lines', name=f'{prefix.rstrip("_")} {win}', line=dict(color=clr, width=1.5), row=1, col=col_idx, showlegend=(col_idx==1))
            fig.update_layout(title=f"{ticker} — Moving Average Comparison", height=450)
            st.plotly_chart(styled(fig), use_container_width=True)

        # ── Tab 3: ADX ──
        with tab3:
            fig = go.Figure()
            fig.add_scatter(x=df['Date'], y=df['ADX'], mode='lines', name='ADX 14', line=dict(color='#ef4444', width=2.5))
            fig.add_scatter(x=df['Date'], y=df['DI+'], mode='lines', name='DI+', line=dict(color='#10b981', width=1.5, dash='dot'))
            fig.add_scatter(x=df['Date'], y=df['DI-'], mode='lines', name='DI−', line=dict(color='#f59e0b', width=1.5, dash='dot'))
            fig.add_hline(y=25, line_dash="dash", line_color="rgba(99,102,241,.4)", annotation_text="Trend threshold")
            fig.update_layout(title=f"{ticker} — ADX & Directional Indicators", yaxis_title="Value")
            st.plotly_chart(styled(fig), use_container_width=True)

        # ── Tab 4: Bollinger ──
        with tab4:
            fig = go.Figure()
            fig.add_scatter(x=df['Date'], y=df['Upper_Band'], mode='lines', name='Upper Band', line=dict(color='#6366f1', width=1))
            fig.add_scatter(x=df['Date'], y=df['Lower_Band'], mode='lines', name='Lower Band', line=dict(color='#6366f1', width=1), fill='tonexty', fillcolor='rgba(99,102,241,.08)')
            fig.add_scatter(x=df['Date'], y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='#f59e0b', width=1.5, dash='dot'))
            fig.add_scatter(x=df['Date'], y=df['High'], mode='lines', name='Price', line=dict(color='#22d3ee', width=2))
            fig.update_layout(title=f"{ticker} — Bollinger Bands (20, 2σ)")
            st.plotly_chart(styled(fig), use_container_width=True)

            fig2 = go.Figure()
            fig2.add_scatter(x=df['Date'], y=df['Z_Score'], mode='lines', name='Z-Score', line=dict(color='#a855f7', width=2))
            fig2.add_hline(y=2, line_dash="dash", line_color="#ef4444", annotation_text="Overbought")
            fig2.add_hline(y=-2, line_dash="dash", line_color="#10b981", annotation_text="Oversold")
            fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.2)")
            fig2.update_layout(title="Z-Score", yaxis_title="σ", height=300)
            st.plotly_chart(styled(fig2), use_container_width=True)

        # ── Tab 5: Strategies ──
        with tab5:
            strat = st.selectbox("Choose Strategy", ["SMA Crossover", "HH/LL Breakout", "Mean Reversion (Z-Score)"])

            if strat == "SMA Crossover":
                tbl = strategy_sma(df)
            elif strat == "HH/LL Breakout":
                tbl = strategy_hh_ll(df)
            else:
                tbl = strategy_mean_rev(df)

            # Strategy metrics
            final_profit = tbl['Profit'].iloc[-1]
            trades = ((tbl['Signal'].diff().abs()) > 0).sum()
            win = tbl[tbl['Profit'] > 0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Profit", f"${final_profit:.2f}", delta_color="normal")
            m2.metric("Total Signals", int(trades))
            m3.metric("Days in Position", int(tbl['Signal'].sum()))

            # Profit chart
            fig = go.Figure()
            fig.add_scatter(x=tbl.index, y=tbl['Profit'], mode='lines+markers', name='Profit', line=dict(color='#10b981' if final_profit >= 0 else '#ef4444', width=2), marker=dict(size=3))
            fig.add_scatter(x=tbl.index, y=tbl['Price'], mode='lines', name='Price', line=dict(color='#6366f1', width=1.5), yaxis='y2')

            buys = tbl[tbl['Signal'].diff() == 1]
            sells = tbl[tbl['Signal'].diff() == -1]
            fig.add_scatter(x=buys.index, y=buys['Price'], mode='markers', name='Buy', marker=dict(color='#10b981', size=10, symbol='triangle-up'), yaxis='y2')
            fig.add_scatter(x=sells.index, y=sells['Price'], mode='markers', name='Sell', marker=dict(color='#ef4444', size=10, symbol='triangle-down'), yaxis='y2')

            fig.update_layout(
                title=f"{ticker} — {strat}",
                yaxis=dict(title="Profit ($)", side="left"),
                yaxis2=dict(title="Price ($)", side="right", overlaying="y"),
                height=500,
            )
            st.plotly_chart(styled(fig), use_container_width=True)

            with st.expander("📋 Full Strategy Table"):
                st.dataframe(tbl, use_container_width=True, height=400)

        # ── Tab 6: Strategy Comparison ──
        with tab6:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Strategy Profit Comparison")
            st.markdown("All three strategies' cumulative profit on the last 100 trading days.")
            st.markdown('</div>', unsafe_allow_html=True)

            with st.spinner("Computing strategies…"):
                tbl_sma = strategy_sma(df)
                tbl_hh  = strategy_hh_ll(df)
                tbl_mr  = strategy_mean_rev(df)

            fig_cmp = go.Figure()

            fig_cmp.add_scatter(
                x=tbl_sma.index, y=tbl_sma['Profit'],
                mode='lines', name='SMA Crossover',
                line=dict(color='#10b981', width=2.5)
            )
            fig_cmp.add_scatter(
                x=tbl_hh.index, y=tbl_hh['Profit'],
                mode='lines', name='HH/LL Breakout',
                line=dict(color='#f59e0b', width=2.5)
            )
            fig_cmp.add_scatter(
                x=tbl_mr.index, y=tbl_mr['Profit'],
                mode='lines', name='Mean Reversion',
                line=dict(color='#22d3ee', width=2.5)
            )

            fig_cmp.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.2)")

            fig_cmp.update_layout(
                title=f"{ticker} — Strategy Profit Comparison",
                xaxis_title="Day",
                yaxis_title="Profit ($)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            st.plotly_chart(styled(fig_cmp), use_container_width=True)

            # Summary metrics
            s1, s2, s3 = st.columns(3)
            p_sma = tbl_sma['Profit'].iloc[-1]
            p_hh  = tbl_hh['Profit'].iloc[-1]
            p_mr  = tbl_mr['Profit'].iloc[-1]
            s1.metric("SMA Crossover", f"${p_sma:.2f}", delta_color="normal")
            s2.metric("HH/LL Breakout", f"${p_hh:.2f}", delta_color="normal")
            s3.metric("Mean Reversion", f"${p_mr:.2f}", delta_color="normal")

            # Data table
            with st.expander("📋 Profit Data"):
                cmp_df = pd.DataFrame({
                    'Day': tbl_sma.index,
                    'SMA Crossover': tbl_sma['Profit'].values,
                })
                cmp_df = cmp_df.set_index('Day')
                cmp_df['HH/LL Breakout'] = tbl_hh['Profit'].values[:len(cmp_df)] if len(tbl_hh) >= len(cmp_df) else pd.Series(tbl_hh['Profit'].values)
                cmp_df['Mean Reversion'] = tbl_mr['Profit'].values[:len(cmp_df)] if len(tbl_mr) >= len(cmp_df) else pd.Series(tbl_mr['Profit'].values)
                st.dataframe(cmp_df, use_container_width=True, height=400)
