import streamlit as st
import requests
import json
import time
import math
import pandas as pd
from datetime import datetime

# --- Configuration and Constants ---
BINANCE_API_BASE = "https://api.binance.com/api/v3"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
# API key is handled by the environment, leave as empty string
GEMINI_API_KEY = ""

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Live Crypto Trading Bot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Initialize Session State (replacing React's useState) ---
# Streamlit reruns the script from top to bottom on every interaction.
# st.session_state is used to persist variables across these reruns.
if 'selected_pair' not in st.session_state:
    st.session_state.selected_pair = 'BTCUSDT'
if 'symbol_input' not in st.session_state:
    st.session_state.symbol_input = 'BTC'
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = '1h'
if 'is_trading' not in st.session_state:
    st.session_state.is_trading = False
if 'is_connected' not in st.session_state:
    st.session_state.is_connected = False # Represents connection to Binance API
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'live_price' not in st.session_state:
    st.session_state.live_price = None
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=['time', 'price'])
if 'order_book' not in st.session_state:
    # Note: Order book will not be live with the current simplified API calls.
    st.session_state.order_book = {'bids': [], 'asks': []}
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'total_value': 10000,
        'pnl': 0,
        'pnl_percent': 0,
        'positions': []
    }
if 'technical_analysis' not in st.session_state:
    st.session_state.technical_analysis = {
        'trend': 'Neutral',
        'strength': 5,
        'support': 0,
        'resistance': 0,
        'rsi': 50,
        'macd': 'Neutral',
        'prediction': 'Analyzing...'
    }
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = 0
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ""
if 'llm_loading' not in st.session_state:
    st.session_state.llm_loading = False
if 'show_llm_modal' not in st.session_state:
    st.session_state.show_llm_modal = False
if 'llm_modal_title' not in st.session_state:
    st.session_state.llm_modal_title = ""

# --- Utility Functions ---

def calculate_ema(prices_arr, period):
    """Calculates Exponential Moving Average (EMA)."""
    if not prices_arr:
        return 0
    k = 2 / (period + 1)
    ema = prices_arr[0]
    for i in range(1, len(prices_arr)):
        ema = (prices_arr[i] * k) + (ema * (1 - k))
    return ema

def calculate_rsi(prices_arr, period=14):
    """Calculates Relative Strength Index (RSI)."""
    if len(prices_arr) < period + 1:
        return 50 # Default if not enough data

    gains = 0
    losses = 0

    for i in range(1, period + 1):
        change = prices_arr[len(prices_arr) - i] - prices_arr[len(prices_arr) - i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100 # Strong uptrend
    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))

# --- Core Data Fetching and Logic ---

def fetch_latest_price(symbol):
    """Fetches the latest 24hr ticker price for a given symbol."""
    try:
        response = requests.get(f"{BINANCE_API_BASE}/ticker/24hr?symbol={symbol}")
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        st.session_state.is_connected = True
        return {
            'price': float(data['lastPrice']),
            'change': float(data['priceChangePercent']),
            'volume': float(data['volume']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'symbol': data['symbol']
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching live price for {symbol}: {e}")
        st.session_state.is_connected = False
        return None

def fetch_historical_data():
    """Fetches historical K-line data from Binance API."""
    try:
        response = requests.get(
            f"{BINANCE_API_BASE}/klines",
            params={
                'symbol': st.session_state.selected_pair,
                'interval': st.session_state.timeframe,
                'limit': 100
            }
        )
        response.raise_for_status()
        data = response.json()

        formatted_data = []
        for item in data:
            timestamp = item[0]
            close_price = float(item[4])
            volume = float(item[5])
            formatted_data.append({
                'time': datetime.fromtimestamp(timestamp / 1000).strftime('%H:%M:%S'),
                'timestamp': timestamp,
                'price': close_price,
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'volume': volume
            })
        st.session_state.price_data = pd.DataFrame(formatted_data)
        st.session_state.price_data['time'] = pd.to_datetime(st.session_state.price_data['timestamp'], unit='ms')
        st.session_state.price_data = st.session_state.price_data.set_index('time')

        calculate_technical_indicators()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data: {e}")
        st.session_state.is_connected = False

def calculate_technical_indicators():
    """Calculates various technical indicators based on current price data."""
    if len(st.session_state.price_data) < 26:
        st.session_state.technical_analysis = {
            'trend': 'Analyzing...',
            'strength': 0,
            'support': 0,
            'resistance': 0,
            'rsi': 0,
            'macd': 'Analyzing...',
            'prediction': 'Not enough data for comprehensive analysis.'
        }
        return

    prices = st.session_state.price_data['price'].tolist()

    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26

    rsi = calculate_rsi(prices)

    recent_prices_for_sr = prices[-20:]
    support = min(recent_prices_for_sr) * 0.995
    resistance = max(recent_prices_for_sr) * 1.005

    trend_lookback = min(10, len(prices))
    recent_prices_for_trend = prices[-trend_lookback:]
    is_uptrend = recent_prices_for_trend[-1] > recent_prices_for_trend[0]
    trend = 'Bullish' if is_uptrend else 'Bearish'

    strength = 5
    if rsi > 70: strength = min(10, strength + 2)
    if rsi < 30: strength = min(10, strength + 2)
    if is_uptrend: strength = min(10, strength + 2)
    else: strength = max(0, strength - 1)

    st.session_state.technical_analysis = {
        'trend': trend,
        'strength': round(strength),
        'support': support,
        'resistance': resistance,
        'rsi': f"{rsi:.1f}",
        'macd': 'Bullish' if macd_line > 0 else 'Bearish',
        'prediction': f"{trend} momentum with {('strong' if strength > 7 else 'moderate' if strength > 4 else 'weak')} signals"
    }

def generate_trading_signal():
    """Generates trading signals based on current market data and TA."""
    if not st.session_state.live_price or len(st.session_state.price_data) < 26:
        return

    now = time.time() * 1000 # Milliseconds
    if now - st.session_state.last_signal_time < 30000: # Limit to one signal per 30 seconds
        return

    ta = st.session_state.technical_analysis
    current_price = st.session_state.live_price['price']
    rsi_value = float(ta['rsi'])

    signal_type = 'HOLD'
    reason = 'Market consolidation or insufficient strong signals'
    signal_strength = 5

    if rsi_value < 30 and ta['trend'] == 'Bullish':
        signal_type = 'BUY'
        reason = 'RSI oversold in bullish trend: potential bounce'
        signal_strength = 8
    elif rsi_value > 70 and ta['trend'] == 'Bearish':
        signal_type = 'SELL'
        reason = 'RSI overbought in bearish trend: potential reversal'
        signal_strength = 8
    elif current_price <= ta['support'] * 1.005 and current_price > ta['support'] * 0.99:
        signal_type = 'BUY'
        reason = 'Price approaching/bouncing off support level'
        signal_strength = 7
    elif current_price >= ta['resistance'] * 0.995 and current_price < ta['resistance'] * 1.01:
        signal_type = 'SELL'
        reason = 'Price approaching/rejecting resistance level'
        signal_strength = 7
    elif ta['trend'] == 'Bullish' and ta['strength'] >= 7:
        signal_type = 'BUY'
        reason = 'Strong bullish momentum detected'
        signal_strength = ta['strength']
    elif ta['trend'] == 'Bearish' and ta['strength'] >= 7:
        signal_type = 'SELL'
        reason = 'Strong bearish momentum detected'
        signal_strength = ta['strength']

    signal = {
        'id': now,
        'type': signal_type,
        'pair': st.session_state.selected_pair,
        'price': current_price,
        'strength': signal_strength,
        'reason': reason,
        'timestamp': datetime.fromtimestamp(now / 1000).strftime('%H:%M:%S'),
        'entry_price': current_price,
        'stop_loss': current_price * (0.98 if signal_type == 'BUY' else 1.02),
        'take_profit': current_price * (1.03 if signal_type == 'BUY' else 0.97),
        'rsi': rsi_value,
        'trend': ta['trend']
    }

    st.session_state.signals.insert(0, signal) # Add to beginning
    st.session_state.signals = st.session_state.signals[:20] # Keep last 20
    st.session_state.last_signal_time = now

    # Update portfolio P&L based on the last signal
    if signal_type != 'HOLD':
        pnl = (st.session_state.live_price['price'] - signal['entry_price']) * (1 if signal_type == 'BUY' else -1)
        pnl_percent = (pnl / signal['entry_price']) * 100
        st.session_state.portfolio['pnl'] = pnl
        st.session_state.portfolio['pnl_percent'] = pnl_percent

def call_gemini_api(prompt, title):
    """Calls the Gemini API to generate text."""
    st.session_state.llm_loading = True
    st.session_state.llm_response = "Generating..."
    st.session_state.llm_modal_title = title
    st.session_state.show_llm_modal = True
    st.rerun() # Rerun to show loading state

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    headers = {'Content-Type': 'application/json'}
    params = {'key': GEMINI_API_KEY}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            text = result['candidates'][0]['content']['parts'][0]['text']
            st.session_state.llm_response = text
        else:
            st.session_state.llm_response = 'Could not generate insight. Please try again.' # ინსაითის გენერირება ვერ მოხერხდა. სცადეთ ხელახლა.
            st.error(f"Gemini API response structure unexpected: {result}")
    except requests.exceptions.RequestException as e:
        st.session_state.llm_response = 'Error connecting to AI service. Please try again later.' # AI სერვისთან დაკავშირების შეცდომა. სცადეთ მოგვიანებით.
        st.error(f"Error calling Gemini API: {e}")
    finally:
        st.session_state.llm_loading = False
        st.rerun() # Rerun to show final response

def generate_crypto_overview(symbol):
    """Generates a crypto overview using Gemini API."""
    prompt = f"Provide a concise, neutral, and informative overview of {symbol}, including its primary purpose, technology, and general market position, suitable for a cryptocurrency trading application. Keep it to 3-4 sentences."
    call_gemini_api(prompt, f"Overview of {symbol}") # მიმოხილვა

def elaborate_signal(signal):
    """Elaborates on a trading signal using Gemini API."""
    prompt = f"Explain the trading signal: {signal['type']} for {signal['pair']} at price {signal['price']:.4f}. The reason provided was '{signal['reason']}'. Technical indicators at the time were RSI: {signal['rsi']}, Trend: {signal['trend']}, Strength: {signal['strength']}/10, Support: {signal['support']:.4f}, Resistance: {signal['resistance']:.4f}. Elaborate on why this signal might be valid or what it implies, in the context of these indicators, suitable for a trading bot user. Keep it concise, around 3-5 sentences."
    call_gemini_api(prompt, f"Insight for {signal['type']} Signal on {signal['pair']}") # ინსაითი სიგნალისთვის

# --- Streamlit UI Components ---

# Custom CSS for styling (approximating Tailwind classes)
st.markdown("""
    <style>
    .stApp {
        background-color: #1a202c; /* bg-gray-900 */
        color: #e2e8f0; /* text-white */
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: #2d3748; /* bg-gray-800 */
        border-radius: 0.5rem; /* rounded-lg */
        padding: 1rem; /* p-4 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-md */
    }
    .stButton>button {
        border-radius: 0.375rem; /* rounded-md */
        font-weight: 600; /* font-semibold */
        transition: background-color 200ms ease-in-out;
        display: flex; /* For icon alignment */
        align-items: center;
        justify-content: center;
        gap: 0.5rem; /* Space between icon and text */
    }
    .stSelectbox>div>div {
        background-color: #4a5568; /* bg-gray-700 */
        color: #e2e8f0; /* text-white */
        border-radius: 0.375rem; /* rounded-md */
        border: 1px solid #4a5568; /* border-gray-600 */
    }
    .stTextInput>div>div>input {
        background-color: #4a5568; /* bg-gray-700 */
        color: #e2e8f0; /* text-white */
        border-radius: 0.375rem; /* rounded-md */
        border: 1px solid #4a5568; /* border-gray-600 */
    }
    .header-container {
        background-color: #2d3748; /* bg-gray-800 */
        border-bottom: 1px solid #4a5568; /* border-b border-gray-700 */
        padding: 1rem; /* p-4 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-lg */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem; /* Reduced gap */
    }
    @media (min-width: 768px) { /* md breakpoint */
        .header-container {
            flex-direction: row;
        }
    }
    .panel {
        background-color: #2d3748; /* bg-gray-800 */
        border-radius: 0.5rem; /* rounded-lg */
        padding: 1rem; /* p-4 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-lg */
    }
    .signal-card {
        background-color: #2d3748; /* bg-gray-800 */
        border-radius: 0.5rem; /* rounded-lg */
        padding: 1rem; /* p-4 */
        border-left: 4px solid #3b82f6; /* border-l-4 border-blue-500 */
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); /* shadow-md */
    }
    .signal-type-buy { background-color: #10b981; } /* bg-green-500 */
    .signal-type-sell { background-color: #ef4444; } /* bg-red-500 */
    .signal-type-hold { background-color: #f59e0b; } /* bg-yellow-500 */
    .text-green-400 { color: #4ade80; }
    .text-red-400 { color: #f87171; }
    .text-yellow-400 { color: #facc15; }
    .text-blue-400 { color: #60a5fa; }
    .text-purple-400 { color: #a78bfa; }
    .text-gray-400 { color: #9ca3af; }
    .text-gray-300 { color: #d1d5db; }
    .text-gray-500 { color: #6b7280; }
    .bg-green-600 { background-color: #16a34a; }
    .bg-red-600 { background-color: #dc2626; }
    .bg-blue-600 { background-color: #2563eb; }
    .bg-indigo-600 { background-color: #4f46e5; }
    .bg-purple-600 { background-color: #9333ea; }
    .hover\\:bg-green-700:hover { background-color: #15803d; }
    .hover\\:bg-red-700:hover { background-color: #b91c1c; }
    .hover\\:bg-blue-700:hover { background-color: #1d4ed8; }
    .hover\\:bg-indigo-700:hover { background-color: #4338ca; }
    .hover\\:bg-purple-700:hover { background-color: #7e22ce; }

    /* Animation for loading spinner */
    .animate-spin {
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .stPlotlyChart {
        height: 300px !important; /* Fixed height for chart */
    }
    /* Adjust Streamlit's default margins for compactness */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTextInput label, .stSelectbox label {
        display: none; /* Hide default Streamlit labels for compactness */
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown('<h1 class="text-2xl font-bold text-center md:text-left">Live Crypto Trading Bot</h1>', unsafe_allow_html=True) # ცოცხალი კრიპტო სავაჭრო ბოტი

# Use st.container for better control over inner spacing
with st.container():
    header_cols = st.columns([2, 1, 1, 1]) # Adjusted column widths for compactness

    with header_cols[0]:
        st.session_state.symbol_input = st.text_input(
            "Enter Symbol (e.g., XRP)", # შეიყვანეთ სიმბოლო (მაგ. XRP)
            value=st.session_state.symbol_input,
            key="symbol_input_widget",
            label_visibility="collapsed" # Hide label
        )
    
    with header_cols[1]:
        # Buttons next to input
        # Group buttons horizontally
        apply_col, overview_col = st.columns(2)
        with apply_col:
            if st.button("Apply", key="apply_symbol_btn", help="Apply selected cryptocurrency symbol"): # გამოყენება
                formatted_symbol = st.session_state.symbol_input.upper()
                if not formatted_symbol.endswith('USDT'):
                    formatted_symbol += 'USDT'
                if formatted_symbol != st.session_state.selected_pair:
                    st.session_state.selected_pair = formatted_symbol
                    st.session_state.signals = [] # Clear old signals
                    st.session_state.last_signal_time = 0 # Reset signal timer
                    fetch_historical_data() # Re-fetch for new pair
                    st.rerun() # Rerun to update UI with new pair
        with overview_col:
            if st.button("✨ Overview", key="crypto_overview_btn", help="Get AI-generated overview of the cryptocurrency"): # ✨ მიმოხილვა
                # Pass base symbol (e.g., BTC from BTCUSDT)
                base_symbol = st.session_state.symbol_input.upper().replace('USDT', '')
                generate_crypto_overview(base_symbol)

    with header_cols[2]:
        st.session_state.timeframe = st.selectbox(
            "Timeframe", # დროის ჩარჩო
            ('1m', '5m', '15m', '1h', '4h', '1d'),
            index=('1m', '5m', '15m', '1h', '4h', '1d').index(st.session_state.timeframe),
            key="timeframe_select_widget",
            label_visibility="collapsed" # Hide label
        )
        if st.session_state.timeframe != st.session_state.get('prev_timeframe', '1h'):
            st.session_state.prev_timeframe = st.session_state.timeframe
            fetch_historical_data()
            st.rerun()

    with header_cols[3]:
        if st.session_state.is_trading:
            if st.button("Stop Signals", key="toggle_trading_btn_stop", help="Stop generating trading signals"): # სიგნალების შეჩერება
                st.session_state.is_trading = False
                st.rerun()
        else:
            if st.button("Start Signals", key="toggle_trading_btn_start", help="Start generating trading signals"): # სიგნალების დაწყება
                st.session_state.is_trading = True
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True) # End header-container

# Main Content Area
st.markdown('<div class="p-6">', unsafe_allow_html=True)

# Portfolio Overview Cards
st.markdown('<h2 class="text-xl font-bold mb-4">Portfolio Overview</h2>', unsafe_allow_html=True) # პორტფოლიოს მიმოხილვა
col_portfolio1, col_portfolio2, col_portfolio3, col_portfolio4 = st.columns(4)

with col_portfolio1:
    st.metric(label="Total Portfolio", value=f"${st.session_state.portfolio['total_value']:,}", help="Your total simulated portfolio value") # მთლიანი პორტფოლიო
with col_portfolio2:
    pnl_color = "green" if st.session_state.portfolio['pnl'] >= 0 else "red"
    st.markdown(f"""
        <div class="stMetric">
            <div class="stMetricLabel">Unrealized P&L</div> <!-- არარეალიზებული მოგება-ზარალი -->
            <div class="stMetricValue" style="color: {pnl_color};">${st.session_state.portfolio['pnl']:.2f}</div>
        </div>
    """, unsafe_allow_html=True)
with col_portfolio3:
    pnl_percent_color = "green" if st.session_state.portfolio['pnl_percent'] >= 0 else "red"
    st.markdown(f"""
        <div class="stMetric">
            <div class="stMetricLabel">P&L %</div> <!-- მოგება-ზარალი % -->
            <div class="stMetricValue" style="color: {pnl_percent_color};">{st.session_state.portfolio['pnl_percent']:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
with col_portfolio4:
    st.metric(label="Live Signals", value=len(st.session_state.signals)) # ცოცხალი სიგნალები

# Main Dashboard Layout
col_main_chart, col_side_panel = st.columns([2, 1])

with col_main_chart:
    st.markdown(f'<div class="panel">', unsafe_allow_html=True) # Removed h-96, using st.plotly_chart height
    st.markdown(f'<h3 class="text-white text-lg font-semibold mb-4">Live Chart - {st.session_state.selected_pair}</h3>', unsafe_allow_html=True) # ცოცხალი გრაფიკი
    
    # Connection status for chart
    connection_icon = "wifi" if st.session_state.is_connected else "wifi-off"
    connection_color = "text-green-400" if st.session_state.is_connected else "text-red-400"
    connection_text = "Live" if st.session_state.is_connected else "Disconnected" # ცოცხალი / გათიშული
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            <i data-lucide="{connection_icon}" class="w-4 h-4 {connection_color}"></i>
            <span class="{connection_color} text-sm">{connection_text}</span>
        </div>
    """, unsafe_allow_html=True)

    # Chart
    if not st.session_state.price_data.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.price_data.index, y=st.session_state.price_data['price'], mode='lines', name='Price', line=dict(color='#3B82F6', width=2))) # ფასი

        # Add Support and Resistance lines
        ta = st.session_state.technical_analysis
        if ta['support'] > 0:
            fig.add_hline(y=ta['support'], line_dash="dash", line_color="#EF4444",
                          annotation_text=f"Support: ${ta['support']:.4f}", # მხარდაჭერა
                          annotation_position="bottom right",
                          annotation_font_color="#EF4444")
        if ta['resistance'] > 0:
            fig.add_hline(y=ta['resistance'], line_dash="dash", line_color="#10B981",
                          annotation_text=f"Resistance: ${ta['resistance']:.4f}", # წინააღმდეგობა
                          annotation_position="top left",
                          annotation_font_color="#10B981")

        fig.update_layout(
            height=300, # Fixed height for chart
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#1F2937", # bg-gray-800
            plot_bgcolor="#1F2937",
            font_color="#F3F4F6", # text-white
            xaxis=dict(showgrid=True, gridcolor="#374151", linecolor="#9CA3AF", ticks="outside", tickfont=dict(color="#9CA3AF")),
            yaxis=dict(showgrid=True, gridcolor="#374151", linecolor="#9CA3AF", ticks="outside", tickfont=dict(color="#9CA3AF")),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Loading chart data...") # გრაფიკის მონაცემების ჩატვირთვა...
    st.markdown('</div>', unsafe_allow_html=True)

    # Live Price Display
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if st.session_state.live_price:
        lp = st.session_state.live_price
        change_color_class = "text-green-400" if lp['change'] >= 0 else "text-red-400"
        change_icon_html = '<i data-lucide="trending-up" class="w-5 h-5 text-green-400"></i>' if lp['change'] >= 0 else '<i data-lucide="trending-down" class="w-5 h-5 text-red-400"></i>'
        
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                <h3 class="text-white text-lg font-semibold">{lp['symbol']}</h3>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div class="w-2 h-2 rounded-full {'bg-green-400' if st.session_state.is_connected else 'bg-red-400'}"></div>
                    <span class="text-sm text-gray-400">Live</span> <!-- ცოცხალი -->
                </div>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span class="text-2xl font-bold text-white">${lp['price']:.4f}</span>
                    {change_icon_html}
                    <span class="font-semibold {change_color_class}">{lp['change']:.2f}%</span>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                <span class="text-gray-400">24h High: ${lp['high']:.4f}</span> <!-- 24 საათიანი მაქსიმუმი -->
                <span class="text-gray-400">24h Low: ${lp['low']:.4f}</span> <!-- 24 საათიანი მინიმუმი -->
            </div>
            <div style="font-size: 0.875rem;" class="text-gray-400">
                Volume: {lp['volume']/1000000:.2f}M <!-- მოცულობა -->
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Waiting for live price data...") # ცოცხალი ფასის მონაცემების მოლოდინში...
    st.markdown('</div>', unsafe_allow_html=True)

with col_side_panel:
    # Technical Analysis Panel
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    ta = st.session_state.technical_analysis
    trend_color = 'text-green-400' if ta['trend'] == 'Bullish' else 'text-red-400' if ta['trend'] == 'Bearish' else 'text-yellow-400'
    rsi_color = 'text-red-400' if float(ta['rsi']) > 70 else 'text-green-400' if float(ta['rsi']) < 30 else 'text-yellow-400'
    macd_color = 'text-green-400' if ta['macd'] == 'Bullish' else 'text-red-400'

    st.markdown(f"""
        <h3 class="text-white text-lg font-semibold mb-4">Live Technical Analysis</h3> <!-- ცოცხალი ტექნიკური ანალიზი -->
        <div class="space-y-4">
            <div class="flex justify-between items-center">
                <span class="text-gray-400">Trend</span> <!-- ტრენდი -->
                <span class="font-semibold {trend_color}">{ta['trend']}</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-gray-400">Strength</span> <!-- სიძლიერე -->
                <span class="text-white font-semibold">{ta['strength']}/10</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-gray-400">Support</span> <!-- მხარდაჭერა -->
                <span class="text-white font-semibold">${ta['support']:.4f}</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-gray-400">Resistance</span> <!-- წინააღმდეგობა -->
                <span class="text-white font-semibold">${ta['resistance']:.4f}</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-gray-400">RSI</span>
                <span class="font-semibold {rsi_color}">{ta['rsi']}</span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-gray-400">MACD</span>
                <span class="font-semibold {macd_color}">{ta['macd']}</span>
            </div>
            <div class="mt-4 p-3 bg-gray-700 rounded-md">
                <p class="text-sm text-gray-300">{ta['prediction']}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Order Book Display (Now in an expander)
    with st.expander("Live Order Book (Static)", expanded=False): # ცოცხალი შეკვეთების წიგნი (სტატიკური)
        col_bids, col_asks = st.columns(2)
        with col_bids:
            st.markdown('<h4 class="text-green-400 font-semibold mb-2">Bids (Buy Orders)</h4>', unsafe_allow_html=True) # ყიდვის ორდერები
            if st.session_state.order_book['bids']:
                for bid in st.session_state.order_book['bids'][:10]:
                    st.markdown(f'<div class="flex justify-between"><span class="text-gray-300">{bid["price"]:.4f}</span><span class="text-gray-500">{bid["quantity"]:.4f}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="text-gray-500">No bids available.</p>', unsafe_allow_html=True) # არ არის ხელმისაწვდომი ყიდვის ორდერები
        with col_asks:
            st.markdown('<h4 class="text-red-400 font-semibold mb-2">Asks (Sell Orders)</h4>', unsafe_allow_html=True) # გაყიდვის ორდერები
            if st.session_state.order_book['asks']:
                for ask in st.session_state.order_book['asks'][:10]:
                    st.markdown(f'<div class="flex justify-between"><span class="text-gray-300">{ask["price"]:.4f}</span><span class="text-gray-500">{ask["quantity"]:.4f}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="text-gray-500">No asks available.</p>', unsafe_allow_html=True) # არ არის ხელმისაწვდომი გაყიდვის ორდერები

    # Trading Signals Panel
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h3 class="text-white text-lg font-semibold mb-4">Live Trading Signals</h3>', unsafe_allow_html=True) # ცოცხალი სავაჭრო სიგნალები
    if not st.session_state.signals:
        st.markdown(f'<p class="text-gray-400 text-center py-8">{ "Analyzing market for signals..." if st.session_state.is_trading else "Start signal generation to see live signals" }</p>', unsafe_allow_html=True) # სიგნალების დაწყება / ბაზრის ანალიზი
    else:
        for signal in st.session_state.signals:
            signal_color_class = "signal-type-buy" if signal['type'] == 'BUY' else "signal-type-sell" if signal['type'] == 'SELL' else "signal-type-hold"
            st.markdown(f"""
                <div class="signal-card">
                    <div class="flex justify-between items-start mb-2">
                        <span class="px-2 py-1 rounded text-xs font-bold text-white {signal_color_class}">
                            {signal['type']}
                        </span>
                        <span class="text-gray-400 text-sm">{signal['timestamp']}</span>
                    </div>
                    <div class="space-y-1">
                        <p class="text-white font-semibold">{signal['pair']}</p>
                        <p class="text-gray-300 text-sm">{signal['reason']}</p>
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-400">Strength: {signal['strength']}/10</span> <!-- სიძლიერე -->
                            <span class="text-gray-400">RSI: {signal['rsi']}</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-400">Entry: ${signal['entry_price']:.4f}</span> <!-- შესვლა -->
                            <span class="text-gray-400">Trend: {signal['trend']}</span> <!-- ტრენდი -->
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-red-400">SL: ${signal['stop_loss']:.4f}</span> <!-- გაჩერების ზარალი -->
                            <span class="text-green-400">TP: ${signal['take_profit']:.4f}</span> <!-- მოგების აღება -->
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            # Streamlit buttons within loops need unique keys.
            # For dynamically created buttons in markdown, we need a workaround
            # to trigger a Python callback. Using `st.session_state.signal_id_to_elaborate`
            # and a hidden button.
            if signal['type'] in ['BUY', 'SELL']:
                if st.button("✨ Get Insight", key=f"insight_btn_{signal['id']}", help="Get AI insight for this signal"): # ✨ მიიღეთ ინსაითი
                    elaborate_signal(signal)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # End p-6

# Footer Connection Status
st.markdown('<div class="panel mt-6">', unsafe_allow_html=True)
connection_dot_color = 'bg-green-400' if st.session_state.is_connected else 'bg-red-400'
connection_status_text = 'Connected to Binance Live Data' if st.session_state.is_connected else 'Disconnected - Reconnecting...' # დაკავშირებულია Binance Live Data-სთან / გათიშულია - ხელახლა დაკავშირება...
st.markdown(f"""
    <div class="flex flex-col sm:flex-row items-center justify-between space-y-2 sm:space-y-0">
        <div class="flex items-center space-x-2">
            <div class="w-3 h-3 rounded-full {connection_dot_color}"></div>
            <span class="text-white font-semibold">{connection_status_text}</span>
        </div>
        <div class="text-sm text-gray-400">
            {len(st.session_state.price_data)} data points • {len(st.session_state.signals)} signals generated <!-- მონაცემთა წერტილები • სიგნალები გენერირებულია -->
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# LLM Response Modal (Simulated with st.empty and st.info/st.warning)
# Streamlit doesn't have native modals. We'll show this in a dedicated area or expander.
if st.session_state.show_llm_modal:
    st.markdown(f"""
        <div class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 p-4">
            <div class="bg-gray-800 rounded-lg shadow-xl max-w-lg w-full p-6 relative">
                <h3 class="text-xl font-bold text-white mb-4">{st.session_state.llm_modal_title}</h3>
                <div class="text-gray-300 text-base leading-relaxed whitespace-pre-wrap">
                    {"<div class='flex justify-center items-center h-32'><div class='animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500'></div><p class='ml-4 text-gray-400'>Loading...</p></div>" if st.session_state.llm_loading else st.session_state.llm_response} <!-- ჩატვირთვა... -->
                </div>
                <button class="stButton" style="margin-top: 1.5rem; width: 100%; background-color: #2563eb; color: white; border-radius: 0.375rem; padding: 0.5rem 1rem; font-weight: 600;"
                    onclick="window.parent.postMessage({{streamlit: {{command: 'SET_PAGE_STATE', args: ['hide_llm_modal_from_js']}}}}, '*');"
                >
                    Close <!-- დახურვა -->
                </button>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Use a hidden button to capture the JS message to close the modal
    if st.button("Hide LLM Modal", key="hide_llm_modal_btn", help="Hidden button to close LLM modal", on_click=lambda: st.session_state.__setitem__('show_llm_modal', False)): # LLM მოდალის დამალვა
        pass # This button's only purpose is its on_click callback

# --- Live Data Polling (replacing WebSocket for Streamlit's model) ---
# Streamlit doesn't support continuous WebSockets directly from Python in the same way as JS.
# We simulate live updates by periodically fetching the latest price and rerunning the app.

# Always attempt to fetch the latest price to update connection status
time.sleep(2) # Still want a delay between updates
latest_price_data = fetch_latest_price(st.session_state.selected_pair)
if latest_price_data:
    st.session_state.live_price = latest_price_data
    # Append to price_data for chart, keeping only last 100
    new_data_point = {
        'time': datetime.now().strftime('%H:%M:%S'),
        'timestamp': datetime.now().timestamp() * 1000,
        'price': latest_price_data['price'],
        'open': latest_price_data['price'], # Simplified for ticker
        'high': latest_price_data['high'],
        'low': latest_price_data['low'],
        'volume': latest_price_data['volume']
    }
    # Convert new_data_point to DataFrame row
    new_row_df = pd.DataFrame([new_data_point])
    new_row_df['time'] = pd.to_datetime(new_row_df['timestamp'], unit='ms')
    new_row_df = new_row_df.set_index('time')

    # Append and keep last 100
    st.session_state.price_data = pd.concat([st.session_state.price_data.tail(99), new_row_df])
    
    calculate_technical_indicators()
    if st.session_state.is_trading:
        generate_trading_signal()
st.rerun() # Always rerun the app to update UI with new data and keep the loop going

# Initial data fetch on first run (if price_data is empty)
if st.session_state.price_data.empty:
    fetch_historical_data()
    st.rerun() # Rerun to display initial data
