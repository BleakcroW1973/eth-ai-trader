import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import plotly.graph_objects as go
import time
import requests
import os

# This tells the code to look for a secure hidden variable on the server
SECRET_KEY = os.environ.get("MUDREX_SECRET_KEY")

if not SECRET_KEY:
    st.error("🚨 CRITICAL: MUDREX_SECRET_KEY environment variable not found!")
    st.stop()

# --- 1. Page Setup ---
st.set_page_config(page_title="AI Hedge Fund Engine", layout="wide", page_icon="🚀")
st.title("🚀 Ethereum Dual-Brain AI Engine")

run_mode = st.sidebar.radio("⚙️ Operation Mode", ["Backtest (CSV)", "Live Trading"], index=1)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🧠 AI Settings")
bull_conf_threshold = st.sidebar.slider("Bull Confidence (LONG)", 0.50, 0.99, 0.67, step=0.01)
bear_conf_threshold = st.sidebar.slider("Bear Confidence (SHORT)", 0.50, 0.99, 0.75, step=0.01)

st.sidebar.markdown("### 🛡️ Risk Parameters")
st.sidebar.markdown("#### 🟢 Bull Parameters (LONG)")
bull_tp_pct = st.sidebar.slider("Bull Take-Profit (%)", 0.5, 10.0, 4.0, step=0.1) / 100
bull_sl_pct = st.sidebar.slider("Bull Stop-Loss (%)", 0.5, 10.0, 3.0, step=0.1) / 100

st.sidebar.markdown("#### 🔴 Bear Parameters (SHORT)")
bear_tp_pct = st.sidebar.slider("Bear Take-Profit (%)", 0.5, 10.0, 4.0, step=0.1) / 100
bear_sl_pct = st.sidebar.slider("Bear Stop-Loss (%)", 0.5, 10.0, 2.0, step=0.1) / 100

if run_mode == "Live Trading":
    st.sidebar.markdown("### 💰 Live Execution Size")
    trade_size_usdt = st.sidebar.number_input("Trade Size (USDT)", min_value=5.0, value=5500.0, step=5.0)


# --- 2. Load BOTH AI Models ---
@st.cache_resource
def load_models():
    bull = xgb.XGBClassifier()
    bull.load_model("eth_model_bull.json")
    bear = xgb.XGBClassifier()
    bear.load_model("eth_model_bear.json")
    return bull, bear


bull_model, bear_model = load_models()

expected_features = [
    'NATR_14', 'MACDs_12_26_9', 'Dist_EMA50', 'MACD_12_26_9', 'RSI_14',
    'ATRr_14', 'Dist_EMA200', 'ADX_14', 'ADXR_14_2', 'DMN_14'
]

# --- 3. DUAL-BRAIN BACKTESTING ENGINE (CSV) ---
if run_mode == "Backtest (CSV)":
    st.header("📊 Dual-Brain Event-Driven Backtest")
    uploaded_file = st.sidebar.file_uploader("Upload historical CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        possible_time_cols = [c for c in df.columns if
                              c in ['timestamp', 'date', 'time', 'datetime', 'open_time', 'close_time']]
        if not possible_time_cols:
            st.error(f"⚠️ Could not find a time column in your CSV! Your columns are: {list(df.columns)}")
            st.stop()
        time_col_name = possible_time_cols[0]
        df.rename(columns={time_col_name: 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        with st.spinner("Calculating 10 Stationary Features..."):
            df.ta.natr(length=14, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            try:
                df.ta.adx(length=14, append=True)
            except:
                pass
            df['Dist_EMA50'] = (df['close'] - ta.ema(df['close'], length=50)) / ta.ema(df['close'], length=50)
            df['Dist_EMA200'] = (df['close'] - ta.ema(df['close'], length=200)) / ta.ema(df['close'], length=200)
            df.dropna(inplace=True)

        with st.spinner("Running AI Inference & Simulating Trades..."):
            X_backtest = df[expected_features]
            df['Bull_Conf'] = bull_model.predict_proba(X_backtest)[:, 1]
            df['Bear_Conf'] = bear_model.predict_proba(X_backtest)[:, 1]

            in_trade = False
            position_type = None
            entry_price = 0.0
            net_profit_pct = 0.0
            FEE_RATE = 0.0015

            equity_curve = [1.0]  # Start with 100% portfolio
            trade_history = []

            for i in range(len(df)):
                current_price = df['close'].iloc[i]
                timestamp = df['timestamp'].iloc[i]

                if in_trade:
                    if position_type == "LONG":
                        current_pnl = (current_price - entry_price) / entry_price
                        if current_pnl >= bull_tp_pct:
                            net_profit_pct += current_pnl - FEE_RATE
                            trade_history.append(
                                {'Time': timestamp, 'Type': 'LONG', 'Result': '🎯 TP', 'PnL': current_pnl})
                            in_trade = False
                        elif current_pnl <= -bull_sl_pct:
                            net_profit_pct += current_pnl - FEE_RATE
                            trade_history.append(
                                {'Time': timestamp, 'Type': 'LONG', 'Result': '🛑 SL', 'PnL': current_pnl})
                            in_trade = False

                    elif position_type == "SHORT":
                        current_pnl = (entry_price - current_price) / entry_price
                        if current_pnl >= bear_tp_pct:
                            net_profit_pct += current_pnl - FEE_RATE
                            trade_history.append(
                                {'Time': timestamp, 'Type': 'SHORT', 'Result': '🎯 TP', 'PnL': current_pnl})
                            in_trade = False
                        elif current_pnl <= -bear_sl_pct:
                            net_profit_pct += current_pnl - FEE_RATE
                            trade_history.append(
                                {'Time': timestamp, 'Type': 'SHORT', 'Result': '🛑 SL', 'PnL': current_pnl})
                            in_trade = False

                else:  # Look for Entry
                    if df['Bull_Conf'].iloc[i] >= bull_conf_threshold and df['Bull_Conf'].iloc[i] > \
                            df['Bear_Conf'].iloc[i]:
                        in_trade = True
                        position_type = "LONG"
                        entry_price = current_price
                        net_profit_pct -= FEE_RATE
                    elif df['Bear_Conf'].iloc[i] >= bear_conf_threshold and df['Bear_Conf'].iloc[i] > \
                            df['Bull_Conf'].iloc[i]:
                        in_trade = True
                        position_type = "SHORT"
                        entry_price = current_price
                        net_profit_pct -= FEE_RATE

                equity_curve.append(1.0 + net_profit_pct)

            df['Portfolio_Multiplier'] = equity_curve[1:]

        # --- Render Results ---
        total_trades = len(trade_history)
        win_count = sum(1 for t in trade_history if 'TP' in t['Result'])
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Net AI Return", f"{net_profit_pct * 100:.2f}%")
        col2.metric("Total Trades Taken", f"{total_trades}")
        col3.metric("AI Win Rate", f"{win_rate:.1f}%")

        fig_eq = go.Figure()
        market_return = (df['close'] / df['close'].iloc[0]) - 1
        fig_eq.add_trace(go.Scatter(x=df['timestamp'], y=market_return * 100, name="Buy & Hold Market (%)"))
        fig_eq.add_trace(
            go.Scatter(x=df['timestamp'], y=df['Portfolio_Multiplier'] * 100 - 100, name="AI Hedge Fund (%)"))
        fig_eq.update_layout(title="Dual-Brain Equity Curve", yaxis_title="Cumulative Return (%)")
        st.plotly_chart(fig_eq, width=True)

        if trade_history:
            st.markdown("### 📜 Virtual Trade Log")
            st.dataframe(pd.DataFrame(trade_history).sort_values(by='Time', ascending=False).reset_index(drop=True),
                         width=True)

# --- 4. LIVE TRADING ENGINE ---
elif run_mode == "Live Trading":

    if 'trade_log' not in st.session_state: st.session_state.trade_log = []
    if 'position_type' not in st.session_state: st.session_state.position_type = None
    if 'position_price' not in st.session_state: st.session_state.position_price = None
    if 'position_size' not in st.session_state: st.session_state.position_size = 0.0

    import math
    import requests

    # Set your global step size for Ethereum
    STEP_SIZE = 0.01


    def sync_mudrex_position():
        """Fetches active ETHUSDT position from Mudrex and restores Streamlit state."""
        url = "https://trade.mudrex.com/fapi/v1/futures/ETHUSDT/position?is_symbol=true"
        headers = {
            "Content-Type": "application/json",
            "X-Authentication": os.environ.get("MUDREX_SECRET_KEY")
        }

        try:
            response = requests.get(url, headers=headers)

            if response.status_code in [200, 201, 202]:
                data = response.json()

                # FAPI endpoints typically return a list of assets or a nested dict
                pos_data = data[0] if isinstance(data, list) else data.get("data", data)

                # Common FAPI keys: 'positionAmt' or 'quantity', 'entryPrice' or 'avgPrice'
                # Shorts are typically returned as negative floats (e.g., -0.010)
                raw_size = float(pos_data.get("positionAmt", pos_data.get("quantity", 0.0)))
                entry_price = float(pos_data.get("entryPrice", pos_data.get("avgPrice", 0.0)))

                if raw_size > 0:
                    st.session_state.position_type = "LONG"
                    st.session_state.position_size = abs(raw_size)
                    st.session_state.position_price = entry_price
                    print(f"🔄 State Restored: LONG | Size: {abs(raw_size)} | Entry: {entry_price}")
                elif raw_size < 0:
                    st.session_state.position_type = "SHORT"
                    st.session_state.position_size = abs(raw_size)
                    st.session_state.position_price = entry_price
                    print(f"🔄 State Restored: SHORT | Size: {abs(raw_size)} | Entry: {entry_price}")
                else:
                    print("🔄 State Restored: FLAT (No active positions)")

            else:
                print(f"❌ Position Sync Failed: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Network Error during Sync: {e}")


    # 🛠️ THE STATE LOCK: Run the sync function exactly ONCE per session startup
    if 'position_synced' not in st.session_state:
        sync_mudrex_position()
        st.session_state.position_synced = True

        
    def format_quantity(amount, step):
        """Forces the quantity to be a perfect multiple of the exchange step size."""
        precision = int(-math.log10(step))
        # Adding 1e-8 prevents Python from rounding 1.9999999 down to 1
        clean_qty = math.floor((amount / step) + 1e-8) * step
        return "{:0.{}f}".format(clean_qty, precision)


    def execute_mudrex_order(action, amount_in_eth):

        # 1. The exact URL provided by Mudrex Support
        url = "https://trade.mudrex.com/fapi/v1/futures/ETHUSDT/order?is_symbol=true"

        # 2. Format the quantity safely
        clean_qty_str = format_quantity(amount_in_eth, STEP_SIZE)

        # 3. Determine order flags
        is_closing = True if "CLOSE" in action else False
        order_type = "LONG" if "LONG" in action else "SHORT"

        # 4. The exact payload quirks defined by Support
        payload = {
            "asset_id": "01903bc9-973a-7106-99e2-08287b632806",  # Support-verified Asset ID
            "symbol": "ETHUSDT",
            "leverage": "3",
            "quantity": clean_qty_str,
            "order_price": "999999999",  # Required dummy price for Market orders
            "order_type": order_type,
            "trigger_type": "MARKET",
            "reduce_only": is_closing
        }

        headers = {
            "Content-Type": "application/json",
            "X-Authentication": SECRET_KEY  # Ensure your SECRET_KEY variable is defined above
        }

        try:
            response = requests.post(url, headers=headers, json=payload)

            # ACCEPT 200 (OK), 201 (Created), and 202 (Accepted)
            if response.status_code in [200, 201, 202]:
                print(f"✅ EXECUTED: {action} | Qty: {clean_qty_str} ETH")
                return True
            else:
                print(f"❌ API REJECTED: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"❌ NETWORK ERROR: {e}")
            return False


    def send_discord_alert(message):
        """Pushes a live execution alert to your private Discord channel."""
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            return  # Silently skip if no URL is configured

        payload = {"content": message}
        try:
            requests.post(webhook_url, json=payload)
        except Exception as e:
            print(f"❌ Discord Webhook Failed: {e}")


    def get_live_data():
        exchange = ccxt.kucoin()
        bars = exchange.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=250)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df.ta.natr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        try:
            df.ta.adx(length=14, append=True)
        except:
            pass

        df['Dist_EMA50'] = (df['close'] - ta.ema(df['close'], length=50)) / ta.ema(df['close'], length=50)
        df['Dist_EMA200'] = (df['close'] - ta.ema(df['close'], length=200)) / ta.ema(df['close'], length=200)
        df.dropna(inplace=True)
        return df


    dashboard_placeholder = st.empty()

    while True:
        try:
            df = get_live_data()
            if df.empty:
                st.warning("⚠️ KuCoin returned empty data or not enough bars for the 200 EMA. Retrying...")
                time.sleep(5)
                continue

            latest_data = df.tail(1).copy()
            current_price = latest_data['close'].values[0]
            current_time = pd.Timestamp.now().strftime("%H:%M:%S")

            X_live = latest_data[expected_features]
            bull_conf = float(bull_model.predict_proba(X_live)[0][1])
            bear_conf = float(bear_model.predict_proba(X_live)[0][1])

            # --- POSITION MANAGEMENT ---
            current_pnl_pct = 0.0
            if st.session_state.position_type is not None:
                entry_price = st.session_state.position_price

                if st.session_state.position_type == "LONG":
                    current_pnl_pct = (current_price - entry_price) / entry_price
                    close_action = "CLOSE LONG"
                    active_tp = bull_tp_pct
                    active_sl = bull_sl_pct
                else:  # SHORT
                    current_pnl_pct = (entry_price - current_price) / entry_price
                    close_action = "CLOSE SHORT"
                    active_tp = bear_tp_pct
                    active_sl = bear_sl_pct

                if current_pnl_pct >= active_tp:
                    if execute_mudrex_order(close_action, st.session_state.position_size):
                        st.session_state.trade_log.append(
                            {'Time': current_time, 'Action': '🎯 TAKE-PROFIT', 'PnL': f"{current_pnl_pct * 100:.2f}%"})
                        st.session_state.position_type = None
                        # 🛠️ DISCORD ALERT
                        send_discord_alert(
                            f"🎯 **TAKE-PROFIT HIT**\nClosed {close_action} at {current_pnl_pct * 100:.2f}% Profit!")
                        st.rerun()  # 🛠️ ADD THIS HERE

                elif current_pnl_pct <= -active_sl:
                    if execute_mudrex_order(close_action, st.session_state.position_size):
                        st.session_state.trade_log.append(
                            {'Time': current_time, 'Action': '🛑 STOP-LOSS', 'PnL': f"{current_pnl_pct * 100:.2f}%"})
                        st.session_state.position_type = None
                        # 🛠️ DISCORD ALERT
                        send_discord_alert(
                            f"🛑 **STOP-LOSS TRIGGERED**\nClosed {close_action} at {current_pnl_pct * 100:.2f}% Loss. Risk managed.")
                        st.rerun()  # 🛠️ ADD THIS HERE

            # --- ENTRY LOGIC ---
            elif st.session_state.position_type is None:
                raw_eth_size = trade_size_usdt / current_price

                # 🛠️ THE FIX: Pre-round the size so your internal state matches the exchange 1-to-1
                # Note: Ensure you have STEP_SIZE = 0.001 defined at the top of this file
                clean_eth_size = float(format_quantity(raw_eth_size, STEP_SIZE))

                if bull_conf >= bull_conf_threshold and bull_conf > bear_conf:
                    if execute_mudrex_order("OPEN LONG", clean_eth_size):
                        st.session_state.position_type = "LONG"
                        st.session_state.position_price = current_price
                        st.session_state.position_size = clean_eth_size
                        st.session_state.trade_log.append(
                            {'Time': current_time, 'Action': '🟢 OPEN LONG', 'PnL': '0.00%'})
                        # 🛠️ DISCORD ALERT
                        send_discord_alert(
                            f"🟢 **OPEN LONG Executed**\n💰 Entry Price: ${current_price:,.2f}\n⚖️ Size: {clean_eth_size} ETH\n🤖 Bull Confidence: {bull_conf:.2%}")
                        st.rerun()  # 🛠️ ADD THIS HERE

                elif bear_conf >= bear_conf_threshold and bear_conf > bull_conf:
                    if execute_mudrex_order("OPEN SHORT", clean_eth_size):
                        st.session_state.position_type = "SHORT"
                        st.session_state.position_price = current_price
                        st.session_state.position_size = clean_eth_size
                        st.session_state.trade_log.append(
                            {'Time': current_time, 'Action': '🔴 OPEN SHORT', 'PnL': '0.00%'})
                        # 🛠️ DISCORD ALERT
                        send_discord_alert(
                            f"🔴 **OPEN SHORT Executed**\n💰 Entry Price: ${current_price:,.2f}\n⚖️ Size: {clean_eth_size} ETH\n🤖 Bear Confidence: {bear_conf:.2%}")
                        st.rerun()  # 🛠️ ADD THIS HERE

            # --- UI RENDERING ---
            fig = go.Figure(data=[
                go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                               name="ETH/USDT")])
            fig.update_layout(title="Live ETH/USDT", height=400, xaxis_rangeslider_visible=False)

            with dashboard_placeholder.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader(f"Live Price: **${current_price:,.2f}**")
                    st.progress(bull_conf, text=f"🟢 BULL Confidence: {bull_conf:.2%}")
                    st.progress(bear_conf, text=f"🔴 BEAR Confidence: {bear_conf:.2%}")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 📊 Active Position")
                    if st.session_state.position_type is not None:
                        st.success(
                            f"**{st.session_state.position_type}** at ${st.session_state.position_price:,.2f}\nPnL: **{current_pnl_pct * 100:.2f}%**")
                    else:
                        st.warning("WAITING FOR SIGNAL")

                    st.markdown("### 📜 Log")
                    # If there are trades in the memory, draw a clean table
                    if len(st.session_state.trade_log) > 0:
                        # Convert the list of dictionaries into a DataFrame for a beautiful UI table
                        log_df = pd.DataFrame(st.session_state.trade_log)
                        st.dataframe(log_df, hide_index=True, use_container_width=True)
                    else:
                        st.write("No trades executed yet.")

            time.sleep(30)

        except Exception as e:
            with dashboard_placeholder.container():
                st.error(f"System Error: {e}")
            time.sleep(5)