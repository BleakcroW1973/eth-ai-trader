import os
import time
import math
import asyncio
import requests
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import ccxt
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# --- 1. ENVIRONMENT & SETUP ---
SECRET_KEY = os.environ.get("MUDREX_SECRET_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not SECRET_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("🚨 CRITICAL: Missing Environment Variables!")
    exit(1)


# --- 2. GLOBAL STATE (Replaces st.session_state) ---
class HedgeFundState:
    def __init__(self):
        self.is_paused = False
        self.position_type = None
        self.position_price = 0.0
        self.position_size = 0.0

        # Hardcoded from your Streamlit defaults
        self.trade_size_usdt = 5500.0
        self.bull_conf_threshold = 0.67
        self.bear_conf_threshold = 0.75
        self.bull_tp_pct = 0.04
        self.bull_sl_pct = 0.03
        self.bear_tp_pct = 0.04
        self.bear_sl_pct = 0.02
        self.step_size = 0.01


state = HedgeFundState()


# --- 3. LOAD MODELS GLOBALLY ---
def load_models():
    print("🧠 Loading AI Models...")
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


# --- 4. CORE QUANT FUNCTIONS ---
def format_quantity(amount, step):
    precision = int(-math.log10(step))
    clean_qty = math.floor((amount / step) + 1e-8) * step
    return "{:0.{}f}".format(clean_qty, precision)


def execute_mudrex_order(action, amount_in_eth):
    url = "https://trade.mudrex.com/fapi/v1/futures/ETHUSDT/order?is_symbol=true"
    clean_qty_str = format_quantity(amount_in_eth, state.step_size)
    is_closing = True if "CLOSE" in action else False
    order_type = "LONG" if "LONG" in action else "SHORT"

    payload = {
        "asset_id": "01903bc9-973a-7106-99e2-08287b632806",
        "symbol": "ETHUSDT",
        "leverage": "3",
        "quantity": clean_qty_str,
        "order_price": "999999999",
        "order_type": order_type,
        "trigger_type": "MARKET",
        "reduce_only": is_closing
    }
    headers = {"Content-Type": "application/json", "X-Authentication": SECRET_KEY}

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code in [200, 201, 202]:
            print(f"✅ EXECUTED: {action} | Qty: {clean_qty_str} ETH")
            return True
        else:
            print(f"❌ API REJECTED: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ NETWORK ERROR: {e}")
        return False


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


def send_discord_alert(message):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url: return
    try:
        requests.post(webhook_url, json={"content": message})
    except Exception as e:
        print(f"❌ Discord Webhook Failed: {e}")


async def sync_mudrex_position(app: Application):
    """Fetches active position from Mudrex and restores state."""
    url = "https://trade.mudrex.com/fapi/v1/futures/ETHUSDT/position?is_symbol=true"
    headers = {"Content-Type": "application/json", "X-Authentication": SECRET_KEY}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code in [200, 201, 202]:
            data = response.json()
            pos_data = data[0] if isinstance(data, list) else data.get("data", data)
            raw_size = float(pos_data.get("positionAmt", pos_data.get("quantity", 0.0)))
            entry_price = float(pos_data.get("entryPrice", pos_data.get("avgPrice", 0.0)))

            if raw_size > 0:
                state.position_type, state.position_size, state.position_price = "LONG", abs(raw_size), entry_price
                msg = f"🔄 Sync Restored: LONG | Size: {abs(raw_size)} | Entry: ${entry_price:,.2f}"
            elif raw_size < 0:
                state.position_type, state.position_size, state.position_price = "SHORT", abs(raw_size), entry_price
                msg = f"🔄 Sync Restored: SHORT | Size: {abs(raw_size)} | Entry: ${entry_price:,.2f}"
            else:
                msg = "🔄 Sync Restored: FLAT (No active positions)"

            print(msg)
            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    except Exception as e:
        print(f"❌ Sync Error: {e}")


# --- 5. ASYNC BACKGROUND SCANNER ---
async def market_scanner(app: Application):
    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚀 Hedge Fund Engine Online & Scanning.")
    await sync_mudrex_position(app)

    while True:
        try:
            if state.is_paused:
                await asyncio.sleep(60)
                continue

            df = get_live_data()
            if df.empty:
                await asyncio.sleep(5)
                continue

            latest_data = df.tail(1).copy()
            current_price = latest_data['close'].values[0]
            X_live = latest_data[expected_features]

            bull_conf = float(bull_model.predict_proba(X_live)[0][1])
            bear_conf = float(bear_model.predict_proba(X_live)[0][1])

            # --- POSITION MANAGEMENT ---
            if state.position_type is not None:
                current_pnl_pct = 0.0
                entry_price = state.position_price

                if state.position_type == "LONG":
                    current_pnl_pct = (current_price - entry_price) / entry_price
                    close_action, active_tp, active_sl = "CLOSE LONG", state.bull_tp_pct, state.bull_sl_pct
                else:
                    current_pnl_pct = (entry_price - current_price) / entry_price
                    close_action, active_tp, active_sl = "CLOSE SHORT", state.bear_tp_pct, state.bear_sl_pct

                if current_pnl_pct >= active_tp:
                    if execute_mudrex_order(close_action, state.position_size):
                        state.position_type = None
                        msg = f"🎯 **TAKE-PROFIT HIT**\nClosed {close_action} at {current_pnl_pct * 100:.2f}% Profit!"
                        send_discord_alert(msg)
                        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                        await asyncio.sleep(5)  # Prevent rate limits before next loop iteration
                        continue

                elif current_pnl_pct <= -active_sl:
                    if execute_mudrex_order(close_action, state.position_size):
                        state.position_type = None
                        msg = f"🛑 **STOP-LOSS TRIGGERED**\nClosed {close_action} at {current_pnl_pct * 100:.2f}% Loss."
                        send_discord_alert(msg)
                        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                        await asyncio.sleep(5)
                        continue

            # --- ENTRY LOGIC ---
            elif state.position_type is None:
                raw_eth_size = state.trade_size_usdt / current_price
                clean_eth_size = float(format_quantity(raw_eth_size, state.step_size))

                if bull_conf >= state.bull_conf_threshold and bull_conf > bear_conf:
                    if execute_mudrex_order("OPEN LONG", clean_eth_size):
                        state.position_type, state.position_price, state.position_size = "LONG", current_price, clean_eth_size
                        msg = f"🟢 **OPEN LONG Executed**\n💰 Entry: ${current_price:,.2f}\n⚖️ Size: {clean_eth_size} ETH\n🤖 Bull Conf: {bull_conf:.2%}"
                        send_discord_alert(msg)
                        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                        await asyncio.sleep(5)
                        continue

                elif bear_conf >= state.bear_conf_threshold and bear_conf > bull_conf:
                    if execute_mudrex_order("OPEN SHORT", clean_eth_size):
                        state.position_type, state.position_price, state.position_size = "SHORT", current_price, clean_eth_size
                        msg = f"🔴 **OPEN SHORT Executed**\n💰 Entry: ${current_price:,.2f}\n⚖️ Size: {clean_eth_size} ETH\n🤖 Bear Conf: {bear_conf:.2%}"
                        send_discord_alert(msg)
                        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                        await asyncio.sleep(5)
                        continue

            # Sleep normally if no actions were taken
            await asyncio.sleep(30)

        except Exception as e:
            print(f"❌ System Error: {e}")
            await asyncio.sleep(10)


# --- 6. TELEGRAM COMMAND HANDLERS ---
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = (
        f"📊 **Hedge Fund Status**\n"
        f"Engine: {'⏸️ PAUSED' if state.is_paused else '▶️ ACTIVE'}\n"
        f"Trade Size: ${state.trade_size_usdt:,.2f}\n"
        f"Position: {state.position_type if state.position_type else 'FLAT'}"
    )
    if state.position_type:
        status_msg += f" at ${state.position_price:,.2f}"
    await update.message.reply_text(status_msg)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_paused = True
    await update.message.reply_text("⏸️ Engine Paused. Monitoring existing positions, but will NOT open new ones.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.is_paused = False
    await update.message.reply_text("▶️ Engine Resumed. Scanning market...")


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        new_size = float(context.args[0])
        state.trade_size_usdt = new_size
        await update.message.reply_text(f"💰 Trade size updated to: ${new_size:,.2f}")
    except (IndexError, ValueError):
        await update.message.reply_text("❌ Please provide a valid number. Example: /size 100")


async def cmd_sync(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Forcing Mudrex Position Sync...")
    await sync_mudrex_position(context.application)


# --- 7. MAIN RUNNER ---
async def post_init(app: Application):
    """Native background task launcher. Bypasses JobQueue entirely."""
    await asyncio.sleep(5) # Give the bot 5 seconds to boot
    asyncio.create_task(market_scanner(app))

def main():
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init) # 🛠️ THE NEW NATIVE LAUNCHER
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )

    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("size", cmd_size))
    app.add_handler(CommandHandler("sync", cmd_sync))

    # NOTE: We entirely deleted the app.job_queue.run_once() line

    print("📡 Connecting to Telegram...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()