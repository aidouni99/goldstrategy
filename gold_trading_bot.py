import os
import time
import logging
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import ccxt
import talib
import telegram
from telegram.ext import Updater, CommandHandler
import schedule
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

class GoldTradingBot:
    def __init__(self):
        self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        self.exchange = ccxt.oanda({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True
        })
        self.model = None
        self.scaler = None
        self.load_or_train_model()
        
    def load_or_train_model(self):
        """Load pre-trained model or train a new one if not available"""
        try:
            self.model = joblib.load('gold_trading_model.pkl')
            self.scaler = joblib.load('gold_trading_scaler.pkl')
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.info("No model found, training a new one")
            self.train_model()
            
    def train_model(self):
        """Train ML model on historical gold data"""
        # Get historical data for training
        historical_data = self.get_historical_data('XAU/USD', '5m', 5000)
        
        # Create features and target
        features, target = self.create_features_and_target(historical_data)
        
        # Split data into training and test sets
        train_size = int(len(features) * 0.8)
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = target[:train_size], target[train_size:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train RandomForest model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        accuracy = self.model.score(X_test_scaled, y_test)
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(self.model, 'gold_trading_model.pkl')
        joblib.dump(self.scaler, 'gold_trading_scaler.pkl')
    
    def get_historical_data(self, symbol, timeframe, limit):
        """Fetch historical OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def create_features_and_target(self, df):
        """Create technical indicators as features and generate target variable"""
        if df is None or df.empty:
            return None, None
        
        # Copy dataframe to avoid modifying the original
        data = df.copy()
        
        # Add technical indicators
        # Moving Averages
        data['ma5'] = talib.SMA(data['close'], timeperiod=5)
        data['ma10'] = talib.SMA(data['close'], timeperiod=10)
        data['ma20'] = talib.SMA(data['close'], timeperiod=20)
        data['ma50'] = talib.SMA(data['close'], timeperiod=50)
        
        # MACD
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # RSI
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        
        # Bollinger Bands
        data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(
            data['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # ATR - Average True Range for volatility
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Stochastic Oscillator
        data['slowk'], data['slowd'] = talib.STOCH(
            data['high'], data['low'], data['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        
        # Price change ratio
        data['price_change'] = data['close'].pct_change()
        
        # Generate target - 1 if price goes up in next candle, 0 otherwise
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        # Extract features and target
        features = data[[
            'ma5', 'ma10', 'ma20', 'ma50', 
            'macd', 'macd_signal', 'macd_hist',
            'rsi', 'upper_band', 'middle_band', 'lower_band',
            'atr', 'slowk', 'slowd', 'price_change'
        ]].values
        target = data['target'].values
        
        return features, target
    
    def analyze_current_market(self):
        """Analyze current market conditions and generate signals"""
        # Get recent data
        data = self.get_historical_data('XAU/USD', '5m', 100)
        if data is None or data.empty:
            logger.error("Failed to get market data")
            return None, 0
        
        # Create features
        features, _ = self.create_features_and_target(data)
        
        if features is None or len(features) == 0:
            logger.error("Failed to create features")
            return None, 0
        
        # Get the latest feature set
        latest_features = features[-1:]
        
        # Scale features
        scaled_features = self.scaler.transform(latest_features)
        
        # Predict
        prediction = self.model.predict(scaled_features)[0]
        probability = max(self.model.predict_proba(scaled_features)[0])
        
        # Get confidence score
        confidence = probability
        
        # Get last close price
        last_close = data['close'].iloc[-1]
        
        # Advanced signal verification using multiple indicators
        rsi = data['rsi'].iloc[-1] if 'rsi' in data else 0
        macd = data['macd'].iloc[-1] if 'macd' in data else 0
        macd_signal = data['macd_signal'].iloc[-1] if 'macd_signal' in data else 0
        ma5 = data['ma5'].iloc[-1] if 'ma5' in data else 0
        ma20 = data['ma20'].iloc[-1] if 'ma20' in data else 0
        
        # Combine ML prediction with traditional indicators
        signal = None
        
        # Strong buy conditions
        if (prediction == 1 and confidence > 0.7 and
            rsi < 70 and  # Not overbought
            macd > macd_signal and  # MACD bullish crossover
            ma5 > ma20):  # Golden cross (short-term)
            signal = "BUY"
            
        # Strong sell conditions
        elif (prediction == 0 and confidence > 0.7 and
             rsi > 30 and  # Not oversold
             macd < macd_signal and  # MACD bearish crossover
             ma5 < ma20):  # Death cross (short-term)
            signal = "SELL"
        
        return signal, confidence, last_close
    
    def calculate_risk_parameters(self, signal, last_close):
        """Calculate stop loss and take profit levels"""
        if signal == "BUY":
            stop_loss = last_close * 0.997  # 0.3% stop loss
            take_profit = last_close * 1.005  # 0.5% take profit
        elif signal == "SELL":
            stop_loss = last_close * 1.003  # 0.3% stop loss
            take_profit = last_close * 0.995  # 0.5% take profit
        else:
            return None, None
        
        return round(stop_loss, 2), round(take_profit, 2)
    
    def send_signal_to_telegram(self, signal, confidence, last_close, stop_loss, take_profit):
        """Send trading signal to Telegram channel"""
        if signal is None:
            return
        
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if signal == "BUY":
            message = (
                f"🟢 GOLD BUY SIGNAL\n"
                f"Time: {timestamp}\n"
                f"Price: ${last_close:.2f}\n"
                f"Stop Loss: ${stop_loss:.2f}\n"
                f"Take Profit: ${take_profit:.2f}\n"
                f"Confidence: {confidence:.2%}\n"
                f"Timeframe: 5 minutes"
            )
        else:  # SELL
            message = (
                f"🔴 GOLD SELL SIGNAL\n"
                f"Time: {timestamp}\n"
                f"Price: ${last_close:.2f}\n"
                f"Stop Loss: ${stop_loss:.2f}\n"
                f"Take Profit: ${take_profit:.2f}\n"
                f"Confidence: {confidence:.2%}\n"
                f"Timeframe: 5 minutes"
            )
        
        try:
            self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info(f"Signal sent: {signal} at ${last_close:.2f}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def check_market_and_send_signals(self):
        """Main function to check market and send signals"""
        logger.info("Analyzing market...")
        signal, confidence, last_close = self.analyze_current_market()
        
        if signal:
            stop_loss, take_profit = self.calculate_risk_parameters(signal, last_close)
            self.send_signal_to_telegram(signal, confidence, last_close, stop_loss, take_profit)
        else:
            logger.info("No trading signal generated")
    
    def start(self):
        """Start the trading bot"""
        logger.info("Gold Trading Bot started")
        
        # Schedule market analysis every 5 minutes
        schedule.every(5).minutes.do(self.check_market_and_send_signals)
        
        # Send startup message
        try:
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text="🤖 Gold Trading Bot has been started!\n"
                     "You will receive trading signals shortly."
            )
        except Exception as e:
            logger.error(f"Failed to send startup message: {e}")
        
        # Run the scheduler
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    bot = GoldTradingBot()
    bot.start()
