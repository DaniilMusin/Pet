import pandas as pd
import numpy as np
import talib
import ccxt
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BTC_Trading_Bot')

class LiveTradingBot:
    """
    Live implementation of the adaptive trading strategy for BTC futures
    """
    
    def __init__(self, 
                 exchange_id='binance', 
                 api_key=None, 
                 api_secret=None,
                 symbol='BTC/USDT',
                 timeframe='4h',
                 params=None):
        """
        Initialize the trading bot
        
        Parameters:
        -----------
        exchange_id : str
            ID of the exchange to use (e.g., 'binance', 'bybit')
        api_key : str
            API key for the exchange
        api_secret : str
            API secret for the exchange
        symbol : str
            Trading symbol (e.g., 'BTC/USDT')
        timeframe : str
            Candlestick timeframe (e.g., '1h', '4h', '1d')
        params : dict
            Strategy parameters (if None, default parameters will be used)
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Default strategy parameters
        self.default_params = {
            'adx_threshold': 25,
            'ema_short': 9,
            'ema_long': 21,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'leverage': 3
        }
        
        # Use provided parameters or defaults
        self.params = params if params else self.default_params
        
        # Initialize exchange connection
        self.initialize_exchange()
        
        # Track current position
        self.current_position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.position_size = 0
        
        logger.info(f"Bot initialized for {self.symbol} on {self.exchange_id}")
        logger.info(f"Strategy parameters: {self.params}")
    
    def initialize_exchange(self):
        """Initialize connection to the exchange"""
        
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures API
                }
            })
            
            # Check if API keys are valid by fetching balance
            if self.api_key and self.api_secret:
                balance = self.exchange.fetch_balance()
                logger.info(f"Connection to {self.exchange_id} established successfully")
                logger.info(f"Available balance: {balance['free']['USDT']} USDT")
                
                # Set leverage
                self.exchange.set_leverage(self.params['leverage'], self.symbol)
                logger.info(f"Leverage set to {self.params['leverage']}x")
            else:
                logger.warning("API keys not provided. Running in simulation mode.")
        
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise
    
    def fetch_data(self, limit=100):
        """Fetch historical OHLCV data from the exchange"""
        
        try:
            logger.info(f"Fetching {limit} {self.timeframe} candles for {self.symbol}")
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched data from {df.index.min()} to {df.index.max()}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators for the strategy"""
        
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # --- Trend indicators ---
            # EMA crossover with custom periods
            df['ema_short'] = talib.EMA(df['close'], timeperiod=self.params['ema_short'])
            df['ema_long'] = talib.EMA(df['close'], timeperiod=self.params['ema_long'])
            
            # --- Mean reversion indicators ---
            # RSI (14 periods)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            
            # --- Market regime indicators ---
            # ADX (Average Directional Index) - for trend strength
            df['adx'] = talib.ADX(
                df['high'], df['low'], df['close'], timeperiod=14
            )
            
            # ATR (Average True Range) - for volatility and stop calculation
            df['atr'] = talib.ATR(
                df['high'], df['low'], df['close'], timeperiod=14
            )
            
            # Identify market regime with custom ADX threshold
            df['trend_regime'] = np.where(df['adx'] > self.params['adx_threshold'], 1, 0)
            
            return df
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def generate_signal(self, data):
        """Generate trading signal based on indicators and market regime"""
        
        try:
            # Get the latest data point and the previous one
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Initialize signal
            signal = 0
            
            # Check market regime
            if current['trend_regime'] == 1:  # Trending market
                logger.info("Current market regime: TRENDING")
                
                # EMA crossover strategy
                if (previous['ema_short'] <= previous['ema_long'] and 
                    current['ema_short'] > current['ema_long']):
                    # Bullish crossover
                    signal = 1
                    logger.info("Signal: BUY (Bullish EMA crossover in trending market)")
                
                elif (previous['ema_short'] >= previous['ema_long'] and 
                      current['ema_short'] < current['ema_long']):
                    # Bearish crossover
                    signal = -1
                    logger.info("Signal: SELL (Bearish EMA crossover in trending market)")
                
                else:
                    logger.info("No new trend signal")
            
            else:  # Range-bound market
                logger.info("Current market regime: RANGING")
                
                # RSI strategy for mean reversion
                if current['rsi'] < self.params['rsi_oversold']:
                    # Oversold condition
                    signal = 1
                    logger.info(f"Signal: BUY (RSI oversold at {current['rsi']:.2f})")
                
                elif current['rsi'] > self.params['rsi_overbought']:
                    # Overbought condition
                    signal = -1
                    logger.info(f"Signal: SELL (RSI overbought at {current['rsi']:.2f})")
                
                else:
                    logger.info(f"No mean reversion signal (RSI: {current['rsi']:.2f})")
            
            return signal, current
        
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0, None
    
    def should_exit_position(self, data):
        """Determine if current position should be exited"""
        
        if self.current_position == 0:
            return False
        
        try:
            # Get the latest data point
            current = data.iloc[-1]
            
            # Exit conditions based on regime
            if current['trend_regime'] == 1:  # Trending market
                if self.current_position == 1:  # Long position
                    # Exit if price crosses below EMA long
                    if current['close'] < current['ema_long']:
                        logger.info("Exit signal: Long position - Price below EMA long")
                        return True
                
                elif self.current_position == -1:  # Short position
                    # Exit if price crosses above EMA long
                    if current['close'] > current['ema_long']:
                        logger.info("Exit signal: Short position - Price above EMA long")
                        return True
            
            else:  # Range-bound market
                if self.current_position == 1:  # Long position
                    # Exit if price reaches middle band or RSI overbought
                    if (current['close'] > current['bb_middle'] or 
                        current['rsi'] > self.params['rsi_overbought']):
                        logger.info("Exit signal: Long position - Target reached or overbought")
                        return True
                
                elif self.current_position == -1:  # Short position
                    # Exit if price reaches middle band or RSI oversold
                    if (current['close'] < current['bb_middle'] or 
                        current['rsi'] < self.params['rsi_oversold']):
                        logger.info("Exit signal: Short position - Target reached or oversold")
                        return True
            
            # Check stop loss (using ATR)
            current_price = current['close']
            atr_multiple = 2.0  # Exit if price moves against us by 2 ATR
            
            if self.current_position == 1:  # Long position
                stop_price = self.entry_price - (atr_multiple * current['atr'])
                if current_price <= stop_price:
                    logger.info(f"Stop loss triggered for LONG at {current_price}")
                    return True
            
            elif self.current_position == -1:  # Short position
                stop_price = self.entry_price + (atr_multiple * current['atr'])
                if current_price >= stop_price:
                    logger.info(f"Stop loss triggered for SHORT at {current_price}")
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def calculate_position_size(self, balance):
        """Calculate position size based on available balance and risk management"""
        
        try:
            # Get current market price
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            
            # Calculate position size
            # We risk 1% of account balance per trade
            risk_percent = 0.01
            risk_amount = balance * risk_percent
            
            # For a 2 ATR stop loss (calculated in should_exit_position)
            # Convert to position size
            position_size_usd = risk_amount * self.params['leverage']
            position_size_btc = position_size_usd / price
            
            logger.info(f"Calculated position size: {position_size_btc:.6f} BTC (${position_size_usd:.2f})")
            return position_size_btc
        
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def execute_trade(self, action, size=None):
        """Execute a trade on the exchange"""
        
        if self.api_key is None or self.api_secret is None:
            logger.warning(f"Simulating {action} trade (no API keys provided)")
            return True
        
        try:
            # Get available balance
            balance = self.exchange.fetch_balance()
            available_balance = balance['free']['USDT']
            
            # Calculate position size if not provided
            if size is None:
                size = self.calculate_position_size(available_balance)
            
            # Skip if position size is too small
            if size <= 0:
                logger.warning("Position size too small, skipping trade")
                return False
            
            # Get current market price
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            
            # Execute trade based on action
            if action == 'buy':
                order = self.exchange.create_market_buy_order(
                    symbol=self.symbol,
                    amount=size
                )
                self.current_position = 1
                self.entry_price = price
                self.position_size = size
                logger.info(f"BUY order executed: {size} BTC at ${price}")
            
            elif action == 'sell':
                order = self.exchange.create_market_sell_order(
                    symbol=self.symbol,
                    amount=size
                )
                self.current_position = -1
                self.entry_price = price
                self.position_size = size
                logger.info(f"SELL order executed: {size} BTC at ${price}")
            
            elif action == 'close':
                if self.current_position == 1:
                    order = self.exchange.create_market_sell_order(
                        symbol=self.symbol,
                        amount=self.position_size
                    )
                    logger.info(f"Closed LONG position: {self.position_size} BTC at ${price}")
                
                elif self.current_position == -1:
                    order = self.exchange.create_market_buy_order(
                        symbol=self.symbol,
                        amount=self.position_size
                    )
                    logger.info(f"Closed SHORT position: {self.position_size} BTC at ${price}")
                
                self.current_position = 0
                self.entry_price = 0
                self.position_size = 0
            
            return True
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def save_state(self):
        """Save bot state to disk"""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'params': self.params
        }
        
        with open('bot_state.json', 'w') as f:
            json.dump(state, f, indent=4)
        
        logger.info(f"Bot state saved")
    
    def load_state(self):
        """Load bot state from disk"""
        
        if not os.path.exists('bot_state.json'):
            logger.info("No saved state found")
            return
        
        with open('bot_state.json', 'r') as f:
            state = json.load(f)
        
        self.current_position = state['current_position']
        self.entry_price = state['entry_price']
        self.position_size = state['position_size']
        
        logger.info(f"Bot state loaded: Position={self.current_position}, Entry=${self.entry_price}")
    
    def run_once(self):
        """Execute a single iteration of the trading strategy"""
        
        try:
            # Fetch market data
            data = self.fetch_data(limit=100)
            if data is None:
                logger.error("Failed to fetch data, skipping iteration")
                return
            
            # Calculate indicators
            data = self.calculate_indicators(data)
            if data is None:
                logger.error("Failed to calculate indicators, skipping iteration")
                return
            
            # Check if we need to exit current position
            if self.current_position != 0 and self.should_exit_position(data):
                self.execute_trade('close')
            
            # Generate new signal if not in a position
            if self.current_position == 0:
                signal, current_data = self.generate_signal(data)
                
                if signal == 1:  # Buy signal
                    self.execute_trade('buy')
                elif signal == -1:  # Sell signal
                    self.execute_trade('sell')
            
            # Save current state
            self.save_state()
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
    
    def run(self, interval_seconds=None):
        """Run the trading bot continuously"""
        
        if interval_seconds is None:
            # Set interval based on timeframe
            if self.timeframe == '1m':
                interval_seconds = 60
            elif self.timeframe == '5m':
                interval_seconds = 300
            elif self.timeframe == '15m':
                interval_seconds = 900
            elif self.timeframe == '1h':
                interval_seconds = 3600
            elif self.timeframe == '4h':
                interval_seconds = 14400
            else:  # Default to 1 hour
                interval_seconds = 3600
        
        # Load previous state if exists
        self.load_state()
        
        logger.info(f"Bot running with {self.timeframe} interval (checking every {interval_seconds} seconds)")
        
        try:
            while True:
                logger.info("-" * 50)
                logger.info(f"Running trading iteration at {datetime.now()}")
                
                # Execute trading logic
                self.run_once()
                
                # Wait for next iteration
                logger.info(f"Sleeping for {interval_seconds} seconds")
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Load optimized parameters (if available)
    params_file = Path('best_parameters.json')
    if params_file.exists():
        with open(params_file, 'r') as f:
            best_params = json.load(f)
    else:
        best_params = None  # Will use defaults
    
    # Initialize bot
    bot = LiveTradingBot(
        exchange_id='binance',
        api_key=os.environ.get('BINANCE_API_KEY'),
        api_secret=os.environ.get('BINANCE_API_SECRET'),
        symbol='BTC/USDT',
        timeframe='4h',
        params=best_params
    )
    
    # Run the bot (check every 30 minutes)
    bot.run(interval_seconds=1800)