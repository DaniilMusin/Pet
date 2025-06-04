import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BalancedAdaptiveStrategy:
    def __init__(self, data_path, 
                 initial_balance=1000, max_leverage=3, 
                 base_risk_per_trade=0.02, 
                 min_trades_interval=6):
        """
        Initialization of balanced adaptive strategy for BTC futures trading
        
        Args:
            data_path: path to CSV file with data
            initial_balance: initial balance in USD
            max_leverage: maximum leverage
            base_risk_per_trade: base risk per trade (% of balance)
            min_trades_interval: minimum interval between trades (in hours)
        """
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.base_risk_per_trade = base_risk_per_trade
        self.min_trades_interval = min_trades_interval
        self.slippage_pct = 0.05  # Default slippage 0.05%
        
        # Indicator parameters (will be optimized later)
        self.params = {
            # EMA parameters
            'short_ema': 9,
            'long_ema': 30,
            
            # RSI parameters
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            
            # ADX parameters
            'adx_period': 14,
            'adx_strong_trend': 25,
            'adx_weak_trend': 20,
            
            # Bollinger Bands parameters
            'bb_period': 20,
            'bb_std': 2,
            
            # MACD parameters
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # ATR parameters
            'atr_period': 14,
            'atr_multiplier_sl': 2.5,
            'atr_multiplier_tp': 5.0,
            
            # Volume filter parameters
            'volume_ma_period': 20,
            'volume_threshold': 1.5,
            
            # Trend parameters
            'trend_lookback': 20,
            'trend_threshold': 0.1,
            
            # Pyramiding parameters
            'pyramid_min_profit': 0.05,
            'pyramid_size_multiplier': 0.5,
            'max_pyramid_entries': 3,
            
            # Time filtering parameters
            'trading_hours_start': 8,
            'trading_hours_end': 16,
            
            # Weighted approach parameters
            'adx_min': 15,
            'adx_max': 35,
            
            # Market regime detection parameters
            'regime_volatility_lookback': 100,
            'regime_direction_short': 20,
            'regime_direction_medium': 50,
            'regime_direction_long': 100,
            
            # Mean reversion parameters
            'mean_reversion_lookback': 20,
            'mean_reversion_threshold': 2.0,
            
            # Multi-timeframe parameters
            'hourly_ema_fast': 9,
            'hourly_ema_slow': 30,
            'four_hour_ema_fast': 9,
            'four_hour_ema_slow': 30,
            
            # Market health parameters
            'health_trend_weight': 0.3,
            'health_volatility_weight': 0.2,
            'health_volume_weight': 0.2,
            'health_breadth_weight': 0.2,
            'health_sr_weight': 0.1,
            
            # Momentum parameters
            'momentum_roc_periods': [5, 10, 20, 50],
            'momentum_reversal_threshold': 5,
            
            # Optimal trading hours and days (will be filled by analysis)
            'optimal_trading_hours': None,
            'optimal_trading_days': None
        }
        
        # Will be populated during processing
        self.data = None
        self.trade_history = []
        self.backtest_results = None
        self.trade_df = None
        self.optimized_params = None
        
        # For tracking trading state
        self.max_price_seen = 0
        self.min_price_seen = float('inf')
    
    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        
        # Load data from CSV
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.tail(50520)  # Approximately 1 year of 15-minute candles
        
        # Convert dates
        self.data['Open time'] = pd.to_datetime(self.data['Open time'])
        self.data.set_index('Open time', inplace=True)
        
        # Ensure all numeric columns have correct type
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
        # Remove rows with NaN values
        self.data.dropna(subset=numeric_cols, inplace=True)
        
        print(f"Loaded {len(self.data)} candles")
        return self.data
    
    def calculate_indicators(self):
        """Calculate expanded set of technical indicators"""
        print("Calculating indicators...")
        
        # --- EMA ---
        self.data[f'EMA_{self.params["short_ema"]}'] = self._calculate_ema(self.data['Close'], self.params['short_ema'])
        self.data[f'EMA_{self.params["long_ema"]}'] = self._calculate_ema(self.data['Close'], self.params['long_ema'])
        
        # --- RSI ---
        self.data['RSI'] = self._calculate_rsi(self.data['Close'], self.params['rsi_period'])
        
        # --- Bollinger Bands ---
        bb_std = self.params['bb_std']
        bb_period = self.params['bb_period']
        self.data['BB_Middle'] = self.data['Close'].rolling(window=bb_period).mean()
        rolling_std = self.data['Close'].rolling(window=bb_period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * bb_std)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * bb_std)
        
        # --- ATR for dynamic stop-losses ---
        self.data['ATR'] = self._calculate_atr(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            self.params['atr_period']
        )
        
        # Add ATR moving average for volatility calculation
        self.data['ATR_MA'] = self.data['ATR'].rolling(20).mean()
        
        # --- ADX ---
        adx_period = self.params['adx_period']
        adx_results = self._calculate_adx(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            adx_period
        )
        self.data['ADX'] = adx_results['ADX']
        self.data['Plus_DI'] = adx_results['Plus_DI']
        self.data['Minus_DI'] = adx_results['Minus_DI']
        
        # --- MACD ---
        macd_results = self._calculate_macd(
            self.data['Close'], 
            self.params['macd_fast'], 
            self.params['macd_slow'], 
            self.params['macd_signal']
        )
        self.data['MACD'] = macd_results['MACD']
        self.data['MACD_Signal'] = macd_results['MACD_Signal']
        self.data['MACD_Hist'] = macd_results['MACD_Hist']
        
        # --- Volume filter ---
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=self.params['volume_ma_period']).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # --- Trend determination by price movement ---
        lookback = self.params['trend_lookback']
        self.data['Price_Change_Pct'] = (self.data['Close'] - self.data['Close'].shift(lookback)) / self.data['Close'].shift(lookback)
        
        # --- RSI divergence calculation ---
        # Look at price and RSI minimums/maximums
        self.data['Price_Min'] = self.data['Close'].rolling(5, center=True).min() == self.data['Close']
        self.data['Price_Max'] = self.data['Close'].rolling(5, center=True).max() == self.data['Close']
        self.data['RSI_Min'] = self.data['RSI'].rolling(5, center=True).min() == self.data['RSI']
        self.data['RSI_Max'] = self.data['RSI'].rolling(5, center=True).max() == self.data['RSI']
        
        # Divergences
        self.data['Bullish_Divergence'] = False
        self.data['Bearish_Divergence'] = False
        
        # Find local minimums and maximums
        price_mins = self.data[self.data['Price_Min']].index
        price_maxs = self.data[self.data['Price_Max']].index
        rsi_mins = self.data[self.data['RSI_Min']].index
        rsi_maxs = self.data[self.data['RSI_Max']].index
        
        for i in range(1, len(price_mins)):
            for j in range(1, len(rsi_mins)):
                if 0 <= (rsi_mins[j] - price_mins[i]).total_seconds() / 3600 <= 3:
                    price_change = self.data.loc[price_mins[i], 'Close'] - self.data.loc[price_mins[i-1], 'Close']
                    rsi_change = self.data.loc[rsi_mins[j], 'RSI'] - self.data.loc[rsi_mins[j-1], 'RSI']
                    if price_change < 0 and rsi_change > 0:
                        self.data.loc[max(price_mins[i], rsi_mins[j]), 'Bullish_Divergence'] = True
        
        for i in range(1, len(price_maxs)):
            for j in range(1, len(rsi_maxs)):
                if 0 <= (rsi_maxs[j] - price_maxs[i]).total_seconds() / 3600 <= 3:
                    price_change = self.data.loc[price_maxs[i], 'Close'] - self.data.loc[price_maxs[i-1], 'Close']
                    rsi_change = self.data.loc[rsi_maxs[j], 'RSI'] - self.data.loc[rsi_maxs[j-1], 'RSI']
                    if price_change > 0 and rsi_change < 0:
                        self.data.loc[max(price_maxs[i], rsi_maxs[j]), 'Bearish_Divergence'] = True
        
        self.data['Bullish_Divergence'].fillna(False, inplace=True)
        self.data['Bearish_Divergence'].fillna(False, inplace=True)
        
        # --- Improved market regime determination ---
        threshold = self.params['trend_threshold']
        self.data['Strong_Trend'] = (self.data['ADX'] > self.params['adx_strong_trend']) & \
                                    (self.data['Price_Change_Pct'].abs() > threshold)
        self.data['Weak_Trend'] = (self.data['ADX'] < self.params['adx_weak_trend']) & \
                                  (self.data['Price_Change_Pct'].abs() < threshold/2)
        
        self.data['Bullish_Trend'] = self.data['Strong_Trend'] & (self.data['Price_Change_Pct'] > 0) & \
                                     (self.data['Plus_DI'] > self.data['Minus_DI'])
        self.data['Bearish_Trend'] = self.data['Strong_Trend'] & (self.data['Price_Change_Pct'] < 0) & \
                                     (self.data['Plus_DI'] < self.data['Minus_DI'])
        
        # Trend weight
        self.data['Trend_Weight'] = np.minimum(1.0, np.maximum(0, 
                                              (self.data['ADX'] - self.params['adx_min']) / 
                                              (self.params['adx_max'] - self.params['adx_min'])))
        
        self.data['Range_Weight'] = 1.0 - self.data['Trend_Weight']
        
        # Time filter
        self.data['Hour'] = self.data.index.hour
        self.data['Active_Hours'] = (self.data['Hour'] >= self.params['trading_hours_start']) & \
                                    (self.data['Hour'] <= self.params['trading_hours_end'])
        
        # MACD signals
        self.data['MACD_Bullish_Cross'] = (self.data['MACD'] > self.data['MACD_Signal']) & \
                                          (self.data['MACD'].shift(1) <= self.data['MACD_Signal'].shift(1))
        self.data['MACD_Bearish_Cross'] = (self.data['MACD'] < self.data['MACD_Signal']) & \
                                          (self.data['MACD'].shift(1) >= self.data['MACD_Signal'].shift(1))
        
        # Higher-timeframe trend detection
        self.data['Daily_Close'] = self.data['Close'].resample('1D').last().reindex(self.data.index, method='ffill')
        self.data['Daily_EMA50'] = self._calculate_ema(self.data['Daily_Close'], 50).fillna(method='ffill')
        self.data['Daily_EMA200'] = self._calculate_ema(self.data['Daily_Close'], 200).fillna(method='ffill')
        self.data['Higher_TF_Bullish'] = self.data['Daily_EMA50'] > self.data['Daily_EMA200']
        self.data['Higher_TF_Bearish'] = self.data['Daily_EMA50'] < self.data['Daily_EMA200']
        
        # Market structure analysis
        self.data['HH'] = self.data['High'].rolling(10).max() > self.data['High'].rolling(20).max().shift(10)
        self.data['HL'] = self.data['Low'].rolling(10).min() > self.data['Low'].rolling(20).min().shift(10)
        self.data['LH'] = self.data['High'].rolling(10).max() < self.data['High'].rolling(20).max().shift(10)
        self.data['LL'] = self.data['Low'].rolling(10).min() < self.data['Low'].rolling(20).min().shift(10)
        self.data['Bullish_Structure'] = self.data['HH'] & self.data['HL']
        self.data['Bearish_Structure'] = self.data['LH'] & self.data['LL']
        
        # Day of week
        self.data['Day_of_Week'] = self.data.index.dayofweek
        self.data['Day_Name'] = self.data['Day_of_Week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        # Volume analysis improvements
        self.data['Volume_MA_3'] = self.data['Volume'].rolling(window=3).mean()
        self.data['Volume_MA_10'] = self.data['Volume'].rolling(window=10).mean()
        self.data['Rising_Volume'] = self.data['Volume'] > self.data['Volume_MA_3'] * 1.2
        self.data['Falling_Volume'] = self.data['Volume'] < self.data['Volume_MA_3'] * 0.8
        
        # Price action patterns
        self.data['Bullish_Engulfing'] = (
            (self.data['Open'] < self.data['Close'].shift(1)) & 
            (self.data['Close'] > self.data['Open'].shift(1)) & 
            (self.data['Close'] > self.data['Open']) &
            (self.data['Open'].shift(1) > self.data['Close'].shift(1))
        )
        
        self.data['Bearish_Engulfing'] = (
            (self.data['Open'] > self.data['Close'].shift(1)) & 
            (self.data['Close'] < self.data['Open'].shift(1)) & 
            (self.data['Close'] < self.data['Open']) &
            (self.data['Open'].shift(1) < self.data['Close'].shift(1))
        )
        
        # Volatility ratio
        self.data['Volatility_Ratio'] = self.data['ATR'] / self.data['ATR_MA'].replace(0, 1e-10)
        
        # 1. Enhanced Market Regime Detection
        self.detect_market_regime(lookback=self.params['regime_volatility_lookback'])
        
        # 2. Add Counter-Cyclical Indicators
        self.data['Price_to_200MA_Ratio'] = self.data['Close'] / self.data['Daily_EMA200']
        self.data['Extreme_Overbought'] = self.data['Price_to_200MA_Ratio'] > 1.3
        self.data['Extreme_Oversold'] = self.data['Price_to_200MA_Ratio'] < 0.7
        
        # Add momentum oscillator divergence
        self.data['Momentum'] = self.data['Close'] - self.data['Close'].shift(14)
        self.data['Momentum_SMA'] = self.data['Momentum'].rolling(window=10).mean()
        self.data['Price_Up_Momentum_Down'] = (self.data['Close'] > self.data['Close'].shift()) & \
                                              (self.data['Momentum'] < self.data['Momentum'].shift())
        self.data['Price_Down_Momentum_Up'] = (self.data['Close'] < self.data['Close'].shift()) & \
                                              (self.data['Momentum'] > self.data['Momentum'].shift())
        
        # 5. Multi-Timeframe Confirmation
        self.calculate_multi_timeframe_confirmation()
        
        # 6. Mean-Reversion Signals
        self.calculate_mean_reversion_signals()
        
        # 7. Market Cycle Phase Detection
        self.identify_market_cycle_phase()
        
        # 8. Market Health Score
        self.calculate_market_health()
        
        # 9. Momentum and Mean Reversion Balance
        self.calculate_momentum_reversion_balance()
        
        print("Indicators calculated!")
        
    def detect_market_regime(self, lookback=100):
        """Detect market regime based on volatility"""
        self.data['Volatility'] = self.data['ATR'].rolling(window=lookback).std()
        self.data['Regime'] = np.where(self.data['Volatility'] > self.data['Volatility'].shift(), 'High Volatility', 'Low Volatility')
    
    def calculate_multi_timeframe_confirmation(self):
        """Calculate multi-timeframe confirmation"""
        self.data['EMA_4H_Fast'] = self._calculate_ema(self.data['Close'], self.params['four_hour_ema_fast'])
        self.data['EMA_4H_Slow'] = self._calculate_ema(self.data['Close'], self.params['four_hour_ema_slow'])
        
        self.data['Hourly_EMA_Fast'] = self._calculate_ema(self.data['Close'], self.params['hourly_ema_fast'])
        self.data['Hourly_EMA_Slow'] = self._calculate_ema(self.data['Close'], self.params['hourly_ema_slow'])
    
    def calculate_mean_reversion_signals(self):
        """Calculate mean-reversion signals"""
        self.data['Mean_Reversion'] = self.data['Close'] - self.data['Close'].rolling(self.params['mean_reversion_lookback']).mean()
        
    def identify_market_cycle_phase(self):
        """Identify market cycle phase"""
        self.data['Market_Cycle_Phase'] = np.where(self.data['Price_to_200MA_Ratio'] > 1, 'Bull Market', 'Bear Market')
        
    def calculate_market_health(self):
        """Calculate market health score"""
        self.data['Market_Health'] = self.data['Trend_Weight'] * self.data['Volume_Ratio'] + \
                                     self.data['Range_Weight'] * self.data['Volatility_Ratio']
        
    def calculate_momentum_reversion_balance(self):
        """Calculate momentum-reversion balance"""
        self.data['Momentum_Reversion_Balance'] = self.data['Momentum'] * self.data['Mean_Reversion']
    
    def _calculate_macd(self, data, fast, slow, signal):
        """Calculate MACD and signal"""
        macd = data.ewm(span=fast, min_periods=fast).mean() - data.ewm(span=slow, min_periods=slow).mean()
        signal_line = macd.ewm(span=signal, min_periods=signal).mean()
        histogram = macd - signal_line
        return pd.DataFrame({'MACD': macd, 'MACD_Signal': signal_line, 'MACD_Hist': histogram})
    
    def _calculate_rsi(self, data, period):
        """Calculate RSI"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, data, span):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=span, adjust=False).mean()
    
    def _calculate_atr(self, high, low, close, period):
        """Calculate ATR (Average True Range)"""
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def backtest(self):
        """Run backtest on strategy"""
        pass  # This function will implement backtesting logic based on data signals.
