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
        self.data = self.data.tail(5520)  # Approximately 1 year of 15-minute candles
        
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
        
        # 9. Momentum and Reversal Metrics
        self.calculate_momentum_metrics()
        
        # 10. Adaptive Trade Management
        self.adapt_to_market_conditions()
        
        # Remove rows with NaN after calculating indicators
        self.data.dropna(inplace=True)
        
        print("Indicators calculated")
        return self.data
    
    def detect_market_regime(self, lookback=100):
        """Enhanced market regime detection with fractal patterns"""
        self.data['Market_Regime'] = 'unknown'
        
        for i in range(lookback, len(self.data)):
            recent = self.data.iloc[i-lookback:i+1]
            
            short_period = self.params['regime_direction_short']
            medium_period = self.params['regime_direction_medium']
            long_period = self.params['regime_direction_long']
            
            if len(recent) >= short_period:
                short_direction = (recent['Close'].iloc[-1] - recent['Close'].iloc[-short_period]) / \
                                   recent['Close'].iloc[-short_period]
            else:
                short_direction = 0
            
            if len(recent) >= medium_period:
                medium_direction = (recent['Close'].iloc[-1] - recent['Close'].iloc[-medium_period]) / \
                                    recent['Close'].iloc[-medium_period]
            else:
                medium_direction = 0
            
            if len(recent) >= long_period:
                long_direction = (recent['Close'].iloc[-1] - recent['Close'].iloc[-long_period]) / \
                                  recent['Close'].iloc[-long_period]
            else:
                long_direction = 0
            
            # Calculate volatility
            if len(recent) >= short_period:
                short_vol = recent['Close'].pct_change().rolling(short_period).std().iloc[-1]
            else:
                short_vol = 0
                
            if len(recent) >= medium_period:
                medium_vol = recent['Close'].pct_change().rolling(medium_period).std().iloc[-1]
            else:
                medium_vol = 0
                
            if len(recent) >= long_period:
                long_vol = recent['Close'].pct_change().rolling(long_period).std().iloc[-1]
            else:
                long_vol = 0
            
            vol_expanding = short_vol > medium_vol > long_vol if (medium_vol > 0 and long_vol > 0) else False
            vol_contracting = short_vol < medium_vol < long_vol if (short_vol > 0 and medium_vol > 0) else False
            
            # Identify regimes
            if short_direction > 0.05 and medium_direction > 0.03 and vol_expanding:
                self.data.iloc[i, self.data.columns.get_loc('Market_Regime')] = "strong_bull"
            elif short_direction < -0.05 and medium_direction < -0.03 and vol_expanding:
                self.data.iloc[i, self.data.columns.get_loc('Market_Regime')] = "strong_bear"
            elif abs(short_direction) < 0.02 and vol_contracting:
                self.data.iloc[i, self.data.columns.get_loc('Market_Regime')] = "choppy_range"
            elif short_direction > 0 and medium_direction < 0:
                self.data.iloc[i, self.data.columns.get_loc('Market_Regime')] = "transition_to_bull"
            elif short_direction < 0 and medium_direction > 0:
                self.data.iloc[i, self.data.columns.get_loc('Market_Regime')] = "transition_to_bear"
            else:
                self.data.iloc[i, self.data.columns.get_loc('Market_Regime')] = "mixed"
                
        self.data['Market_Regime'].fillna('unknown', inplace=True)
        
        # Set position size multipliers based on regime
        self.data['Regime_Long_Multiplier'] = 1.0
        self.data['Regime_Short_Multiplier'] = 1.0
        
        regime_multipliers = {
            "strong_bull": {"LONG": 1.2, "SHORT": 0.6},
            "strong_bear": {"LONG": 0.6, "SHORT": 1.2},
            "choppy_range": {"LONG": 0.8, "SHORT": 0.8},
            "transition_to_bull": {"LONG": 1.0, "SHORT": 0.8},
            "transition_to_bear": {"LONG": 0.8, "SHORT": 1.0},
            "mixed": {"LONG": 0.7, "SHORT": 0.7},
            "unknown": {"LONG": 0.7, "SHORT": 0.7}
        }
        
        for regime, multipliers in regime_multipliers.items():
            mask = self.data['Market_Regime'] == regime
            self.data.loc[mask, 'Regime_Long_Multiplier'] = multipliers["LONG"]
            self.data.loc[mask, 'Regime_Short_Multiplier'] = multipliers["SHORT"]
    
    def calculate_multi_timeframe_confirmation(self):
        """Create signals from multiple timeframes for stronger confirmation"""
        self.data['Hourly_Bullish'] = False
        self.data['4H_Bullish'] = False
        
        try:
            hourly = self.data['Close'].resample('1H').ohlc()
            hourly_ema_fast = self._calculate_ema(hourly['close'], self.params['hourly_ema_fast'])
            hourly_ema_slow = self._calculate_ema(hourly['close'], self.params['hourly_ema_slow'])
            hourly_signal = hourly_ema_fast > hourly_ema_slow
            
            four_hour = self.data['Close'].resample('4H').ohlc()
            four_hour_ema_fast = self._calculate_ema(four_hour['close'], self.params['four_hour_ema_fast'])
            four_hour_ema_slow = self._calculate_ema(four_hour['close'], self.params['four_hour_ema_slow'])
            four_hour_signal = four_hour_ema_fast > four_hour_ema_slow
            
            for idx, row in self.data.iterrows():
                hour_idx = idx.floor('H')
                four_hour_idx = idx.floor('4H')
                
                if hour_idx in hourly_signal.index:
                    self.data.at[idx, 'Hourly_Bullish'] = hourly_signal[hour_idx]
                if four_hour_idx in four_hour_signal.index:
                    self.data.at[idx, '4H_Bullish'] = four_hour_signal[four_hour_idx]
                    
            self.data['MTF_Bull_Strength'] = (
                (self.data['Hourly_Bullish'].astype(int) + 
                 self.data['4H_Bullish'].astype(int) + 
                 self.data['Higher_TF_Bullish'].astype(int)) / 3
            )
            self.data['MTF_Bear_Strength'] = (
                ((~self.data['Hourly_Bullish']).astype(int) + 
                 (~self.data['4H_Bullish']).astype(int) + 
                 self.data['Higher_TF_Bearish'].astype(int)) / 3
            )
        except Exception as e:
            print(f"Error in multi-timeframe calculation: {e}")
            # Fallback if resampling fails
            self.data['MTF_Bull_Strength'] = self.data['Higher_TF_Bullish'].astype(int)
            self.data['MTF_Bear_Strength'] = self.data['Higher_TF_Bearish'].astype(int)
    
    def calculate_mean_reversion_signals(self):
        """Calculate mean reversion indicators for range markets"""
        lookback = self.params['mean_reversion_lookback']
        threshold = self.params['mean_reversion_threshold']
        
        self.data['Close_SMA20'] = self.data['Close'].rolling(window=lookback).mean()
        self.data['Price_Deviation'] = self.data['Close'] - self.data['Close_SMA20']
        self.data['Price_Deviation_Std'] = self.data['Price_Deviation'].rolling(window=lookback).std()
        self.data['Z_Score'] = self.data['Price_Deviation'] / self.data['Price_Deviation_Std'].replace(0, np.nan)
        
        self.data['Stat_Overbought'] = self.data['Z_Score'] > threshold
        self.data['Stat_Oversold'] = self.data['Z_Score'] < -threshold
        
        self.data['MR_Long_Signal'] = (
            (self.data['Z_Score'] < -threshold) & 
            (self.data['Z_Score'].shift(1) >= -threshold)
        )
        self.data['MR_Short_Signal'] = (
            (self.data['Z_Score'] > threshold) & 
            (self.data['Z_Score'].shift(1) <= threshold)
        )
    
    def identify_market_cycle_phase(self):
        """Identify market cycle phase using multiple metrics"""
        self.data['Cycle_Phase'] = 'Unknown'
        
        accumulation_mask = (
            (self.data['RSI'] < 40) & 
            (self.data['Volume_Ratio'] > 1.2) & 
            (self.data['Close'] < self.data['Daily_EMA50']) & 
            (self.data['Close'] > self.data['Close'].shift(10))
        )
        self.data.loc[accumulation_mask, 'Cycle_Phase'] = 'Accumulation'
        
        markup_mask = (
            self.data['Bullish_Trend'] & 
            self.data['Higher_TF_Bullish'] & 
            (self.data['Volume_Ratio'] > 1.0)
        )
        self.data.loc[markup_mask, 'Cycle_Phase'] = 'Markup'
        
        distribution_mask = (
            (self.data['RSI'] > 60) & 
            (self.data['Volume_Ratio'] > 1.2) & 
            (self.data['Close'] > self.data['Daily_EMA50']) & 
            (self.data['Close'] < self.data['Close'].shift(10))
        )
        self.data.loc[distribution_mask, 'Cycle_Phase'] = 'Distribution'
        
        markdown_mask = (
            self.data['Bearish_Trend'] & 
            self.data['Higher_TF_Bearish'] & 
            (self.data['Volume_Ratio'] > 1.0)
        )
        self.data.loc[markdown_mask, 'Cycle_Phase'] = 'Markdown'
        
        self.data['Long_Phase_Weight'] = 1.0
        self.data['Short_Phase_Weight'] = 1.0
        
        self.data.loc[self.data['Cycle_Phase'] == 'Accumulation', 'Long_Phase_Weight'] = 1.3
        self.data.loc[self.data['Cycle_Phase'] == 'Accumulation', 'Short_Phase_Weight'] = 0.7
        
        self.data.loc[self.data['Cycle_Phase'] == 'Markup', 'Long_Phase_Weight'] = 1.5
        self.data.loc[self.data['Cycle_Phase'] == 'Markup', 'Short_Phase_Weight'] = 0.5
        
        self.data.loc[self.data['Cycle_Phase'] == 'Distribution', 'Long_Phase_Weight'] = 0.7
        self.data.loc[self.data['Cycle_Phase'] == 'Distribution', 'Short_Phase_Weight'] = 1.3
        
        self.data.loc[self.data['Cycle_Phase'] == 'Markdown', 'Long_Phase_Weight'] = 0.5
        self.data.loc[self.data['Cycle_Phase'] == 'Markdown', 'Short_Phase_Weight'] = 1.5
    
    def calculate_market_health(self):
        """Calculate overall market health score (0-100)"""
        self.data['Trend_Health'] = (self.data['Close'] > self.data['Daily_EMA50']).astype(int) * 20
        
        vol_ratio = self.data['ATR'] / self.data['ATR'].rolling(100).mean().replace(0, 1e-10)
        self.data['Volatility_Health'] = 20 - (vol_ratio - 1).clip(0, 2) * 10
        
        self.data['Volume_Health'] = self.data['Volume_Ratio'].clip(0, 2) * 10
        
        indicators_bullish = (
            (self.data['RSI'] > 50).astype(int) + 
            (self.data['MACD'] > 0).astype(int) + 
            (self.data[f'EMA_{self.params["short_ema"]}'] > self.data[f'EMA_{self.params["long_ema"]}']).astype(int) +
            (self.data['Bullish_Structure']).astype(int)
        ) / 4 * 20
        self.data['Breadth_Health'] = indicators_bullish
        
        bb_position = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        bb_position = bb_position.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        self.data['Support_Resistance_Health'] = (0.5 - abs(bb_position - 0.5)) * 2 * 20
        
        self.data['Market_Health'] = (
            self.data['Trend_Health'] * self.params['health_trend_weight'] + 
            self.data['Volatility_Health'] * self.params['health_volatility_weight'] + 
            self.data['Volume_Health'] * self.params['health_volume_weight'] + 
            self.data['Breadth_Health'] * self.params['health_breadth_weight'] + 
            self.data['Support_Resistance_Health'] * self.params['health_sr_weight']
        ).clip(0, 100)
        
        self.data['Health_Long_Bias'] = self.data['Market_Health'] / 100
        self.data['Health_Short_Bias'] = 1 - (self.data['Market_Health'] / 100)
    
    def calculate_momentum_metrics(self):
        """Calculate momentum strength and detect potential reversals"""
        periods = self.params['momentum_roc_periods']
        for period in periods:
            self.data[f'ROC_{period}'] = self.data['Close'].pct_change(period) * 100
        
        momentum_components = [
            (np.sign(self.data[f'ROC_{period}']) * self.data[f'ROC_{period}'].abs() ** 0.5) 
            for period in periods
        ]
        self.data['Momentum_Score'] = sum(momentum_components) / len(periods)
        
        max_val = self.data['Momentum_Score'].abs().max()
        if max_val > 0:
            self.data['Momentum_Score'] = self.data['Momentum_Score'] * (100 / max_val)
        
        self.data['Mom_Acceleration'] = self.data['Momentum_Score'].diff(3)
        reversal_threshold = self.params['momentum_reversal_threshold']
        self.data['Potential_Momentum_Reversal'] = (
            ((self.data['Momentum_Score'] > 80) & (self.data['Mom_Acceleration'] < -reversal_threshold)) |
            ((self.data['Momentum_Score'] < -80) & (self.data['Mom_Acceleration'] > reversal_threshold))
        )
        
        self.data['Momentum_Long_Bias'] = ((self.data['Momentum_Score'] + 100) / 200).clip(0.3, 0.7)
        self.data['Momentum_Short_Bias'] = 1 - self.data['Momentum_Long_Bias']
    
    def adapt_to_market_conditions(self):
        """Apply all market condition metrics to create balanced strategy bias"""
        
        self.data['Final_Long_Bias'] = (
            self.data['Health_Long_Bias'] * 0.3 +
            self.data['Momentum_Long_Bias'] * 0.3 +
            self.data['Long_Phase_Weight'] / 2 * 0.2 +
            self.data['MTF_Bull_Strength'] * 0.2
        )
        
        self.data['Final_Short_Bias'] = (
            self.data['Health_Short_Bias'] * 0.3 +
            self.data['Momentum_Short_Bias'] * 0.3 +
            self.data['Short_Phase_Weight'] / 2 * 0.2 +
            self.data['MTF_Bear_Strength'] * 0.2
        )
        
        signal_threshold = 0.65
        
        self.data['Choppy_Market'] = (
            (self.data['Final_Long_Bias'] < signal_threshold) & 
            (self.data['Final_Short_Bias'] < signal_threshold)
        )
        
        self.data['MR_Signal_Weight'] = 0.5
        self.data.loc[self.data['Choppy_Market'], 'MR_Signal_Weight'] = 1.5
        
        self.data['Balanced_Long_Signal'] = (
            (self.data['Final_Long_Bias'] > signal_threshold) |
            (self.data['Choppy_Market'] & self.data['MR_Long_Signal'])
        )
        
        self.data['Balanced_Short_Signal'] = (
            (self.data['Final_Short_Bias'] > signal_threshold) |
            (self.data['Choppy_Market'] & self.data['MR_Short_Signal'])
        )
        
        self.data['Adaptive_Stop_Multiplier'] = np.where(
            self.data['Choppy_Market'],
            self.params['atr_multiplier_sl'] * 1.2,
            self.params['atr_multiplier_sl'] * 0.9
        )
        
        self.data['Adaptive_TP_Multiplier'] = np.where(
            self.data['Choppy_Market'],
            self.params['atr_multiplier_tp'] * 0.8,
            self.params['atr_multiplier_tp'] * 1.2
        )
    
    def adaptive_risk_per_trade(self, current_market_regime, win_rate_long, win_rate_short):
        base_risk = self.base_risk_per_trade
        long_adjustment = 1.0
        short_adjustment = 1.0
        
        if win_rate_long > 0.6:
            long_adjustment = 1.2
        elif win_rate_long < 0.4:
            long_adjustment = 0.8
            
        if win_rate_short > 0.6:
            short_adjustment = 1.2
        elif win_rate_short < 0.4:
            short_adjustment = 0.8
        
        regime_factors = {
            "strong_bull": {"LONG": 1.1, "SHORT": 0.7},
            "strong_bear": {"LONG": 0.7, "SHORT": 1.1},
            "choppy_range": {"LONG": 0.8, "SHORT": 0.8},
            "transition_to_bull": {"LONG": 0.9, "SHORT": 0.8},
            "transition_to_bear": {"LONG": 0.8, "SHORT": 0.9},
            "mixed": {"LONG": 0.7, "SHORT": 0.7},
            "unknown": {"LONG": 0.7, "SHORT": 0.7}
        }
        
        if current_market_regime not in regime_factors:
            current_market_regime = "mixed"
        
        return {
            "LONG": base_risk * long_adjustment * regime_factors[current_market_regime]["LONG"],
            "SHORT": base_risk * short_adjustment * regime_factors[current_market_regime]["SHORT"]
        }
    
    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_adx(self, high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        smooth_plus_dm = plus_dm.rolling(window=period).mean()
        smooth_minus_dm = minus_dm.rolling(window=period).mean()
        
        plus_di = 100 * (smooth_plus_dm / atr.replace(0, 1e-10))
        minus_di = 100 * (smooth_minus_dm / atr.replace(0, 1e-10))
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
        adx = dx.rolling(window=period).mean()
        
        return {
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di
        }
    
    def _calculate_macd(self, series, fast_period, slow_period, signal_period):
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return {
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Hist': macd_hist
        }
    
    def get_trading_signals(self, current, previous, regime_type):
        long_signals = []
        short_signals = []
        
        volume_multiplier = 1.0
        if current['Volume_Ratio'] > self.params['volume_threshold']:
            volume_multiplier = min(2.0, current['Volume_Ratio'] / self.params['volume_threshold'])
        
        health_factor_long = current['Health_Long_Bias']
        health_factor_short = current['Health_Short_Bias']
        
        momentum_factor_long = current['Momentum_Long_Bias']
        momentum_factor_short = current['Momentum_Short_Bias']
        
        phase_factor_long = current.get('Long_Phase_Weight', 1.0)
        phase_factor_short = current.get('Short_Phase_Weight', 1.0)
        
        regime_multiplier_long = current.get('Regime_Long_Multiplier', 1.0)
        regime_multiplier_short = current.get('Regime_Short_Multiplier', 1.0)
        
        if regime_type == 'trend':
            if current['Trend_Weight'] > 0.6:
                # Long signals
                if ((previous[f"EMA_{self.params['short_ema']}"] < previous[f"EMA_{self.params['long_ema']}"]) and 
                    (current[f"EMA_{self.params['short_ema']}"] >= current[f"EMA_{self.params['long_ema']}"])):
                    signal_weight = (current['Trend_Weight'] * 1.2 *
                                     health_factor_long * momentum_factor_long *
                                     phase_factor_long * regime_multiplier_long)
                    long_signals.append(('EMA Crossover', signal_weight))
                
                if (current['MACD_Bullish_Cross'] and 
                    current['MACD_Hist'] > 0 and 
                    current['MACD_Hist'] > previous['MACD_Hist']):
                    signal_weight = (current['Trend_Weight'] * 1.3 *
                                     health_factor_long * momentum_factor_long *
                                     phase_factor_long * regime_multiplier_long)
                    long_signals.append(('MACD Bullish Cross', signal_weight))
                
                if (current['Bullish_Trend'] and not previous['Bullish_Trend'] and 
                    current['Plus_DI'] > current['Minus_DI'] * 1.2):
                    signal_weight = (current['Trend_Weight'] * 1.5 *
                                     health_factor_long * momentum_factor_long *
                                     phase_factor_long * regime_multiplier_long)
                    long_signals.append(('Strong Bullish Trend', signal_weight))
                
                if current['Higher_TF_Bullish']:
                    for i in range(len(long_signals)):
                        signal, weight = long_signals[i]
                        long_signals[i] = (signal, weight * 1.3)
                
                if current['Balanced_Long_Signal'] and current['Final_Long_Bias'] > 0.65:
                    signal_weight = current['Final_Long_Bias'] * 1.5 * regime_multiplier_long
                    long_signals.append(('Balanced Long Signal', signal_weight))
                
                # Short signals
                if ((previous[f"EMA_{self.params['short_ema']}"] > previous[f"EMA_{self.params['long_ema']}"]) and 
                    (current[f"EMA_{self.params['short_ema']}"] <= current[f"EMA_{self.params['long_ema']}"])):
                    signal_weight = (current['Trend_Weight'] * 1.2 *
                                     health_factor_short * momentum_factor_short *
                                     phase_factor_short * regime_multiplier_short)
                    short_signals.append(('EMA Crossover', signal_weight))
                
                if (current['MACD_Bearish_Cross'] and 
                    current['MACD_Hist'] < 0 and 
                    current['MACD_Hist'] < previous['MACD_Hist']):
                    signal_weight = (current['Trend_Weight'] * 1.3 *
                                     health_factor_short * momentum_factor_short *
                                     phase_factor_short * regime_multiplier_short)
                    short_signals.append(('MACD Bearish Cross', signal_weight))
                
                if (current['Bearish_Trend'] and not previous['Bearish_Trend'] and 
                    current['Minus_DI'] > current['Plus_DI'] * 1.2):
                    signal_weight = (current['Trend_Weight'] * 1.5 *
                                     health_factor_short * momentum_factor_short *
                                     phase_factor_short * regime_multiplier_short)
                    short_signals.append(('Strong Bearish Trend', signal_weight))
                
                if current['Higher_TF_Bearish']:
                    for i in range(len(short_signals)):
                        signal, weight = short_signals[i]
                        short_signals[i] = (signal, weight * 1.3)
                
                if current['Balanced_Short_Signal'] and current['Final_Short_Bias'] > 0.65:
                    signal_weight = current['Final_Short_Bias'] * 1.5 * regime_multiplier_short
                    short_signals.append(('Balanced Short Signal', signal_weight))
                    
        else:
            # RANGING MARKET STRATEGY
            if current['Range_Weight'] > 0.6:
                # Long signals
                if (current['RSI'] < self.params['rsi_oversold'] and 
                    current['Close'] < current['BB_Lower'] * 1.01):
                    signal_weight = (current['Range_Weight'] * 1.3 *
                                     health_factor_long * phase_factor_long *
                                     regime_multiplier_long)
                    long_signals.append(('RSI Oversold + BB Lower', signal_weight))
                
                if current['Bullish_Divergence'] and current['RSI'] < 40:
                    signal_weight = (current['Range_Weight'] * 1.6 *
                                     health_factor_long * phase_factor_long *
                                     regime_multiplier_long)
                    long_signals.append(('Strong Bullish Divergence', signal_weight))
                
                if (current['Close'] > current['Open'] and 
                    previous['Close'] < previous['Open'] and
                    current['Low'] > previous['Low'] * 0.998 and
                    current['Volume'] > previous['Volume'] * 1.2):
                    signal_weight = (current['Range_Weight'] * 1.2 *
                                     health_factor_long * phase_factor_long *
                                     regime_multiplier_long)
                    long_signals.append(('Support Bounce', signal_weight))
                
                if current['MR_Long_Signal'] and current['Z_Score'] < -2.0:
                    signal_weight = (current['Range_Weight'] * 1.4 *
                                     current['MR_Signal_Weight'] * regime_multiplier_long)
                    long_signals.append(('Mean Reversion Long', signal_weight))
                
                # Short signals
                if (current['RSI'] > self.params['rsi_overbought'] and 
                    current['Close'] > current['BB_Upper'] * 0.99):
                    signal_weight = (current['Range_Weight'] * 1.3 *
                                     health_factor_short * phase_factor_short *
                                     regime_multiplier_short)
                    short_signals.append(('RSI Overbought + BB Upper', signal_weight))
                
                if current['Bearish_Divergence'] and current['RSI'] > 60:
                    signal_weight = (current['Range_Weight'] * 1.6 *
                                     health_factor_short * phase_factor_short *
                                     regime_multiplier_short)
                    short_signals.append(('Strong Bearish Divergence', signal_weight))
                
                if (current['Close'] < current['Open'] and 
                    previous['Close'] > previous['Open'] and
                    current['High'] < previous['High'] * 1.002 and
                    current['Volume'] > previous['Volume'] * 1.2):
                    signal_weight = (current['Range_Weight'] * 1.2 *
                                     health_factor_short * phase_factor_short *
                                     regime_multiplier_short)
                    short_signals.append(('Resistance Rejection', signal_weight))
                
                if current['MR_Short_Signal'] and current['Z_Score'] > 2.0:
                    signal_weight = (current['Range_Weight'] * 1.4 *
                                     current['MR_Signal_Weight'] * regime_multiplier_short)
                    short_signals.append(('Mean Reversion Short', signal_weight))
        
        long_signals = [(signal, weight * volume_multiplier) for signal, weight in long_signals]
        short_signals = [(signal, weight * volume_multiplier) for signal, weight in short_signals]
        
        long_weight = sum(weight for _, weight in long_signals) / len(long_signals) if long_signals else 0
        short_weight = sum(weight for _, weight in short_signals) / len(short_signals) if short_signals else 0
        
        return {
            'long_signals': long_signals,
            'short_signals': short_signals,
            'long_weight': long_weight,
            'short_weight': short_weight
        }

    def apply_advanced_filtering(self, current, signals):
        long_weight = signals['long_weight']
        short_weight = signals['short_weight']
        
        # Market volatility filter
        atr_ma = current['ATR_MA'] if (current['ATR_MA'] != 0 and not pd.isna(current['ATR_MA'])) else 1e-10
        vol_ratio = current['ATR'] / atr_ma
        
        if vol_ratio > 1.5:
            long_weight *= 0.7
            short_weight *= 0.7
        
        hour = current.name.hour
        if hour >= 0 and hour < 6:
            long_weight *= 0.8
            short_weight *= 0.8
        
        if current['Close'] > current['Open']:
            long_weight *= 1.1
            short_weight *= 0.9
        else:
            long_weight *= 0.9
            short_weight *= 1.1
        
        if hasattr(self, 'trade_df') and self.trade_df is not None and len(self.trade_df) >= 5:
            recent_trades = self.trade_df.tail(5)
            long_trades = recent_trades[recent_trades['position'] == 'LONG']
            short_trades = recent_trades[recent_trades['position'] == 'SHORT']
            
            if len(long_trades) > 0:
                long_win_rate = sum(1 for p in long_trades['pnl'] if p > 0) / len(long_trades)
                if long_win_rate > 0.6:
                    long_weight *= 1.2
                elif long_win_rate < 0.4:
                    long_weight *= 0.8
            
            if len(short_trades) > 0:
                short_win_rate = sum(1 for p in short_trades['pnl'] if p > 0) / len(short_trades)
                if short_win_rate > 0.6:
                    short_weight *= 1.2
                elif short_win_rate < 0.4:
                    short_weight *= 0.8
        
        if 'Final_Long_Bias' in current and 'Final_Short_Bias' in current:
            long_weight *= current['Final_Long_Bias']
            short_weight *= current['Final_Short_Bias']
        
        if 'Market_Regime' in current:
            regime = current['Market_Regime']
            
            if regime == 'strong_bull':
                long_weight *= 1.2
                short_weight *= 0.8
            elif regime == 'strong_bear':
                long_weight *= 0.8
                short_weight *= 1.2
            elif regime == 'choppy_range':
                if not any('Mean Reversion' in signal for signal, _ in signals['long_signals']):
                    long_weight *= 0.8
                if not any('Mean Reversion' in signal for signal, _ in signals['short_signals']):
                    short_weight *= 0.8
            elif regime == 'transition_to_bull':
                long_weight *= 1.1
            elif regime == 'transition_to_bear':
                short_weight *= 1.1
        
        return {
            'long_weight': long_weight,
            'short_weight': short_weight
        }

    def calculate_dynamic_exit_levels(self, position_type, entry_price, current_candle, trade_age_hours=0):
        if 'Adaptive_Stop_Multiplier' in current_candle and not pd.isna(current_candle['Adaptive_Stop_Multiplier']):
            sl_multiplier = current_candle['Adaptive_Stop_Multiplier']
        else:
            sl_multiplier = self.params['atr_multiplier_sl']
            
        if 'Adaptive_TP_Multiplier' in current_candle and not pd.isna(current_candle['Adaptive_TP_Multiplier']):
            tp_multiplier = current_candle['Adaptive_TP_Multiplier']
        else:
            tp_multiplier = self.params['atr_multiplier_tp']
        
        atr_value = current_candle['ATR']
        
        # Доп. защита
        if 'ATR_MA' in current_candle and not pd.isna(current_candle['ATR_MA']):
            if current_candle['ATR_MA'] == 0:
                pass  # уже есть защита выше, при делении заменили бы на 1e-10
        # ...
        
        if 'Market_Regime' in current_candle:
            regime = current_candle['Market_Regime']
            if regime == 'strong_bull' and position_type == 'LONG':
                tp_multiplier *= 1.2
            elif regime == 'strong_bear' and position_type == 'SHORT':
                tp_multiplier *= 1.2
            elif regime == 'choppy_range':
                tp_multiplier *= 0.8
                sl_multiplier *= 1.2
        
        if trade_age_hours > 4:
            age_factor = min(3.0, 1.0 + (trade_age_hours - 4) / 20)
            sl_multiplier = sl_multiplier / age_factor
        
        if position_type == 'LONG':
            stop_loss = entry_price * (1 - atr_value * sl_multiplier / entry_price)
            take_profit = entry_price * (1 + atr_value * tp_multiplier / entry_price)
        else:
            stop_loss = entry_price * (1 + atr_value * sl_multiplier / entry_price)
            take_profit = entry_price * (1 - atr_value * tp_multiplier / entry_price)
        
        if 'Range_Weight' in current_candle and current_candle['Range_Weight'] > 0.7:
            if position_type == 'LONG' and current_candle['BB_Upper'] < take_profit:
                take_profit = current_candle['BB_Upper']
            elif position_type == 'SHORT' and current_candle['BB_Lower'] > take_profit:
                take_profit = current_candle['BB_Lower']
        
        if position_type == 'LONG':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < 2.0:
            if position_type == 'LONG':
                take_profit = entry_price + (risk * 2.0)
            else:
                take_profit = entry_price - (risk * 2.0)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def apply_trailing_stop(self, position_type, entry_price, current_price, max_price, min_price, unrealized_pnl_pct):
        if unrealized_pnl_pct <= 0:
            return None
        
        if position_type == 'LONG':
            if unrealized_pnl_pct >= 0.03:
                if unrealized_pnl_pct >= 0.10:
                    trail_pct = unrealized_pnl_pct * 0.4
                    new_stop = entry_price * (1 + trail_pct)
                elif unrealized_pnl_pct >= 0.05:
                    trail_pct = unrealized_pnl_pct * 0.3
                    new_stop = entry_price * (1 + trail_pct)
                else:
                    trail_pct = unrealized_pnl_pct * 0.2
                    new_stop = entry_price * (1 + trail_pct)
                return new_stop
        
        else:  # SHORT
            if unrealized_pnl_pct >= 0.03:
                if unrealized_pnl_pct >= 0.10:
                    trail_pct = unrealized_pnl_pct * 0.4
                    new_stop = entry_price * (1 - trail_pct)
                elif unrealized_pnl_pct >= 0.05:
                    trail_pct = unrealized_pnl_pct * 0.3
                    new_stop = entry_price * (1 - trail_pct)
                else:
                    trail_pct = unrealized_pnl_pct * 0.2
                    new_stop = entry_price * (1 - trail_pct)
                return new_stop
        
        return None

    def calculate_optimal_leverage(self, current_candle, trade_direction, max_allowed_leverage=3):
        base_leverage = 2
        
        atr_ma = current_candle['ATR_MA'] if (not pd.isna(current_candle['ATR_MA']) and current_candle['ATR_MA'] != 0) else 1e-10
        vol_ratio = current_candle['ATR'] / atr_ma
        
        if vol_ratio > 1.5:
            vol_adjustment = 0.7
        elif vol_ratio < 0.8:
            vol_adjustment = 1.3
        else:
            vol_adjustment = 1.0
        
        trend_adjustment = 1.0
        if current_candle['ADX'] > 35:
            if (trade_direction == 'LONG' and current_candle['Plus_DI'] > current_candle['Minus_DI']) or \
               (trade_direction == 'SHORT' and current_candle['Minus_DI'] > current_candle['Plus_DI']):
                trend_adjustment = 1.2
            else:
                trend_adjustment = 0.7
        
        regime_adjustment = 1.0
        if 'Market_Regime' in current_candle:
            regime = current_candle['Market_Regime']
            if regime == 'strong_bull' and trade_direction == 'LONG':
                regime_adjustment = 1.2
            elif regime == 'strong_bear' and trade_direction == 'SHORT':
                regime_adjustment = 1.2
            elif regime == 'choppy_range':
                regime_adjustment = 0.8
            elif regime == 'transition_to_bull' and trade_direction == 'SHORT':
                regime_adjustment = 0.8
            elif regime == 'transition_to_bear' and trade_direction == 'LONG':
                regime_adjustment = 0.8
        
        health_adjustment = 1.0
        if 'Market_Health' in current_candle:
            health = current_candle['Market_Health']
            if trade_direction == 'LONG':
                health_adjustment = 0.8 + (health / 100) * 0.4
            else:
                health_adjustment = 1.2 - (health / 100) * 0.4
        
        optimal_leverage = base_leverage * vol_adjustment * trend_adjustment * regime_adjustment * health_adjustment
        return min(max_allowed_leverage, optimal_leverage)

    def adaptive_position_sizing(self, balance, risk_per_trade, entry_price, stop_loss_price, optimal_leverage):
        risk_amount = balance * risk_per_trade
        price_risk_pct = abs(entry_price - stop_loss_price) / entry_price
        
        leveraged_risk_pct = price_risk_pct * optimal_leverage
        
        if leveraged_risk_pct > 0:
            position_size = risk_amount / leveraged_risk_pct
        else:
            position_size = 0
        
        max_position = balance * optimal_leverage
        position_size = min(position_size, max_position)
        
        if balance <= 1000 and position_size < 100:
            position_size = max(100, position_size)
        
        return position_size
    
    def is_optimal_trading_time(self, timestamp):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        day_name = day_map[day_of_week]
        
        hour_optimal = True
        day_optimal = True
        
        if self.params['optimal_trading_hours'] is not None:
            hour_optimal = hour in self.params['optimal_trading_hours']
        
        if self.params['optimal_trading_days'] is not None:
            day_optimal = day_name in self.params['optimal_trading_days']
        
        return hour_optimal and day_optimal
    
    def calculate_kelly_criterion(self, win_rate, avg_win_pct, avg_loss_pct):
        if avg_loss_pct == 0:
            avg_loss_pct = 0.001
        win_loss_ratio = avg_win_pct / avg_loss_pct
        
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        kelly_pct = min(0.25, max(0, kelly_pct))
        
        return kelly_pct
    
    def dynamically_adjust_risk_parameters(self):
        # Проверяем, существует ли trade_df и является ли он не None
        if not hasattr(self, 'trade_df') or self.trade_df is None or len(self.trade_df) < 20:
            # Здесь, например, можно задать какое-то значение по умолчанию или обработать ошибку
            print("Trade DataFrame не инициализирован или недостаточно данных.")
            return
        
        recent_trades = self.trade_df.tail(20)
        
        win_rate = sum(1 for p in recent_trades['pnl'] if p > 0) / len(recent_trades)
        profit_sum = sum(p for p in recent_trades['pnl'] if p > 0)
        loss_sum = abs(sum(p for p in recent_trades['pnl'] if p <= 0))
        
        win_trades = [p for p in recent_trades['pnl'] if p > 0]
        loss_trades = [abs(p) for p in recent_trades['pnl'] if p <= 0]
        
        avg_win = sum(win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(loss_trades) / len(loss_trades) if loss_trades else 0
        
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
        
        if profit_factor > 1.5 and win_rate > 0.5:
            self.base_risk_per_trade = min(0.03, self.base_risk_per_trade * 1.1)
        elif profit_factor < 1.0 or win_rate < 0.4:
            self.base_risk_per_trade = max(0.01, self.base_risk_per_trade * 0.9)
        
        if avg_win > 0 and avg_loss > 0:
            current_rr_ratio = avg_win / avg_loss
            if current_rr_ratio < 1.5:
                self.params['atr_multiplier_tp'] = min(7.0, self.params['atr_multiplier_tp'] * 1.05)
            elif current_rr_ratio > 3.0 and win_rate < 0.4:
                self.params['atr_multiplier_tp'] = max(3.0, self.params['atr_multiplier_tp'] * 0.95)
        
        long_trades = recent_trades[recent_trades['position'] == 'LONG']
        short_trades = recent_trades[recent_trades['position'] == 'SHORT']
        
        long_win_rate = sum(1 for p in long_trades['pnl'] if p > 0) / len(long_trades) if len(long_trades) > 0 else 0.5
        short_win_rate = sum(1 for p in short_trades['pnl'] if p > 0) / len(short_trades) if len(short_trades) > 0 else 0.5
        
        self.recent_long_win_rate = long_win_rate
        self.recent_short_win_rate = short_win_rate
    
    def analyze_hour_performance(self):
        if not hasattr(self, 'trade_df') or len(self.trade_df) == 0:
            return None
        self.trade_df['entry_hour'] = pd.to_datetime(self.trade_df['entry_date']).dt.hour
        
        hour_stats = self.trade_df.groupby('entry_hour').agg({
            'pnl': ['count', 'mean', 'sum'],
            'position': 'count'
        }).reset_index()
        
        hour_stats.columns = ['hour', 'num_trades', 'avg_pnl', 'total_pnl', 'position_count']
        
        win_rates = []
        for hour in hour_stats['hour'].unique():
            hour_trades = self.trade_df[self.trade_df['entry_hour'] == hour]
            wins = sum(1 for pnl in hour_trades['pnl'] if pnl > 0)
            total = len(hour_trades)
            win_rate = wins / total if total > 0 else 0
            win_rates.append(win_rate)
        
        hour_stats['win_rate'] = win_rates
        hour_stats = hour_stats.sort_values('win_rate', ascending=False)
        
        optimal_hours = hour_stats[(hour_stats['win_rate'] > 0.5) & (hour_stats['num_trades'] >= 5)]['hour'].tolist()
        if optimal_hours:
            self.params['optimal_trading_hours'] = optimal_hours
        
        return hour_stats
    
    def analyze_day_performance(self):
        if not hasattr(self, 'trade_df') or len(self.trade_df) == 0:
            return None
        self.trade_df['entry_day'] = pd.to_datetime(self.trade_df['entry_date']).dt.dayofweek
        
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        self.trade_df['day_name'] = self.trade_df['entry_day'].map(day_map)
        
        day_stats = self.trade_df.groupby('day_name').agg({
            'pnl': ['count', 'mean', 'sum'],
            'position': 'count'
        }).reset_index()
        
        day_stats.columns = ['day', 'num_trades', 'avg_pnl', 'total_pnl', 'position_count']
        
        win_rates = []
        for day in day_stats['day'].unique():
            day_trades = self.trade_df[self.trade_df['day_name'] == day]
            wins = sum(1 for pnl in day_trades['pnl'] if pnl > 0)
            total = len(day_trades)
            win_rate = wins / total if total > 0 else 0
            win_rates.append(win_rate)
        
        day_stats['win_rate'] = win_rates
        day_stats = day_stats.sort_values('win_rate', ascending=False)
        
        optimal_days = day_stats[(day_stats['win_rate'] > 0.5) & (day_stats['num_trades'] >= 5)]['day'].tolist()
        if optimal_days:
            self.params['optimal_trading_days'] = optimal_days
        
        return day_stats
    
    def calculate_correlation_metrics(self, benchmark_path=None):
        if benchmark_path is None:
            self.data['Benchmark_Return'] = self.data['Close'].pct_change()
        else:
            try:
                benchmark = pd.read_csv(benchmark_path)
                benchmark['Date'] = pd.to_datetime(benchmark['Date'])
                benchmark.set_index('Date', inplace=True)
                benchmark_returns = benchmark['Close'].pct_change()
                self.data['Benchmark_Return'] = benchmark_returns.resample(
                    f"{int(24*60/len(self.data))}H"
                ).last().reindex(self.data.index, method='ffill')
            except Exception as e:
                print(f"Error loading benchmark data: {e}")
                self.data['Benchmark_Return'] = self.data['Close'].pct_change()
        
        if hasattr(self, 'backtest_results') and 'equity' in self.backtest_results.columns:
            strategy_returns = self.backtest_results['equity'].pct_change().fillna(0)
            window = min(100, len(strategy_returns)//4)
            min_length = min(len(strategy_returns), len(self.data['Benchmark_Return']))
            strategy_returns = strategy_returns.iloc[:min_length]
            benchmark_returns = self.data['Benchmark_Return'].iloc[:min_length]
            
            rolling_corr = strategy_returns.rolling(window).corr(benchmark_returns)
            autocorr = pd.Series(
                [strategy_returns.autocorr(lag=i) for i in range(1, 11)],
                index=[f'lag_{i}' for i in range(1, 11)]
            )
            
            print("\n===== CORRELATION METRICS =====")
            print(f"Average correlation with BTC price: {rolling_corr.mean():.4f}")
            print("Strategy autocorrelation:")
            for lag, value in autocorr.items():
                print(f"  {lag}: {value:.4f}")
            
            if abs(rolling_corr.mean()) < 0.3:
                print("Strategy shows low correlation with BTC price, suggesting market-neutral characteristics")
            elif rolling_corr.mean() > 0.7:
                print("Strategy is highly correlated with BTC price, may struggle in bear markets")
            elif rolling_corr.mean() < -0.7:
                print("Strategy is highly negatively correlated with BTC price, may struggle in bull markets")

    def run_backtest(self):
        print("Running backtest...")
        
        balance = self.initial_balance
        position = 0
        entry_price = 0.0
        position_size = 0.0
        entry_date = None
        last_trade_date = None
        pyramid_entries = 0
        stop_loss_price = 0.0
        take_profit_price = 0.0
        
        max_trade_price = 0.0
        min_trade_price = float('inf')
        
        results = []
        self.trade_history = []
        
        self.recent_long_win_rate = 0.5
        self.recent_short_win_rate = 0.5
        
        for i in range(1, len(self.data)):
            current = self.data.iloc[i]
            previous = self.data.iloc[i-1]
            
            current_balance = balance
            current_equity = balance
            
            if position != 0:
                if entry_price <= 0:
                    position = 0
                    entry_price = 0.0
                    position_size = 0.0
                    entry_date = None
                    pyramid_entries = 0
                    max_trade_price = 0.0
                    min_trade_price = float('inf')
                    continue
                
                max_trade_price = max(max_trade_price, current['High'])
                min_trade_price = min(min_trade_price, current['Low'])
                
                trade_age_hours = (current.name - entry_date).total_seconds() / 3600
                
                if position == 1:
                    unrealized_pnl_pct = (current['Close'] / entry_price) - 1
                    unrealized_pnl = unrealized_pnl_pct * position_size * self.max_leverage
                    current_equity = balance + unrealized_pnl
                    
                    new_stop = self.apply_trailing_stop('LONG', entry_price, current['Close'], max_trade_price, min_trade_price, unrealized_pnl_pct)
                    if new_stop is not None and new_stop > stop_loss_price:
                        stop_loss_price = new_stop
                    
                    if current['Low'] <= stop_loss_price:
                        pnl = ((stop_loss_price / entry_price) - 1) * position_size * self.max_leverage
                        balance += pnl
                        
                        self.trade_history.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current.name,
                            'exit_price': stop_loss_price,
                            'position': 'LONG',
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Stop Loss',
                            'pyramid_entries': pyramid_entries,
                            'trade_duration': trade_age_hours
                        })
                        
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    if current['High'] >= take_profit_price:
                        pnl = ((take_profit_price / entry_price) - 1) * position_size * self.max_leverage
                        balance += pnl
                        
                        self.trade_history.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current.name,
                            'exit_price': take_profit_price,
                            'position': 'LONG',
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Take Profit',
                            'pyramid_entries': pyramid_entries,
                            'trade_duration': trade_age_hours
                        })
                        
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    if (pyramid_entries < self.params['max_pyramid_entries'] and 
                        current['Bullish_Trend'] and 
                        current['ADX'] > 40 and 
                        current['Close'] > entry_price * (1 + self.params['pyramid_min_profit']) and
                        (current['MACD_Bullish_Cross'] or current['Final_Long_Bias'] > 0.7)):
                        
                        additional_size = position_size * self.params['pyramid_size_multiplier']
                        old_value = entry_price * position_size
                        new_value = current['Close'] * additional_size
                        position_size += additional_size
                        entry_price = (old_value + new_value) / position_size
                        
                        exit_levels = self.calculate_dynamic_exit_levels('LONG', entry_price, current, trade_age_hours)
                        stop_loss_price = exit_levels['stop_loss']
                        take_profit_price = exit_levels['take_profit']
                        
                        pyramid_entries += 1
                        continue
                    
                    exit_signals = []
                    current_regime = 'trend' if current['Trend_Weight'] > 0.5 else 'range'
                    
                    if current_regime == 'trend':
                        if ((previous[f'EMA_{self.params["short_ema"]}'] >= previous[f'EMA_{self.params["long_ema"]}']) and 
                            (current[f'EMA_{self.params["short_ema"]}'] < current[f'EMA_{self.params["long_ema"]}'])):
                            exit_signals.append('EMA Crossover')
                        
                        if current['MACD_Bearish_Cross']:
                            exit_signals.append('MACD Crossover')
                        
                        if (current['Bearish_Trend'] and not previous['Bearish_Trend']):
                            exit_signals.append('Trend Change')
                        
                        if current['Higher_TF_Bearish'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Higher TF Trend Change')
                        
                        if 'Market_Regime' in current:
                            if current['Market_Regime'] == 'transition_to_bear' and unrealized_pnl_pct > 0.02:
                                exit_signals.append('Regime Change to Bear')
                    
                    else:
                        if current['RSI'] > self.params['rsi_overbought']:
                            exit_signals.append('RSI Overbought')
                        if current['Close'] > current['BB_Upper']:
                            exit_signals.append('Upper Bollinger')
                        if current['Bearish_Divergence']:
                            exit_signals.append('Bearish Divergence')
                        if current['Bearish_Engulfing'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Bearish Engulfing')
                        if current['Stat_Overbought'] and unrealized_pnl_pct > 0.03:
                            exit_signals.append('Statistical Overbought')
                    
                    if current['Volatility_Ratio'] > 2.0 and unrealized_pnl_pct > 0.03:
                        exit_signals.append('Volatility Spike')
                    
                    if (unrealized_pnl_pct > 0.08 and 
                        previous['Close'] > current['Close'] and 
                        previous['MACD'] > current['MACD']):
                        exit_signals.append('Profit Protection')
                    
                    if 'Potential_Momentum_Reversal' in current and current['Potential_Momentum_Reversal'] and unrealized_pnl_pct > 0.04:
                        exit_signals.append('Momentum Reversal')
                    
                    if 'Final_Short_Bias' in current and current['Final_Short_Bias'] > 0.7 and unrealized_pnl_pct > 0.03:
                        exit_signals.append('Bias Shift')
                    
                    if exit_signals and trade_age_hours > 4:
                        pnl = ((current['Close'] / entry_price) - 1) * position_size * self.max_leverage
                        balance += pnl
                        
                        self.trade_history.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current.name,
                            'exit_price': current['Close'],
                            'position': 'LONG',
                            'pnl': pnl,
                            'balance': balance,
                            'reason': ', '.join(exit_signals),
                            'pyramid_entries': pyramid_entries,
                            'trade_duration': trade_age_hours
                        })
                        
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                
                elif position == -1:
                    unrealized_pnl_pct = 1 - (current['Close'] / entry_price)
                    unrealized_pnl = unrealized_pnl_pct * position_size * self.max_leverage
                    current_equity = balance + unrealized_pnl
                    
                    new_stop = self.apply_trailing_stop('SHORT', entry_price, current['Close'], max_trade_price, min_trade_price, unrealized_pnl_pct)
                    if new_stop is not None and new_stop < stop_loss_price:
                        stop_loss_price = new_stop
                    
                    if current['High'] >= stop_loss_price:
                        pnl = (1 - (stop_loss_price / entry_price)) * position_size * self.max_leverage
                        balance += pnl
                        
                        self.trade_history.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current.name,
                            'exit_price': stop_loss_price,
                            'position': 'SHORT',
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Stop Loss',
                            'pyramid_entries': pyramid_entries,
                            'trade_duration': trade_age_hours
                        })
                        
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    if current['Low'] <= take_profit_price:
                        pnl = (1 - (take_profit_price / entry_price)) * position_size * self.max_leverage
                        balance += pnl
                        
                        self.trade_history.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current.name,
                            'exit_price': take_profit_price,
                            'position': 'SHORT',
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Take Profit',
                            'pyramid_entries': pyramid_entries,
                            'trade_duration': trade_age_hours
                        })
                        
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    if (pyramid_entries < self.params['max_pyramid_entries'] and 
                        current['Bearish_Trend'] and 
                        current['ADX'] > 40 and 
                        current['Close'] < entry_price * (1 - self.params['pyramid_min_profit']) and
                        (current['MACD_Bearish_Cross'] or current['Final_Short_Bias'] > 0.7)):
                        
                        additional_size = position_size * self.params['pyramid_size_multiplier']
                        old_value = entry_price * position_size
                        new_value = current['Close'] * additional_size
                        position_size += additional_size
                        entry_price = (old_value + new_value) / position_size
                        
                        exit_levels = self.calculate_dynamic_exit_levels('SHORT', entry_price, current, trade_age_hours)
                        stop_loss_price = exit_levels['stop_loss']
                        take_profit_price = exit_levels['take_profit']
                        
                        pyramid_entries += 1
                        continue
                    
                    exit_signals = []
                    current_regime = 'trend' if current['Trend_Weight'] > 0.5 else 'range'
                    
                    if current_regime == 'trend':
                        if ((previous[f'EMA_{self.params["short_ema"]}'] <= previous[f'EMA_{self.params["long_ema"]}']) and 
                            (current[f'EMA_{self.params["short_ema"]}'] > current[f'EMA_{self.params["long_ema"]}'])):
                            exit_signals.append('EMA Crossover')
                        
                        if current['MACD_Bullish_Cross']:
                            exit_signals.append('MACD Crossover')
                        
                        if current['Bullish_Trend'] and not previous['Bullish_Trend']:
                            exit_signals.append('Trend Change')
                        
                        if current['Higher_TF_Bullish'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Higher TF Trend Change')
                        
                        if 'Market_Regime' in current:
                            if current['Market_Regime'] == 'transition_to_bull' and unrealized_pnl_pct > 0.02:
                                exit_signals.append('Regime Change to Bull')
                    
                    else:
                        if current['RSI'] < self.params['rsi_oversold']:
                            exit_signals.append('RSI Oversold')
                        if current['Close'] < current['BB_Lower']:
                            exit_signals.append('Lower Bollinger')
                        if current['Bullish_Divergence']:
                            exit_signals.append('Bullish Divergence')
                        if current['Bullish_Engulfing'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Bullish Engulfing')
                        if current['Stat_Oversold'] and unrealized_pnl_pct > 0.03:
                            exit_signals.append('Statistical Oversold')
                    
                    if current['Volatility_Ratio'] > 2.0 and unrealized_pnl_pct > 0.03:
                        exit_signals.append('Volatility Spike')
                    
                    if (unrealized_pnl_pct > 0.08 and 
                        previous['Close'] < current['Close'] and 
                        previous['MACD'] < current['MACD']):
                        exit_signals.append('Profit Protection')
                    
                    if 'Potential_Momentum_Reversal' in current and current['Potential_Momentum_Reversal'] and unrealized_pnl_pct > 0.04:
                        exit_signals.append('Momentum Reversal')
                    
                    if 'Final_Long_Bias' in current and current['Final_Long_Bias'] > 0.7 and unrealized_pnl_pct > 0.03:
                        exit_signals.append('Bias Shift')
                    
                    if exit_signals and trade_age_hours > 4:
                        pnl = (1 - (current['Close'] / entry_price)) * position_size * self.max_leverage
                        balance += pnl
                        
                        self.trade_history.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current.name,
                            'exit_price': current['Close'],
                            'position': 'SHORT',
                            'pnl': pnl,
                            'balance': balance,
                            'reason': ', '.join(exit_signals),
                            'pyramid_entries': pyramid_entries,
                            'trade_duration': trade_age_hours
                        })
                        
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
            
            if position == 0:
                if last_trade_date is not None:
                    hours_since_last_trade = (current.name - last_trade_date).total_seconds() / 3600
                    if hours_since_last_trade < self.min_trades_interval:
                        continue
                
                if not current['Active_Hours']:
                    continue
                
                if not self.is_optimal_trading_time(current.name):
                    continue
                
                current_regime = 'trend' if current['Trend_Weight'] > 0.5 else 'range'
                signals = self.get_trading_signals(current, previous, current_regime)
                filtered_signals = self.apply_advanced_filtering(current, signals)
                
                market_regime = current.get('Market_Regime', 'unknown')
                
                adaptive_risk = self.adaptive_risk_per_trade(
                    market_regime, 
                    self.recent_long_win_rate, 
                    self.recent_short_win_rate
                )
                
                min_signal_threshold = 0.6
                
                if (filtered_signals['long_weight'] > filtered_signals['short_weight'] and 
                    filtered_signals['long_weight'] >= min_signal_threshold):
                    
                    position = 1
                    entry_price = current['Close']
                    if entry_price <= 0:
                        entry_price = 1.0
                    entry_date = current.name
                    
                    optimal_leverage = self.calculate_optimal_leverage(current, 'LONG', self.max_leverage)
                    exit_levels = self.calculate_dynamic_exit_levels('LONG', entry_price, current)
                    stop_loss_price = exit_levels['stop_loss']
                    take_profit_price = exit_levels['take_profit']
                    
                    risk_per_trade = adaptive_risk['LONG']
                    position_size = self.adaptive_position_sizing(balance, risk_per_trade, entry_price, stop_loss_price, optimal_leverage)
                    
                    max_trade_price = current['High']
                    min_trade_price = current['Low']
                    
                    signal_info = ', '.join(signal for signal, _ in signals['long_signals'])
                    
                    self.trade_history.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': None,
                        'exit_price': None,
                        'position': 'LONG',
                        'pnl': None,
                        'balance': balance,
                        'reason': f'Entry: {signal_info}',
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'weight': filtered_signals['long_weight'],
                        'risk_per_trade': risk_per_trade,
                        'leverage': optimal_leverage,
                        'market_regime': market_regime,
                        'market_health': current.get('Market_Health', None)
                    })
                
                elif (filtered_signals['short_weight'] > filtered_signals['long_weight'] and 
                      filtered_signals['short_weight'] >= min_signal_threshold):
                    
                    position = -1
                    entry_price = current['Close']
                    if entry_price <= 0:
                        entry_price = 1.0
                    entry_date = current.name
                    
                    optimal_leverage = self.calculate_optimal_leverage(current, 'SHORT', self.max_leverage)
                    exit_levels = self.calculate_dynamic_exit_levels('SHORT', entry_price, current)
                    stop_loss_price = exit_levels['stop_loss']
                    take_profit_price = exit_levels['take_profit']
                    
                    risk_per_trade = adaptive_risk['SHORT']
                    position_size = self.adaptive_position_sizing(balance, risk_per_trade, entry_price, stop_loss_price, optimal_leverage)
                    
                    max_trade_price = current['High']
                    min_trade_price = current['Low']
                    
                    signal_info = ', '.join(signal for signal, _ in signals['short_signals'])
                    
                    self.trade_history.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': None,
                        'exit_price': None,
                        'position': 'SHORT',
                        'pnl': None,
                        'balance': balance,
                        'reason': f'Entry: {signal_info}',
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'weight': filtered_signals['short_weight'],
                        'risk_per_trade': risk_per_trade,
                        'leverage': optimal_leverage,
                        'market_regime': market_regime,
                        'market_health': current.get('Market_Health', None)
                    })
            
            results.append({
                'date': current.name,
                'close': current['Close'],
                'balance': balance,
                'equity': current_equity,
                'position': position,
                'entry_price': entry_price,
                'position_size': position_size,
                'rsi': current['RSI'],
                'adx': current['ADX'],
                'is_trending': current['Strong_Trend'],
                'is_ranging': current['Weak_Trend'],
                'trend_weight': current['Trend_Weight'],
                'range_weight': current['Range_Weight'],
                'market_regime': current.get('Market_Regime', 'unknown'),
                'market_health': current.get('Market_Health', None),
                'long_bias': current.get('Final_Long_Bias', None),
                'short_bias': current.get('Final_Short_Bias', None)
            })
            
            if len(self.trade_history) % 20 == 0 and len(self.trade_history) > 0:
                self.dynamically_adjust_risk_parameters()
        
        # Закрыть позицию в конце
        if position != 0 and entry_price > 0:
            last_candle = self.data.iloc[-1]
            exit_price = last_candle['Close']
            trade_age_hours = (last_candle.name - entry_date).total_seconds() / 3600
            
            if position == 1:
                pnl = ((exit_price / entry_price) - 1) * position_size * self.max_leverage
            else:
                pnl = (1 - (exit_price / entry_price)) * position_size * self.max_leverage
            balance += pnl
            
            for trade in reversed(self.trade_history):
                if trade['exit_date'] is None:
                    trade['exit_date'] = last_candle.name
                    trade['exit_price'] = exit_price
                    trade['pnl'] = pnl
                    trade['balance'] = balance
                    trade['reason'] = trade['reason'] + ', End of Backtest'
                    trade['trade_duration'] = trade_age_hours
                    break
        
        self.backtest_results = pd.DataFrame(results)
        
        if self.trade_history:
            self.trade_df = pd.DataFrame(self.trade_history)
            self.trade_df['exit_date'].fillna(self.data.index[-1], inplace=True)
            self.trade_df['exit_price'].fillna(self.data['Close'].iloc[-1], inplace=True)
            
            for i, row in self.trade_df.iterrows():
                if pd.isna(row['pnl']):
                    if row['position'] == 'LONG':
                        pnl = ((row['exit_price'] / row['entry_price']) - 1) * row['balance'] * self.max_leverage
                    else:
                        pnl = (1 - (row['exit_price'] / row['entry_price'])) * row['balance'] * self.max_leverage
                    self.trade_df.at[i, 'pnl'] = pnl
        else:
            self.trade_df = pd.DataFrame()
        
        self.analyze_hour_performance()
        self.analyze_day_performance()
        
        print("Backtest completed")
        return self.backtest_results
    
    def plot_equity_curve(self):
        if self.backtest_results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        ax1.plot(self.backtest_results['date'], self.backtest_results['equity'], label='Equity', color='blue')
        ax1.plot(self.backtest_results['date'], self.backtest_results['balance'], label='Balance', color='green')
        
        equity_curve = self.backtest_results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max * 100
        
        ax2.fill_between(self.backtest_results['date'], 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylim(bottom=0, top=max(drawdown) * 1.5)
        ax2.invert_yaxis()
        
        if 'market_health' in self.backtest_results.columns and not self.backtest_results['market_health'].isnull().all():
            ax3.plot(self.backtest_results['date'], self.backtest_results['market_health'], 
                     label='Market Health', color='purple', alpha=0.7)
            ax3.set_ylim(bottom=0, top=100)
            ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax3.set_title('Market Health Score (0-100)')
        else:
            if 'market_regime' in self.backtest_results.columns and not self.backtest_results['market_regime'].isnull().all():
                regime_indicator = self.backtest_results['market_regime'].map({
                    'strong_bull': 1.0,
                    'transition_to_bull': 0.5,
                    'choppy_range': 0,
                    'transition_to_bear': -0.5,
                    'strong_bear': -1.0,
                    'mixed': 0,
                    'unknown': np.nan
                }).fillna(0)
                
                ax3.plot(self.backtest_results['date'], regime_indicator, 
                         label='Market Regime', color='orange', alpha=0.7)
                ax3.set_ylim(bottom=-1.2, top=1.2)
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax3.set_title('Market Regime (1=Bull, 0=Range, -1=Bear)')
        
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_formatter(date_format)
        ax3.xaxis.set_major_formatter(date_format)
        
        def currency_formatter(x, pos):
            return f'${x:,.0f}'
        ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        def percentage_formatter(x, pos):
            return f'{x:.0f}%'
        ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        initial_balance = self.initial_balance
        final_balance = self.backtest_results['balance'].iloc[-1]
        total_return = ((final_balance / initial_balance) - 1) * 100
        max_drawdown_val = drawdown.max()
        
        if hasattr(self, 'trade_df') and self.trade_df is not None and len(self.trade_df) > 0:
            win_rate = len(self.trade_df[self.trade_df['pnl'] > 0]) / len(self.trade_df) * 100
        else:
            win_rate = 0
        
        textstr = '\n'.join((
            f'Initial Balance: ${initial_balance:,.2f}',
            f'Final Balance: ${final_balance:,.2f}',
            f'Total Return: {total_return:.2f}%',
            f'Max Drawdown: {max_drawdown_val:.2f}%',
            f'Win Rate: {win_rate:.2f}%',
            f'Total Trades: {len(self.trade_df) if hasattr(self, "trade_df") else 0}'
        ))
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax1.text(0.02, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=props)
        
        if hasattr(self, 'trade_df') and len(self.trade_df) > 0:
            sample_trades = self.trade_df.sample(min(50, len(self.trade_df))) if len(self.trade_df) > 50 else self.trade_df
            for idx, trade in sample_trades.iterrows():
                if trade['position'] == 'LONG':
                    color = 'green'
                    marker = '^'
                else:
                    color = 'red'
                    marker = 'v'
                
                entry_date = pd.to_datetime(trade['entry_date'])
                exit_date = pd.to_datetime(trade['exit_date'])
                
                closest_entry = min(self.backtest_results['date'], key=lambda x: abs(x - entry_date))
                entry_equity = self.backtest_results.loc[self.backtest_results['date'] == closest_entry, 'equity'].values
                if len(entry_equity) > 0:
                    ax1.scatter(closest_entry, entry_equity[0], color=color, s=50, marker=marker, alpha=0.6)
                
                closest_exit = min(self.backtest_results['date'], key=lambda x: abs(x - exit_date))
                exit_equity = self.backtest_results.loc[self.backtest_results['date'] == closest_exit, 'equity'].values
                if len(exit_equity) > 0:
                    ax1.scatter(closest_exit, exit_equity[0], color='black', s=30, marker='o', alpha=0.6)
        
        ax1.set_title('Equity Curve and Performance')
        ax1.set_ylabel('Account Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_regime_performance(self):
        if not hasattr(self, 'trade_df') or len(self.trade_df) == 0 or 'market_regime' not in self.trade_df.columns:
            print("No regime data available.")
            return
        
        import matplotlib.pyplot as plt
        
        regime_stats = self.trade_df.groupby('market_regime').agg({
            'pnl': ['count', 'mean', 'sum'],
            'position': 'count'
        }).reset_index()
        
        regime_stats.columns = ['regime', 'num_trades', 'avg_pnl', 'total_pnl', 'position_count']
        
        regime_win_rates = []
        for regime in regime_stats['regime'].unique():
            regime_trades = self.trade_df[self.trade_df['market_regime'] == regime]
            wins = sum(1 for pnl in regime_trades['pnl'] if pnl > 0)
            total = len(regime_trades)
            win_rate = wins / total if total > 0 else 0
            regime_win_rates.append(win_rate * 100)
        
        regime_stats['win_rate'] = regime_win_rates
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        regime_colors = {
            'strong_bull': 'green',
            'transition_to_bull': 'lightgreen',
            'choppy_range': 'gray',
            'transition_to_bear': 'salmon',
            'strong_bear': 'red',
            'mixed': 'blue',
            'unknown': 'lightgray'
        }
        
        bars1 = ax1.bar(regime_stats['regime'], regime_stats['num_trades'], 
                        color=[regime_colors.get(r, 'gray') for r in regime_stats['regime']])
        ax1.set_title('Number of Trades by Market Regime')
        ax1.set_xlabel('Market Regime')
        ax1.set_ylabel('Number of Trades')
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        bars2 = ax2.bar(regime_stats['regime'], regime_stats['win_rate'], 
                        color=[regime_colors.get(r, 'gray') for r in regime_stats['regime']])
        ax2.set_title('Win Rate by Market Regime')
        ax2.set_xlabel('Market Regime')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, 100)
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        bars3 = ax3.bar(regime_stats['regime'], regime_stats['total_pnl'], 
                        color=[regime_colors.get(r, 'gray') for r in regime_stats['regime']])
        ax3.set_title('Total P&L by Market Regime')
        ax3.set_xlabel('Market Regime')
        ax3.set_ylabel('P&L ($)')
        for bar in bars3:
            height = bar.get_height()
            ax3.annotate(f'${height:.1f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_results(self):
        if self.backtest_results is None or len(self.trade_df) == 0:
            print("No data for analysis. Run backtest first.")
            return None
        
        print("\n===== BACKTEST RESULTS =====")
        
        initial_balance = self.initial_balance
        final_balance = self.backtest_results['balance'].iloc[-1]
        total_return = ((final_balance / initial_balance) - 1) * 100
        
        start_date = self.backtest_results['date'].iloc[0]
        end_date = self.backtest_results['date'].iloc[-1]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        if months_diff == 0:
            months_diff = 1
        monthly_return = (((final_balance / initial_balance) ** (1 / months_diff)) - 1) * 100
        
        equity_curve = self.backtest_results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max * 100
        max_drawdown = drawdown.max()
        
        total_trades = len(self.trade_df)
        profitable_trades = len(self.trade_df[self.trade_df['pnl'] > 0])
        losing_trades = len(self.trade_df[self.trade_df['pnl'] <= 0])
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit = self.trade_df[self.trade_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = self.trade_df[self.trade_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_sum = self.trade_df[self.trade_df['pnl'] > 0]['pnl'].sum()
        loss_sum = abs(self.trade_df[self.trade_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
        
        if 'equity' in self.backtest_results.columns:
            daily_returns = self.backtest_results['equity'].pct_change().dropna()
            if len(daily_returns) > 0:
                sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
                negative_returns = daily_returns[daily_returns < 0]
                downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-10
                sortino_ratio = (daily_returns.mean() * 252) / downside_deviation
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        if 'trade_duration' in self.trade_df.columns:
            median_duration_hours = self.trade_df['trade_duration'].median()
        else:
            self.trade_df['duration'] = pd.to_datetime(self.trade_df['exit_date']) - pd.to_datetime(self.trade_df['entry_date'])
            median_duration_hours = self.trade_df['duration'].median().total_seconds() / 3600
        
        long_trades = len(self.trade_df[self.trade_df['position'] == 'LONG'])
        short_trades = len(self.trade_df[self.trade_df['position'] == 'SHORT'])
        
        # Для вкладов позиций используем упрощённый расчёт:
        long_contribution = self.trade_df[self.trade_df['position'] == 'LONG']['pnl'].sum()
        short_contribution = self.trade_df[self.trade_df['position'] == 'SHORT']['pnl'].sum()
        
        avg_long_pnl = self.trade_df[self.trade_df['position'] == 'LONG']['pnl'].mean() if long_trades > 0 else 0
        avg_short_pnl = self.trade_df[self.trade_df['position'] == 'SHORT']['pnl'].mean() if short_trades > 0 else 0
        
        long_win_rate = len(self.trade_df[(self.trade_df['position'] == 'LONG') & (self.trade_df['pnl'] > 0)]) / long_trades * 100 if long_trades > 0 else 0
        short_win_rate = len(self.trade_df[(self.trade_df['position'] == 'SHORT') & (self.trade_df['pnl'] > 0)]) / short_trades * 100 if short_trades > 0 else 0
        
        # Подсчитаем причины выхода напрямую (без str.extract):
        exit_reason_counts = self.trade_df['reason'].value_counts()
        
        # Анализ по часам и дням
        hour_stats = self.analyze_hour_performance()
        day_stats = self.analyze_day_performance()
        
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Monthly Return: {monthly_return:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Profitable Trades: {profitable_trades} ({win_rate:.2f}%)")
        print(f"Losing Trades: {losing_trades} ({100 - win_rate:.2f}%)")
        print(f"Average Profit: ${avg_profit:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Median Trade Duration: {median_duration_hours:.2f} hours")
        
        print("\n===== STATISTICS BY TRADE TYPE =====")
        print(f"Long Trades: {long_trades} (Win Rate: {long_win_rate:.2f}%)")
        print(f"Short Trades: {short_trades} (Win Rate: {short_win_rate:.2f}%)")
        print(f"Average P&L Long Trade: ${avg_long_pnl:.2f}")
        print(f"Average P&L Short Trade: ${avg_short_pnl:.2f}")
        print(f"Total P&L (Long): ${long_contribution:.2f}")
        print(f"Total P&L (Short): ${short_contribution:.2f}")
        
        print("\n===== EXIT REASON DISTRIBUTION =====")
        for reason, count in exit_reason_counts.items():
            print(f"{reason}: {count}")
        
        self.backtest_results['year'] = pd.to_datetime(self.backtest_results['date']).dt.year
        yearly_performance = {}
        for year in self.backtest_results['year'].unique():
            year_data = self.backtest_results[self.backtest_results['year'] == year]
            start_balance_yr = year_data['balance'].iloc[0]
            end_balance_yr = year_data['balance'].iloc[-1]
            yearly_return = ((end_balance_yr / start_balance_yr) - 1) * 100
            yearly_performance[year] = yearly_return
        
        print("\n===== YEARLY PERFORMANCE =====")
        for year, yearly_return in yearly_performance.items():
            print(f"{year}: {yearly_return:.2f}%")
        
        if 'market_regime' in self.trade_df.columns:
            print("\n===== PERFORMANCE BY MARKET REGIME =====")
            regime_stats = self.trade_df.groupby('market_regime').agg({
                'pnl': ['count', 'mean', 'sum'],
                'position': 'count'
            }).reset_index()
            
            regime_stats.columns = ['regime', 'num_trades', 'avg_pnl', 'total_pnl', 'position_count']
            
            regime_win_rates = []
            for regime in regime_stats['regime'].unique():
                regime_trades = self.trade_df[self.trade_df['market_regime'] == regime]
                wins = sum(1 for pnl in regime_trades['pnl'] if pnl > 0)
                total = len(regime_trades)
                rw = wins / total if total > 0 else 0
                regime_win_rates.append(rw * 100)
            
            regime_stats['win_rate'] = regime_win_rates
            for _, row in regime_stats.iterrows():
                print(f"{row['regime']}: Win Rate {row['win_rate']:.2f}%, Avg P&L ${row['avg_pnl']:.2f}, Total P&L ${row['total_pnl']:.2f}, Trades: {row['num_trades']}")
        
        if 'market_health' in self.trade_df.columns and not self.trade_df['market_health'].isnull().all():
            print("\n===== PERFORMANCE BY MARKET HEALTH =====")
            self.trade_df['health_bin'] = pd.cut(
                self.trade_df['market_health'].fillna(50),
                bins=[0, 20, 40, 60, 80, 100],
                labels=['Very Poor (0-20)', 'Poor (20-40)', 'Neutral (40-60)', 'Good (60-80)', 'Excellent (80-100)']
            )
            
            health_stats = self.trade_df.groupby('health_bin').agg({
                'pnl': ['count', 'mean', 'sum'],
                'position': 'count'
            }).reset_index()
            
            health_stats.columns = ['health_range', 'num_trades', 'avg_pnl', 'total_pnl', 'position_count']
            
            hw_rates = []
            for health_range in health_stats['health_range'].unique():
                h_trades = self.trade_df[self.trade_df['health_bin'] == health_range]
                wins = sum(1 for pnl in h_trades['pnl'] if pnl > 0)
                total = len(h_trades)
                wr = wins / total if total > 0 else 0
                hw_rates.append(wr * 100)
            
            health_stats['win_rate'] = hw_rates
            for _, row in health_stats.iterrows():
                print(f"{row['health_range']}: Win Rate {row['win_rate']:.2f}%, Avg P&L ${row['avg_pnl']:.2f}, Total P&L ${row['total_pnl']:.2f}, Trades: {row['num_trades']}")
        
        if 'pyramid_entries' in self.trade_df.columns:
            pyramid_trades = self.trade_df[self.trade_df['pyramid_entries'] > 0].copy()
            non_pyramid_trades = self.trade_df[self.trade_df['pyramid_entries'] == 0].copy()
            
            pyramid_win_rate = (len(pyramid_trades[pyramid_trades['pnl'] > 0]) / len(pyramid_trades) * 100
                                if len(pyramid_trades) > 0 else 0)
            non_pyramid_win_rate = (len(non_pyramid_trades[non_pyramid_trades['pnl'] > 0]) / len(non_pyramid_trades) * 100
                                    if len(non_pyramid_trades) > 0 else 0)
            
            pyramid_profit = pyramid_trades['pnl'].sum()
            non_pyramid_profit = non_pyramid_trades['pnl'].sum()
            
            print("\n===== PYRAMIDING EFFECT =====")
            print(f"Trades with Pyramiding: {len(pyramid_trades)} (Win Rate: {pyramid_win_rate:.2f}%)")
            print(f"Trades without Pyramiding: {len(non_pyramid_trades)} (Win Rate: {non_pyramid_win_rate:.2f}%)")
            print(f"P&L with Pyramiding: ${pyramid_profit:.2f}")
            print(f"P&L without Pyramiding: ${non_pyramid_profit:.2f}")
        
        if hour_stats is not None:
            print("\n===== PERFORMANCE BY TRADING HOUR =====")
            best_hours = hour_stats.sort_values('win_rate', ascending=False).head(5)
            print("Best 5 Trading Hours (by win rate):")
            for _, row in best_hours.iterrows():
                print(f"Hour {row['hour']}: Win Rate {row['win_rate']*100:.2f}%, P&L ${row['total_pnl']:.2f}, Trades: {row['num_trades']}")
        
        if day_stats is not None:
            print("\n===== PERFORMANCE BY DAY OF WEEK =====")
            for _, row in day_stats.sort_values('win_rate', ascending=False).iterrows():
                print(f"{row['day']}: Win Rate {row['win_rate']*100:.2f}%, P&L ${row['total_pnl']:.2f}, Trades: {row['num_trades']}")
        
        if 'leverage' in self.trade_df.columns:
            avg_leverage = self.trade_df['leverage'].mean()
            print(f"\nAverage Used Leverage: {avg_leverage:.2f}x")
        
        if 'risk_per_trade' in self.trade_df.columns:
            avg_risk = self.trade_df['risk_per_trade'].mean() * 100
            print(f"Average Risk per Trade: {avg_risk:.2f}%")
        
        self.calculate_correlation_metrics()
        
        stats = {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'monthly_return': monthly_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
        }
        
        return stats
    
    def optimize_parameters(self, param_ranges=None, num_trials=20):
        if param_ranges is None:
            param_ranges = {
                'short_ema': [5, 7, 9, 12, 15],
                'long_ema': [20, 25, 30, 35, 40],
                'rsi_oversold': [25, 30, 35],
                'rsi_overbought': [65, 70, 75],
                'adx_strong_trend': [20, 25, 30],
                'adx_weak_trend': [15, 20, 25],
                'atr_multiplier_sl': [2.0, 2.5, 3.0],
                'atr_multiplier_tp': [4.0, 5.0, 6.0],
                'volume_threshold': [1.2, 1.5, 1.8],
                'trend_threshold': [0.05, 0.1, 0.15],
                'pyramid_min_profit': [0.03, 0.05, 0.07],
                'pyramid_size_multiplier': [0.3, 0.5, 0.7],
                'max_pyramid_entries': [2, 3, 4]
            }
        
        print(f"Starting parameter optimization with {num_trials} trials...")
        
        best_profit = -float('inf')
        best_params = None
        best_stats = None
        
        for trial in range(num_trials):
            trial_params = {}
            for param, values in param_ranges.items():
                trial_params[param] = np.random.choice(values)
            
            for param, value in trial_params.items():
                self.params[param] = value
            
            self.calculate_indicators()
            self.run_backtest()
            stats = self.analyze_results()
            
            profit = stats['final_balance'] - stats['initial_balance']
            
            if profit > best_profit:
                best_profit = profit
                best_params = trial_params.copy()
                best_stats = stats.copy()
                print(f"New best profit: ${best_profit:.2f} (Trial {trial+1}/{num_trials})")
            else:
                print(f"Trial {trial+1}/{num_trials} - Profit: ${profit:.2f} (Best: ${best_profit:.2f})")
        
        self.optimized_params = best_params
        
        print("\n===== OPTIMIZATION RESULTS =====")
        print(f"Best Profit: ${best_profit:.2f}")
        print("Best Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return best_params, best_stats


def main():
    """Main function to execute the strategy"""
    import os
    
    base_dir = r"C:\Diploma\Pet"
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {base_dir}. Please ensure your data file is in this directory.")
        return
    
    data_file = csv_files[0]
    data_path = os.path.join(base_dir, data_file)
    
    print(f"Using data file: {data_path}")
    
    strategy = BalancedAdaptiveStrategy(
        data_path=data_path,
        initial_balance=1000,
        max_leverage=3,
        base_risk_per_trade=0.02,
        min_trades_interval=6
    )
    
    strategy.load_data()
    strategy.calculate_indicators()
    strategy.run_backtest()
    stats = strategy.analyze_results()
    
    strategy.plot_equity_curve()
    strategy.plot_regime_performance()
    
    return strategy

if __name__ == "__main__":
    main()
