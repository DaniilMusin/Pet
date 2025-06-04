import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedAdaptiveStrategy:
    def __init__(self, data_path, 
                 initial_balance=1000, max_leverage=3, 
                 base_risk_per_trade=0.02, 
                 min_trades_interval=6):
        """
        Инициализация улучшенной адаптивной стратегии для торговли BTC фьючерсами
        
        Args:
            data_path: путь к CSV файлу с данными
            initial_balance: начальный баланс в USD
            max_leverage: максимальное плечо
            base_risk_per_trade: базовый риск на сделку (% от баланса)
            min_trades_interval: минимальный интервал между сделками (в часах)
        """
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.base_risk_per_trade = base_risk_per_trade
        self.min_trades_interval = min_trades_interval
        self.slippage_pct = 0.05  # Добавляем проскальзывание 0.05% по умолчанию
        
        # Параметры индикаторов (будут оптимизированы позже)
        self.params = {
            # EMA параметры
            'short_ema': 9,
            'long_ema': 30,
            
            # RSI параметры
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            
            # ADX параметры
            'adx_period': 14,
            'adx_strong_trend': 25,
            'adx_weak_trend': 20,
            
            # Bollinger Bands параметры
            'bb_period': 20,
            'bb_std': 2,
            
            # MACD параметры
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # ATR параметры
            'atr_period': 14,
            'atr_multiplier_sl': 2.5,
            'atr_multiplier_tp': 5.0,
            
            # Параметры для объемного фильтра
            'volume_ma_period': 20,
            'volume_threshold': 1.5,
            
            # Параметры для определения тренда
            'trend_lookback': 20,
            'trend_threshold': 0.1,
            
            # Параметры для пирамидинга
            'pyramid_min_profit': 0.05,
            'pyramid_size_multiplier': 0.5,
            'max_pyramid_entries': 3,
            
            # Параметры для временной фильтрации
            'trading_hours_start': 8,
            'trading_hours_end': 16,
            
            # Параметры для взвешенного подхода
            'adx_min': 15,
            'adx_max': 35,
            
            # Optimal trading hours and days (will be filled by analysis)
            'optimal_trading_hours': None,
            'optimal_trading_days': None
        }
        
        # Будет заполнено в процессе
        self.data = None
        self.trade_history = []
        self.backtest_results = None
        self.trade_df = None
        self.optimized_params = None
        
        # Для отслеживания состояния торгов
        self.max_price_seen = 0
        self.min_price_seen = float('inf')
    
    def load_data(self):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        
        # Загрузка данных из CSV
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.tail(50520)  # Примерно год данных по 15-минутным свечам
        
        # Конвертация дат
        self.data['Open time'] = pd.to_datetime(self.data['Open time'])
        self.data.set_index('Open time', inplace=True)
        
        # Убедимся, что все числовые колонки имеют правильный тип
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
        # Удаление строк с NaN-значениями
        self.data.dropna(subset=numeric_cols, inplace=True)
        
        print(f"Загружено {len(self.data)} свечей")
        return self.data
    
    def calculate_indicators(self):
        """Расчет расширенного набора технических индикаторов"""
        print("Расчет индикаторов...")
        
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
        
        # --- ATR для динамических стоп-лоссов ---
        self.data['ATR'] = self._calculate_atr(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            self.params['atr_period']
        )
        
        # Добавляем скользящее среднее ATR для расчета волатильности
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
        
        # --- Объемный фильтр ---
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=self.params['volume_ma_period']).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # --- Определение тренда по ценовому движению ---
        lookback = self.params['trend_lookback']
        self.data['Price_Change_Pct'] = (self.data['Close'] - self.data['Close'].shift(lookback)) / self.data['Close'].shift(lookback)
        
        # --- Расчет дивергенций RSI ---
        # Смотрим на минимумы/максимумы цены и RSI
        self.data['Price_Min'] = self.data['Close'].rolling(5, center=True).min() == self.data['Close']
        self.data['Price_Max'] = self.data['Close'].rolling(5, center=True).max() == self.data['Close']
        self.data['RSI_Min'] = self.data['RSI'].rolling(5, center=True).min() == self.data['RSI']
        self.data['RSI_Max'] = self.data['RSI'].rolling(5, center=True).max() == self.data['RSI']
        
        # Дивергенции (позитивная: цена падает, RSI растет; негативная: цена растет, RSI падает)
        self.data['Bullish_Divergence'] = False  # Инициализация
        self.data['Bearish_Divergence'] = False  # Инициализация
        
        # Находим локальные минимумы и максимумы
        price_mins = self.data[self.data['Price_Min']].index
        price_maxs = self.data[self.data['Price_Max']].index
        rsi_mins = self.data[self.data['RSI_Min']].index
        rsi_maxs = self.data[self.data['RSI_Max']].index
        
        # Для простоты ищем точки, где минимумы/максимумы близки по времени
        for i in range(1, len(price_mins)):
            for j in range(1, len(rsi_mins)):
                # Если временной промежуток небольшой (например, до 3 свечей)
                if 0 <= (rsi_mins[j] - price_mins[i]).total_seconds() / 3600 <= 3:
                    # Проверяем дивергенцию: цена делает более низкий минимум, а RSI - более высокий
                    price_change = self.data.loc[price_mins[i], 'Close'] - self.data.loc[price_mins[i-1], 'Close']
                    rsi_change = self.data.loc[rsi_mins[j], 'RSI'] - self.data.loc[rsi_mins[j-1], 'RSI']
                    if price_change < 0 and rsi_change > 0:
                        self.data.loc[max(price_mins[i], rsi_mins[j]), 'Bullish_Divergence'] = True
        
        for i in range(1, len(price_maxs)):
            for j in range(1, len(rsi_maxs)):
                # Если временной промежуток небольшой
                if 0 <= (rsi_maxs[j] - price_maxs[i]).total_seconds() / 3600 <= 3:
                    # Проверяем дивергенцию: цена делает более высокий максимум, а RSI - более низкий
                    price_change = self.data.loc[price_maxs[i], 'Close'] - self.data.loc[price_maxs[i-1], 'Close']
                    rsi_change = self.data.loc[rsi_maxs[j], 'RSI'] - self.data.loc[rsi_maxs[j-1], 'RSI']
                    if price_change > 0 and rsi_change < 0:
                        self.data.loc[max(price_maxs[i], rsi_maxs[j]), 'Bearish_Divergence'] = True
        
        # Заполняем пропуски
        self.data['Bullish_Divergence'].fillna(False, inplace=True)
        self.data['Bearish_Divergence'].fillna(False, inplace=True)
        
        # --- Улучшенное определение режима рынка ---
        # 1. Расширенное определение тренда (ADX + ценовое движение)
        threshold = self.params['trend_threshold']
        self.data['Strong_Trend'] = (self.data['ADX'] > self.params['adx_strong_trend']) & \
                                   (self.data['Price_Change_Pct'].abs() > threshold)
        
        self.data['Weak_Trend'] = (self.data['ADX'] < self.params['adx_weak_trend']) & \
                                  (self.data['Price_Change_Pct'].abs() < threshold/2)
        
        # 2. Определение направления тренда
        self.data['Bullish_Trend'] = self.data['Strong_Trend'] & (self.data['Price_Change_Pct'] > 0) & \
                                     (self.data['Plus_DI'] > self.data['Minus_DI'])
        
        self.data['Bearish_Trend'] = self.data['Strong_Trend'] & (self.data['Price_Change_Pct'] < 0) & \
                                     (self.data['Plus_DI'] < self.data['Minus_DI'])
        
        # 3. Коэффициент "трендовости" (для взвешенного подхода)
        self.data['Trend_Weight'] = np.minimum(1.0, np.maximum(0, 
                                              (self.data['ADX'] - self.params['adx_min']) / 
                                              (self.params['adx_max'] - self.params['adx_min'])))
        
        self.data['Range_Weight'] = 1.0 - self.data['Trend_Weight']
        
        # Добавляем временной фильтр (часы активной торговли)
        self.data['Hour'] = self.data.index.hour
        self.data['Active_Hours'] = (self.data['Hour'] >= self.params['trading_hours_start']) & \
                                    (self.data['Hour'] <= self.params['trading_hours_end'])
        
        # --- MACD-сигналы ---
        self.data['MACD_Bullish_Cross'] = (self.data['MACD'] > self.data['MACD_Signal']) & \
                                          (self.data['MACD'].shift(1) <= self.data['MACD_Signal'].shift(1))
        self.data['MACD_Bearish_Cross'] = (self.data['MACD'] < self.data['MACD_Signal']) & \
                                          (self.data['MACD'].shift(1) >= self.data['MACD_Signal'].shift(1))
        
        # --- НОВЫЕ ИНДИКАТОРЫ И УЛУЧШЕНИЯ ---

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
        
        # Day of week for time-based analysis
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
        self.data['Bullish_Engulfing'] = (self.data['Open'] < self.data['Close'].shift(1)) & \
                                        (self.data['Close'] > self.data['Open'].shift(1)) & \
                                        (self.data['Close'] > self.data['Open']) & \
                                        (self.data['Open'].shift(1) > self.data['Close'].shift(1))
        
        self.data['Bearish_Engulfing'] = (self.data['Open'] > self.data['Close'].shift(1)) & \
                                        (self.data['Close'] < self.data['Open'].shift(1)) & \
                                        (self.data['Close'] < self.data['Open']) & \
                                        (self.data['Open'].shift(1) < self.data['Close'].shift(1))
        
        # Volatility ratio for dynamic position sizing
        self.data['Volatility_Ratio'] = self.data['ATR'] / self.data['ATR_MA'].replace(0, 1e-10)
        
        # Удаляем строки с NaN после расчета индикаторов
        self.data.dropna(inplace=True)
        
        print("Индикаторы рассчитаны")
        return self.data
    
    def _calculate_ema(self, series, period):
        """Расчет EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series, period):
        """Расчет RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Рассчитываем RS (relative strength) и предотвращаем деление на 0
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        
        # Рассчитываем RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, high, low, close, period):
        """Расчет ATR"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Используем простое скользящее среднее для ATR
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_adx(self, high, low, close, period):
        """Расчет ADX, +DI, -DI"""
        # Рассчитываем истинный диапазон
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Рассчитываем +DM и -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0, index=high.index, dtype=float)
        minus_dm = pd.Series(0, index=high.index, dtype=float)
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Преобразуем в серии pandas
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Сглаживаем +DM и -DM
        smooth_plus_dm = plus_dm.rolling(window=period).mean()
        smooth_minus_dm = minus_dm.rolling(window=period).mean()
        
        # Рассчитываем +DI и -DI
        plus_di = 100 * (smooth_plus_dm / atr.replace(0, 1e-10))
        minus_di = 100 * (smooth_minus_dm / atr.replace(0, 1e-10))
        
        # Рассчитываем DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
        
        # Сглаживаем DX для получения ADX
        adx = dx.rolling(window=period).mean()
        
        return {
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di
        }
    
    def _calculate_macd(self, series, fast_period, slow_period, signal_period):
        """Расчет MACD"""
        # Рассчитываем EMA
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        
        # Рассчитываем MACD
        macd = fast_ema - slow_ema
        
        # Рассчитываем сигнальную линию
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Рассчитываем гистограмму
        macd_hist = macd - macd_signal
        
        return {
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Hist': macd_hist
        }
    
    def get_trading_signals(self, current, previous, regime_type):
        """
        Generate trading signals based on the current market regime
        
        Args:
            current: Current candle data
            previous: Previous candle data
            regime_type: 'trend' or 'range'
            
        Returns:
            dict with long_signals, short_signals, and their weights
        """
        long_signals = []
        short_signals = []
        
        # Common signals for both regimes
        volume_multiplier = min(2.0, current['Volume_Ratio'] / self.params['volume_threshold']) if current['Volume_Ratio'] > self.params['volume_threshold'] else 1.0
        
        if regime_type == 'trend':
            # TRENDING MARKET STRATEGY
            
            # Long signals in trending markets
            if current['Trend_Weight'] > 0.6:
                # EMA crossover
                if ((previous[f"EMA_{self.params['short_ema']}"] < previous[f"EMA_{self.params['long_ema']}"]) and 
                    (current[f"EMA_{self.params['short_ema']}"] >= current[f"EMA_{self.params['long_ema']}"])):
                    long_signals.append(('EMA Crossover', current['Trend_Weight'] * 1.2))
                
                # MACD crossover with histogram confirmation
                if (current['MACD_Bullish_Cross'] and 
                    current['MACD_Hist'] > 0 and 
                    current['MACD_Hist'] > previous['MACD_Hist']):
                    long_signals.append(('MACD Bullish Cross', current['Trend_Weight'] * 1.3))
                
                # Strong trend confirmation
                if (current['Bullish_Trend'] and not previous['Bullish_Trend'] and 
                    current['Plus_DI'] > current['Minus_DI'] * 1.2):
                    long_signals.append(('Strong Bullish Trend', current['Trend_Weight'] * 1.5))
                    
                # Higher timeframe alignment for stronger signals
                if current['Higher_TF_Bullish']:
                    for i in range(len(long_signals)):
                        signal, weight = long_signals[i]
                        long_signals[i] = (signal, weight * 1.3)  # 30% boost when aligned with higher TF
            
            # Short signals in trending markets
            if current['Trend_Weight'] > 0.6:
                # EMA crossover for shorts
                if ((previous[f"EMA_{self.params['short_ema']}"] > previous[f"EMA_{self.params['long_ema']}"]) and 
                    (current[f"EMA_{self.params['short_ema']}"] <= current[f"EMA_{self.params['long_ema']}"])):
                    short_signals.append(('EMA Crossover', current['Trend_Weight'] * 1.2))
                
                # MACD crossover with histogram confirmation for shorts
                if (current['MACD_Bearish_Cross'] and 
                    current['MACD_Hist'] < 0 and 
                    current['MACD_Hist'] < previous['MACD_Hist']):
                    short_signals.append(('MACD Bearish Cross', current['Trend_Weight'] * 1.3))
                
                # Strong bearish trend confirmation
                if (current['Bearish_Trend'] and not previous['Bearish_Trend'] and 
                    current['Minus_DI'] > current['Plus_DI'] * 1.2):
                    short_signals.append(('Strong Bearish Trend', current['Trend_Weight'] * 1.5))
                    
                # Higher timeframe alignment for stronger signals
                if current['Higher_TF_Bearish']:
                    for i in range(len(short_signals)):
                        signal, weight = short_signals[i]
                        short_signals[i] = (signal, weight * 1.3)  # 30% boost when aligned with higher TF
                        
        else:
            # RANGING MARKET STRATEGY
            
            # Long signals in ranging markets
            if current['Range_Weight'] > 0.6:
                # RSI oversold with Bollinger Band confirmation
                if (current['RSI'] < self.params['rsi_oversold'] and 
                    current['Close'] < current['BB_Lower'] * 1.01):
                    long_signals.append(('RSI Oversold + BB Lower', current['Range_Weight'] * 1.3))
                
                # Bullish divergence with stronger signal
                if current['Bullish_Divergence'] and current['RSI'] < 40:
                    long_signals.append(('Strong Bullish Divergence', current['Range_Weight'] * 1.6))
                
                # Support bounce with confirmation
                if (current['Close'] > current['Open'] and 
                    previous['Close'] < previous['Open'] and
                    current['Low'] > previous['Low'] * 0.998 and  # Support level holding
                    current['Volume'] > previous['Volume'] * 1.2):  # Volume confirmation
                    long_signals.append(('Support Bounce', current['Range_Weight'] * 1.2))
                    
            # Short signals in ranging markets
            if current['Range_Weight'] > 0.6:
                # RSI overbought with Bollinger Band confirmation
                if (current['RSI'] > self.params['rsi_overbought'] and 
                    current['Close'] > current['BB_Upper'] * 0.99):
                    short_signals.append(('RSI Overbought + BB Upper', current['Range_Weight'] * 1.3))
                
                # Bearish divergence with stronger signal
                if current['Bearish_Divergence'] and current['RSI'] > 60:
                    short_signals.append(('Strong Bearish Divergence', current['Range_Weight'] * 1.6))
                
                # Resistance rejection with confirmation
                if (current['Close'] < current['Open'] and 
                    previous['Close'] > previous['Open'] and
                    current['High'] < previous['High'] * 1.002 and  # Resistance level holding
                    current['Volume'] > previous['Volume'] * 1.2):  # Volume confirmation
                    short_signals.append(('Resistance Rejection', current['Range_Weight'] * 1.2))
        
        # Apply volume multiplier to all signals
        long_signals = [(signal, weight * volume_multiplier) for signal, weight in long_signals]
        short_signals = [(signal, weight * volume_multiplier) for signal, weight in short_signals]
        
        # Calculate average weights
        long_weight = sum(weight for _, weight in long_signals) / len(long_signals) if long_signals else 0
        short_weight = sum(weight for _, weight in short_signals) / len(short_signals) if short_signals else 0
        
        return {
            'long_signals': long_signals,
            'short_signals': short_signals,
            'long_weight': long_weight,
            'short_weight': short_weight
        }

    def apply_advanced_filtering(self, current, signals):
        """
        Apply advanced filtering to trading signals
        """
        long_weight = signals['long_weight']
        short_weight = signals['short_weight']
        
        # Market volatility filter
        if 'ATR_MA' in current and current['ATR_MA'] > 0:
            volatility_ratio = current['ATR'] / current['ATR_MA']
            if volatility_ratio > 1.5:
                # In high volatility, reduce all signal weights
                long_weight *= 0.7
                short_weight *= 0.7
        
        # Time-based filter (avoid trading during potentially low-volume periods)
        hour = current.name.hour
        if hour >= 0 and hour < 6:  # Between midnight and 6 AM (using your local timezone)
            long_weight *= 0.8
            short_weight *= 0.8
        
        # Price action confirmation
        if current['Close'] > current['Open']:  # Bullish candle
            long_weight *= 1.1
            short_weight *= 0.9
        else:  # Bearish candle
            long_weight *= 0.9
            short_weight *= 1.1
            
        # Recent performance filter - favor what's been working
        if hasattr(self, 'trade_df') and self.trade_df is not None and len(self.trade_df) >= 5:
            recent_trades = self.trade_df.tail(5)
            long_trades = recent_trades[recent_trades['position'] == 'LONG']
            short_trades = recent_trades[recent_trades['position'] == 'SHORT']
            
            if len(long_trades) > 0:
                long_win_rate = sum(1 for p in long_trades['pnl'] if p > 0) / len(long_trades)
                if long_win_rate > 0.6:
                    long_weight *= 1.2  # Boost long signals if recent long trades were successful
                elif long_win_rate < 0.4:
                    long_weight *= 0.8  # Reduce long signals if recent long trades were unsuccessful
                    
            if len(short_trades) > 0:
                short_win_rate = sum(1 for p in short_trades['pnl'] if p > 0) / len(short_trades)
                if short_win_rate > 0.6:
                    short_weight *= 1.2  # Boost short signals if recent short trades were successful
                elif short_win_rate < 0.4:
                    short_weight *= 0.8  # Reduce short signals if recent short trades were unsuccessful
        
        return {
            'long_weight': long_weight,
            'short_weight': short_weight
        }

    def calculate_dynamic_exit_levels(self, position_type, entry_price, current_candle, trade_age_hours=0):
        """
        Calculate dynamic take-profit and stop-loss levels based on market conditions
        
        Args:
            position_type: 'LONG' or 'SHORT'
            entry_price: Position entry price
            current_candle: Current price data
            trade_age_hours: How long the trade has been open in hours
            
        Returns:
            dict with stop_loss and take_profit prices
        """
        # Base ATR value for volatility-based exits
        atr_value = current_candle['ATR']
        
        # Base multipliers from parameters
        base_sl_multiplier = self.params['atr_multiplier_sl']
        base_tp_multiplier = self.params['atr_multiplier_tp']
        
        # Adjust multipliers based on trend strength
        if current_candle['ADX'] > 40:  # Strong trend
            # In strong trends, we want wider stops and targets
            sl_multiplier = base_sl_multiplier * 1.2
            tp_multiplier = base_tp_multiplier * 1.3
        elif current_candle['ADX'] < 20:  # Weak trend / range
            # In ranges, tighter targets can be better
            sl_multiplier = base_sl_multiplier * 0.9
            tp_multiplier = base_tp_multiplier * 0.8
        else:
            sl_multiplier = base_sl_multiplier
            tp_multiplier = base_tp_multiplier
        
        # Adjust stop-loss based on trade age (trailing stop logic)
        if trade_age_hours > 4:
            # Start tightening stop after 4 hours
            age_factor = min(3.0, 1.0 + (trade_age_hours - 4) / 20)  # Caps at 3x tighter after 24 hours
            sl_multiplier = sl_multiplier / age_factor
        
        # Calculate base levels
        if position_type == 'LONG':
            stop_loss = entry_price * (1 - atr_value * sl_multiplier / entry_price)
            take_profit = entry_price * (1 + atr_value * tp_multiplier / entry_price)
        else:  # SHORT
            stop_loss = entry_price * (1 + atr_value * sl_multiplier / entry_price)
            take_profit = entry_price * (1 - atr_value * tp_multiplier / entry_price)
        
        # Advanced adjustment: Use Bollinger Bands to influence take-profit in ranging markets
        if current_candle['Range_Weight'] > 0.7:
            if position_type == 'LONG' and current_candle['BB_Upper'] < take_profit:
                # In ranges, consider taking profit at upper Bollinger Band
                take_profit = current_candle['BB_Upper']
            elif position_type == 'SHORT' and current_candle['BB_Lower'] > take_profit:
                # In ranges, consider taking profit at lower Bollinger Band
                take_profit = current_candle['BB_Lower']
        
        # Risk-reward sanity check
        if position_type == 'LONG':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Ensure minimum 2:1 reward-to-risk ratio
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
        """
        Calculate trailing stop-loss based on trade performance
        
        Args:
            position_type: 'LONG' or 'SHORT'
            entry_price: Position entry price
            current_price: Current market price
            max_price: Maximum price seen during the trade (for longs)
            min_price: Minimum price seen during the trade (for shorts)
            unrealized_pnl_pct: Current unrealized profit/loss percentage
            
        Returns:
            New stop-loss price or None if no adjustment needed
        """
        # Only apply trailing stop if in profit
        if unrealized_pnl_pct <= 0:
            return None
        
        if position_type == 'LONG':
            # For long positions
            if unrealized_pnl_pct >= 0.03:  # 3% or more profit
                # Trail by different amounts based on profit level
                if unrealized_pnl_pct >= 0.10:  # 10%+ profit
                    # Trail by 40% of the profit
                    trail_pct = unrealized_pnl_pct * 0.4
                    new_stop = entry_price * (1 + trail_pct)
                elif unrealized_pnl_pct >= 0.05:  # 5-10% profit
                    # Trail by 30% of the profit
                    trail_pct = unrealized_pnl_pct * 0.3
                    new_stop = entry_price * (1 + trail_pct)
                else:  # 3-5% profit
                    # Trail by 20% of the profit
                    trail_pct = unrealized_pnl_pct * 0.2
                    new_stop = entry_price * (1 + trail_pct)
                    
                return new_stop
        else:  # SHORT
            # For short positions
            if unrealized_pnl_pct >= 0.03:  # 3% or more profit
                # Trail by different amounts based on profit level
                if unrealized_pnl_pct >= 0.10:  # 10%+ profit
                    # Trail by 40% of the profit
                    trail_pct = unrealized_pnl_pct * 0.4
                    new_stop = entry_price * (1 - trail_pct)
                elif unrealized_pnl_pct >= 0.05:  # 5-10% profit
                    # Trail by 30% of the profit
                    trail_pct = unrealized_pnl_pct * 0.3
                    new_stop = entry_price * (1 - trail_pct)
                else:  # 3-5% profit
                    # Trail by 20% of the profit
                    trail_pct = unrealized_pnl_pct * 0.2
                    new_stop = entry_price * (1 - trail_pct)
                    
                return new_stop
        
        return None  # No adjustment needed

    def calculate_optimal_leverage(self, current_candle, trade_direction, max_allowed_leverage=3):
        """
        Calculate optimal leverage based on market conditions
        
        Args:
            current_candle: Current market data
            trade_direction: 'LONG' or 'SHORT'
            max_allowed_leverage: Maximum allowed leverage
            
        Returns:
            Optimal leverage value
        """
        # Base leverage
        base_leverage = 2
        
        # Adjust leverage based on volatility
        if 'ATR_MA' in current_candle and current_candle['ATR_MA'] > 0:
            vol_ratio = current_candle['ATR'] / current_candle['ATR_MA']
            
            if vol_ratio > 1.5:  # High volatility
                vol_adjustment = 0.7  # Reduce leverage
            elif vol_ratio < 0.8:  # Low volatility
                vol_adjustment = 1.3  # Increase leverage
            else:
                vol_adjustment = 1.0  # No change
        else:
            vol_adjustment = 1.0
        
        # Adjust leverage based on trend strength
        if current_candle['ADX'] > 35:  # Strong trend
            if (trade_direction == 'LONG' and current_candle['Plus_DI'] > current_candle['Minus_DI']) or \
               (trade_direction == 'SHORT' and current_candle['Minus_DI'] > current_candle['Plus_DI']):
                # Trade is in the direction of the trend
                trend_adjustment = 1.2  # Increase leverage
            else:
                # Trade is against the trend
                trend_adjustment = 0.7  # Reduce leverage
        else:
            trend_adjustment = 1.0  # No change
        
        # Adjust leverage based on market regime
        if trade_direction == 'LONG' and current_candle['Range_Weight'] > 0.7:
            # Long in a range - be more conservative
            regime_adjustment = 0.9
        elif trade_direction == 'SHORT' and current_candle['Trend_Weight'] > 0.7:
            # Short in a strong trend - can be more aggressive
            regime_adjustment = 1.1
        else:
            regime_adjustment = 1.0
        
        # Calculate final leverage
        optimal_leverage = base_leverage * vol_adjustment * trend_adjustment * regime_adjustment
        
        # Cap at maximum allowed leverage
        return min(max_allowed_leverage, optimal_leverage)

    def adaptive_position_sizing(self, balance, risk_per_trade, entry_price, stop_loss_price, optimal_leverage):
        """
        Calculate position size with adaptive risk management
        
        Args:
            balance: Account balance
            risk_per_trade: Risk percentage per trade
            entry_price: Position entry price
            stop_loss_price: Stop-loss price
            optimal_leverage: Calculated optimal leverage
            
        Returns:
            Position size in USD
        """
        # Calculate risk amount in USD
        risk_amount = balance * risk_per_trade
        
        # Calculate percentage risk from entry to stop
        price_risk_pct = abs(entry_price - stop_loss_price) / entry_price
        
        # Apply leverage
        leveraged_risk_pct = price_risk_pct * optimal_leverage
        
        # Calculate position size
        if leveraged_risk_pct > 0:
            position_size = risk_amount / leveraged_risk_pct
        else:
            position_size = 0
        
        # Cap position size at balance * leverage
        max_position = balance * optimal_leverage
        position_size = min(position_size, max_position)
        
        # For small accounts, ensure minimum effective position size
        if balance <= 1000 and position_size < 100:
            # Minimum $100 position for trades to be meaningful
            position_size = max(100, position_size)
        
        return position_size
    
    def is_optimal_trading_time(self, timestamp):
        """
        Check if current time is in the optimal trading window
        """
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        day_name = day_map[day_of_week]
        
        hour_optimal = True
        day_optimal = True
        
        if hasattr(self, 'params') and 'optimal_trading_hours' in self.params and self.params['optimal_trading_hours'] is not None:
            hour_optimal = hour in self.params['optimal_trading_hours']
        
        if hasattr(self, 'params') and 'optimal_trading_days' in self.params and self.params['optimal_trading_days'] is not None:
            day_optimal = day_name in self.params['optimal_trading_days']
        
        return hour_optimal and day_optimal
    
    def calculate_kelly_criterion(self, win_rate, avg_win_pct, avg_loss_pct):
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Win rate as decimal (e.g., 0.6 for 60%)
            avg_win_pct: Average win percentage (e.g., 0.05 for 5%)
            avg_loss_pct: Average loss percentage as positive number (e.g., 0.03 for 3%)
            
        Returns:
            Kelly percentage as decimal
        """
        # Classic Kelly formula: K% = W - [(1-W)/R]
        # Where:
        # W = Win rate
        # R = Win/Loss ratio (average win / average loss)
        
        if avg_loss_pct == 0:
            avg_loss_pct = 0.001  # Prevent division by zero
        
        win_loss_ratio = avg_win_pct / avg_loss_pct
        
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Cap Kelly at 25% to be conservative
        kelly_pct = min(0.25, max(0, kelly_pct))
        
        return kelly_pct
    
    def dynamically_adjust_risk_parameters(self):
        """
        Dynamically adjust risk parameters based on recent performance
        """
        if not hasattr(self, 'trade_df') or len(self.trade_df) < 20:
            # Not enough trades to make adjustments
            return
        
        # Get last 20 trades
        recent_trades = self.trade_df.tail(20)
        
        # Calculate recent performance metrics
        win_rate = sum(1 for p in recent_trades['pnl'] if p > 0) / len(recent_trades)
        profit_sum = sum(p for p in recent_trades['pnl'] if p > 0)
        loss_sum = abs(sum(p for p in recent_trades['pnl'] if p <= 0))
        
        win_trades = [p for p in recent_trades['pnl'] if p > 0]
        loss_trades = [abs(p) for p in recent_trades['pnl'] if p <= 0]
        
        avg_win = sum(win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(loss_trades) / len(loss_trades) if loss_trades else 0
        
        # Calculate profit factor
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
        
        # Adjust parameters based on performance
        if profit_factor > 1.5 and win_rate > 0.5:
            # Strategy is performing well, increase risk slightly
            self.base_risk_per_trade = min(0.03, self.base_risk_per_trade * 1.1)
        elif profit_factor < 1.0 or win_rate < 0.4:
            # Strategy is performing poorly, reduce risk
            self.base_risk_per_trade = max(0.01, self.base_risk_per_trade * 0.9)
        
        # Adjust take-profit and stop-loss multipliers
        if avg_win > 0 and avg_loss > 0:
            current_rr_ratio = avg_win / avg_loss
            
            if current_rr_ratio < 1.5:
                # Reward-to-risk ratio is too low, adjust for more favorable ratio
                self.params['atr_multiplier_tp'] = min(7.0, self.params['atr_multiplier_tp'] * 1.05)
            elif current_rr_ratio > 3.0 and win_rate < 0.4:
                # Reward-to-risk ratio is too high with low win rate, tighten targets
                self.params['atr_multiplier_tp'] = max(3.0, self.params['atr_multiplier_tp'] * 0.95)
    
    def analyze_hour_performance(self):
        """
        Analyze performance by hour of day to identify optimal trading hours
        """
        if not hasattr(self, 'trade_df') or len(self.trade_df) == 0:
            return None
        
        # Extract hour from entry_date
        self.trade_df['entry_hour'] = pd.to_datetime(self.trade_df['entry_date']).dt.hour
        
        # Group by hour and calculate performance metrics
        hour_stats = self.trade_df.groupby('entry_hour').agg({
            'pnl': ['count', 'mean', 'sum'],
            'position': 'count'
        }).reset_index()
        
        hour_stats.columns = ['hour', 'num_trades', 'avg_pnl', 'total_pnl', 'position_count']
        
        # Calculate win rate by hour
        win_rates = []
        for hour in hour_stats['hour'].unique():
            hour_trades = self.trade_df[self.trade_df['entry_hour'] == hour]
            wins = sum(1 for pnl in hour_trades['pnl'] if pnl > 0)
            total = len(hour_trades)
            win_rate = wins / total if total > 0 else 0
            win_rates.append(win_rate)
        
        hour_stats['win_rate'] = win_rates
        
        # Sort by win rate descending
        hour_stats = hour_stats.sort_values('win_rate', ascending=False)
        
        # Identify optimal trading hours (win rate > 50% and at least 5 trades)
        optimal_hours = hour_stats[(hour_stats['win_rate'] > 0.5) & (hour_stats['num_trades'] >= 5)]['hour'].tolist()
        
        # Update trading_hours in parameters
        if optimal_hours:
            self.params['optimal_trading_hours'] = optimal_hours
        
        return hour_stats

    def analyze_day_performance(self):
        """
        Analyze performance by day of week to identify optimal trading days
        """
        if not hasattr(self, 'trade_df') or len(self.trade_df) == 0:
            return None
        
        # Extract day of week from entry_date
        self.trade_df['entry_day'] = pd.to_datetime(self.trade_df['entry_date']).dt.dayofweek
        
        # Map day number to day name
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        self.trade_df['day_name'] = self.trade_df['entry_day'].map(day_map)
        
        # Group by day and calculate performance metrics
        day_stats = self.trade_df.groupby('day_name').agg({
            'pnl': ['count', 'mean', 'sum'],
            'position': 'count'
        }).reset_index()
        
        day_stats.columns = ['day', 'num_trades', 'avg_pnl', 'total_pnl', 'position_count']
        
        # Calculate win rate by day
        win_rates = []
        for day in day_stats['day'].unique():
            day_trades = self.trade_df[self.trade_df['day_name'] == day]
            wins = sum(1 for pnl in day_trades['pnl'] if pnl > 0)
            total = len(day_trades)
            win_rate = wins / total if total > 0 else 0
            win_rates.append(win_rate)
        
        day_stats['win_rate'] = win_rates
        
        # Sort by win rate descending
        day_stats = day_stats.sort_values('win_rate', ascending=False)
        
        # Identify optimal trading days (win rate > 50% and at least 5 trades)
        optimal_days = day_stats[(day_stats['win_rate'] > 0.5) & (day_stats['num_trades'] >= 5)]['day'].tolist()
        
        # Update optimal_trading_days in parameters
        if optimal_days:
            self.params['optimal_trading_days'] = optimal_days
        
        return day_stats

    def run_backtest(self):
        """Запуск улучшенного бэктеста стратегии"""
        print("Запуск бэктеста...")
        
        # Инициализируем переменные
        balance = self.initial_balance
        position = 0  # 0 - нет позиции, 1 - лонг, -1 - шорт
        entry_price = 0.0
        position_size = 0.0
        entry_date = None
        last_trade_date = None  # Для отслеживания минимального интервала между сделками
        pyramid_entries = 0  # Счетчик дополнительных входов для пирамидинга
        stop_loss_price = 0.0
        take_profit_price = 0.0
        
        # Для отслеживания максимальной/минимальной цены во время торговли
        max_trade_price = 0.0
        min_trade_price = float('inf')
        
        # Создаем DataFrame для результатов и историю сделок
        results = []
        self.trade_history = []
        
        # Цикл по всем свечам
        for i in range(1, len(self.data)):
            current = self.data.iloc[i]
            previous = self.data.iloc[i-1]
            
            # Сохраняем текущий баланс
            current_balance = balance
            current_equity = balance
            
            # ------ ОБРАБОТКА СУЩЕСТВУЮЩЕЙ ПОЗИЦИИ -------
            if position != 0:  # Если есть открытая позиция
                if entry_price <= 0:
                    # Исправляем состояние - закрываем неправильную позицию
                    position = 0
                    entry_price = 0.0
                    position_size = 0.0
                    entry_date = None
                    pyramid_entries = 0
                    max_trade_price = 0.0
                    min_trade_price = float('inf')
                    continue
                
                # Обновляем максимальную/минимальную цену
                max_trade_price = max(max_trade_price, current['High'])
                min_trade_price = min(min_trade_price, current['Low'])
                
                # Расчет текущего нереализованного P/L и возраста сделки
                trade_age_hours = (current.name - entry_date).total_seconds() / 3600
                
                if position == 1:  # Лонг
                    unrealized_pnl = ((current['Close'] / entry_price) - 1) * position_size * self.max_leverage
                    unrealized_pnl_pct = (current['Close'] / entry_price) - 1
                    current_equity = balance + unrealized_pnl
                    
                    # Проверка на обновление трейлинг-стопа
                    new_stop = self.apply_trailing_stop('LONG', entry_price, current['Close'], max_trade_price, min_trade_price, unrealized_pnl_pct)
                    if new_stop is not None and new_stop > stop_loss_price:
                        stop_loss_price = new_stop  # Обновляем стоп-лосс
                    
                    # Проверка стоп-лосса
                    if current['Low'] <= stop_loss_price:
                        # Сработал стоп-лосс для лонга
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
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    # Проверка тейк-профита
                    if current['High'] >= take_profit_price:
                        # Сработал тейк-профит для лонга
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
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    # --- ПИРАМИДИНГ ДЛЯ ЛОНГА ---
                    # Если в прибыльной позиции и сильный тренд, добавляем к позиции
                    if (pyramid_entries < self.params['max_pyramid_entries'] and 
                        current['Bullish_Trend'] and 
                        current['ADX'] > 40 and 
                        current['Close'] > entry_price * (1 + self.params['pyramid_min_profit']) and
                        current['MACD_Bullish_Cross']):
                        
                        # Рассчитываем размер дополнительной позиции (меньше основной)
                        additional_size = position_size * self.params['pyramid_size_multiplier']
                        
                        # Обновляем средневзвешенную цену входа и размер позиции
                        old_value = entry_price * position_size
                        new_value = current['Close'] * additional_size
                        position_size += additional_size
                        entry_price = (old_value + new_value) / position_size
                        
                        # Обновляем стоп-лосс и тейк-профит для новой средней цены
                        exit_levels = self.calculate_dynamic_exit_levels('LONG', entry_price, current, trade_age_hours)
                        stop_loss_price = exit_levels['stop_loss']
                        take_profit_price = exit_levels['take_profit']
                        
                        pyramid_entries += 1
                        continue
                    
                    # Проверка сигналов на выход
                    exit_signals = []
                    
                    # Определим текущий режим рынка
                    current_regime = 'trend' if current['Trend_Weight'] > 0.5 else 'range'
                    
                    # Трендовые сигналы выхода из лонга
                    if current_regime == 'trend':
                        # Сильный трендовый сигнал - выход по кроссоверу EMA или MACD
                        if ((previous[f'EMA_{self.params["short_ema"]}'] >= previous[f'EMA_{self.params["long_ema"]}']) and 
                            (current[f'EMA_{self.params["short_ema"]}'] < current[f'EMA_{self.params["long_ema"]}'])):
                            exit_signals.append('EMA Crossover')
                        
                        if current['MACD_Bearish_Cross']:
                            exit_signals.append('MACD Crossover')
                            
                        if current['Bearish_Trend'] and not previous['Bearish_Trend']:
                            exit_signals.append('Trend Change')
                            
                        # Новые улучшенные сигналы
                        if current['Higher_TF_Bearish'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Higher TF Trend Change')
                    
                    # Контртрендовые сигналы выхода из лонга
                    else:  # current_regime == 'range'
                        # Сильный флетовый сигнал - выход по RSI, Боллинджеру или дивергенции
                        if current['RSI'] > self.params['rsi_overbought']:
                            exit_signals.append('RSI Overbought')
                            
                        if current['Close'] > current['BB_Upper']:
                            exit_signals.append('Upper Bollinger')
                            
                        if current['Bearish_Divergence']:
                            exit_signals.append('Bearish Divergence')
                            
                        # Новые улучшенные сигналы
                        if current['Bearish_Engulfing'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Bearish Engulfing')
                            
                    # Общие сигналы выхода вне зависимости от режима
                    
                    # Выход на основе волатильности
                    if current['Volatility_Ratio'] > 2.0 and unrealized_pnl_pct > 0.03:
                        exit_signals.append('Volatility Spike')
                        
                    # Если прибыль достигла определенного уровня, но рост замедлился
                    if (unrealized_pnl_pct > 0.08 and 
                        previous['Close'] > current['Close'] and 
                        previous['MACD'] > current['MACD']):
                        exit_signals.append('Profit Protection')
                    
                    # Если есть сигнал на выход и прошло достаточно времени с момента входа
                    if exit_signals and trade_age_hours > 4:
                        # Закрываем лонг
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
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                
                elif position == -1:  # Шорт
                    unrealized_pnl = (1 - (current['Close'] / entry_price)) * position_size * self.max_leverage
                    unrealized_pnl_pct = 1 - (current['Close'] / entry_price)
                    current_equity = balance + unrealized_pnl
                    
                    # Проверка на обновление трейлинг-стопа
                    new_stop = self.apply_trailing_stop('SHORT', entry_price, current['Close'], max_trade_price, min_trade_price, unrealized_pnl_pct)
                    if new_stop is not None and new_stop < stop_loss_price:
                        stop_loss_price = new_stop  # Обновляем стоп-лосс
                    
                    # Проверка стоп-лосса
                    if current['High'] >= stop_loss_price:
                        # Сработал стоп-лосс для шорта
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
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    # Проверка тейк-профита
                    if current['Low'] <= take_profit_price:
                        # Сработал тейк-профит для шорта
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
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
                    
                    # --- ПИРАМИДИНГ ДЛЯ ШОРТА ---
                    # Если в прибыльной позиции и сильный тренд, добавляем к позиции
                    if (pyramid_entries < self.params['max_pyramid_entries'] and 
                        current['Bearish_Trend'] and 
                        current['ADX'] > 40 and 
                        current['Close'] < entry_price * (1 - self.params['pyramid_min_profit']) and
                        current['MACD_Bearish_Cross']):
                        
                        # Рассчитываем размер дополнительной позиции (меньше основной)
                        additional_size = position_size * self.params['pyramid_size_multiplier']
                        
                        # Обновляем средневзвешенную цену входа и размер позиции
                        old_value = entry_price * position_size
                        new_value = current['Close'] * additional_size
                        position_size += additional_size
                        entry_price = (old_value + new_value) / position_size
                        
                        # Обновляем стоп-лосс и тейк-профит для новой средней цены
                        exit_levels = self.calculate_dynamic_exit_levels('SHORT', entry_price, current, trade_age_hours)
                        stop_loss_price = exit_levels['stop_loss']
                        take_profit_price = exit_levels['take_profit']
                        
                        pyramid_entries += 1
                        continue
                    
                    # Проверка сигналов на выход
                    exit_signals = []
                    
                    # Определим текущий режим рынка
                    current_regime = 'trend' if current['Trend_Weight'] > 0.5 else 'range'
                    
                    # Трендовые сигналы выхода из шорта
                    if current_regime == 'trend':
                        # Сильный трендовый сигнал - выход по кроссоверу EMA или MACD
                        if ((previous[f'EMA_{self.params["short_ema"]}'] <= previous[f'EMA_{self.params["long_ema"]}']) and 
                            (current[f'EMA_{self.params["short_ema"]}'] > current[f'EMA_{self.params["long_ema"]}'])):
                            exit_signals.append('EMA Crossover')
                        
                        if current['MACD_Bullish_Cross']:
                            exit_signals.append('MACD Crossover')
                            
                        if current['Bullish_Trend'] and not previous['Bullish_Trend']:
                            exit_signals.append('Trend Change')
                            
                        # Новые улучшенные сигналы
                        if current['Higher_TF_Bullish'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Higher TF Trend Change')
                    
                    # Контртрендовые сигналы выхода из шорта
                    else:  # current_regime == 'range'
                        # Сильный флетовый сигнал - выход по RSI, Боллинджеру или дивергенции
                        if current['RSI'] < self.params['rsi_oversold']:
                            exit_signals.append('RSI Oversold')
                            
                        if current['Close'] < current['BB_Lower']:
                            exit_signals.append('Lower Bollinger')
                            
                        if current['Bullish_Divergence']:
                            exit_signals.append('Bullish Divergence')
                            
                        # Новые улучшенные сигналы
                        if current['Bullish_Engulfing'] and unrealized_pnl_pct > 0.02:
                            exit_signals.append('Bullish Engulfing')
                    
                    # Общие сигналы выхода вне зависимости от режима
                    
                    # Выход на основе волатильности
                    if current['Volatility_Ratio'] > 2.0 and unrealized_pnl_pct > 0.03:
                        exit_signals.append('Volatility Spike')
                        
                    # Если прибыль достигла определенного уровня, но падение замедлилось
                    if (unrealized_pnl_pct > 0.08 and 
                        previous['Close'] < current['Close'] and 
                        previous['MACD'] < current['MACD']):
                        exit_signals.append('Profit Protection')
                    
                    # Если есть сигнал на выход и прошло достаточно времени с момента входа
                    if exit_signals and trade_age_hours > 4:
                        # Закрываем шорт
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
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        max_trade_price = 0.0
                        min_trade_price = float('inf')
                        continue
            
            # ------ ОТКРЫТИЕ НОВЫХ ПОЗИЦИЙ -------
            if position == 0:
                # Проверяем минимальный интервал между сделками
                if last_trade_date is not None:
                    hours_since_last_trade = (current.name - last_trade_date).total_seconds() / 3600
                    if hours_since_last_trade < self.min_trades_interval:
                        # Пропускаем сигнал, если не прошло достаточно времени
                        continue
                
                # Проверяем временной фильтр (если мы не в активные часы, пропускаем сигнал)
                if not current['Active_Hours']:
                    continue
                
                # Проверяем оптимальное время для торговли на основе анализа
                if not self.is_optimal_trading_time(current.name):
                    continue
                
                # Определяем текущий режим рынка
                current_regime = 'trend' if current['Trend_Weight'] > 0.5 else 'range'
                
                # Получаем торговые сигналы, специфичные для режима рынка
                signals = self.get_trading_signals(current, previous, current_regime)
                
                # Применяем дополнительную фильтрацию
                filtered_signals = self.apply_advanced_filtering(current, signals)
                
                # Динамический расчет риска на сделку
                # 1. Рассчитываем базовый риск на основе волатильности
                volatility_multiplier = 1.0
                if current['ATR'] > 0 and not pd.isna(current['ATR_MA']) and current['ATR_MA'] > 0:
                    # Нормализуем ATR относительно его среднего значения
                    atr_ratio = current['ATR'] / current['ATR_MA']
                    volatility_multiplier = 1.0 / atr_ratio  # Меньше риска при высокой волатильности
                
                risk_per_trade = self.base_risk_per_trade * volatility_multiplier
                
                # 2. Корректируем риск на основе режима рынка
                if current_regime == 'trend':
                    # В тренде можно рисковать немного больше
                    risk_per_trade *= 1.1
                else:
                    # Во флете нужно быть более осторожным
                    risk_per_trade *= 0.9
                
                # 3. Учитываем тренд на старшем таймфрейме
                if (current['Higher_TF_Bullish'] and filtered_signals['long_weight'] > filtered_signals['short_weight']) or \
                   (current['Higher_TF_Bearish'] and filtered_signals['short_weight'] > filtered_signals['long_weight']):
                    # Если торгуем по направлению тренда на старшем таймфрейме
                    risk_per_trade *= 1.2
                
                # Ограничиваем мин и макс риск
                risk_per_trade = max(0.01, min(0.03, risk_per_trade))
                
                # -- ПРИНЯТИЕ РЕШЕНИЯ О ВХОДЕ --
                # Минимальный порог силы сигнала для входа
                min_signal_threshold = 0.6
                
                # Вход в ЛОНГ, если его сигнал сильнее и превышает порог
                if filtered_signals['long_weight'] > filtered_signals['short_weight'] and filtered_signals['long_weight'] >= min_signal_threshold:
                    # Сигнал на лонг
                    position = 1
                    entry_price = current['Close']
                    if entry_price <= 0:
                        # Защита от нулевой цены
                        entry_price = 1.0
                    entry_date = current.name
                    
                    # Рассчитываем оптимальное плечо
                    optimal_leverage = self.calculate_optimal_leverage(current, 'LONG', self.max_leverage)
                    
                    # Динамические стоп-лосс и тейк-профит
                    exit_levels = self.calculate_dynamic_exit_levels('LONG', entry_price, current)
                    stop_loss_price = exit_levels['stop_loss']
                    take_profit_price = exit_levels['take_profit']
                    
                    # Адаптивный размер позиции
                    position_size = self.adaptive_position_sizing(balance, risk_per_trade, entry_price, stop_loss_price, optimal_leverage)
                    
                    # Инициализируем отслеживание максимальной/минимальной цены
                    max_trade_price = current['High']
                    min_trade_price = current['Low']
                    
                    # Сохраняем информацию о сигналах
                    signal_info = ', '.join(signal for signal, _ in signals['long_signals'])
                    
                    # Добавляем запись о входе в лонг
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
                        'leverage': optimal_leverage
                    })
                
                # Вход в ШОРТ, если его сигнал сильнее и превышает порог
                elif filtered_signals['short_weight'] > filtered_signals['long_weight'] and filtered_signals['short_weight'] >= min_signal_threshold:
                    # Сигнал на шорт
                    position = -1
                    entry_price = current['Close']
                    if entry_price <= 0:
                        # Защита от нулевой цены
                        entry_price = 1.0
                    entry_date = current.name
                    
                    # Рассчитываем оптимальное плечо
                    optimal_leverage = self.calculate_optimal_leverage(current, 'SHORT', self.max_leverage)
                    
                    # Динамические стоп-лосс и тейк-профит
                    exit_levels = self.calculate_dynamic_exit_levels('SHORT', entry_price, current)
                    stop_loss_price = exit_levels['stop_loss']
                    take_profit_price = exit_levels['take_profit']
                    
                    # Адаптивный размер позиции
                    position_size = self.adaptive_position_sizing(balance, risk_per_trade, entry_price, stop_loss_price, optimal_leverage)
                    
                    # Инициализируем отслеживание максимальной/минимальной цены
                    max_trade_price = current['High']
                    min_trade_price = current['Low']
                    
                    # Сохраняем информацию о сигналах
                    signal_info = ', '.join(signal for signal, _ in signals['short_signals'])
                    
                    # Добавляем запись о входе в шорт
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
                        'leverage': optimal_leverage
                    })
            
            # ------ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ -------
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
                'range_weight': current['Range_Weight']
            })
        
        # ------ ЗАКРЫТИЕ ПОЗИЦИИ В КОНЦЕ БЭКТЕСТА -------
        if position != 0 and entry_price > 0:
            last_candle = self.data.iloc[-1]
            exit_price = last_candle['Close']
            
            # Рассчитываем длительность сделки
            trade_age_hours = (last_candle.name - entry_date).total_seconds() / 3600
            
            if position == 1:  # Лонг
                pnl = ((exit_price / entry_price) - 1) * position_size * self.max_leverage
            else:  # Шорт
                pnl = (1 - (exit_price / entry_price)) * position_size * self.max_leverage
                
            balance += pnl
            
            # Находим запись о входе в эту позицию и добавляем в нее информацию о выходе
            for trade in reversed(self.trade_history):
                if trade['exit_date'] is None:
                    trade['exit_date'] = last_candle.name
                    trade['exit_price'] = exit_price
                    trade['pnl'] = pnl
                    trade['balance'] = balance
                    trade['reason'] = trade['reason'] + ', End of Backtest'
                    trade['trade_duration'] = trade_age_hours
                    break
        
        # Создаем DataFrame с результатами
        self.backtest_results = pd.DataFrame(results)
        
        # Обрабатываем историю сделок
        if self.trade_history:
            self.trade_df = pd.DataFrame(self.trade_history)
            
            # Заполняем пропущенные значения для незавершенных сделок
            self.trade_df['exit_date'].fillna(self.data.index[-1], inplace=True)
            self.trade_df['exit_price'].fillna(self.data['Close'].iloc[-1], inplace=True)
            
            # Рассчитываем P&L для незавершенных сделок
            for i, row in self.trade_df.iterrows():
                if pd.isna(row['pnl']):
                    if row['position'] == 'LONG':
                        pnl = ((row['exit_price'] / row['entry_price']) - 1) * row['balance'] * self.max_leverage
                    else:  # 'SHORT'
                        pnl = (1 - (row['exit_price'] / row['entry_price'])) * row['balance'] * self.max_leverage
                    
                    self.trade_df.at[i, 'pnl'] = pnl
        else:
            self.trade_df = pd.DataFrame()
        
        print("Бэктест завершен")
        return self.backtest_results
    
    def plot_equity_curve(self):
        """Plot equity curve with additional performance metrics"""
        if self.backtest_results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter
        
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(self.backtest_results['date'], self.backtest_results['equity'], label='Equity', color='blue')
        ax1.plot(self.backtest_results['date'], self.backtest_results['balance'], label='Balance', color='green')
        
        # Calculate and plot drawdown
        equity_curve = self.backtest_results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max * 100
        
        # Plot drawdown
        ax2.fill_between(self.backtest_results['date'], 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylim(bottom=0, top=max(drawdown) * 1.5)
        ax2.invert_yaxis()  # Invert y-axis so drawdowns go down
        
        # Format x-axis for dates
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_formatter(date_format)
        
        # Format y-axis for equity as currency
        def currency_formatter(x, pos):
            return f'${x:,.0f}'
        
        ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        # Format y-axis for drawdown as percentage
        def percentage_formatter(x, pos):
            return f'{x:.0f}%'
        
        ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        # Add annotations for key metrics
        initial_balance = self.initial_balance
        final_balance = self.backtest_results['balance'].iloc[-1]
        total_return = ((final_balance / initial_balance) - 1) * 100
        max_drawdown = drawdown.max()
        
        # Calculate winning trades percentage
        if hasattr(self, 'trade_df') and self.trade_df is not None and len(self.trade_df) > 0:
            win_rate = len(self.trade_df[self.trade_df['pnl'] > 0]) / len(self.trade_df) * 100
        else:
            win_rate = 0
        
        # Add text box with performance metrics
        textstr = '\n'.join((
            f'Initial Balance: ${initial_balance:,.2f}',
            f'Final Balance: ${final_balance:,.2f}',
            f'Total Return: {total_return:.2f}%',
            f'Max Drawdown: {max_drawdown:.2f}%',
            f'Win Rate: {win_rate:.2f}%'
        ))
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax1.text(0.02, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Add vertical lines for trades
        if hasattr(self, 'trade_df') and self.trade_df is not None and len(self.trade_df) > 0:
            # Plot only a subset of trades to avoid overcrowding
            sample_trades = self.trade_df.sample(min(50, len(self.trade_df))) if len(self.trade_df) > 50 else self.trade_df
            
            for idx, trade in sample_trades.iterrows():
                if trade['position'] == 'LONG':
                    color = 'green'
                    marker = '^'
                else:  # SHORT
                    color = 'red'
                    marker = 'v'
                    
                # Convert to datetime if it's a string
                entry_date = pd.to_datetime(trade['entry_date']) if isinstance(trade['entry_date'], str) else trade['entry_date']
                exit_date = pd.to_datetime(trade['exit_date']) if isinstance(trade['exit_date'], str) else trade['exit_date']
                
                # Find closest dates in backtest_results
                if entry_date is not None:
                    closest_entry = min(self.backtest_results['date'], key=lambda x: abs(x - entry_date))
                    entry_equity = self.backtest_results.loc[self.backtest_results['date'] == closest_entry, 'equity'].values
                    if len(entry_equity) > 0:
                        ax1.scatter(closest_entry, entry_equity[0], color=color, s=50, marker=marker, alpha=0.6)
                
                if exit_date is not None:
                    closest_exit = min(self.backtest_results['date'], key=lambda x: abs(x - exit_date))
                    exit_equity = self.backtest_results.loc[self.backtest_results['date'] == closest_exit, 'equity'].values
                    if len(exit_equity) > 0:
                        ax1.scatter(closest_exit, exit_equity[0], color='black', s=30, marker='o', alpha=0.6)
        
        # Set titles and labels
        ax1.set_title('Equity Curve and Performance')
        ax1.set_ylabel('Account Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
        return fig
    
    def analyze_results(self):
        """Анализ результатов бэктеста с правильным расчетом P&L"""
        if self.backtest_results is None or len(self.trade_df) == 0:
            print("Нет данных для анализа. Сначала запустите бэктест.")
            return None
        
        print("\n===== РЕЗУЛЬТАТЫ БЭКТЕСТА =====")
        
        # Общие показатели
        initial_balance = self.initial_balance
        final_balance = self.backtest_results['balance'].iloc[-1]
        total_return = ((final_balance / initial_balance) - 1) * 100
        
        # Расчет месячной доходности
        start_date = self.backtest_results['date'].iloc[0]
        end_date = self.backtest_results['date'].iloc[-1]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        if months_diff == 0:
            months_diff = 1  # Минимум 1 месяц
            
        monthly_return = (((final_balance / initial_balance) ** (1 / months_diff)) - 1) * 100
        
        # Расчет максимальной просадки
        equity_curve = self.backtest_results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max * 100
        max_drawdown = drawdown.max()
        
        # Статистика по сделкам
        total_trades = len(self.trade_df)
        profitable_trades = len(self.trade_df[self.trade_df['pnl'] > 0])
        losing_trades = len(self.trade_df[self.trade_df['pnl'] <= 0])
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit = self.trade_df[self.trade_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = self.trade_df[self.trade_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(self.trade_df[self.trade_df['pnl'] > 0]['pnl'].sum() / 
                        self.trade_df[self.trade_df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 and self.trade_df[self.trade_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
        
        # Расчет коэффициентов Шарпа и Сортино
        if 'equity' in self.backtest_results.columns:
            daily_returns = self.backtest_results['equity'].pct_change().dropna()
            
            if len(daily_returns) > 0:
                # Годовой коэффициент Шарпа (предполагаемая безрисковая ставка 0%)
                sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
                
                # Годовой коэффициент Сортино (только отрицательные доходности)
                negative_returns = daily_returns[daily_returns < 0]
                downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() > 0 else 1e-10
                sortino_ratio = (daily_returns.mean() * 252) / downside_deviation
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Медианная продолжительность сделок
        if 'trade_duration' in self.trade_df.columns:
            median_duration_hours = self.trade_df['trade_duration'].median()
        else:
            self.trade_df['duration'] = pd.to_datetime(self.trade_df['exit_date']) - pd.to_datetime(self.trade_df['entry_date'])
            median_duration_hours = self.trade_df['duration'].median().total_seconds() / 3600
        
        # Статистика по типам позиций с УЛУЧШЕННЫМ расчетом P&L
        long_trades = len(self.trade_df[self.trade_df['position'] == 'LONG'])
        short_trades = len(self.trade_df[self.trade_df['position'] == 'SHORT'])
        
        # ИСПРАВЛЕННЫЙ МЕТОД для расчета вклада в доходность
        # Создаем временную таблицу с накопительным балансом
        balance_df = pd.DataFrame({'date': [], 'balance': [], 'position': []})
        
        # Добавляем начальный баланс
        balance_df = balance_df._append({'date': self.backtest_results['date'].iloc[0], 
                                        'balance': initial_balance, 
                                        'position': 'INITIAL'}, 
                                    ignore_index=True)
        
        # Сортируем сделки по дате выхода
        sorted_trades = self.trade_df.sort_values('exit_date').copy()
        
        # Строим таблицу с изменением баланса после каждой сделки
        for _, trade in sorted_trades.iterrows():
            if pd.notna(trade['pnl']):
                new_balance = balance_df['balance'].iloc[-1] + trade['pnl']
                balance_df = balance_df._append({'date': trade['exit_date'], 
                                            'balance': new_balance, 
                                            'position': trade['position'],
                                            'pnl': trade['pnl']}, 
                                            ignore_index=True)
        
        # Рассчитываем процентное изменение после каждой сделки
        balance_df['pct_change'] = balance_df['balance'].pct_change() * 100
        balance_df.loc[0, 'pct_change'] = 0  # Для начального баланса
        
        # Суммируем процентные изменения по типам позиций
        long_contribution_pct = balance_df[balance_df['position'] == 'LONG']['pct_change'].sum()
        short_contribution_pct = balance_df[balance_df['position'] == 'SHORT']['pct_change'].sum()
        
        # Рассчитываем абсолютный вклад в долларах
        long_contribution = initial_balance * (long_contribution_pct / 100)
        short_contribution = initial_balance * (short_contribution_pct / 100)
        
        # Сохраняем старый метод для сравнения
        old_long_profit = self.trade_df[self.trade_df['position'] == 'LONG']['pnl'].sum()
        old_short_profit = self.trade_df[self.trade_df['position'] == 'SHORT']['pnl'].sum()
        
        # Вычисляем средний P&L для лонгов и шортов
        avg_long_pnl = self.trade_df[self.trade_df['position'] == 'LONG']['pnl'].mean() if long_trades > 0 else 0
        avg_short_pnl = self.trade_df[self.trade_df['position'] == 'SHORT']['pnl'].mean() if short_trades > 0 else 0
        
        # Вычисляем процент выигрышных сделок для лонгов и шортов
        long_win_rate = len(self.trade_df[(self.trade_df['position'] == 'LONG') & (self.trade_df['pnl'] > 0)]) / long_trades * 100 if long_trades > 0 else 0
        short_win_rate = len(self.trade_df[(self.trade_df['position'] == 'SHORT') & (self.trade_df['pnl'] > 0)]) / short_trades * 100 if short_trades > 0 else 0
        
        # Анализ по причинам выхода
        exit_reasons = self.trade_df['reason'].str.extract(r'(?:, )([^,]+)', expand=False)
        exit_reasons_counts = exit_reasons.value_counts()
        
        # Анализ по часам и дням
        hour_stats = self.analyze_hour_performance()
        day_stats = self.analyze_day_performance()
        
        # Вывод результатов
        print(f"Начальный баланс: ${initial_balance:.2f}")
        print(f"Конечный баланс: ${final_balance:.2f}")
        print(f"Общая доходность: {total_return:.2f}%")
        print(f"Месячная доходность: {monthly_return:.2f}%")
        print(f"Максимальная просадка: {max_drawdown:.2f}%")
        print(f"Всего сделок: {total_trades}")
        print(f"Прибыльных сделок: {profitable_trades} ({win_rate:.2f}%)")
        print(f"Убыточных сделок: {losing_trades} ({100 - win_rate:.2f}%)")
        print(f"Средняя прибыль: ${avg_profit:.2f}")
        print(f"Средний убыток: ${avg_loss:.2f}")
        print(f"Профит-фактор: {profit_factor:.2f}")
        print(f"Коэффициент Шарпа: {sharpe_ratio:.2f}")
        print(f"Коэффициент Сортино: {sortino_ratio:.2f}")
        print(f"Медианная продолжительность сделки: {median_duration_hours:.2f} часов")
        
        # Улучшенная статистика по типам сделок
        print("\n===== УЛУЧШЕННАЯ СТАТИСТИКА ПО ТИПАМ СДЕЛОК =====")
        print(f"Лонг сделок: {long_trades} (Win Rate: {long_win_rate:.2f}%)")
        print(f"Шорт сделок: {short_trades} (Win Rate: {short_win_rate:.2f}%)")
        print(f"Средний P&L лонг сделки: ${avg_long_pnl:.2f}")
        print(f"Средний P&L шорт сделки: ${avg_short_pnl:.2f}")
        print(f"Вклад лонг сделок в доходность: {long_contribution_pct:.2f}% (${long_contribution:.2f})")
        print(f"Вклад шорт сделок в доходность: {short_contribution_pct:.2f}% (${short_contribution:.2f})")
        print(f"Сумма вкладов: {long_contribution_pct + short_contribution_pct:.2f}% (${long_contribution + short_contribution:.2f})")
        print(f"Проверка: итоговое изменение баланса ${final_balance - initial_balance:.2f}")
        
        # Для сравнения
        print(f"\nСтарый метод расчета (суммирование P&L):")
        print(f"Суммарный P&L лонг сделок: ${old_long_profit:.2f}")
        print(f"Суммарный P&L шорт сделок: ${old_short_profit:.2f}")
        print(f"Общий P&L: ${old_long_profit + old_short_profit:.2f}")
        
        print("\n===== РАСПРЕДЕЛЕНИЕ ПРИЧИН ВЫХОДА =====")
        for reason, count in exit_reasons_counts.items():
            print(f"{reason}: {count} ({count/total_trades*100:.2f}%)")
        
        # Годовая эффективность
        self.backtest_results['year'] = pd.to_datetime(self.backtest_results['date']).dt.year
        yearly_performance = {}
        
        for year in self.backtest_results['year'].unique():
            year_data = self.backtest_results[self.backtest_results['year'] == year]
            start_balance = year_data['balance'].iloc[0]
            end_balance = year_data['balance'].iloc[-1]
            yearly_return = ((end_balance / start_balance) - 1) * 100
            yearly_performance[year] = yearly_return
        
        print("\n===== ГОДОВАЯ ЭФФЕКТИВНОСТЬ =====")
        for year, yearly_return in yearly_performance.items():
            print(f"{year}: {yearly_return:.2f}%")
        
        # Эффективность в разных режимах рынка с УЛУЧШЕННЫМ расчетом
        # Разделяем сделки на трендовые и флетовые на основе причины входа
        trend_trades_idx = self.trade_df['reason'].str.contains('Trend|EMA|MACD')
        trend_trades = self.trade_df[trend_trades_idx].copy()
        range_trades = self.trade_df[~trend_trades_idx].copy()
        
        # Вычисляем винрейт для каждого типа
        trend_win_rate = len(trend_trades[trend_trades['pnl'] > 0]) / len(trend_trades) * 100 if len(trend_trades) > 0 else 0
        range_win_rate = len(range_trades[range_trades['pnl'] > 0]) / len(range_trades) * 100 if len(range_trades) > 0 else 0
        
        # Фильтруем таблицу баланса для вычисления вклада
        trend_trades_exits = set(trend_trades['exit_date'].tolist())
        range_trades_exits = set(range_trades['exit_date'].tolist())
        
        trend_balance_changes = balance_df[balance_df['date'].isin(trend_trades_exits)]
        range_balance_changes = balance_df[balance_df['date'].isin(range_trades_exits)]
        
        # Вычисляем вклад в процентах
        trend_contribution_pct = trend_balance_changes['pct_change'].sum()
        range_contribution_pct = range_balance_changes['pct_change'].sum()
        
        # Конвертируем в абсолютные значения
        trend_contribution = initial_balance * (trend_contribution_pct / 100)
        range_contribution = initial_balance * (range_contribution_pct / 100)
        
        # Старый метод для сравнения
        trend_profit = trend_trades['pnl'].sum()
        range_profit = range_trades['pnl'].sum()
        
        print("\n===== ЭФФЕКТИВНОСТЬ ПО ТИПАМ РЫНКА =====")
        print(f"Трендовые сделки: {len(trend_trades)} (Win Rate: {trend_win_rate:.2f}%)")
        print(f"Флетовые сделки: {len(range_trades)} (Win Rate: {range_win_rate:.2f}%)")
        print(f"Вклад трендовых сделок в доходность: {trend_contribution_pct:.2f}% (${trend_contribution:.2f})")
        print(f"Вклад флетовых сделок в доходность: {range_contribution_pct:.2f}% (${range_contribution:.2f})")
        print(f"Старый метод - Суммарный P&L трендовых сделок: ${trend_profit:.2f}")
        print(f"Старый метод - Суммарный P&L флетовых сделок: ${range_profit:.2f}")
        
        # Анализ влияния пирамидинга - по аналогии 
        if 'pyramid_entries' in self.trade_df.columns:
            pyramid_trades = self.trade_df[self.trade_df['pyramid_entries'] > 0].copy()
            non_pyramid_trades = self.trade_df[self.trade_df['pyramid_entries'] == 0].copy()
            
            pyramid_win_rate = len(pyramid_trades[pyramid_trades['pnl'] > 0]) / len(pyramid_trades) * 100 if len(pyramid_trades) > 0 else 0
            non_pyramid_win_rate = len(non_pyramid_trades[non_pyramid_trades['pnl'] > 0]) / len(non_pyramid_trades) * 100 if len(non_pyramid_trades) > 0 else 0
            
            # Фильтруем таблицу баланса для вычисления вклада
            pyramid_trades_exits = set(pyramid_trades['exit_date'].tolist())
            non_pyramid_trades_exits = set(non_pyramid_trades['exit_date'].tolist())
            
            pyramid_balance_changes = balance_df[balance_df['date'].isin(pyramid_trades_exits)]
            non_pyramid_balance_changes = balance_df[balance_df['date'].isin(non_pyramid_trades_exits)]
            
            # Вычисляем вклад в процентах
            pyramid_contribution_pct = pyramid_balance_changes['pct_change'].sum() if len(pyramid_balance_changes) > 0 else 0
            non_pyramid_contribution_pct = non_pyramid_balance_changes['pct_change'].sum() if len(non_pyramid_balance_changes) > 0 else 0
            
            # Конвертируем в абсолютные значения
            pyramid_contribution = initial_balance * (pyramid_contribution_pct / 100)
            non_pyramid_contribution = initial_balance * (non_pyramid_contribution_pct / 100)
            
            # Старый метод для сравнения
            pyramid_profit = pyramid_trades['pnl'].sum()
            non_pyramid_profit = non_pyramid_trades['pnl'].sum()
            
            print("\n===== ВЛИЯНИЕ ПИРАМИДИНГА =====")
            print(f"Сделки с пирамидингом: {len(pyramid_trades)} (Win Rate: {pyramid_win_rate:.2f}%)")
            print(f"Сделки без пирамидинга: {len(non_pyramid_trades)} (Win Rate: {non_pyramid_win_rate:.2f}%)")
            print(f"Вклад сделок с пирамидингом в доходность: {pyramid_contribution_pct:.2f}% (${pyramid_contribution:.2f})")
            print(f"Вклад сделок без пирамидинга в доходность: {non_pyramid_contribution_pct:.2f}% (${non_pyramid_contribution:.2f})")
            print(f"Старый метод - P&L сделок с пирамидингом: ${pyramid_profit:.2f}")
            print(f"Старый метод - P&L сделок без пирамидинга: ${non_pyramid_profit:.2f}")
        
        # Информация о временной эффективности
        if hour_stats is not None:
            print("\n===== ЭФФЕКТИВНОСТЬ ПО ЧАСАМ ТОРГОВЛИ =====")
            best_hours = hour_stats.sort_values('win_rate', ascending=False).head(5)
            print("Лучшие 5 часов для торговли (по win rate):")
            for _, row in best_hours.iterrows():
                print(f"Час {row['hour']}: Win Rate {row['win_rate']*100:.2f}%, PnL ${row['total_pnl']:.2f}, Кол-во сделок: {row['num_trades']}")
        
        if day_stats is not None:
            print("\n===== ЭФФЕКТИВНОСТЬ ПО ДНЯМ НЕДЕЛИ =====")
            for _, row in day_stats.sort_values('win_rate', ascending=False).iterrows():
                print(f"{row['day']}: Win Rate {row['win_rate']*100:.2f}%, PnL ${row['total_pnl']:.2f}, Кол-во сделок: {row['num_trades']}")
        
        # Информация о среднем плече
        if 'leverage' in self.trade_df.columns:
            avg_leverage = self.trade_df['leverage'].mean()
            print(f"\nСреднее используемое плечо: {avg_leverage:.2f}x")
                
        # Динамический анализ риска
        if 'risk_per_trade' in self.trade_df.columns:
            avg_risk = self.trade_df['risk_per_trade'].mean() * 100
            print(f"Средний риск на сделку: {avg_risk:.2f}%")
        
        # Сводная статистика
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
            'yearly_performance': yearly_performance,
            'trend_win_rate': trend_win_rate,
            'range_win_rate': range_win_rate
        }
        
        return stats
    


# Add this at the end of your file, after the class definition

def main():
    """Main function to execute the strategy"""
    import os
    
    # Define the path to your data file
    base_dir = r"C:\Diploma\Pet"
    
    # List CSV files in the directory
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {base_dir}. Please ensure your data file is in this directory.")
        return
    
    # Use the first CSV file found
    data_file = csv_files[0]
    data_path = os.path.join(base_dir, data_file)
    
    print(f"Using data file: {data_path}")
    
    # Create a strategy instance
    strategy = AdvancedAdaptiveStrategy(
        data_path=data_path,
        initial_balance=1000,
        max_leverage=3,
        base_risk_per_trade=0.02,
        min_trades_interval=6
    )
    
    # Load and prepare data
    strategy.load_data()
    
    # Calculate indicators
    strategy.calculate_indicators()
    
    # Run backtest
    strategy.run_backtest()
    
    # Analyze results
    stats = strategy.analyze_results()
    
    # Plot equity curve
    strategy.plot_equity_curve()
    
    return strategy

# Execute main function when script is run directly
if __name__ == "__main__":
    main()