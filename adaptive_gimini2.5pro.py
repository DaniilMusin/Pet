import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import warnings
# Вместо глобального подавления - обрабатывать по месту или игнорировать конкретные типы
# warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None # Отключить предупреждение Chained Assignment

# --- Константы ---
DEFAULT_SLIPPAGE_PCT = 0.0005  # 0.05%
DEFAULT_COMMISSION_PCT = 0.0004 # 0.04% (Примерная комиссия Binance Futures для мейкера/тейкера)
MIN_RISK_PER_TRADE = 0.005
MAX_RISK_PER_TRADE = 0.05
MIN_ATR_MULTIPLIER_SL = 1.5
MAX_ATR_MULTIPLIER_SL = 4.0
MIN_ATR_MULTIPLIER_TP = 2.0
MAX_ATR_MULTIPLIER_TP = 8.0
KELLY_FRACTION = 0.5 # Использовать только часть от рассчитанной доли Келли

# --- Вспомогательные классы ---

class Config:
    """Класс для хранения конфигурации стратегии."""
    def __init__(self, initial_balance=1000, max_leverage=3,
                 base_risk_per_trade=0.02, min_trades_interval_hours=6):
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.base_risk_per_trade = base_risk_per_trade
        self.min_trades_interval_hours = min_trades_interval_hours
        self.slippage_pct = DEFAULT_SLIPPAGE_PCT
        self.commission_pct = DEFAULT_COMMISSION_PCT

        # Параметры индикаторов и стратегии по умолчанию
        self.params = {
            'short_ema': 9, 'long_ema': 30,
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'adx_period': 14, 'adx_strong_trend': 25, 'adx_weak_trend': 20,
            'bb_period': 20, 'bb_std': 2,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'atr_period': 14, 'atr_multiplier_sl': 2.5, 'atr_multiplier_tp': 5.0,
            'volume_ma_period': 20, 'volume_threshold': 1.5,
            'trend_lookback': 20, 'trend_threshold': 0.1,
            'pyramid_min_profit': 0.05, 'pyramid_size_multiplier': 0.5, 'max_pyramid_entries': 3,
            'trading_hours_start': 8, 'trading_hours_end': 16,
            'adx_min': 15, 'adx_max': 35,
            'regime_volatility_lookback': 100, 'regime_direction_short': 20,
            'regime_direction_medium': 50, 'regime_direction_long': 100,
            'mean_reversion_lookback': 20, 'mean_reversion_threshold': 2.0,
            'hourly_ema_fast': 9, 'hourly_ema_slow': 30,
            'four_hour_ema_fast': 9, 'four_hour_ema_slow': 30,
            'health_trend_weight': 0.3, 'health_volatility_weight': 0.2,
            'health_volume_weight': 0.2, 'health_breadth_weight': 0.2,
            'health_sr_weight': 0.1,
            'momentum_roc_periods': [5, 10, 20, 50], 'momentum_reversal_threshold': 5,
            'optimal_trading_hours': None, 'optimal_trading_days': None
        }

    def update_params(self, new_params: dict):
        """Обновление параметров стратегии."""
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value
            else:
                print(f"Warning: Parameter '{key}' not found in default config.")

class DataHandler:
    """Загрузка и предварительная обработка данных."""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None

    def load_data(self, num_candles: int = 5520) -> pd.DataFrame:
        """Загрузка и подготовка данных."""
        print("Loading data...")
        try:
            self.data = pd.read_csv(self.data_path)
            if num_candles > 0 and num_candles < len(self.data):
                 self.data = self.data.tail(num_candles) # Ограничение по количеству свечей

            # Преобразование дат и установка индекса
            self.data['Open time'] = pd.to_datetime(self.data['Open time'])
            self.data.set_index('Open time', inplace=True)

            # Преобразование типов и обработка пропусков
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data.dropna(subset=numeric_cols, inplace=True)

            print(f"Loaded {len(self.data)} candles")
            return self.data.copy() # Возвращаем копию для изоляции
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

class IndicatorCalculator:
    """Расчет всех технических индикаторов."""
    def __init__(self, params: dict):
        self.params = params

    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех индикаторов для DataFrame."""
        print("Calculating indicators...")
        df = data.copy() # Работаем с копией

        # --- Базовые индикаторы ---
        df = self._calculate_ema(df)
        df = self._calculate_rsi(df)
        df = self._calculate_bb(df)
        df = self._calculate_atr(df)
        df = self._calculate_adx(df)
        df = self._calculate_macd(df)
        df = self._calculate_volume_features(df)
        df = self._calculate_trend_features(df)

        # --- Продвинутые индикаторы ---
        df = self._calculate_divergences(df) # Осторожно, использует цикл
        df = self._calculate_price_action(df)
        df = self._calculate_higher_tf_trend(df)
        df = self._calculate_market_structure(df)
        df = self._calculate_time_features(df)

        # --- Векторизованные адаптивные метрики ---
        df = self.detect_market_regime(df) # Векторизованный
        df = self.calculate_multi_timeframe_confirmation(df) # Векторизованный
        df = self.calculate_mean_reversion_signals(df) # Векторизованный
        df = self.identify_market_cycle_phase(df)
        df = self.calculate_market_health(df)
        df = self.calculate_momentum_metrics(df)
        df = self.adapt_to_market_conditions(df) # Зависит от предыдущих

        # Удаление NaN после всех расчетов
        initial_len = len(df)
        df.dropna(inplace=True)
        print(f"Indicators calculated. Dropped {initial_len - len(df)} rows with NaNs.")
        return df

    # --- Методы расчета отдельных индикаторов (приватные) ---
    def _calculate_ema(self, df):
        df[f'EMA_{self.params["short_ema"]}'] = df['Close'].ewm(span=self.params["short_ema"], adjust=False).mean()
        df[f'EMA_{self.params["long_ema"]}'] = df['Close'].ewm(span=self.params["long_ema"], adjust=False).mean()
        return df

    def _calculate_rsi(self, df):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Используем ewm для сглаживания как в стандартной реализации RSI
        avg_gain = gain.ewm(com=self.params['rsi_period'] - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.params['rsi_period'] - 1, adjust=False).mean()

        # Обработка деления на ноль
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        df['RSI'] = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
        return df

    def _calculate_bb(self, df):
        bb_period = self.params['bb_period']
        bb_std = self.params['bb_std']
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        rolling_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * bb_std)
        return df

    def _calculate_atr(self, df):
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Используем EMA для ATR, как это часто делается
        df['ATR'] = tr.ewm(alpha=1/self.params['atr_period'], adjust=False).mean()
        df['ATR_MA'] = df['ATR'].rolling(20).mean() # Для доп. анализа волатильности
        return df

    def _calculate_adx(self, df):
        period = self.params['adx_period']
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_series = pd.Series(plus_dm, index=df.index)
        minus_dm_series = pd.Series(minus_dm, index=df.index)

        smooth_plus_dm = plus_dm_series.ewm(alpha=1/period, adjust=False).mean()
        smooth_minus_dm = minus_dm_series.ewm(alpha=1/period, adjust=False).mean()

        # Обработка деления на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di = 100 * (smooth_plus_dm / atr)
            minus_di = 100 * (smooth_minus_dm / atr)
            dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))

        df['Plus_DI'] = plus_di.fillna(0)
        df['Minus_DI'] = minus_di.fillna(0)
        df['ADX'] = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)
        return df

    def _calculate_macd(self, df):
        fast = self.params['macd_fast']
        slow = self.params['macd_slow']
        signal = self.params['macd_signal']
        fast_ema = df['Close'].ewm(span=fast, adjust=False).mean()
        slow_ema = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df

    def _calculate_volume_features(self, df):
        vol_ma_period = self.params['volume_ma_period']
        df['Volume_MA'] = df['Volume'].rolling(window=vol_ma_period).mean()

        # Рассчитать соотношение напрямую как Pandas Series
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # Заменить бесконечности (возникающие при делении на 0) на 1.0
        # Затем заменить NaN (которые могли быть в исходных данных или после rolling().mean()) на 1.0
        df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)

        # Остальные расчеты без изменений
        df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Rising_Volume'] = df['Volume'] > df['Volume_MA_3'] * 1.2
        df['Falling_Volume'] = df['Volume'] < df['Volume_MA_3'] * 0.8
        return df

    def _calculate_trend_features(self, df):
        lookback = self.params['trend_lookback']
        # Обработка деления на ноль
        shifted_close = df['Close'].shift(lookback)
        df['Price_Change_Pct'] = np.where(
            shifted_close == 0, 0.0,
            (df['Close'] - shifted_close) / shifted_close
        ).fillna(0.0)

        # Условия тренда (векторизованные)
        threshold = self.params['trend_threshold']
        df['Strong_Trend_ADX'] = df['ADX'] > self.params['adx_strong_trend']
        df['Weak_Trend_ADX'] = df['ADX'] < self.params['adx_weak_trend']

        df['Strong_Trend_Price'] = df['Price_Change_Pct'].abs() > threshold
        df['Weak_Trend_Price'] = df['Price_Change_Pct'].abs() < threshold / 2

        df['Strong_Trend'] = df['Strong_Trend_ADX'] & df['Strong_Trend_Price']
        df['Weak_Trend'] = df['Weak_Trend_ADX'] & df['Weak_Trend_Price']

        df['Bullish_Trend'] = df['Strong_Trend'] & (df['Price_Change_Pct'] > 0) & (df['Plus_DI'] > df['Minus_DI'])
        df['Bearish_Trend'] = df['Strong_Trend'] & (df['Price_Change_Pct'] < 0) & (df['Plus_DI'] < df['Minus_DI'])

        # Взвешивание тренда/флэта
        df['Trend_Weight'] = np.clip(
            (df['ADX'] - self.params['adx_min']) / (self.params['adx_max'] - self.params['adx_min']), 0, 1
        ).fillna(0.5) # Заполнить NaN средним значением
        df['Range_Weight'] = 1.0 - df['Trend_Weight']
        return df

    def _calculate_divergences(self, df, lookback=15, threshold_pct=0.005):
        """
        Расчет RSI дивергенций (ПРОСТАЯ реализация с циклом, может быть медленной).
        Рассмотрите `scipy.signal.find_peaks` для более быстрой, но сложной реализации.
        """
        print("Calculating divergences (may be slow)...")
        df['Bullish_Divergence'] = False
        df['Bearish_Divergence'] = False

        # Простой поиск локальных минимумов/максимумов
        df['Price_Min'] = df['Close'].rolling(lookback, center=True).min() == df['Close']
        df['Price_Max'] = df['Close'].rolling(lookback, center=True).max() == df['Close']
        df['RSI_Min'] = df['RSI'].rolling(lookback, center=True).min() == df['RSI']
        df['RSI_Max'] = df['RSI'].rolling(lookback, center=True).max() == df['RSI']

        price_mins_idx = df[df['Price_Min']].index
        price_maxs_idx = df[df['Price_Max']].index
        rsi_mins_idx = df[df['RSI_Min']].index
        rsi_maxs_idx = df[df['RSI_Max']].index

        # Поиск бычьих дивергенций (LL на цене, HL на RSI)
        for i in range(1, len(price_mins_idx)):
            current_price_min_idx = price_mins_idx[i]
            prev_price_min_idx = price_mins_idx[i-1]

            # Находим ближайший предыдущий минимум RSI
            relevant_rsi_mins = rsi_mins_idx[(rsi_mins_idx < current_price_min_idx) & (rsi_mins_idx >= prev_price_min_idx)]
            if not relevant_rsi_mins.empty:
                prev_rsi_min_idx = relevant_rsi_mins[-1] # Берем самый последний из релевантных

                # Проверяем условия дивергенции
                if (df.loc[current_price_min_idx, 'Close'] < df.loc[prev_price_min_idx, 'Close'] * (1 - threshold_pct) and
                    df.loc[current_price_min_idx, 'RSI'] > df.loc[prev_rsi_min_idx, 'RSI'] * (1 + threshold_pct)):
                    df.loc[current_price_min_idx, 'Bullish_Divergence'] = True

        # Поиск медвежьих дивергенций (HH на цене, LH на RSI)
        for i in range(1, len(price_maxs_idx)):
            current_price_max_idx = price_maxs_idx[i]
            prev_price_max_idx = price_maxs_idx[i-1]

            # Находим ближайший предыдущий максимум RSI
            relevant_rsi_maxs = rsi_maxs_idx[(rsi_maxs_idx < current_price_max_idx) & (rsi_maxs_idx >= prev_price_max_idx)]
            if not relevant_rsi_maxs.empty:
                prev_rsi_max_idx = relevant_rsi_maxs[-1]

                # Проверяем условия дивергенции
                if (df.loc[current_price_max_idx, 'Close'] > df.loc[prev_price_max_idx, 'Close'] * (1 + threshold_pct) and
                    df.loc[current_price_max_idx, 'RSI'] < df.loc[prev_rsi_max_idx, 'RSI'] * (1 - threshold_pct)):
                    df.loc[current_price_max_idx, 'Bearish_Divergence'] = True

        df.drop(columns=['Price_Min', 'Price_Max', 'RSI_Min', 'RSI_Max'], inplace=True)
        return df


    def _calculate_price_action(self, df):
        # Паттерны Price Action
        df['Bullish_Engulfing'] = (
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1)) &
            (df['Close'] > df['Open']) &
            (df['Open'].shift(1) > df['Close'].shift(1))
        )
        df['Bearish_Engulfing'] = (
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1)) &
            (df['Close'] < df['Open']) &
            (df['Open'].shift(1) < df['Close'].shift(1))
        )
        df['MACD_Bullish_Cross'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df['MACD_Bearish_Cross'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))

        return df

    def _calculate_higher_tf_trend(self, df):
        # Тренд старшего таймфрейма (дневной)
        try:
            daily_close = df['Close'].resample('1D').last()
            daily_ema50 = daily_close.ewm(span=50, adjust=False).mean()
            daily_ema200 = daily_close.ewm(span=200, adjust=False).mean()
            # Заполняем пропуски после ресемплинга/реиндексации
            # ИСПРАВЛЕНО: Используем .ffill() и .bfill() вместо fillna(method=...)
            df['Daily_EMA50'] = daily_ema50.reindex(df.index).ffill().bfill() # Сначала вперед, потом назад
            df['Daily_EMA200'] = daily_ema200.reindex(df.index).ffill().bfill() # Сначала вперед, потом назад
            df['Higher_TF_Bullish'] = df['Daily_EMA50'] > df['Daily_EMA200']
            df['Higher_TF_Bearish'] = df['Daily_EMA50'] < df['Daily_EMA200']
        except Exception as e:
            print(f"Warning: Could not calculate daily EMAs: {e}. Setting higher TF trend to neutral.")
            df['Daily_EMA50'] = df['Close'] # Fallback
            df['Daily_EMA200'] = df['Close'] # Fallback
            df['Higher_TF_Bullish'] = False
            df['Higher_TF_Bearish'] = False
        return df

    def _calculate_market_structure(self, df):
        # Анализ структуры рынка (HH, HL, LH, LL)
        # Эти расчеты могут быть не очень надежными на коротких периодах
        roll_short = 10
        roll_long = 20
        df['HH'] = df['High'].rolling(roll_short).max() > df['High'].rolling(roll_long).max().shift(roll_short)
        df['HL'] = df['Low'].rolling(roll_short).min() > df['Low'].rolling(roll_long).min().shift(roll_short)
        df['LH'] = df['High'].rolling(roll_short).max() < df['High'].rolling(roll_long).max().shift(roll_short)
        df['LL'] = df['Low'].rolling(roll_short).min() < df['Low'].rolling(roll_long).min().shift(roll_short)
        df['Bullish_Structure'] = df['HH'] & df['HL']
        df['Bearish_Structure'] = df['LH'] & df['LL']
        return df

    def _calculate_time_features(self, df):
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek # Monday=0, Sunday=6
        df['Active_Hours'] = (df['Hour'] >= self.params['trading_hours_start']) & \
                             (df['Hour'] <= self.params['trading_hours_end'])
        return df

    # --- Векторизованные методы для адаптивных метрик ---

    def detect_market_regime(self, df):
        """Векторизованное определение рыночного режима."""
        lookback = self.params['regime_volatility_lookback'] # Максимальный период для сдвига
        short_p = self.params['regime_direction_short']
        medium_p = self.params['regime_direction_medium']
        long_p = self.params['regime_direction_long']

        # Расчет изменений цены
        df[f'chg_{short_p}'] = df['Close'].pct_change(periods=short_p)
        df[f'chg_{medium_p}'] = df['Close'].pct_change(periods=medium_p)
        df[f'chg_{long_p}'] = df['Close'].pct_change(periods=long_p)

        # Расчет волатильности (стандартное отклонение процентных изменений)
        pct_change = df['Close'].pct_change()
        df[f'vol_{short_p}'] = pct_change.rolling(window=short_p).std()
        df[f'vol_{medium_p}'] = pct_change.rolling(window=medium_p).std()
        df[f'vol_{long_p}'] = pct_change.rolling(window=long_p).std()

        # Условия расширения/сжатия волатильности
        vol_expanding = (df[f'vol_{short_p}'] > df[f'vol_{medium_p}']) & \
                        (df[f'vol_{medium_p}'] > df[f'vol_{long_p}'])
        vol_contracting = (df[f'vol_{short_p}'] < df[f'vol_{medium_p}']) & \
                          (df[f'vol_{medium_p}'] < df[f'vol_{long_p}'])

        # Определение режимов с помощью масок
        strong_bull_mask = (df[f'chg_{short_p}'] > 0.05) & (df[f'chg_{medium_p}'] > 0.03) & vol_expanding
        strong_bear_mask = (df[f'chg_{short_p}'] < -0.05) & (df[f'chg_{medium_p}'] < -0.03) & vol_expanding
        choppy_range_mask = (df[f'chg_{short_p}'].abs() < 0.02) & vol_contracting
        transition_bull_mask = (df[f'chg_{short_p}'] > 0) & (df[f'chg_{medium_p}'] < 0)
        transition_bear_mask = (df[f'chg_{short_p}'] < 0) & (df[f'chg_{medium_p}'] > 0)

        conditions = [strong_bull_mask, strong_bear_mask, choppy_range_mask, transition_bull_mask, transition_bear_mask]
        choices = ["strong_bull", "strong_bear", "choppy_range", "transition_to_bull", "transition_to_bear"]
        df['Market_Regime'] = np.select(conditions, choices, default='mixed')

        # Удаляем временные столбцы
        df.drop(columns=[f'chg_{short_p}', f'chg_{medium_p}', f'chg_{long_p}',
                         f'vol_{short_p}', f'vol_{medium_p}', f'vol_{long_p}'], inplace=True)

        # Мультипликаторы размера позиции (остаются как в оригинале)
        regime_multipliers_map = {
            "strong_bull": {"LONG": 1.2, "SHORT": 0.6}, "strong_bear": {"LONG": 0.6, "SHORT": 1.2},
            "choppy_range": {"LONG": 0.8, "SHORT": 0.8}, "transition_to_bull": {"LONG": 1.0, "SHORT": 0.8},
            "transition_to_bear": {"LONG": 0.8, "SHORT": 1.0}, "mixed": {"LONG": 0.7, "SHORT": 0.7},
            "unknown": {"LONG": 0.7, "SHORT": 0.7} # Добавлено для случая NaN или непредвиденных значений
        }
        df['Regime_Long_Multiplier'] = df['Market_Regime'].map(lambda x: regime_multipliers_map.get(x, regime_multipliers_map['unknown'])['LONG'])
        df['Regime_Short_Multiplier'] = df['Market_Regime'].map(lambda x: regime_multipliers_map.get(x, regime_multipliers_map['unknown'])['SHORT'])

        # Заполняем возможные пропуски после map
        df['Regime_Long_Multiplier'].fillna(regime_multipliers_map['unknown']['LONG'], inplace=True)
        df['Regime_Short_Multiplier'].fillna(regime_multipliers_map['unknown']['SHORT'], inplace=True)

        return df

    def calculate_multi_timeframe_confirmation(self, df):
        """Векторизованное подтверждение на нескольких таймфреймах."""
        try:
            # Hourly
            hourly = df['Close'].resample('1H').last() # Берем последнюю цену закрытия часа
            hourly_ema_fast = hourly.ewm(span=self.params['hourly_ema_fast'], adjust=False).mean()
            hourly_ema_slow = hourly.ewm(span=self.params['hourly_ema_slow'], adjust=False).mean()
            # ИСПРАВЛЕНО: Используем .ffill() и .fillna(False)
            df['Hourly_Bullish'] = (hourly_ema_fast > hourly_ema_slow).reindex(df.index).ffill().fillna(False) # Используем ffill

            # 4-Hourly
            four_hour = df['Close'].resample('4H').last()
            four_hour_ema_fast = four_hour.ewm(span=self.params['four_hour_ema_fast'], adjust=False).mean()
            four_hour_ema_slow = four_hour.ewm(span=self.params['four_hour_ema_slow'], adjust=False).mean()
            # ИСПРАВЛЕНО: Используем .ffill() и .fillna(False)
            df['4H_Bullish'] = (four_hour_ema_fast > four_hour_ema_slow).reindex(df.index).ffill().fillna(False) # Используем ffill

            # Combined Strength
            df['MTF_Bull_Strength'] = (df['Hourly_Bullish'].astype(int) +
                                       df['4H_Bullish'].astype(int) +
                                       df['Higher_TF_Bullish'].astype(int)) / 3.0
            df['MTF_Bear_Strength'] = ( (~df['Hourly_Bullish']).astype(int) +
                                        (~df['4H_Bullish']).astype(int) +
                                         df['Higher_TF_Bearish'].astype(int)) / 3.0

        except Exception as e:
            print(f"Warning: Error in multi-timeframe calculation: {e}. Using daily trend only.")
            df['Hourly_Bullish'] = False
            df['4H_Bullish'] = False
            df['MTF_Bull_Strength'] = df['Higher_TF_Bullish'].astype(float)
            df['MTF_Bear_Strength'] = df['Higher_TF_Bearish'].astype(float)

        return df

    def calculate_mean_reversion_signals(self, df):
        """Векторизованный расчет сигналов возврата к среднему."""
        lookback = self.params['mean_reversion_lookback']
        threshold = self.params['mean_reversion_threshold']

        df['Close_SMA20'] = df['Close'].rolling(window=lookback).mean()
        df['Price_Deviation'] = df['Close'] - df['Close_SMA20']
        df['Price_Deviation_Std'] = df['Price_Deviation'].rolling(window=lookback).std()

        # Обработка деления на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Z_Score'] = (df['Price_Deviation'] / df['Price_Deviation_Std']).fillna(0)

        df['Stat_Overbought'] = df['Z_Score'] > threshold
        df['Stat_Oversold'] = df['Z_Score'] < -threshold

        # Сигналы пересечения порога Z-Score
        df['MR_Long_Signal'] = (df['Z_Score'] < -threshold) & (df['Z_Score'].shift(1) >= -threshold)
        df['MR_Short_Signal'] = (df['Z_Score'] > threshold) & (df['Z_Score'].shift(1) <= threshold)
        return df

    def identify_market_cycle_phase(self, df):
        """Определение фазы рыночного цикла (векторизованно)."""
        # Маски для фаз
        accumulation_mask = (
            (df['RSI'] < 40) &
            (df['Volume_Ratio'] > 1.2) &
            (df['Close'] < df['Daily_EMA50']) &
            (df['Close'] > df['Close'].shift(10)) # Краткосрочный рост
        )
        markup_mask = (
            df['Bullish_Trend'] &
            df['Higher_TF_Bullish'] &
            (df['Volume_Ratio'] > 1.0)
        )
        distribution_mask = (
            (df['RSI'] > 60) &
            (df['Volume_Ratio'] > 1.2) &
            (df['Close'] > df['Daily_EMA50']) &
            (df['Close'] < df['Close'].shift(10)) # Краткосрочное падение
        )
        markdown_mask = (
            df['Bearish_Trend'] &
            df['Higher_TF_Bearish'] &
            (df['Volume_Ratio'] > 1.0)
        )

        conditions = [accumulation_mask, markup_mask, distribution_mask, markdown_mask]
        choices = ['Accumulation', 'Markup', 'Distribution', 'Markdown']
        df['Cycle_Phase'] = np.select(conditions, choices, default='Unknown')

        # Веса для фаз
        phase_weights = {
            'Accumulation': {'LONG': 1.3, 'SHORT': 0.7},
            'Markup': {'LONG': 1.5, 'SHORT': 0.5},
            'Distribution': {'LONG': 0.7, 'SHORT': 1.3},
            'Markdown': {'LONG': 0.5, 'SHORT': 1.5},
            'Unknown': {'LONG': 1.0, 'SHORT': 1.0}
        }
        df['Long_Phase_Weight'] = df['Cycle_Phase'].map(lambda x: phase_weights.get(x, phase_weights['Unknown'])['LONG'])
        df['Short_Phase_Weight'] = df['Cycle_Phase'].map(lambda x: phase_weights.get(x, phase_weights['Unknown'])['SHORT'])

        # Заполняем возможные пропуски
        df['Long_Phase_Weight'].fillna(phase_weights['Unknown']['LONG'], inplace=True)
        df['Short_Phase_Weight'].fillna(phase_weights['Unknown']['SHORT'], inplace=True)
        return df

    def calculate_market_health(self, df):
        """Расчет индекса здоровья рынка (0-100)."""
        # Trend Health (20 points)
        df['Trend_Health'] = (df['Close'] > df['Daily_EMA50']).astype(int) * 20

        # Volatility Health (20 points, lower volatility is better unless extremely low)
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_ratio = df['ATR'] / df['ATR'].rolling(100).mean()
        atr_ratio.fillna(1.0, inplace=True) # Заполняем NaN, возникшие из-за rolling
        # Штраф за высокую волатильность, бонус за низкую, но не слишком
        df['Volatility_Health'] = np.clip(20 - (atr_ratio - 1).clip(0, 2) * 10 + (1 - atr_ratio).clip(0, 0.5)*10, 0, 20)

        # Volume Health (20 points, higher volume is better, capped)
        df['Volume_Health'] = df['Volume_Ratio'].clip(0.5, 2.0) * 10 # Даем очки за объем > 0.5 * MA

        # Breadth Health (20 points, agreement among indicators)
        indicators_bullish = (
            (df['RSI'] > 50).astype(int) +
            (df['MACD'] > 0).astype(int) +
            (df[f'EMA_{self.params["short_ema"]}'] > df[f'EMA_{self.params["long_ema"]}']).astype(int) +
            (df['Bullish_Structure']).astype(int) # Используем структуру рынка
        )
        df['Breadth_Health'] = (indicators_bullish / 4.0) * 20

        # Support/Resistance Health (20 points, price near BB middle is neutral)
        with np.errstate(divide='ignore', invalid='ignore'):
            bb_range = df['BB_Upper'] - df['BB_Lower']
            bb_position = np.where(bb_range == 0, 0.5, (df['Close'] - df['BB_Lower']) / bb_range)
        bb_position = pd.Series(bb_position, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        df['Support_Resistance_Health'] = (0.5 - abs(bb_position - 0.5)) * 2 * 20 # Чем ближе к середине, тем выше балл

        # Weighted Health Score
        weights = self.params
        df['Market_Health'] = np.clip(
            (df['Trend_Health'] * weights['health_trend_weight'] +
             df['Volatility_Health'] * weights['health_volatility_weight'] +
             df['Volume_Health'] * weights['health_volume_weight'] +
             df['Breadth_Health'] * weights['health_breadth_weight'] +
             df['Support_Resistance_Health'] * weights['health_sr_weight'])
            / (weights['health_trend_weight'] + weights['health_volatility_weight'] + # Нормализация на сумму весов
               weights['health_volume_weight'] + weights['health_breadth_weight'] +
               weights['health_sr_weight']),
            0, 100
        ).fillna(50) # Заполняем NaN значением 50 (нейтрально)

        # Health Bias
        df['Health_Long_Bias'] = df['Market_Health'] / 100.0
        df['Health_Short_Bias'] = 1.0 - df['Health_Long_Bias']
        return df

    def calculate_momentum_metrics(self, df):
        """Расчет метрик моментума и разворотов."""
        periods = self.params['momentum_roc_periods']
        roc_cols = []
        for period in periods:
            col_name = f'ROC_{period}'
            # Обработка деления на ноль
            shifted_close = df['Close'].shift(period)
            df[col_name] = np.where(
                shifted_close == 0, 0.0,
                (df['Close'] - shifted_close) / shifted_close * 100
            ).fillna(0.0)
            roc_cols.append(col_name)

        # Weighted momentum score (using sign and sqrt for normalization)
        momentum_components = [ (np.sign(df[col]) * np.sqrt(np.abs(df[col]))) for col in roc_cols ]
        # ИСПРАВЛЕНО: Используем временное имя, чтобы избежать перезаписи перед расчетом max_abs_score
        df['Momentum_Score_Raw'] = sum(momentum_components) / len(periods) if periods else 0

        # Scale score to approx -100 to 100
        with warnings.catch_warnings(): # Подавляем RuntimeWarning от nanargmax/nanargmin если все NaN
             warnings.simplefilter("ignore", category=RuntimeWarning)
             # ИСПРАВЛЕНО: Используем .ffill() вместо fillna(method='ffill')
             max_abs_score = df['Momentum_Score_Raw'].abs().rolling(window=100, min_periods=20).max().ffill().fillna(1.0) # Используем rolling max

        # ИСПРАВЛЕНО: Рассчитываем финальный Momentum_Score
        df['Momentum_Score'] = np.clip(df['Momentum_Score_Raw'] * (100 / max_abs_score.replace(0, 1.0)), -100, 100).fillna(0)
        df.drop(columns=['Momentum_Score_Raw'], inplace=True) # Удаляем временный столбец

        # Momentum Acceleration
        df['Mom_Acceleration'] = df['Momentum_Score'].diff(3).fillna(0)
        reversal_threshold = self.params['momentum_reversal_threshold']

        # Potential Reversal Signal
        df['Potential_Momentum_Reversal'] = (
            ((df['Momentum_Score'] > 80) & (df['Mom_Acceleration'] < -reversal_threshold)) |
            ((df['Momentum_Score'] < -80) & (df['Mom_Acceleration'] > reversal_threshold))
        )

        # Momentum Bias (more conservative range, e.g., 0.3 to 0.7)
        df['Momentum_Long_Bias'] = np.clip((df['Momentum_Score'] + 100) / 200, 0.3, 0.7)
        df['Momentum_Short_Bias'] = 1.0 - df['Momentum_Long_Bias']
        return df


    def adapt_to_market_conditions(self, df):
        """Применение адаптивных метрик для создания финального смещения."""
        # Финальное смещение (взвешенное)
        df['Final_Long_Bias'] = (
            df['Health_Long_Bias'] * 0.3 +
            df['Momentum_Long_Bias'] * 0.3 +
            df['Long_Phase_Weight'].clip(0.5, 1.5) * 0.2 + # Ограничиваем влияние фазы
            df['MTF_Bull_Strength'] * 0.2
        ).fillna(0.5) # Заполняем NaN средним

        df['Final_Short_Bias'] = (
            df['Health_Short_Bias'] * 0.3 +
            df['Momentum_Short_Bias'] * 0.3 +
            df['Short_Phase_Weight'].clip(0.5, 1.5) * 0.2 + # Ограничиваем влияние фазы
            df['MTF_Bear_Strength'] * 0.2
        ).fillna(0.5) # Заполняем NaN средним

        # Определение состояния рынка для выбора стратегии (Тренд/Флэт)
        signal_threshold = 0.60 # Немного понизил порог для большей активности
        df['Choppy_Market'] = (df['Final_Long_Bias'] < signal_threshold) & (df['Final_Short_Bias'] < signal_threshold)

        # Вес для сигналов Mean Reversion
        df['MR_Signal_Weight'] = np.where(df['Choppy_Market'], 1.5, 0.5)

        # Финальные сигналы (Комбинация смещения и MR в боковике)
        df['Balanced_Long_Signal_Condition'] = (df['Final_Long_Bias'] > signal_threshold) | \
                                               (df['Choppy_Market'] & df['MR_Long_Signal'])
        df['Balanced_Short_Signal_Condition'] = (df['Final_Short_Bias'] > signal_threshold) | \
                                                (df['Choppy_Market'] & df['MR_Short_Signal'])

        # Адаптивные множители для SL/TP
        df['Adaptive_Stop_Multiplier'] = np.where(
            df['Choppy_Market'],
            self.params['atr_multiplier_sl'] * 1.2, # Более широкий стоп в боковике
            self.params['atr_multiplier_sl'] * 0.9  # Более узкий стоп в тренде
        ).clip(MIN_ATR_MULTIPLIER_SL, MAX_ATR_MULTIPLIER_SL) # Ограничиваем

        df['Adaptive_TP_Multiplier'] = np.where(
            df['Choppy_Market'],
            self.params['atr_multiplier_tp'] * 0.8, # Более близкий тейк в боковике
            self.params['atr_multiplier_tp'] * 1.2  # Более дальний тейк в тренде
        ).clip(MIN_ATR_MULTIPLIER_TP, MAX_ATR_MULTIPLIER_TP) # Ограничиваем

        return df

class SignalGenerator:
    """Генерация базовых торговых сигналов на основе индикаторов."""
    def __init__(self, params: dict):
        self.params = params

    def get_signals(self, current: pd.Series, previous: pd.Series) -> dict:
        """Генерация и взвешивание сигналов для текущей свечи."""
        long_signals = []
        short_signals = []

        # --- Факторы для взвешивания ---
        volume_multiplier = np.clip(current['Volume_Ratio'] / self.params['volume_threshold'], 0.5, 2.0) if self.params['volume_threshold'] > 0 else 1.0
        health_factor_long = current.get('Health_Long_Bias', 0.5) # Безопасный доступ с default
        health_factor_short = current.get('Health_Short_Bias', 0.5)
        momentum_factor_long = current.get('Momentum_Long_Bias', 0.5)
        momentum_factor_short = current.get('Momentum_Short_Bias', 0.5)
        phase_factor_long = current.get('Long_Phase_Weight', 1.0)
        phase_factor_short = current.get('Short_Phase_Weight', 1.0)
        regime_multiplier_long = current.get('Regime_Long_Multiplier', 1.0)
        regime_multiplier_short = current.get('Regime_Short_Multiplier', 1.0)

        is_trending = current['Trend_Weight'] > 0.6
        is_ranging = current['Range_Weight'] > 0.6

        # --- Сигналы Трендовой Стратегии ---
        if is_trending:
            trend_w = current['Trend_Weight']
            # Long
            if (previous[f'EMA_{self.params["short_ema"]}'] <= previous[f'EMA_{self.params["long_ema"]}']) and \
               (current[f'EMA_{self.params["short_ema"]}'] > current[f'EMA_{self.params["long_ema"]}']):
                weight = trend_w * 1.2 * health_factor_long * momentum_factor_long * phase_factor_long * regime_multiplier_long
                long_signals.append(('EMA Cross', weight))

            if current['MACD_Bullish_Cross'] and current['MACD_Hist'] > 0:
                weight = trend_w * 1.3 * health_factor_long * momentum_factor_long * phase_factor_long * regime_multiplier_long
                long_signals.append(('MACD Cross', weight))

            if current['Bullish_Trend'] and not previous['Bullish_Trend'] and current['Plus_DI'] > current['Minus_DI'] * 1.1:
                 weight = trend_w * 1.5 * health_factor_long * momentum_factor_long * phase_factor_long * regime_multiplier_long
                 long_signals.append(('Strong Bull Trend Start', weight))

            # Short
            if (previous[f'EMA_{self.params["short_ema"]}'] >= previous[f'EMA_{self.params["long_ema"]}']) and \
               (current[f'EMA_{self.params["short_ema"]}'] < current[f'EMA_{self.params["long_ema"]}']):
                weight = trend_w * 1.2 * health_factor_short * momentum_factor_short * phase_factor_short * regime_multiplier_short
                short_signals.append(('EMA Cross', weight))

            if current['MACD_Bearish_Cross'] and current['MACD_Hist'] < 0:
                weight = trend_w * 1.3 * health_factor_short * momentum_factor_short * phase_factor_short * regime_multiplier_short
                short_signals.append(('MACD Cross', weight))

            if current['Bearish_Trend'] and not previous['Bearish_Trend'] and current['Minus_DI'] > current['Plus_DI'] * 1.1:
                 weight = trend_w * 1.5 * health_factor_short * momentum_factor_short * phase_factor_short * regime_multiplier_short
                 short_signals.append(('Strong Bear Trend Start', weight))

        # --- Сигналы Флэтовой Стратегии (Mean Reversion, Divergence) ---
        if is_ranging:
            range_w = current['Range_Weight']
            mr_w = current.get('MR_Signal_Weight', 1.0)

            # Long
            if current['RSI'] < self.params['rsi_oversold'] and current['Close'] < current['BB_Lower']:
                weight = range_w * 1.3 * health_factor_long * phase_factor_long * regime_multiplier_long
                long_signals.append(('RSI Oversold + BB Lower', weight))

            if current.get('Bullish_Divergence', False) and current['RSI'] < 45: # Используем .get для безопасности
                weight = range_w * 1.6 * health_factor_long * phase_factor_long * regime_multiplier_long
                long_signals.append(('Bullish Divergence', weight))

            if current.get('MR_Long_Signal', False) and current.get('Z_Score', 0) < -self.params['mean_reversion_threshold']:
                weight = range_w * 1.4 * mr_w * regime_multiplier_long
                long_signals.append(('Mean Reversion Long', weight))

            # Short
            if current['RSI'] > self.params['rsi_overbought'] and current['Close'] > current['BB_Upper']:
                weight = range_w * 1.3 * health_factor_short * phase_factor_short * regime_multiplier_short
                short_signals.append(('RSI Overbought + BB Upper', weight))

            if current.get('Bearish_Divergence', False) and current['RSI'] > 55:
                weight = range_w * 1.6 * health_factor_short * phase_factor_short * regime_multiplier_short
                short_signals.append(('Bearish Divergence', weight))

            if current.get('MR_Short_Signal', False) and current.get('Z_Score', 0) > self.params['mean_reversion_threshold']:
                weight = range_w * 1.4 * mr_w * regime_multiplier_short
                short_signals.append(('Mean Reversion Short', weight))

        # --- Общие Сигналы (например, Price Action) ---
        if current.get('Bullish_Engulfing', False):
            weight = 1.0 * health_factor_long * phase_factor_long * regime_multiplier_long
            long_signals.append(('Bullish Engulfing', weight))
        if current.get('Bearish_Engulfing', False):
            weight = 1.0 * health_factor_short * phase_factor_short * regime_multiplier_short
            short_signals.append(('Bearish Engulfing', weight))

        # --- Усиление сигналов по направлению старшего ТФ ---
        if current.get('Higher_TF_Bullish', False):
            long_signals = [(signal, weight * 1.2) for signal, weight in long_signals]
            short_signals = [(signal, weight * 0.8) for signal, weight in short_signals]
        elif current.get('Higher_TF_Bearish', False):
            long_signals = [(signal, weight * 0.8) for signal, weight in long_signals]
            short_signals = [(signal, weight * 1.2) for signal, weight in short_signals]

        # --- Применение общих фильтров и финальный расчет веса ---
        # Фильтр по объему
        long_signals = [(signal, weight * volume_multiplier) for signal, weight in long_signals]
        short_signals = [(signal, weight * volume_multiplier) for signal, weight in short_signals]

        # Фильтр по волатильности (штраф за экстремальную волатильность)
        vol_ratio = current['ATR'] / current['ATR_MA'] if current['ATR_MA'] > 0 else 1.0
        if vol_ratio > 2.0: # Сильно повышенная волатильность
            vol_filter_mult = 0.7
            long_signals = [(signal, weight * vol_filter_mult) for signal, weight in long_signals]
            short_signals = [(signal, weight * vol_filter_mult) for signal, weight in short_signals]

        # Фильтр по направлению свечи
        candle_dir_mult_long = 1.1 if current['Close'] > current['Open'] else 0.9
        candle_dir_mult_short = 1.1 if current['Close'] < current['Open'] else 0.9
        long_signals = [(signal, weight * candle_dir_mult_long) for signal, weight in long_signals]
        short_signals = [(signal, weight * candle_dir_mult_short) for signal, weight in short_signals]

        # Агрегация весов (усреднение)
        long_weight = sum(weight for _, weight in long_signals) / len(long_signals) if long_signals else 0
        short_weight = sum(weight for _, weight in short_signals) / len(short_signals) if short_signals else 0

        # Применение финального смещения (Final Bias) как множителя
        long_weight *= current.get('Final_Long_Bias', 0.5) * 2 # Умножаем на 2, т.к. bias [0, 1]
        short_weight *= current.get('Final_Short_Bias', 0.5) * 2

        return {
            'long_signals': long_signals,
            'short_signals': short_signals,
            'long_weight': long_weight,
            'short_weight': short_weight
        }

class RiskManager:
    """Управление риском, размером позиции, плечом, SL/TP, трейлингом."""
    def __init__(self, config: Config):
        self.config = config
        # Состояние для адаптивных параметров
        self.current_base_risk = config.base_risk_per_trade
        self.current_atr_multiplier_tp = config.params['atr_multiplier_tp']
        self.recent_long_win_rate = 0.5
        self.recent_short_win_rate = 0.5

    def calculate_kelly_criterion(self, win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
        """Расчет доли Келли."""
        if avg_loss_pct <= 0: return 0.0 # Избегаем деления на ноль
        win_loss_ratio = avg_win_pct / avg_loss_pct
        if win_loss_ratio <= 0: return 0.0 # Избегаем деления на ноль или отрицательный результат

        kelly_pct = win_rate - ((1.0 - win_rate) / win_loss_ratio)
        # Используем дробное Келли для снижения агрессивности
        return max(0.0, min(0.5, kelly_pct * KELLY_FRACTION)) # Ограничиваем сверху (например, 50%)

    def update_adaptive_params(self, trade_history: pd.DataFrame):
        """Динамическая корректировка параметров риска на основе недавних сделок."""
        if trade_history is None or len(trade_history) < 20:
            # Недостаточно данных для адаптации, используем базовые значения
            self.current_base_risk = self.config.base_risk_per_trade
            self.current_atr_multiplier_tp = self.config.params['atr_multiplier_tp']
            self.recent_long_win_rate = 0.5
            self.recent_short_win_rate = 0.5
            return

        recent_trades = trade_history.tail(20)
        win_rate = sum(1 for pnl in recent_trades['pnl'] if pnl > 0) / len(recent_trades)
        profit_sum = recent_trades.loc[recent_trades['pnl'] > 0, 'pnl'].sum()
        loss_sum = abs(recent_trades.loc[recent_trades['pnl'] <= 0, 'pnl'].sum())
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

        # Адаптация базового риска
        if profit_factor > 1.5 and win_rate > 0.55:
            self.current_base_risk = min(MAX_RISK_PER_TRADE, self.current_base_risk * 1.1)
        elif profit_factor < 1.1 or win_rate < 0.45:
            self.current_base_risk = max(MIN_RISK_PER_TRADE, self.current_base_risk * 0.9)

        # Адаптация множителя TP
        avg_win = recent_trades.loc[recent_trades['pnl'] > 0, 'pnl'].mean()
        avg_loss = abs(recent_trades.loc[recent_trades['pnl'] <= 0, 'pnl'].mean())
        if avg_win > 0 and avg_loss > 0:
            current_rr_ratio = avg_win / avg_loss
            if current_rr_ratio < 1.8: # Если средний R:R низкий, увеличиваем цель TP
                self.current_atr_multiplier_tp = min(MAX_ATR_MULTIPLIER_TP, self.current_atr_multiplier_tp * 1.05)
            elif current_rr_ratio > 3.5 and win_rate < 0.4: # Если R:R высокий, но винрейт низкий, уменьшаем цель TP
                self.current_atr_multiplier_tp = max(MIN_ATR_MULTIPLIER_TP, self.current_atr_multiplier_tp * 0.95)

        # Сохраняем недавний винрейт для лонгов/шортов
        long_trades = recent_trades[recent_trades['position'] == 'LONG']
        short_trades = recent_trades[recent_trades['position'] == 'SHORT']
        self.recent_long_win_rate = sum(1 for pnl in long_trades['pnl'] if pnl > 0) / len(long_trades) if len(long_trades) > 0 else 0.5
        self.recent_short_win_rate = sum(1 for pnl in short_trades['pnl'] if pnl > 0) / len(short_trades) if len(short_trades) > 0 else 0.5

    def get_adaptive_risk_per_trade(self, market_regime: str) -> dict:
        """Получение адаптивного риска для Long/Short на основе винрейта и режима."""
        long_adjustment = 1.0
        short_adjustment = 1.0

        # Коррекция на основе недавнего винрейта
        if self.recent_long_win_rate > 0.6: long_adjustment = 1.2
        elif self.recent_long_win_rate < 0.4: long_adjustment = 0.8
        if self.recent_short_win_rate > 0.6: short_adjustment = 1.2
        elif self.recent_short_win_rate < 0.4: short_adjustment = 0.8

        # Коррекция на основе рыночного режима (можно вынести в Config)
        regime_factors = {
            "strong_bull": {"LONG": 1.1, "SHORT": 0.7}, "strong_bear": {"LONG": 0.7, "SHORT": 1.1},
            "choppy_range": {"LONG": 0.8, "SHORT": 0.8}, "transition_to_bull": {"LONG": 1.0, "SHORT": 0.8},
            "transition_to_bear": {"LONG": 0.8, "SHORT": 1.0}, "mixed": {"LONG": 0.9, "SHORT": 0.9},
            "unknown": {"LONG": 0.7, "SHORT": 0.7}
        }
        factors = regime_factors.get(market_regime, regime_factors['unknown'])

        risk_long = np.clip(self.current_base_risk * long_adjustment * factors["LONG"], MIN_RISK_PER_TRADE, MAX_RISK_PER_TRADE)
        risk_short = np.clip(self.current_base_risk * short_adjustment * factors["SHORT"], MIN_RISK_PER_TRADE, MAX_RISK_PER_TRADE)

        return {"LONG": risk_long, "SHORT": risk_short}

    def calculate_optimal_leverage(self, current_candle: pd.Series, trade_direction: str) -> float:
        """Расчет оптимального плеча на основе волатильности, тренда, режима, здоровья."""
        base_leverage = 1.5 # Стартуем с более консервативного плеча
        max_leverage = self.config.max_leverage

        # Волатильность (ATR / ATR_MA)
        atr_ma = current_candle['ATR_MA'] if pd.notna(current_candle['ATR_MA']) and current_candle['ATR_MA'] > 0 else current_candle['ATR']
        vol_ratio = current_candle['ATR'] / atr_ma if atr_ma > 0 else 1.0
        vol_adjustment = np.clip(1.0 / vol_ratio, 0.5, 1.5) # Обратная зависимость: выше волатильность -> ниже плечо

        # Сила тренда (ADX)
        trend_adjustment = 1.0
        if current_candle['ADX'] > 30: # Сильный тренд
            if (trade_direction == 'LONG' and current_candle['Plus_DI'] > current_candle['Minus_DI']) or \
               (trade_direction == 'SHORT' and current_candle['Minus_DI'] > current_candle['Plus_DI']):
                trend_adjustment = 1.2 # Увеличиваем плечо по тренду
            else:
                trend_adjustment = 0.8 # Уменьшаем против сильного тренда
        elif current_candle['ADX'] < 20: # Слабый тренд / флэт
             trend_adjustment = 0.9

        # Рыночный режим
        regime = current_candle.get('Market_Regime', 'unknown')
        regime_adjustment = 1.0
        if regime == 'strong_bull' and trade_direction == 'LONG': regime_adjustment = 1.2
        elif regime == 'strong_bear' and trade_direction == 'SHORT': regime_adjustment = 1.2
        elif regime == 'choppy_range': regime_adjustment = 0.8 # Снижаем плечо в боковике
        elif regime == 'transition_to_bull' and trade_direction == 'SHORT': regime_adjustment = 0.9
        elif regime == 'transition_to_bear' and trade_direction == 'LONG': regime_adjustment = 0.9

        # Здоровье рынка
        health = current_candle.get('Market_Health', 50) # Нейтральное значение по умолчанию
        # Линейная интерполяция от 0.8 до 1.2 в зависимости от здоровья
        health_adjustment = 0.8 + (health / 100.0) * 0.4 if trade_direction == 'LONG' else 1.2 - (health / 100.0) * 0.4

        # Финальный расчет плеча
        optimal_leverage = base_leverage * vol_adjustment * trend_adjustment * regime_adjustment * health_adjustment
        # Ограничиваем плечо снизу (1x) и сверху (max_leverage)
        return np.clip(optimal_leverage, 1.0, max_leverage)

    def calculate_dynamic_exit_levels(self, position_type: str, entry_price: float, current_candle: pd.Series, trade_age_hours: float = 0) -> dict:
        """Расчет динамических уровней Stop Loss и Take Profit."""
        if entry_price <= 0: # Защита от некорректной цены входа
            return {'stop_loss': 0, 'take_profit': 0}

        # Адаптивные множители ATR
        sl_multiplier = current_candle.get('Adaptive_Stop_Multiplier', self.config.params['atr_multiplier_sl'])
        tp_multiplier = current_candle.get('Adaptive_TP_Multiplier', self.current_atr_multiplier_tp) # Используем адаптированный TP
        atr_value = current_candle['ATR'] if pd.notna(current_candle['ATR']) else 0

        if atr_value <= 0: # Если ATR некорректен, используем % от цены
            atr_value = entry_price * 0.01 # Например, 1% от цены

        # Коррекция множителей на основе возраста сделки (сужение стопа со временем)
        if trade_age_hours > 4:
             age_factor = min(2.0, 1.0 + (trade_age_hours - 4) / 24) # Сужаем стоп до 2 раз быстрее за сутки
             sl_multiplier = max(MIN_ATR_MULTIPLIER_SL * 0.7, sl_multiplier / age_factor) # Не слишком узко

        # Базовые SL/TP на основе ATR
        if position_type == 'LONG':
            stop_loss = entry_price - atr_value * sl_multiplier
            take_profit = entry_price + atr_value * tp_multiplier
        else: # SHORT
            stop_loss = entry_price + atr_value * sl_multiplier
            take_profit = entry_price - atr_value * tp_multiplier

        # Коррекция TP во флэте (ограничение по BB)
        if current_candle.get('Range_Weight', 0) > 0.7:
            if position_type == 'LONG' and pd.notna(current_candle['BB_Upper']) and current_candle['BB_Upper'] < take_profit:
                take_profit = current_candle['BB_Upper'] * 0.998 # Чуть ниже верхней границы
            elif position_type == 'SHORT' and pd.notna(current_candle['BB_Lower']) and current_candle['BB_Lower'] > take_profit:
                take_profit = current_candle['BB_Lower'] * 1.002 # Чуть выше нижней границы

        # Гарантируем минимальное соотношение R:R (например, 1.5)
        min_rr_ratio = 1.5
        if position_type == 'LONG':
            risk = entry_price - stop_loss
            if risk > 0:
                 reward = take_profit - entry_price
                 if reward / risk < min_rr_ratio:
                     take_profit = entry_price + risk * min_rr_ratio
            else: # Если стоп некорректен
                stop_loss = entry_price * 0.99 # Резервный стоп 1%
                take_profit = entry_price * (1 + 0.01 * min_rr_ratio)
        else: # SHORT
            risk = stop_loss - entry_price
            if risk > 0:
                reward = entry_price - take_profit
                if reward / risk < min_rr_ratio:
                    take_profit = entry_price - risk * min_rr_ratio
            else: # Если стоп некорректен
                stop_loss = entry_price * 1.01 # Резервный стоп 1%
                take_profit = entry_price * (1 - 0.01 * min_rr_ratio)

        # Защита от нулевых или отрицательных цен
        stop_loss = max(0.01, stop_loss)
        take_profit = max(0.01, take_profit)

        return {'stop_loss': stop_loss, 'take_profit': take_profit}

    def apply_trailing_stop(self, position_type: str, entry_price: float, stop_loss: float, current_price: float, highest_price_in_trade: float, lowest_price_in_trade: float) -> float:
        """Применение простого процентного трейлинг-стопа от пика."""
        if entry_price <= 0: return stop_loss

        new_stop = stop_loss # По умолчанию не меняем стоп

        if position_type == 'LONG':
            unrealized_gain = highest_price_in_trade - entry_price
            if unrealized_gain > 0:
                # Пример: трейлить на 30% от достигнутой прибыли после достижения 2% профита
                profit_threshold = entry_price * 0.02
                trail_percentage = 0.30
                if highest_price_in_trade > entry_price + profit_threshold:
                    potential_new_stop = highest_price_in_trade - (unrealized_gain * trail_percentage)
                    # Передвигаем стоп только если он выше текущего и выше точки безубытка
                    new_stop = max(stop_loss, entry_price, potential_new_stop)
        else: # SHORT
            unrealized_gain = entry_price - lowest_price_in_trade
            if unrealized_gain > 0:
                profit_threshold = entry_price * 0.02
                trail_percentage = 0.30
                if lowest_price_in_trade < entry_price - profit_threshold:
                    potential_new_stop = lowest_price_in_trade + (unrealized_gain * trail_percentage)
                    # Передвигаем стоп только если он ниже текущего и ниже точки безубытка
                    new_stop = min(stop_loss, entry_price, potential_new_stop)

        return max(0.01, new_stop) # Защита от нулевого стопа

    def calculate_position_size(self, balance: float, risk_per_trade: float, entry_price: float, stop_loss_price: float, leverage: float) -> float:
        """Расчет размера позиции на основе риска, стопа и плеча."""
        if entry_price <= 0 or stop_loss_price <= 0 or entry_price == stop_loss_price:
            print("Warning: Invalid entry or stop price for position sizing. Returning 0.")
            return 0.0

        risk_amount_usd = balance * risk_per_trade
        price_risk_per_unit = abs(entry_price - stop_loss_price)

        # Размер позиции в базовом активе (например, BTC)
        # units = risk_amount_usd / price_risk_per_unit
        # Размер позиции в USD (стоимость контрактов)
        # position_size_usd = units * entry_price

        # Прямой расчет размера позиции в USD через процентный риск на капитал
        price_risk_pct = abs(entry_price - stop_loss_price) / entry_price
        if price_risk_pct * leverage <= 0:
             print("Warning: Zero or negative leveraged risk percentage. Returning 0.")
             return 0.0

        # Максимальный размер позиции исходя из риска
        position_size_usd = risk_amount_usd / (price_risk_pct * leverage)

        # Ограничение максимальным плечом от баланса
        max_position_by_leverage = balance * leverage
        position_size_usd = min(position_size_usd, max_position_by_leverage)

        # Минимальный размер позиции (например, 10 USD)
        min_position_size = 10.0
        if position_size_usd < min_position_size and balance > min_position_size * 2: # Только если баланс позволяет
             position_size_usd = min_position_size

        # Не можем открыть позицию больше, чем позволяет баланс с плечом
        position_size_usd = min(position_size_usd, balance * leverage)

        return max(0.0, position_size_usd) # Не может быть отрицательным

class Backtester:
    """Основной класс для выполнения бэктеста."""
    def __init__(self, data: pd.DataFrame, config: Config, risk_manager: RiskManager):
        self.data = data
        self.config = config
        self.risk_manager = risk_manager
        self.signal_generator = SignalGenerator(config.params) # Используем параметры из config
        self.trade_history = []
        self.backtest_results = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Запуск итерационного бэктеста."""
        print("Running backtest...")
        if self.data is None or self.data.empty:
            raise ValueError("Data is not loaded or empty.")

        balance = self.config.initial_balance
        equity = balance
        position = 0 # 0: нет позиции, 1: LONG, -1: SHORT
        entry_price = 0.0
        position_size_usd = 0.0 # Размер позиции в USD
        position_size_units = 0.0 # Размер позиции в базовом активе (BTC)
        entry_timestamp = None
        last_trade_close_timestamp = None
        pyramid_entries = 0
        stop_loss_price = 0.0
        take_profit_price = 0.0
        current_leverage = 1.0

        # Для трейлинг стопа
        highest_price_in_trade = 0.0
        lowest_price_in_trade = float('inf')

        results_list = [] # Список для хранения результатов по каждой свече

        for i in range(1, len(self.data)): # Начинаем со второй свечи для доступа к previous
            current_candle = self.data.iloc[i]
            previous_candle = self.data.iloc[i-1]
            current_timestamp = self.data.index[i]

            current_unrealized_pnl_pct = 0.0 # Инициализация

            # Обновление эквити в открытой позиции
            if position != 0:
                if entry_price > 0:
                    if position == 1: # LONG
                        unrealized_pnl_pct = (current_candle['Close'] / entry_price) - 1.0
                    else: # SHORT
                        unrealized_pnl_pct = 1.0 - (current_candle['Close'] / entry_price)
                    current_unrealized_pnl_pct = unrealized_pnl_pct # Сохраняем для _check_exit_signals
                    # PnL считается от размера позиции в USD
                    unrealized_pnl_usd = unrealized_pnl_pct * position_size_usd
                    equity = balance + unrealized_pnl_usd
                else: equity = balance # Если цена входа некорректна, считаем эквити равным балансу
            else:
                equity = balance # Если нет позиции, эквити равно балансу

            # --- Логика выхода из позиции ---
            if position != 0 and entry_timestamp is not None:
                trade_age_hours = (current_timestamp - entry_timestamp).total_seconds() / 3600

                # Обновляем highest/lowest price
                highest_price_in_trade = max(highest_price_in_trade, current_candle['High'])
                lowest_price_in_trade = min(lowest_price_in_trade, current_candle['Low'])

                # Применяем трейлинг стоп
                stop_loss_price = self.risk_manager.apply_trailing_stop(
                    'LONG' if position == 1 else 'SHORT',
                    entry_price, stop_loss_price, current_candle['Close'],
                    highest_price_in_trade, lowest_price_in_trade
                )

                exit_price = 0.0
                exit_reason = None

                # Проверка Stop Loss / Take Profit
                if position == 1: # LONG
                    if current_candle['Low'] <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_reason = 'Stop Loss'
                    elif current_candle['High'] >= take_profit_price:
                        exit_price = take_profit_price
                        exit_reason = 'Take Profit'
                elif position == -1: # SHORT
                    if current_candle['High'] >= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_reason = 'Stop Loss'
                    elif current_candle['Low'] <= take_profit_price:
                        exit_price = take_profit_price
                        exit_reason = 'Take Profit'

                # Проверка сигналов на выход (если SL/TP не сработал)
                if exit_reason is None and trade_age_hours > 2: # Даем сделке немного подышать
                   exit_signals_list = self._check_exit_signals(position, current_candle, previous_candle, current_unrealized_pnl_pct)
                   if exit_signals_list:
                       exit_price = current_candle['Close'] # Выход по рынку
                       exit_reason = ', '.join(exit_signals_list)

                # Если есть причина для выхода
                if exit_reason:
                    # --- Расчет PnL с учетом комиссии и проскальзывания ---
                    exit_price_with_slippage = exit_price * (1 - self.config.slippage_pct * position) # Ухудшаем цену выхода

                    if position == 1: # LONG
                        pnl_pct = (exit_price_with_slippage / entry_price) - 1.0
                    else: # SHORT
                        pnl_pct = 1.0 - (exit_price_with_slippage / entry_price)

                    gross_pnl_usd = pnl_pct * position_size_usd

                    # Комиссия (на вход и выход, от полной стоимости позиции с плечом)
                    entry_commission = self.config.commission_pct * position_size_usd
                    exit_commission = self.config.commission_pct * position_size_usd * (exit_price_with_slippage / entry_price) # Примерный расчет на стоимость при выходе
                    total_commission = entry_commission + exit_commission

                    net_pnl_usd = gross_pnl_usd - total_commission
                    balance += net_pnl_usd
                    equity = balance # После закрытия сделки эквити равно балансу

                    # Запись истории сделки
                    self.trade_history.append({
                        'entry_date': entry_timestamp, 'entry_price': entry_price,
                        'exit_date': current_timestamp, 'exit_price': exit_price_with_slippage,
                        'position': 'LONG' if position == 1 else 'SHORT',
                        'pnl': net_pnl_usd, 'balance': balance, 'reason': exit_reason,
                        'pyramid_entries': pyramid_entries, 'trade_duration': trade_age_hours,
                        'leverage': current_leverage, 'commission': total_commission,
                        'position_size_usd': position_size_usd
                    })

                    # Сброс состояния позиции
                    position = 0
                    entry_price = 0.0
                    position_size_usd = 0.0
                    position_size_units = 0.0
                    entry_timestamp = None
                    last_trade_close_timestamp = current_timestamp
                    pyramid_entries = 0
                    highest_price_in_trade = 0.0
                    lowest_price_in_trade = float('inf')

                    # Динамическая корректировка параметров риска после закрытия сделки
                    if len(self.trade_history) >= 20:
                        self.risk_manager.update_adaptive_params(pd.DataFrame(self.trade_history))

            # --- Логика входа в позицию ---
            if position == 0:
                # Проверка минимального интервала между сделками
                if last_trade_close_timestamp and (current_timestamp - last_trade_close_timestamp) < timedelta(hours=self.config.min_trades_interval_hours):
                    results_list.append({ # Записываем состояние даже если не торгуем
                         'timestamp': current_timestamp, 'close': current_candle['Close'],
                         'balance': balance, 'equity': equity, 'position': position,
                         'entry_price': np.nan, 'position_size_usd': 0,
                         'stop_loss': np.nan, 'take_profit': np.nan,
                         'market_regime': current_candle.get('Market_Regime', 'unknown'),
                         'market_health': current_candle.get('Market_Health', np.nan),
                         'long_bias': current_candle.get('Final_Long_Bias', np.nan),
                         'short_bias': current_candle.get('Final_Short_Bias', np.nan),
                         'leverage': 1.0
                     })
                    continue

                # Проверка активных торговых часов (если заданы)
                # hour_optimal = self._is_optimal_trading_hour(current_timestamp)
                # day_optimal = self._is_optimal_trading_day(current_timestamp)
                # if not (hour_optimal and day_optimal):
                #      continue

                # Получение сигналов
                signals = self.signal_generator.get_signals(current_candle, previous_candle)
                long_weight = signals['long_weight']
                short_weight = signals['short_weight']
                min_signal_threshold = 0.65 # Порог для входа

                trade_direction = 0 # 1 for LONG, -1 for SHORT

                # Выбор направления сделки
                if long_weight > short_weight and long_weight >= min_signal_threshold:
                    trade_direction = 1
                elif short_weight > long_weight and short_weight >= min_signal_threshold:
                    trade_direction = -1

                # Если есть сигнал на вход
                if trade_direction != 0:
                    entry_timestamp = current_timestamp
                    pos_type = 'LONG' if trade_direction == 1 else 'SHORT'

                    # Цена входа с учетом проскальзывания
                    entry_price = current_candle['Close'] * (1 + self.config.slippage_pct * trade_direction) # Ухудшаем цену входа

                    if entry_price <= 0:
                         results_list.append({ # Записываем состояние
                              'timestamp': current_timestamp, 'close': current_candle['Close'],
                              'balance': balance, 'equity': equity, 'position': position,
                              'entry_price': np.nan, 'position_size_usd': 0,
                              'stop_loss': np.nan, 'take_profit': np.nan,
                              'market_regime': current_candle.get('Market_Regime', 'unknown'),
                              'market_health': current_candle.get('Market_Health', np.nan),
                              'long_bias': current_candle.get('Final_Long_Bias', np.nan),
                              'short_bias': current_candle.get('Final_Short_Bias', np.nan),
                              'leverage': 1.0
                          })
                         continue # Не входим, если цена некорректна

                    # Расчет плеча, SL/TP
                    current_leverage = self.risk_manager.calculate_optimal_leverage(current_candle, pos_type)
                    exit_levels = self.risk_manager.calculate_dynamic_exit_levels(pos_type, entry_price, current_candle)
                    stop_loss_price = exit_levels['stop_loss']
                    take_profit_price = exit_levels['take_profit']

                    if stop_loss_price <= 0 or take_profit_price <= 0:
                         results_list.append({ # Записываем состояние
                              'timestamp': current_timestamp, 'close': current_candle['Close'],
                              'balance': balance, 'equity': equity, 'position': position,
                              'entry_price': np.nan, 'position_size_usd': 0,
                              'stop_loss': np.nan, 'take_profit': np.nan,
                              'market_regime': current_candle.get('Market_Regime', 'unknown'),
                              'market_health': current_candle.get('Market_Health', np.nan),
                              'long_bias': current_candle.get('Final_Long_Bias', np.nan),
                              'short_bias': current_candle.get('Final_Short_Bias', np.nan),
                              'leverage': 1.0
                          })
                         continue # Не входим, если SL/TP некорректны

                    # Адаптивный риск
                    market_regime = current_candle.get('Market_Regime', 'unknown')
                    adaptive_risk_map = self.risk_manager.get_adaptive_risk_per_trade(market_regime)
                    risk_per_trade = adaptive_risk_map[pos_type]

                    # Размер позиции
                    position_size_usd = self.risk_manager.calculate_position_size(
                        balance, risk_per_trade, entry_price, stop_loss_price, current_leverage
                    )

                    if position_size_usd <= 0:
                        results_list.append({ # Записываем состояние
                             'timestamp': current_timestamp, 'close': current_candle['Close'],
                             'balance': balance, 'equity': equity, 'position': position,
                             'entry_price': np.nan, 'position_size_usd': 0,
                             'stop_loss': np.nan, 'take_profit': np.nan,
                             'market_regime': current_candle.get('Market_Regime', 'unknown'),
                             'market_health': current_candle.get('Market_Health', np.nan),
                             'long_bias': current_candle.get('Final_Long_Bias', np.nan),
                             'short_bias': current_candle.get('Final_Short_Bias', np.nan),
                             'leverage': 1.0
                         })
                        continue # Не входим, если размер позиции нулевой

                    position_size_units = position_size_usd / entry_price
                    position = trade_direction
                    pyramid_entries = 0 # Начальная позиция
                    highest_price_in_trade = current_candle['High']
                    lowest_price_in_trade = current_candle['Low']

                    # Запись об открытии (можно добавить в history для логгирования)
                    # print(f"{current_timestamp}: Enter {pos_type} @ {entry_price:.2f}, Size: {position_size_usd:.2f} USD, SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}, Lev: {current_leverage:.2f}x")

            # --- Логика Пирамидинга (упрощенная) ---
            # Пирамидинг временно отключен для упрощения, можно добавить позже
            # ...

            # Запись результатов для текущей свечи
            results_list.append({
                'timestamp': current_timestamp, 'close': current_candle['Close'],
                'balance': balance, 'equity': equity, 'position': position,
                'entry_price': entry_price if position != 0 else np.nan,
                'position_size_usd': position_size_usd if position != 0 else 0,
                'stop_loss': stop_loss_price if position != 0 else np.nan,
                'take_profit': take_profit_price if position != 0 else np.nan,
                'market_regime': current_candle.get('Market_Regime', 'unknown'),
                'market_health': current_candle.get('Market_Health', np.nan),
                'long_bias': current_candle.get('Final_Long_Bias', np.nan),
                'short_bias': current_candle.get('Final_Short_Bias', np.nan),
                'leverage': current_leverage if position != 0 else 1.0
            })

        # --- Обработка конца бэктеста ---
        if position != 0 and entry_timestamp is not None and entry_price > 0: # Добавлена проверка entry_price
            # Закрываем оставшуюся позицию по последней цене закрытия
            last_candle = self.data.iloc[-1]
            last_timestamp = self.data.index[-1]
            exit_price = last_candle['Close']
            exit_price_with_slippage = exit_price * (1 - self.config.slippage_pct * position)
            exit_reason = 'End of Backtest'
            trade_age_hours = (last_timestamp - entry_timestamp).total_seconds() / 3600

            if position == 1: # LONG
                pnl_pct = (exit_price_with_slippage / entry_price) - 1.0
            else: # SHORT
                pnl_pct = 1.0 - (exit_price_with_slippage / entry_price)

            gross_pnl_usd = pnl_pct * position_size_usd
            entry_commission = self.config.commission_pct * position_size_usd
            exit_commission = self.config.commission_pct * position_size_usd * (exit_price_with_slippage / entry_price)
            total_commission = entry_commission + exit_commission
            net_pnl_usd = gross_pnl_usd - total_commission
            balance += net_pnl_usd

            self.trade_history.append({
                'entry_date': entry_timestamp, 'entry_price': entry_price,
                'exit_date': last_timestamp, 'exit_price': exit_price_with_slippage,
                'position': 'LONG' if position == 1 else 'SHORT',
                'pnl': net_pnl_usd, 'balance': balance, 'reason': exit_reason,
                'pyramid_entries': pyramid_entries, 'trade_duration': trade_age_hours,
                'leverage': current_leverage, 'commission': total_commission,
                'position_size_usd': position_size_usd
            })
        elif results_list: # Если последняя строка не добавлена из-за закрытия позиции
              # Добавляем последнюю строку, если она еще не была добавлена
              last_saved_timestamp = results_list[-1]['timestamp']
              last_data_timestamp = self.data.index[-1]
              if last_saved_timestamp < last_data_timestamp:
                   last_candle = self.data.iloc[-1]
                   results_list.append({
                       'timestamp': last_data_timestamp, 'close': last_candle['Close'],
                       'balance': balance, 'equity': balance, 'position': 0, # Закрыто
                       'entry_price': np.nan, 'position_size_usd': 0,
                       'stop_loss': np.nan, 'take_profit': np.nan,
                       'market_regime': last_candle.get('Market_Regime', 'unknown'),
                       'market_health': last_candle.get('Market_Health', np.nan),
                       'long_bias': last_candle.get('Final_Long_Bias', np.nan),
                       'short_bias': last_candle.get('Final_Short_Bias', np.nan),
                       'leverage': 1.0
                   })

        if not results_list:
            print("Warning: No results generated during backtest.")
            self.backtest_results = pd.DataFrame() # Пустой DataFrame
        else:
            self.backtest_results = pd.DataFrame(results_list).set_index('timestamp')

        trade_df = pd.DataFrame(self.trade_history)

        print(f"Backtest completed. Total trades: {len(trade_df)}")
        return self.backtest_results, trade_df

    def _check_exit_signals(self, position: int, current: pd.Series, previous: pd.Series, unrealized_pnl_pct: float) -> list:
        """Проверка условий для принудительного выхода из позиции."""
        exit_signals = []
        short_ema = f'EMA_{self.config.params["short_ema"]}'
        long_ema = f'EMA_{self.config.params["long_ema"]}'
        min_profit_pct_for_exit = 0.01 # Минимальный профит для выхода по слабому сигналу

        if position == 1: # Exit LONG
            # Сигналы разворота
            if previous[short_ema] >= previous[long_ema] and current[short_ema] < current[long_ema]:
                exit_signals.append('EMA Bear Cross')
            if current['MACD_Bearish_Cross']:
                exit_signals.append('MACD Bear Cross')
            if current['Bearish_Trend'] and not previous['Bearish_Trend']:
                exit_signals.append('Bear Trend Start')
            if current.get('Bearish_Divergence', False) and current['RSI'] > 60:
                 exit_signals.append('Bearish Divergence')
            if current.get('Bearish_Engulfing', False) and unrealized_pnl_pct > min_profit_pct_for_exit:
                 exit_signals.append('Bearish Engulfing')
            # ИСПРАВЛЕНО: Используем current.get() для Market_Regime
            if current.get('Market_Regime', 'unknown') == 'transition_to_bear' and unrealized_pnl_pct > min_profit_pct_for_exit:
                 exit_signals.append('Regime Change Bear')
            if current.get('Higher_TF_Bearish', False) and not previous.get('Higher_TF_Bearish', False) and unrealized_pnl_pct > min_profit_pct_for_exit:
                 exit_signals.append('Higher TF Bear')
            if current.get('Potential_Momentum_Reversal', False) and current.get('Momentum_Score', 0) > 70 and unrealized_pnl_pct > 0.03:
                 exit_signals.append('Momentum Reversal')
            if current.get('Final_Short_Bias', 0) > 0.7 and unrealized_pnl_pct > 0.02:
                 exit_signals.append('Strong Short Bias')
            # Условия перекупленности во флэте
            if current.get('Choppy_Market', False):
                 if current['RSI'] > self.config.params['rsi_overbought'] + 5: # Более строгий порог
                     exit_signals.append('RSI Extreme Overbought')
                 if current['Close'] > current['BB_Upper'] * 1.005: # Выход за BB
                     exit_signals.append('BB Upper Break')
                 if current.get('Stat_Overbought', False) and unrealized_pnl_pct > min_profit_pct_for_exit:
                     exit_signals.append('Stat Overbought')

        elif position == -1: # Exit SHORT
            # Сигналы разворота
            if previous[short_ema] <= previous[long_ema] and current[short_ema] > current[long_ema]:
                exit_signals.append('EMA Bull Cross')
            if current['MACD_Bullish_Cross']:
                exit_signals.append('MACD Bull Cross')
            if current['Bullish_Trend'] and not previous['Bullish_Trend']:
                exit_signals.append('Bull Trend Start')
            if current.get('Bullish_Divergence', False) and current['RSI'] < 40:
                exit_signals.append('Bullish Divergence')
            if current.get('Bullish_Engulfing', False) and unrealized_pnl_pct > min_profit_pct_for_exit:
                exit_signals.append('Bullish Engulfing')
            # ИСПРАВЛЕНО: Используем current.get() для Market_Regime
            if current.get('Market_Regime', 'unknown') == 'transition_to_bull' and unrealized_pnl_pct > min_profit_pct_for_exit:
                exit_signals.append('Regime Change Bull')
            if current.get('Higher_TF_Bullish', False) and not previous.get('Higher_TF_Bullish', False) and unrealized_pnl_pct > min_profit_pct_for_exit:
                 exit_signals.append('Higher TF Bull')
            if current.get('Potential_Momentum_Reversal', False) and current.get('Momentum_Score', 0) < -70 and unrealized_pnl_pct > 0.03:
                exit_signals.append('Momentum Reversal')
            if current.get('Final_Long_Bias', 0) > 0.7 and unrealized_pnl_pct > 0.02:
                exit_signals.append('Strong Long Bias')
             # Условия перепроданности во флэте
            if current.get('Choppy_Market', False):
                 if current['RSI'] < self.config.params['rsi_oversold'] - 5:
                     exit_signals.append('RSI Extreme Oversold')
                 if current['Close'] < current['BB_Lower'] * 0.995:
                     exit_signals.append('BB Lower Break')
                 if current.get('Stat_Oversold', False) and unrealized_pnl_pct > min_profit_pct_for_exit:
                     exit_signals.append('Stat Oversold')

        # Общий выход при резком росте волатильности в профите
        vol_ratio = current['ATR'] / current['ATR_MA'] if current['ATR_MA'] > 0 else 1.0
        if vol_ratio > 2.5 and unrealized_pnl_pct > 0.03:
             exit_signals.append('Volatility Spike Exit')

        return exit_signals

    # Методы для проверки оптимального времени (можно вынести в Config или RiskManager)
    # def _is_optimal_trading_hour(self, timestamp): ...
    # def _is_optimal_trading_day(self, timestamp): ...

class PerformanceAnalyzer:
    """Анализ результатов бэктеста и построение графиков."""
    def __init__(self, results: pd.DataFrame, trades: pd.DataFrame, config: Config):
        if results is None or results.empty or trades is None:
             raise ValueError("Results or trades data is missing for analysis.")
        self.results = results
        self.trades = trades
        self.config = config
        self.stats = {}

    def calculate_metrics(self) -> dict:
        """Расчет основных метрик производительности."""
        print("\n===== BACKTEST ANALYSIS =====")
        self.stats = {}
        initial_balance = self.config.initial_balance

        # Проверяем, не пуст ли DataFrame results
        if self.results.empty:
            print("Backtest results are empty. Cannot calculate metrics.")
            # Заполняем статы нулями или NaN
            self.stats['Initial Balance'] = initial_balance
            self.stats['Final Balance'] = initial_balance
            self.stats['Total Return %'] = 0.0
            self.stats['Max Drawdown %'] = 0.0
            self.stats['Total Trades'] = 0
            # Добавляем остальные ключи со значениями по умолчанию
            keys_to_default = ['Profitable Trades', 'Losing Trades', 'Win Rate %',
                               'Avg Profit $', 'Avg Loss $', 'Profit Factor',
                               'Sharpe Ratio (ann.)', 'Sortino Ratio (ann.)',
                               'Avg Trade Duration (Hours)', 'Long Trades', 'Short Trades',
                               'Long Win Rate %', 'Short Win Rate %', 'Total PnL Long $',
                               'Total PnL Short $', 'Avg Leverage Used']
            for key in keys_to_default:
                self.stats[key] = 0.0 if 'Rate' in key or 'Ratio' in key or 'Factor' in key or 'Avg' in key else 0

            # Вывод пустой статистики
            for key, value in self.stats.items():
                if isinstance(value, float): print(f"{key}: {value:.2f}")
                else: print(f"{key}: {value}")
            return self.stats.copy()

        final_balance = self.results['balance'].iloc[-1]

        self.stats['Initial Balance'] = initial_balance
        self.stats['Final Balance'] = final_balance
        self.stats['Total Return %'] = ((final_balance / initial_balance) - 1) * 100 if initial_balance > 0 else 0

        # Расчет Max Drawdown по эквити
        equity_curve = self.results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max.replace(0, 1) * 100 # В процентах, избегаем деления на ноль
        self.stats['Max Drawdown %'] = drawdown.max() if not drawdown.empty else 0

        total_trades = len(self.trades)
        self.stats['Total Trades'] = total_trades

        if total_trades > 0:
            profitable_trades = self.trades[self.trades['pnl'] > 0]
            losing_trades = self.trades[self.trades['pnl'] <= 0]

            self.stats['Profitable Trades'] = len(profitable_trades)
            self.stats['Losing Trades'] = len(losing_trades)
            self.stats['Win Rate %'] = (len(profitable_trades) / total_trades * 100) if total_trades > 0 else 0

            self.stats['Avg Profit $'] = profitable_trades['pnl'].mean() if not profitable_trades.empty else 0
            self.stats['Avg Loss $'] = losing_trades['pnl'].mean() if not losing_trades.empty else 0 # Будет отрицательным

            # Profit Factor
            total_profit = profitable_trades['pnl'].sum()
            total_loss = abs(losing_trades['pnl'].sum())
            self.stats['Profit Factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

            # Sharpe & Sortino (приблизительно, нужны безрисковые данные для точности)
            # Используем дневные доходности эквити
            daily_results = self.results['equity'].resample('D').last()
            daily_returns = daily_results.pct_change().dropna()

            if not daily_returns.empty and daily_returns.std() > 0:
                 # Годовая доходность и стандартное отклонение
                 annualized_return = daily_returns.mean() * 365
                 annualized_std = daily_returns.std() * np.sqrt(365)
                 self.stats['Sharpe Ratio (ann.)'] = annualized_return / annualized_std if annualized_std > 0 else 0

                 # Sortino
                 negative_returns = daily_returns[daily_returns < 0]
                 downside_std = negative_returns.std() * np.sqrt(365) if not negative_returns.empty else 0
                 self.stats['Sortino Ratio (ann.)'] = annualized_return / downside_std if downside_std > 0 else 0
            else:
                 self.stats['Sharpe Ratio (ann.)'] = 0
                 self.stats['Sortino Ratio (ann.)'] = 0

            # Среднее время удержания позиции
            if 'trade_duration' in self.trades.columns:
                 self.stats['Avg Trade Duration (Hours)'] = self.trades['trade_duration'].mean()

            # Статистика по Long/Short
            long_trades = self.trades[self.trades['position'] == 'LONG']
            short_trades = self.trades[self.trades['position'] == 'SHORT']
            self.stats['Long Trades'] = len(long_trades)
            self.stats['Short Trades'] = len(short_trades)
            self.stats['Long Win Rate %'] = (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if not long_trades.empty else 0
            self.stats['Short Win Rate %'] = (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if not short_trades.empty else 0
            self.stats['Total PnL Long $'] = long_trades['pnl'].sum()
            self.stats['Total PnL Short $'] = short_trades['pnl'].sum()

            # Среднее плечо и риск
            if 'leverage' in self.trades.columns:
                 self.stats['Avg Leverage Used'] = self.trades['leverage'].mean()
            # if 'risk_per_trade' in self.trades.columns: # Нужно добавить risk_per_trade в history
            #      self.stats['Avg Risk Per Trade %'] = self.trades['risk_per_trade'].mean() * 100

        else:
            # Заполняем нулями или NaN, если сделок не было
            for key in ['Profitable Trades', 'Losing Trades', 'Win Rate %', 'Avg Profit $', 'Avg Loss $',
                        'Profit Factor', 'Sharpe Ratio (ann.)', 'Sortino Ratio (ann.)', 'Avg Trade Duration (Hours)',
                        'Long Trades', 'Short Trades', 'Long Win Rate %', 'Short Win Rate %',
                        'Total PnL Long $', 'Total PnL Short $', 'Avg Leverage Used']:
                self.stats[key] = 0.0 if 'Rate' in key or 'Ratio' in key or 'Factor' in key or 'Avg' in key else 0


        # Вывод статистики
        for key, value in self.stats.items():
            if isinstance(value, float): print(f"{key}: {value:.2f}")
            else: print(f"{key}: {value}")

        return self.stats.copy()

    def plot_equity_curve(self, plot_trades=True, plot_health=True):
        """Построение графика эквити, баланса и просадки."""
        if self.results is None or self.results.empty:
            print("No backtest results to plot.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1, 1]})
        ax1, ax2, ax3 = axes # Equity, Drawdown, Indicator

        # 1. Equity and Balance
        ax1.plot(self.results.index, self.results['equity'], label='Equity', color='blue', linewidth=1.5)
        ax1.plot(self.results.index, self.results['balance'], label='Balance', color='green', linestyle='--', linewidth=1.0)
        ax1.set_ylabel('Account Value ($)')
        ax1.set_title('Strategy Performance')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Добавление статистики на график
        stats_text = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in self.stats.items() if k in [
                                    'Total Return %', 'Max Drawdown %', 'Win Rate %',
                                    'Profit Factor', 'Sharpe Ratio (ann.)', 'Total Trades'
                                ]])
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='bottom', bbox=props)

        # Отметки сделок на графике эквити
        if plot_trades and not self.trades.empty:
            trades_to_plot = self.trades.copy()
            # Преобразуем даты в datetime, если они еще не такие
            trades_to_plot['entry_dt'] = pd.to_datetime(trades_to_plot['entry_date'])
            trades_to_plot['exit_dt'] = pd.to_datetime(trades_to_plot['exit_date'])

            # Используем reindex для получения значений эквити в точках входа/выхода
            equity_indexed = self.results['equity'].reindex(trades_to_plot['entry_dt'], method='nearest')
            # exit_equity_indexed = self.results['equity'].reindex(trades_to_plot['exit_dt'], method='nearest') # Пока не используется

            long_entries = trades_to_plot[trades_to_plot['position'] == 'LONG']
            short_entries = trades_to_plot[trades_to_plot['position'] == 'SHORT']

            # Проверка наличия индексов перед доступом
            valid_long_indices = long_entries.index.intersection(equity_indexed.index)
            valid_short_indices = short_entries.index.intersection(equity_indexed.index)

            if not valid_long_indices.empty:
                 entry_equity_long = equity_indexed.loc[valid_long_indices]
                 ax1.scatter(long_entries.loc[valid_long_indices, 'entry_dt'], entry_equity_long, color='lime', marker='^', s=40, alpha=0.8, label='Long Entry')
            if not valid_short_indices.empty:
                 entry_equity_short = equity_indexed.loc[valid_short_indices]
                 ax1.scatter(short_entries.loc[valid_short_indices, 'entry_dt'], entry_equity_short, color='red', marker='v', s=40, alpha=0.8, label='Short Entry')


        ax1.legend(loc='upper left')


        # 2. Drawdown
        equity_curve = self.results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max.replace(0, 1) * 100 # В процентах, избегаем деления на ноль
        drawdown.fillna(0, inplace=True)

        ax2.fill_between(self.results.index, 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(bottom=0) # Просадка не может быть меньше 0
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))

        # 3. Market Health or Regime Indicator
        if plot_health and 'market_health' in self.results.columns and not self.results['market_health'].isnull().all():
            ax3.plot(self.results.index, self.results['market_health'], label='Market Health', color='purple', alpha=0.7, linewidth=1.0)
            ax3.set_ylim(0, 100)
            ax3.axhline(50, color='grey', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Market Health (0-100)')
        elif 'market_regime' in self.results.columns and not self.results['market_regime'].isnull().all():
             # Преобразуем режим в числовой индикатор для графика
             regime_map = {'strong_bull': 1.0, 'transition_to_bull': 0.5, 'choppy_range': 0, 'mixed': 0,
                           'transition_to_bear': -0.5, 'strong_bear': -1.0, 'unknown': np.nan}
             regime_indicator = self.results['market_regime'].map(regime_map).fillna(0)
             ax3.plot(self.results.index, regime_indicator, label='Market Regime', color='orange', alpha=0.7, linewidth=1.0)
             ax3.set_ylim(-1.2, 1.2)
             ax3.axhline(0, color='grey', linestyle='--', alpha=0.5)
             ax3.set_ylabel('Market Regime')
             ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
             ax3.set_yticklabels(['Bear', 'Trans. Bear', 'Range/Mixed', 'Trans. Bull', 'Bull'])
        else:
             ax3.text(0.5, 0.5, 'No Health/Regime Data', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3.set_xlabel('Date')

        # Форматирование оси X (даты)
        fig.autofmt_xdate()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Немного места для заголовка
        plt.show()
        return fig

    def plot_regime_performance(self):
        """График производительности по рыночным режимам."""
        if self.trades.empty or 'pnl' not in self.trades.columns:
             print("No trades or PnL data to plot regime performance.")
             return

        # Используем результаты бэктеста для связи режима и сделки, если нет в trades
        trades_with_regime = self.trades.copy()
        if 'market_regime' not in trades_with_regime.columns:
            print("Market regime column missing in trades, trying to merge from results...")
            try:
                 # Находим ближайший режим к моменту входа в сделку
                 trades_with_regime['entry_dt_temp'] = pd.to_datetime(trades_with_regime['entry_date'])
                 regimes_series = self.results['market_regime'].reindex(trades_with_regime['entry_dt_temp'], method='nearest')
                 trades_with_regime['market_regime'] = regimes_series.values
                 trades_with_regime.drop(columns=['entry_dt_temp'], inplace=True)
                 if trades_with_regime['market_regime'].isnull().any():
                     print("Warning: Could not find market regime for all trades.")
                     trades_with_regime['market_regime'].fillna('unknown', inplace=True)
            except Exception as e:
                print(f"Error merging market regime: {e}. Cannot plot regime performance.")
                return
        else:
             # Если колонка есть, заполним пропуски
             trades_with_regime['market_regime'].fillna('unknown', inplace=True)


        regime_stats = trades_with_regime.groupby('market_regime').agg(
            num_trades=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            avg_pnl=('pnl', 'mean')
        ).reset_index()

        # Расчет Win Rate для каждого режима
        win_rates = []
        for regime in regime_stats['market_regime']:
            regime_trades = trades_with_regime[trades_with_regime['market_regime'] == regime]
            wins = len(regime_trades[regime_trades['pnl'] > 0])
            total = len(regime_trades)
            win_rate = (wins / total * 100) if total > 0 else 0
            win_rates.append(win_rate)
        regime_stats['win_rate'] = win_rates

        # Сортировка для лучшего вида
        regime_order = ['strong_bull', 'transition_to_bull', 'choppy_range', 'mixed', 'transition_to_bear', 'strong_bear', 'unknown']
        # Используем pd.Categorical для сортировки
        regime_stats['market_regime_cat'] = pd.Categorical(regime_stats['market_regime'], categories=regime_order, ordered=True)
        regime_stats = regime_stats.sort_values('market_regime_cat')


        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False) # Не шарим Y
        ax1, ax2, ax3 = axes

        regime_colors = {
            'strong_bull': 'green', 'transition_to_bull': 'lightgreen',
            'choppy_range': 'gray', 'mixed': 'lightblue',
            'transition_to_bear': 'salmon', 'strong_bear': 'red',
            'unknown': 'lightgray'
        }
        # Используем отсортированный порядок для цветов
        colors = [regime_colors.get(r, 'gray') for r in regime_stats['market_regime']]


        # 1. Number of Trades
        bars1 = ax1.bar(regime_stats['market_regime'], regime_stats['num_trades'], color=colors)
        ax1.set_title('Trades per Regime')
        ax1.set_ylabel('Number of Trades')
        ax1.tick_params(axis='x', rotation=45)
        ax1.bar_label(bars1, fmt='%d') # Добавляем подписи

        # 2. Win Rate
        bars2 = ax2.bar(regime_stats['market_regime'], regime_stats['win_rate'], color=colors)
        ax2.set_title('Win Rate per Regime')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.axhline(50, color='grey', linestyle='--', alpha=0.7)
        ax2.tick_params(axis='x', rotation=45)
        ax2.bar_label(bars2, fmt='%.1f%%') # Добавляем подписи

        # 3. Total PnL
        bars3 = ax3.bar(regime_stats['market_regime'], regime_stats['total_pnl'], color=colors)
        ax3.set_title('Total PnL per Regime')
        ax3.set_ylabel('Total PnL ($)')
        ax3.axhline(0, color='grey', linestyle='-', alpha=0.7)
        ax3.tick_params(axis='x', rotation=45)
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax3.bar_label(bars3, fmt='$%.0f', padding=3) # Добавляем подписи

        plt.tight_layout()
        plt.show()
        return fig

    # --- Другие возможные методы анализа ---
    # def analyze_hourly_performance(self): ...
    # def analyze_daily_performance(self): ...
    # def plot_pnl_distribution(self): ...
    # def calculate_correlation_metrics(self, benchmark_path=None): ...

# --- Основная функция для запуска ---
def main():
    """Основная функция для инициализации и запуска стратегии."""
    import os

    # --- Настройка ---
    base_dir = r"C:\Diploma\Pet" # Укажите ваш путь
    # Ищем первый CSV файл в директории
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in {base_dir}")
        return

    data_file = csv_files[0]
    data_path = os.path.join(base_dir, data_file)
    print(f"Using data file: {data_path}")

    # Конфигурация стратегии
    strategy_config = Config(
        initial_balance=1000,
        max_leverage=5, # Увеличил плечо для примера
        base_risk_per_trade=0.015, # Немного снизил базовый риск
        min_trades_interval_hours=4 # Снизил интервал
    )
    # Можно переопределить параметры индикаторов здесь, если нужно
    # strategy_config.update_params({'short_ema': 7, 'long_ema': 25})

    try:
        # 1. Загрузка данных
        data_handler = DataHandler(data_path)
        raw_data = data_handler.load_data(num_candles=10000) # Загрузим больше данных

        # 2. Расчет индикаторов
        indicator_calculator = IndicatorCalculator(strategy_config.params)
        data_with_indicators = indicator_calculator.calculate_all(raw_data)

        # 3. Инициализация менеджера риска
        risk_manager = RiskManager(strategy_config)

        # 4. Запуск бэктеста
        backtester = Backtester(data_with_indicators, strategy_config, risk_manager)
        results_df, trades_df = backtester.run()

        # 5. Анализ результатов
        if results_df is not None and not results_df.empty and trades_df is not None:
            analyzer = PerformanceAnalyzer(results_df, trades_df, strategy_config)
            stats = analyzer.calculate_metrics()

            # 6. Визуализация
            analyzer.plot_equity_curve(plot_trades=True, plot_health=True)
            analyzer.plot_regime_performance()
        else:
            print("Backtest did not produce results or trades for analysis.")

    except Exception as e:
        print(f"\n--- An error occurred during execution ---")
        import traceback
        print(traceback.format_exc())
        print(f"Error details: {e}")

if __name__ == "__main__":
    main()
