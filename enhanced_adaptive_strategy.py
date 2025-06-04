
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedAdaptiveStrategy:
    def __init__(self, data_path, 
                 initial_balance=10000, max_leverage=3, 
                 base_risk_per_trade=0.02, 
                 min_trades_interval=24,
                 timeframe='1h'):
        """
        Инициализация улучшенной адаптивной стратегии для торговли BTC фьючерсами
        
        Args:
            data_path: путь к CSV файлу с данными
            initial_balance: начальный баланс в USD
            max_leverage: максимальное плечо
            base_risk_per_trade: базовый риск на сделку (% от баланса)
            min_trades_interval: минимальный интервал между сделками (в часах)
            timeframe: таймфрейм данных ('1h' или '15m')
        """
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.base_risk_per_trade = base_risk_per_trade
        self.min_trades_interval = min_trades_interval
        self.timeframe = timeframe
        
        # Параметры индикаторов (будут оптимизированы позже)
        self.params = {
            # EMA параметры
            'short_ema': 9,
            'long_ema': 30,
            
            # RSI параметры
            'rsi_period': 14,
            'rsi_oversold': 25,  # Более жесткие условия (было 30)
            'rsi_overbought': 75,  # Более жесткие условия (было 70)
            
            # ADX параметры
            'adx_period': 14,
            'adx_strong_trend': 25,
            'adx_weak_trend': 15,  # Более строгое условие для флета (было 20)
            
            # Bollinger Bands параметры
            'bb_period': 20,
            'bb_std': 2.2,  # Увеличенное стандартное отклонение для снижения ложных сигналов
            
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
            'volume_filter_enabled': True,  # Добавлена опция включения/отключения фильтра
            
            # Параметры для определения тренда
            'trend_lookback': 20,
            'trend_threshold': 0.1,
            'trend_confirmation_period': 3,  # Добавлен период подтверждения тренда
            
            # Параметры для пирамидинга
            'pyramid_min_profit': 0.02,  # Снижено требование (было 0.05)
            'pyramid_size_multiplier': 0.5,
            'max_pyramid_entries': 2,
            
            # Параметры для временной фильтрации
            'trading_hours_start': 0,  # Для криптовалюты можно использовать 24/7
            'trading_hours_end': 23,
            'time_filter_enabled': False,  # Добавлена опция включения/отключения фильтра
            
            # Параметры для взвешенного подхода
            'adx_min': 15,
            'adx_max': 35,
            
            # Максимальный размер позиции (% от баланса)
            'max_position_size': 0.5,  # Максимум 50% от баланса
            
            # Параметры для защиты прибыли (трейлинг-стопы)
            'trailing_stop_activation': 0.03,  # Активация при прибыли 3%
            'trailing_stop_distance': 0.01,    # Трейлинг-стоп на расстоянии 1% от цены
            
            # Параметры для фильтрации во флете
            'range_bounce_threshold': 0.005,   # Требование отскока для подтверждения
            'range_volume_threshold': 1.2,     # Требование по объему для флета
            
            # Параметры для фильтрации в тренде
            'trend_persistence': 3,           # Сколько баров должен держаться тренд
            'trend_pullback_allowed': 0.01,   # Допустимый откат в тренде
            
            # Параметры для управления риском
            'vol_risk_adjustment': True,       # Динамическая корректировка риска
            'max_daily_trades': 5,             # Ограничение сделок в день
            'max_consecutive_losses': 3,       # Максимум последовательных убытков
            
            # Дополнительные фильтры для шортов
            'short_filter_ma_period': 50,      # Период МА для фильтрации шортов
            'short_extra_volume_req': 1.2      # Дополнительное требование по объему для шортов
        }
        
        # Будет заполнено в процессе
        self.data = None
        self.trade_history = []
        self.backtest_results = None
        self.trade_df = None
        self.optimized_params = None
        
        # Переменные для отслеживания последовательных убытков
        self.consecutive_losses = 0
        self.daily_trades_count = {}  # Словарь для отслеживания сделок по дням
    
    def load_data(self):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        
        # Загрузка данных из CSV
        self.data = pd.read_csv(self.data_path)
        
        # Конвертация дат
        self.data['Open time'] = pd.to_datetime(self.data['Open time'])
        self.data.set_index('Open time', inplace=True)
        
        # Убедимся, что все числовые колонки имеют правильный тип
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
        # Удаление строк с NaN-значениями
        self.data.dropna(subset=numeric_cols, inplace=True)
        
        # Добавляем данные о дне (для ограничения сделок в день)
        self.data['Date'] = self.data.index.date
        
        print(f"Загружено {len(self.data)} свечей")
        return self.data
    
    def calculate_indicators(self):
        """Расчет расширенного набора технических индикаторов"""
        print("Расчет индикаторов...")
        
        # --- EMA ---
        print("Расчет EMA...")
        self.data[f'EMA_{self.params["short_ema"]}'] = self._calculate_ema(self.data['Close'], self.params['short_ema'])
        self.data[f'EMA_{self.params["long_ema"]}'] = self._calculate_ema(self.data['Close'], self.params['long_ema'])
        
        # --- RSI ---
        print("Расчет RSI...")
        self.data['RSI'] = self._calculate_rsi(self.data['Close'], self.params['rsi_period'])
        
        # --- Bollinger Bands ---
        print("Расчет Bollinger Bands...")
        bb_std = self.params['bb_std']
        bb_period = self.params['bb_period']
        self.data['BB_Middle'] = self.data['Close'].rolling(window=bb_period).mean()
        rolling_std = self.data['Close'].rolling(window=bb_period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * bb_std)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * bb_std)
        
        # --- ATR для динамических стоп-лоссов ---
        print("Расчет ATR...")
        self.data['ATR'] = self._calculate_atr(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            self.params['atr_period']
        )
        
        # Добавляем скользящее среднее ATR для расчета волатильности
        self.data['ATR_MA'] = self.data['ATR'].rolling(20).mean()
        
        # --- ADX ---
        print("Расчет ADX...")
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
        print("Расчет MACD...")
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
        print("Расчет объемных индикаторов...")
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=self.params['volume_ma_period']).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # --- Определение тренда по ценовому движению ---
        lookback = self.params['trend_lookback']
        self.data['Price_Change_Pct'] = (self.data['Close'] - self.data['Close'].shift(lookback)) / self.data['Close'].shift(lookback)
        
        # --- Дополнительная скользящая средняя для фильтрации шортов ---
        self.data['MA_Short_Filter'] = self.data['Close'].rolling(window=self.params['short_filter_ma_period']).mean()
        
        # --- Расчет дивергенций RSI ---
        # Смотрим на минимумы/максимумы цены и RSI
        print("Расчет дивергенций...")
        self.data['Price_Min'] = self.data['Close'].rolling(5, center=True).min() == self.data['Close']
        self.data['Price_Max'] = self.data['Close'].rolling(5, center=True).max() == self.data['Close']
        self.data['RSI_Min'] = self.data['RSI'].rolling(5, center=True).min() == self.data['RSI']
        self.data['RSI_Max'] = self.data['RSI'].rolling(5, center=True).max() == self.data['RSI']
        
        # Дивергенции (позитивная: цена падает, RSI растет; негативная: цена растет, RSI падает)
        self.data['Bullish_Divergence'] = False  # Инициализация
        self.data['Bearish_Divergence'] = False  # Инициализация
        
        # Находим локальные минимумы и максимумы (ограничиваем количество для ускорения)
        price_mins = self.data[self.data['Price_Min']].index[-50:]  # Только последние 50 минимумов
        price_maxs = self.data[self.data['Price_Max']].index[-50:]  # Только последние 50 максимумов
        rsi_mins = self.data[self.data['RSI_Min']].index[-50:]      # Только последние 50 минимумов RSI
        rsi_maxs = self.data[self.data['RSI_Max']].index[-50:]      # Только последние 50 максимумов RSI
        
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
        print("Определение режима рынка...")
        threshold = self.params['trend_threshold']
        self.data['Strong_Trend'] = (self.data['ADX'] > self.params['adx_strong_trend']) & \
                                   (self.data['Price_Change_Pct'].abs() > threshold)
        
        # Требуем более строгого определения флета
        self.data['True_Range'] = (self.data['ADX'] < self.params['adx_weak_trend']) & \
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
        
        # --- Подтверждение тренда - проверка нескольких последовательных баров ---
        for i in range(1, self.params['trend_confirmation_period'] + 1):
            self.data[f'Bullish_Trend_{i}'] = self.data['Bullish_Trend'].shift(i)
            self.data[f'Bearish_Trend_{i}'] = self.data['Bearish_Trend'].shift(i)
        
        # Создаем индикаторы продолжительного тренда
        self.data['Confirmed_Bullish_Trend'] = self.data['Bullish_Trend']
        self.data['Confirmed_Bearish_Trend'] = self.data['Bearish_Trend']
        
        for i in range(1, self.params['trend_confirmation_period']):
            self.data['Confirmed_Bullish_Trend'] &= self.data[f'Bullish_Trend_{i}']
            self.data['Confirmed_Bearish_Trend'] &= self.data[f'Bearish_Trend_{i}']
        
        # --- Подтверждение разворота в боковике ---
        # Проверка отскока цены для подтверждения разворота
        self.data['Range_Bounce_Up'] = (self.data['Close'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1) > self.params['range_bounce_threshold']
        self.data['Range_Bounce_Down'] = (self.data['Close'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1) < -self.params['range_bounce_threshold']
        
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
        plus_di = 100 * (smooth_plus_dm / atr)
        minus_di = 100 * (smooth_minus_dm / atr)
        
        # Рассчитываем DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        
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
        
    def _check_daily_trade_limit(self, date):
        """Проверка, не превышен ли дневной лимит сделок"""
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str not in self.daily_trades_count:
            self.daily_trades_count[date_str] = 0
            
        if self.daily_trades_count[date_str] >= self.params['max_daily_trades']:
            return False  # Лимит превышен
            
        self.daily_trades_count[date_str] += 1
        return True  # Лимит не превышен
        
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
        self.consecutive_losses = 0
        self.daily_trades_count = {}
        
        # Создаем DataFrame для результатов и историю сделок
        results = []
        self.trade_history = []
        
        # Для корректного подсчета P&L и баланса
        total_pnl = 0.0
        trend_pnl = 0.0
        range_pnl = 0.0
        long_pnl = 0.0
        short_pnl = 0.0
        
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
                    continue
                
                # Расчет текущего нереализованного P/L
                unrealized_pnl = 0.0
                
                if position == 1:  # Лонг
                    unrealized_pnl = ((current['Close'] / entry_price) - 1) * position_size * self.max_leverage
                    current_equity = balance + unrealized_pnl
                    
                    # ------ ПРОВЕРКА СТОП-ЛОССА ДЛЯ ЛОНГА -------
                    if current['Low'] <= stop_loss_price:
                        # Сработал стоп-лосс для лонга
                        pnl = ((stop_loss_price / entry_price) - 1) * position_size * self.max_leverage
                        
                        # Ограничиваем максимальный убыток до 10% от депозита
                        if pnl < -0.1 * self.initial_balance:
                            pnl = -0.1 * self.initial_balance
                        
                        balance += pnl
                        total_pnl += pnl
                        
                        # Обновляем P&L по категориям
                        if 'trend_entry' in self.trade_history[-1]:
                            if self.trade_history[-1]['trend_entry']:
                                trend_pnl += pnl
                            else:
                                range_pnl += pnl
                        
                        long_pnl += pnl
                        
                        # Обновляем счетчик последовательных убытков
                        if pnl < 0:
                            self.consecutive_losses += 1
                        else:
                            self.consecutive_losses = 0
                        
                        self.trade_history[-1].update({
                            'exit_date': current.name,
                            'exit_price': stop_loss_price,
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Stop Loss'
                        })
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        continue
                    
                    # --- ТРЕЙЛИНГ-СТОП ДЛЯ ЛОНГА (новая функциональность) ---
                    if current['Close'] > entry_price * (1 + self.params['trailing_stop_activation']):
                        # Активируем трейлинг-стоп если прибыль превышает порог
                        new_stop = current['Close'] * (1 - self.params['trailing_stop_distance'])
                        if new_stop > stop_loss_price:
                            stop_loss_price = new_stop
                    
                    # ------ ПРОВЕРКА ТЕЙК-ПРОФИТА ДЛЯ ЛОНГА -------
                    if current['High'] >= take_profit_price:
                        # Сработал тейк-профит для лонга
                        pnl = ((take_profit_price / entry_price) - 1) * position_size * self.max_leverage
                        balance += pnl
                        total_pnl += pnl
                        
                        # Обновляем P&L по категориям
                        if 'trend_entry' in self.trade_history[-1]:
                            if self.trade_history[-1]['trend_entry']:
                                trend_pnl += pnl
                            else:
                                range_pnl += pnl
                        
                        long_pnl += pnl
                        
                        # Сбрасываем счетчик последовательных убытков
                        self.consecutive_losses = 0
                        
                        self.trade_history[-1].update({
                            'exit_date': current.name,
                            'exit_price': take_profit_price,
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Take Profit'
                        })
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        continue
                    
                    # --- ПИРАМИДИНГ ДЛЯ ЛОНГА (улучшенная версия) ---
                    # Если в прибыльной позиции и сильный тренд, добавляем к позиции
                    if (pyramid_entries < self.params['max_pyramid_entries'] and 
                        (current['Confirmed_Bullish_Trend'] or 
                        (current['Bullish_Trend'] and current['MACD_Bullish_Cross'])) and
                        current['Close'] > entry_price * (1 + self.params['pyramid_min_profit'])):
                        
                        # Рассчитываем размер дополнительной позиции (меньше основной)
                        additional_size = position_size * self.params['pyramid_size_multiplier']
                        
                        # Обновляем средневзвешенную цену входа и размер позиции
                        old_value = entry_price * position_size
                        new_value = current['Close'] * additional_size
                        position_size += additional_size
                        entry_price = (old_value + new_value) / position_size
                        
                        # Обновляем стоп-лосс и тейк-профит для новой средней цены
                        stop_loss_price = entry_price * (1 - current['ATR'] * self.params['atr_multiplier_sl'] / current['Close'])
                        take_profit_price = entry_price * (1 + current['ATR'] * self.params['atr_multiplier_tp'] / current['Close'])
                        
                        pyramid_entries += 1
                        
                        # Обновляем информацию о позиции
                        self.trade_history[-1]['pyramid_entries'] = pyramid_entries
                        self.trade_history[-1]['avg_entry_price'] = entry_price
                        continue
                    
                    # Проверка сигналов на выход
                    exit_signals = []
                    
                    # Трендовые сигналы выхода из лонга
                    if current['Trend_Weight'] > 0.7:
                        # Сильный трендовый сигнал - выход по кроссоверу EMA или MACD
                        if ((previous[f'EMA_{self.params["short_ema"]}'] >= previous[f'EMA_{self.params["long_ema"]}']) and 
                            (current[f'EMA_{self.params["short_ema"]}'] < current[f'EMA_{self.params["long_ema"]}'])):
                            exit_signals.append('EMA Crossover')
                        
                        if current['MACD_Bearish_Cross']:
                            exit_signals.append('MACD Crossover')
                            
                        if current['Bearish_Trend'] and not previous['Bearish_Trend']:
                            exit_signals.append('Trend Change')
                    
                    # Контртрендовые сигналы выхода из лонга
                    if current['Range_Weight'] > 0.7:
                        # Сильный флетовый сигнал - выход по RSI, Боллинджеру или дивергенции
                        if current['RSI'] > self.params['rsi_overbought']:
                            exit_signals.append('RSI Overbought')
                            
                        if current['Close'] > current['BB_Upper']:
                            exit_signals.append('Upper Bollinger')
                            
                        if current['Bearish_Divergence']:
                            exit_signals.append('Bearish Divergence')
                    
                    # Если есть сигнал на выход и прошло достаточно времени с момента входа
                    if exit_signals and (current.name - entry_date).total_seconds() / 3600 > 4:
                        # Закрываем лонг
                        pnl = ((current['Close'] / entry_price) - 1) * position_size * self.max_leverage
                        balance += pnl
                        total_pnl += pnl
                        
                        # Обновляем P&L по категориям
                        if 'trend_entry' in self.trade_history[-1]:
                            if self.trade_history[-1]['trend_entry']:
                                trend_pnl += pnl
                            else:
                                range_pnl += pnl
                        
                        long_pnl += pnl
                        
                        # Обновляем счетчик последовательных убытков
                        if pnl < 0:
                            self.consecutive_losses += 1
                        else:
                            self.consecutive_losses = 0
                        
                        self.trade_history[-1].update({
                            'exit_date': current.name,
                            'exit_price': current['Close'],
                            'pnl': pnl,
                            'balance': balance,
                            'reason': ', '.join(exit_signals)
                        })
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        continue
                
                elif position == -1:  # Шорт
                    unrealized_pnl = (1 - (current['Close'] / entry_price)) * position_size * self.max_leverage
                    current_equity = balance + unrealized_pnl
                    
                    # ------ ПРОВЕРКА СТОП-ЛОССА ДЛЯ ШОРТА -------
                    if current['High'] >= stop_loss_price:
                        # Сработал стоп-лосс для шорта
                        pnl = (1 - (stop_loss_price / entry_price)) * position_size * self.max_leverage
                        
                        # Ограничиваем максимальный убыток до 10% от депозита
                        if pnl < -0.1 * self.initial_balance:
                            pnl = -0.1 * self.initial_balance
                            
                        balance += pnl
                        total_pnl += pnl
                        
                        # Обновляем P&L по категориям
                        if 'trend_entry' in self.trade_history[-1]:
                            if self.trade_history[-1]['trend_entry']:
                                trend_pnl += pnl
                            else:
                                range_pnl += pnl
                        
                        short_pnl += pnl
                        
                        # Обновляем счетчик последовательных убытков
                        if pnl < 0:
                            self.consecutive_losses += 1
                        else:
                            self.consecutive_losses = 0
                        
                        self.trade_history[-1].update({
                            'exit_date': current.name,
                            'exit_price': stop_loss_price,
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Stop Loss'
                        })
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        continue
                    
                    # --- ТРЕЙЛИНГ-СТОП ДЛЯ ШОРТА (новая функциональность) ---
                    if current['Close'] < entry_price * (1 - self.params['trailing_stop_activation']):
                        # Активируем трейлинг-стоп если прибыль превышает порог
                        new_stop = current['Close'] * (1 + self.params['trailing_stop_distance'])
                        if new_stop < stop_loss_price or stop_loss_price == 0:
                            stop_loss_price = new_stop
                    
                    # ------ ПРОВЕРКА ТЕЙК-ПРОФИТА ДЛЯ ШОРТА -------
                    if current['Low'] <= take_profit_price:
                        # Сработал тейк-профит для шорта
                        pnl = (1 - (take_profit_price / entry_price)) * position_size * self.max_leverage
                        balance += pnl
                        total_pnl += pnl
                        
                        # Обновляем P&L по категориям
                        if 'trend_entry' in self.trade_history[-1]:
                            if self.trade_history[-1]['trend_entry']:
                                trend_pnl += pnl
                            else:
                                range_pnl += pnl
                        
                        short_pnl += pnl
                        
                        # Сбрасываем счетчик последовательных убытков
                        self.consecutive_losses = 0
                        
                        self.trade_history[-1].update({
                            'exit_date': current.name,
                            'exit_price': take_profit_price,
                            'pnl': pnl,
                            'balance': balance,
                            'reason': 'Take Profit'
                        })
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        continue
                    
                    # --- ПИРАМИДИНГ ДЛЯ ШОРТА (улучшенная версия) ---
                    # Если в прибыльной позиции и сильный тренд, добавляем к позиции
                    if (pyramid_entries < self.params['max_pyramid_entries'] and 
                        (current['Confirmed_Bearish_Trend'] or 
                        (current['Bearish_Trend'] and current['MACD_Bearish_Cross'])) and
                        current['Close'] < entry_price * (1 - self.params['pyramid_min_profit'])):
                        
                        # Рассчитываем размер дополнительной позиции (меньше основной)
                        additional_size = position_size * self.params['pyramid_size_multiplier']
                        
                        # Обновляем средневзвешенную цену входа и размер позиции
                        old_value = entry_price * position_size
                        new_value = current['Close'] * additional_size
                        position_size += additional_size
                        entry_price = (old_value + new_value) / position_size
                        
                        # Обновляем стоп-лосс и тейк-профит для новой средней цены
                        stop_loss_price = entry_price * (1 + current['ATR'] * self.params['atr_multiplier_sl'] / current['Close'])
                        take_profit_price = entry_price * (1 - current['ATR'] * self.params['atr_multiplier_tp'] / current['Close'])
                        
                        pyramid_entries += 1
                        
                        # Обновляем информацию о позиции
                        self.trade_history[-1]['pyramid_entries'] = pyramid_entries
                        self.trade_history[-1]['avg_entry_price'] = entry_price
                        continue
                    
                    # Проверка сигналов на выход
                    exit_signals = []
                    
                    # Трендовые сигналы выхода из шорта
                    if current['Trend_Weight'] > 0.7:
                        # Сильный трендовый сигнал - выход по кроссоверу EMA или MACD
                        if ((previous[f'EMA_{self.params["short_ema"]}'] <= previous[f'EMA_{self.params["long_ema"]}']) and 
                            (current[f'EMA_{self.params["short_ema"]}'] > current[f'EMA_{self.params["long_ema"]}'])):
                            exit_signals.append('EMA Crossover')
                        
                        if current['MACD_Bullish_Cross']:
                            exit_signals.append('MACD Crossover')
                            
                        if current['Bullish_Trend'] and not previous['Bullish_Trend']:
                            exit_signals.append('Trend Change')
                    
                    # Контртрендовые сигналы выхода из шорта
                    if current['Range_Weight'] > 0.7:
                        # Сильный флетовый сигнал - выход по RSI, Боллинджеру или дивергенции
                        if current['RSI'] < self.params['rsi_oversold']:
                            exit_signals.append('RSI Oversold')
                            
                        if current['Close'] < current['BB_Lower']:
                            exit_signals.append('Lower Bollinger')
                            
                        if current['Bullish_Divergence']:
                            exit_signals.append('Bullish Divergence')
                    
                    # Если есть сигнал на выход и прошло достаточно времени с момента входа
                    if exit_signals and (current.name - entry_date).total_seconds() / 3600 > 4:
                        # Закрываем шорт
                        pnl = (1 - (current['Close'] / entry_price)) * position_size * self.max_leverage
                        balance += pnl
                        total_pnl += pnl
                        
                        # Обновляем P&L по категориям
                        if 'trend_entry' in self.trade_history[-1]:
                            if self.trade_history[-1]['trend_entry']:
                                trend_pnl += pnl
                            else:
                                range_pnl += pnl
                        
                        short_pnl += pnl
                        
                        # Обновляем счетчик последовательных убытков
                        if pnl < 0:
                            self.consecutive_losses += 1
                        else:
                            self.consecutive_losses = 0
                        
                        self.trade_history[-1].update({
                            'exit_date': current.name,
                            'exit_price': current['Close'],
                            'pnl': pnl,
                            'balance': balance,
                            'reason': ', '.join(exit_signals)
                        })
                        
                        # Сбрасываем состояние позиции
                        position = 0
                        entry_price = 0.0
                        position_size = 0.0
                        entry_date = None
                        last_trade_date = current.name
                        pyramid_entries = 0
                        continue
            
            # ------ ОТКРЫТИЕ НОВЫХ ПОЗИЦИЙ -------
            if position == 0:  # Если нет открытой позиции
                # Проверка временного фильтра
                if self.params['time_filter_enabled'] and not current['Active_Hours']:
                    continue
                
                # Проверяем минимальный интервал между сделками
                if last_trade_date is not None:
                    hours_since_last_trade = (current.name - last_trade_date).total_seconds() / 3600
                    if hours_since_last_trade < self.min_trades_interval:
                        # Пропускаем сигнал, если не прошло достаточно времени
                        continue
                
                # Проверяем дневной лимит сделок
                if not self._check_daily_trade_limit(current.name.date()):
                    continue
                
                # Проверяем счетчик последовательных убытков
                if self.consecutive_losses >= self.params['max_consecutive_losses']:
                    continue  # Пропускаем новые сделки после серии убытков
                
                # Вычисляем динамический риск на сделку (base_risk * volatility_multiplier)
                volatility_multiplier = 1.0
                if self.params['vol_risk_adjustment'] and current['ATR'] > 0 and not pd.isna(current['ATR_MA']) and current['ATR_MA'] > 0:
                    # Нормализуем ATR относительно его среднего значения
                    atr_ratio = current['ATR'] / current['ATR_MA']
                    volatility_multiplier = 1.0 / atr_ratio  # Меньше риска при высокой волатильности
                
                risk_per_trade = self.base_risk_per_trade * volatility_multiplier
                # Ограничиваем мин и макс риск
                risk_per_trade = max(0.01, min(0.03, risk_per_trade))
                
                # Сигналы для входа в ЛОНГ
                long_signals = []
                
                # --- ТРЕНДОВЫЕ СИГНАЛЫ ДЛЯ ЛОНГА (с весом тренда) ---
                if current['Trend_Weight'] > 0.5:
                    # EMA-кроссовер
                    if ((previous[f'EMA_{self.params["short_ema"]}'] < previous[f'EMA_{self.params["long_ema"]}']) and 
                        (current[f'EMA_{self.params["short_ema"]}'] >= current[f'EMA_{self.params["long_ema"]}'])):
                        long_signals.append(('EMA Crossover', current['Trend_Weight']))
                    
                    # MACD-кроссовер с подтверждением тренда
                    if current['MACD_Bullish_Cross'] and current['Bullish_Trend']:
                        long_signals.append(('MACD Crossover', current['Trend_Weight'] * 1.2))
                    
                    # Бычий тренд начинается с продолжительным подтверждением
                    if current['Confirmed_Bullish_Trend']:
                        long_signals.append(('Confirmed Bullish Trend', current['Trend_Weight'] * 1.5))
                
                # --- КОНТРТРЕНДОВЫЕ СИГНАЛЫ ДЛЯ ЛОНГА (с весом флета) ---
                if current['Range_Weight'] > 0.5:
                    # RSI перепродан + нижняя полоса Боллинджера + подтверждение отскока
                    if (current['RSI'] < self.params['rsi_oversold'] and 
                        current['Close'] <= current['BB_Lower'] * 0.99 and
                        current['Range_Bounce_Up']):
                        long_signals.append(('RSI Oversold + BB Lower + Bounce', current['Range_Weight'] * 1.3))
                    
                    # Бычья дивергенция с подтверждением объема
                    if current['Bullish_Divergence'] and current['Volume'] > current['Volume_MA'] * self.params['range_volume_threshold']:
                        long_signals.append(('Bullish Divergence + Volume', current['Range_Weight'] * 1.5))
                
                # Взвешенный сигнал для лонга (средний вес сигналов)
                long_weight = 0
                long_is_trend_entry = False  # Флаг для обозначения сигнала как трендового
                
                if long_signals:
                    # Определяем, является ли сигнал преимущественно трендовым
                    trend_signals = sum(1 for signal, _ in long_signals if 'EMA' in signal or 'MACD' in signal or 'Trend' in signal)
                    range_signals = len(long_signals) - trend_signals
                    long_is_trend_entry = trend_signals > range_signals
                    
                    # Расчет общего веса сигнала
                    long_weight = sum(weight for _, weight in long_signals) / len(long_signals)
                
                # Объемный фильтр (применяем как множитель к весу сигнала)
                volume_multiplier = 1.0
                if self.params['volume_filter_enabled'] and current['Volume_Ratio'] > self.params['volume_threshold']:
                    volume_multiplier = min(2.0, current['Volume_Ratio'] / self.params['volume_threshold'])
                
                # Финальный вес сигнала для лонга
                long_weight = long_weight * volume_multiplier
                
                # Сигналы для входа в ШОРТ
                short_signals = []
                
                # --- ТРЕНДОВЫЕ СИГНАЛЫ ДЛЯ ШОРТА (с весом тренда) ---
                if current['Trend_Weight'] > 0.6:  # Более высокий порог для шортов
                    # EMA-кроссовер с подтверждением
                    if ((previous[f'EMA_{self.params["short_ema"]}'] > previous[f'EMA_{self.params["long_ema"]}']) and 
                        (current[f'EMA_{self.params["short_ema"]}'] <= current[f'EMA_{self.params["long_ema"]}']) and
                        current['Close'] < current['MA_Short_Filter']):  # Подтверждение по долгосрочной MA
                        short_signals.append(('EMA Crossover + MA Filter', current['Trend_Weight']))
                    
                    # MACD-кроссовер с подтверждением тренда
                    if current['MACD_Bearish_Cross'] and current['Bearish_Trend']:
                        short_signals.append(('MACD Crossover', current['Trend_Weight'] * 1.2))
                    
                    # Медвежий тренд начинается с продолжительным подтверждением
                    if current['Confirmed_Bearish_Trend']:
                        short_signals.append(('Confirmed Bearish Trend', current['Trend_Weight'] * 1.5))
                
                # --- КОНТРТРЕНДОВЫЕ СИГНАЛЫ ДЛЯ ШОРТА (с весом флета) ---
                if current['Range_Weight'] > 0.6:  # Более высокий порог для шортов
                    # RSI перекуплен + верхняя полоса Боллинджера + подтверждение снижения
                    if (current['RSI'] > self.params['rsi_overbought'] and 
                        current['Close'] >= current['BB_Upper'] * 1.01 and
                        current['Range_Bounce_Down']):
                        short_signals.append(('RSI Overbought + BB Upper + Down', current['Range_Weight'] * 1.3))
                    
                    # Медвежья дивергенция с подтверждением объема
                    if current['Bearish_Divergence'] and current['Volume'] > current['Volume_MA'] * self.params['range_volume_threshold']:
                        short_signals.append(('Bearish Divergence + Volume', current['Range_Weight'] * 1.5))
                
                # Взвешенный сигнал для шорта (средний вес сигналов)
                short_weight = 0
                short_is_trend_entry = False  # Флаг для обозначения сигнала как трендового
                
                if short_signals:
                    # Определяем, является ли сигнал преимущественно трендовым
                    trend_signals = sum(1 for signal, _ in short_signals if 'EMA' in signal or 'MACD' in signal or 'Trend' in signal)
                    range_signals = len(short_signals) - trend_signals
                    short_is_trend_entry = trend_signals > range_signals
                    
                    # Расчет общего веса сигнала
                    short_weight = sum(weight for _, weight in short_signals) / len(short_signals)
                
                # Объемный фильтр для шортов (усиленное требование)
                short_volume_multiplier = 1.0
                if self.params['volume_filter_enabled']:
                    vol_threshold = self.params['volume_threshold'] * self.params['short_extra_volume_req']
                    if current['Volume_Ratio'] > vol_threshold:
                        short_volume_multiplier = min(2.0, current['Volume_Ratio'] / vol_threshold)
                    else:
                        short_volume_multiplier = 0.8  # Штраф при недостаточном объеме
                
                # Финальный вес сигнала для шорта
                short_weight = short_weight * short_volume_multiplier
                
                # -- ПРИНЯТИЕ РЕШЕНИЯ О ВХОДЕ --
                # Минимальный порог силы сигнала для входа
                min_signal_threshold = 0.65  # Повышенный порог для фильтрации слабых сигналов
                
                # Вход в ЛОНГ, если его сигнал сильнее и превышает порог
                if long_weight > short_weight and long_weight >= min_signal_threshold:
                    # Сигнал на лонг
                    position = 1
                    entry_price = current['Close']
                    if entry_price <= 0:
                        # Защита от нулевой цены
                        entry_price = 1.0
                    entry_date = current.name
                    
                    # Динамический стоп-лосс на основе ATR
                    atr_value = current['ATR']
                    atr_multiplier = self.params['atr_multiplier_sl']
                    
                    # Уменьшаем размер стоп-лосса для трендовых входов
                    if long_is_trend_entry:
                        atr_multiplier *= 0.8
                    
                    stop_loss_price = entry_price * (1 - atr_value * atr_multiplier / entry_price)
                    take_profit_price = entry_price * (1 + atr_value * self.params['atr_multiplier_tp'] / entry_price)
                    
                    # Расчет размера позиции с учетом риска
                    max_risk_amount = balance * risk_per_trade
                    max_loss_percentage = (entry_price - stop_loss_price) / entry_price * self.max_leverage
                    
                    # Защита от деления на ноль и очень маленьких значений
                    if max_loss_percentage < 0.001:
                        max_loss_percentage = 0.001
                        
                    position_size = max_risk_amount / max_loss_percentage
                    
                    # Проверка на максимальный размер позиции
                    max_position = balance * self.params['max_position_size']
                    if position_size > max_position:
                        position_size = max_position
                    
                    # Сохраняем информацию о сигналах
                    signal_info = ', '.join(signal for signal, _ in long_signals)
                    
                    # Добавляем запись о входе в лонг
                    self.trade_history.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': 'LONG',
                        'balance': balance,
                        'reason': f'Entry: {signal_info}',
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'weight': long_weight,
                        'risk_per_trade': risk_per_trade,
                        'position_size': position_size,
                        'trend_entry': long_is_trend_entry,
                        'pyramid_entries': 0
                    })
                
                # Вход в ШОРТ, если его сигнал сильнее и превышает порог
                elif short_weight > long_weight and short_weight >= min_signal_threshold:
                    # Сигнал на шорт
                    position = -1
                    entry_price = current['Close']
                    if entry_price <= 0:
                        # Защита от нулевой цены
                        entry_price = 1.0
                    entry_date = current.name
                    
                    # Динамический стоп-лосс на основе ATR
                    atr_value = current['ATR']
                    atr_multiplier = self.params['atr_multiplier_sl']
                    
                    # Уменьшаем размер стоп-лосса для трендовых входов
                    if short_is_trend_entry:
                        atr_multiplier *= 0.8
                    
                    stop_loss_price = entry_price * (1 + atr_value * atr_multiplier / entry_price)
                    take_profit_price = entry_price * (1 - atr_value * self.params['atr_multiplier_tp'] / entry_price)
                    
                    # Расчет размера позиции с учетом риска
                    max_risk_amount = balance * risk_per_trade
                    max_loss_percentage = (stop_loss_price - entry_price) / entry_price * self.max_leverage
                    
                    # Защита от деления на ноль и очень маленьких значений
                    if max_loss_percentage < 0.001:
                        max_loss_percentage = 0.001
                        
                    position_size = max_risk_amount / max_loss_percentage
                    
                    # Проверка на максимальный размер позиции
                    max_position = balance * self.params['max_position_size']
                    if position_size > max_position:
                        position_size = max_position
                    
                    # Сохраняем информацию о сигналах
                    signal_info = ', '.join(signal for signal, _ in short_signals)
                    
                    # Добавляем запись о входе в шорт
                    self.trade_history.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': 'SHORT',
                        'balance': balance,
                        'reason': f'Entry: {signal_info}',
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'weight': short_weight,
                        'risk_per_trade': risk_per_trade,
                        'position_size': position_size,
                        'trend_entry': short_is_trend_entry,
                        'pyramid_entries': 0
                    })
            
            # ------ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ -------
            # Сохраняем текущее состояние для анализа
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
                'is_ranging': current['True_Range'],
                'trend_weight': current['Trend_Weight'],
                'range_weight': current['Range_Weight'],
                'total_pnl': total_pnl,
                'trend_pnl': trend_pnl,
                'range_pnl': range_pnl,
                'long_pnl': long_pnl,
                'short_pnl': short_pnl
            })
        
        # ------ ЗАКРЫТИЕ ПОЗИЦИИ В КОНЦЕ БЭКТЕСТА -------
        if position != 0 and entry_price > 0:
            last_candle = self.data.iloc[-1]
            exit_price = last_candle['Close']
            
            if position == 1:  # Лонг
                pnl = ((exit_price / entry_price) - 1) * position_size * self.max_leverage
            else:  # Шорт
                pnl = (1 - (exit_price / entry_price)) * position_size * self.max_leverage
                
            balance += pnl
            total_pnl += pnl
            
            # Обновляем P&L по категориям
            if 'trend_entry' in self.trade_history[-1]:
                if self.trade_history[-1]['trend_entry']:
                    trend_pnl += pnl
                else:
                    range_pnl += pnl
            
            if position == 1:
                long_pnl += pnl
            else:
                short_pnl += pnl
            
            # Обновляем последнюю запись в истории сделок
            for trade in reversed(self.trade_history):
                if 'exit_date' not in trade or trade['exit_date'] is None:
                    trade.update({
                        'exit_date': last_candle.name,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'balance': balance,
                        'reason': 'End of Backtest'
                    })
                    break
        
        # Создаем DataFrame с результатами
        self.backtest_results = pd.DataFrame(results)
        
        # Проверяем и обрабатываем историю сделок
        if self.trade_history:
            self.trade_df = pd.DataFrame(self.trade_history)
            
            # Заполняем пропущенные значения для незавершенных сделок
            if 'exit_date' in self.trade_df.columns:
                self.trade_df['exit_date'].fillna(self.data.index[-1], inplace=True)
            
            if 'exit_price' in self.trade_df.columns:
                self.trade_df['exit_price'].fillna(self.data['Close'].iloc[-1], inplace=True)
            
            # Рассчитываем P&L для незавершенных сделок
            for i, row in self.trade_df.iterrows():
                if 'pnl' not in row or pd.isna(row['pnl']):
                    if row['position'] == 'LONG':
                        pnl = ((row['exit_price'] / row['entry_price']) - 1) * row.get('position_size', 0) * self.max_leverage
                    else:  # 'SHORT'
                        pnl = (1 - (row['exit_price'] / row['entry_price'])) * row.get('position_size', 0) * self.max_leverage
                    
                    self.trade_df.at[i, 'pnl'] = pnl
        else:
            self.trade_df = pd.DataFrame()
        
        print("Бэктест завершен")
        print(f"Трендовые сделки P&L: ${trend_pnl:.2f}")
        print(f"Флетовые сделки P&L: ${range_pnl:.2f}")
        print(f"Лонг сделки P&L: ${long_pnl:.2f}")
        print(f"Шорт сделки P&L: ${short_pnl:.2f}")
        print(f"Общий P&L: ${total_pnl:.2f}")
        print(f"Конечный баланс: ${balance:.2f}")
        
        return self.backtest_results

    def walk_forward_analysis(self, train_size=0.7, window_size=None, step_size=None):
        """
        Проведение walk-forward анализа для проверки устойчивости стратегии
        
        Args:
            train_size: доля данных для обучения (по умолчанию 0.7)
            window_size: размер окна в днях (если None, используется train_size)
            step_size: шаг продвижения окна в днях (если None, используется 30 дней)
            
        Returns:
            DataFrame с результатами по периодам
        """
        print("Проведение walk-forward анализа...")
        
        # Проверяем, нужно ли использовать скользящее окно или фиксированное разделение
        use_sliding_window = window_size is not None
        
        if use_sliding_window:
            # Для скользящего окна
            # Преобразуем индекс в datetime, если это еще не сделано
            if not isinstance(self.data.index, pd.DatetimeIndex):
                self.data.index = pd.to_datetime(self.data.index)
            
            # Определяем шаг продвижения окна
            if step_size is None:
                step_size = 30  # 30 дней по умолчанию
            
            # Получаем уникальные даты
            dates = self.data.index.sort_values().unique()
            
            # Определяем временные периоды
            periods = []
            start_idx = 0
            
            while start_idx + window_size < len(dates):
                train_end_idx = start_idx + int(window_size * train_size)
                test_end_idx = min(start_idx + window_size, len(dates) - 1)
                
                train_start = dates[start_idx]
                train_end = dates[train_end_idx]
                test_start = dates[train_end_idx + 1]
                test_end = dates[test_end_idx]
                
                periods.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
                
                start_idx += step_size
                
                # Если следующее окно выходит за пределы данных, прерываем цикл
                if start_idx + window_size > len(dates):
                    break
        else:
            # Для фиксированного разделения
            # Определяем временные периоды
            # Определяем размер периода на основе имеющихся данных
            total_days = (self.data.index.max() - self.data.index.min()).days
            period_days = total_days // 4  # Например, делим на 4 периода
            
            start_date = self.data.index.min()
            periods = []
            
            for i in range(3):  # 3 периода (последний охватывает до конца)
                period_start = start_date + timedelta(days=i * period_days)
                period_end = start_date + timedelta(days=(i + 1) * period_days)
                
                train_end = period_start + timedelta(days=int(period_days * train_size))
                
                periods.append({
                    'train_start': period_start,
                    'train_end': train_end,
                    'test_start': train_end + timedelta(days=1),
                    'test_end': period_end
                })
        
        # Запускаем анализ для каждого периода
        results = []
        
        for i, period in enumerate(periods):
            print(f"\nПериод {i+1}/{len(periods)}")
            print(f"Обучение: {period['train_start']} - {period['train_end']}")
            print(f"Тестирование: {period['test_start']} - {period['test_end']}")
            
            # Фильтруем данные для обучения
            train_data = self.data[(self.data.index >= period['train_start']) & 
                                    (self.data.index <= period['train_end'])]
            
            # Фильтруем данные для тестирования
            test_data = self.data[(self.data.index >= period['test_start']) & 
                                  (self.data.index <= period['test_end'])]
            
            # Проверяем, что есть достаточно данных
            if len(train_data) < 100 or len(test_data) < 100:
                print("Недостаточно данных для периода, пропускаем...")
                continue
            
            # Сохраняем копию исходных данных
            original_data = self.data.copy()
            
            # Устанавливаем данные для обучения
            self.data = train_data.copy()
            
            # Оптимизируем параметры на обучающем наборе
            param_ranges = {
                'short_ema': (5, 20),
                'long_ema': (25, 50),
                'rsi_period': (10, 20),
                'adx_period': (10, 20),
                'atr_multiplier_sl': (1.5, 3.5),
                'atr_multiplier_tp': (3.0, 7.0),
                'volume_threshold': (1.2, 2.0)
            }
            
            best_params, _ = self.optimize_parameters(param_ranges, n_trials=5, scoring='combined')
            
            # Устанавливаем данные для тестирования
            self.data = test_data.copy()
            
            # Применяем оптимизированные параметры
            for param, value in best_params.items():
                self.params[param] = value
            
            # Пересчитываем индикаторы и запускаем бэктест
            self.calculate_indicators()
            self.run_backtest()
            stats = self.analyze_results()
            
            # Сохраняем результаты
            period_result = {
                'period': i+1,
                'train_start': period['train_start'],
                'train_end': period['train_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'monthly_return': stats['monthly_return'],
                'max_drawdown': stats['max_drawdown'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'parameters': best_params
            }
            
            results.append(period_result)
            
            # Восстанавливаем исходные данные
            self.data = original_data.copy()
        
        # Создаем DataFrame с результатами
        results_df = pd.DataFrame(results)
        
        # Выводим средние показатели
        print("\n===== СРЕДНИЕ ПОКАЗАТЕЛИ WALK-FORWARD АНАЛИЗА =====")
        print(f"Средняя месячная доходность: {results_df['monthly_return'].mean():.2f}%")
        print(f"Средняя максимальная просадка: {results_df['max_drawdown'].mean():.2f}%")
        print(f"Средний Win Rate: {results_df['win_rate'].mean():.2f}%")
        print(f"Средний Профит-фактор: {results_df['profit_factor'].mean():.2f}")
        
        # Проверяем устойчивость по периодам
        consistency = results_df['monthly_return'].std() / results_df['monthly_return'].mean()
        print(f"Коэффициент вариации доходности: {consistency:.2f} (чем ниже, тем стабильнее)")
        
        return results_df    
    def apply_optimized_parameters(self):
        """Применяет оптимизированные параметры и запускает бэктест"""
        if self.optimized_params is None:
            print("Нет оптимизированных параметров. Сначала запустите оптимизацию.")
            return None
        
        # Применяем оптимизированные параметры
        for param, value in self.optimized_params.items():
            self.params[param] = value
        
        print("Оптимизированные параметры применены:")
        for param, value in self.optimized_params.items():
            print(f"  {param}: {value}")
        
        # Пересчитываем индикаторы
        self.calculate_indicators()
        
        # Запускаем бэктест
        self.run_backtest()
        
        # Анализируем результаты
        stats = self.analyze_results()
        
        # Строим график
        self.plot_equity_curve()
        
        return stats    



    def analyze_results(self):
        """Анализ результатов бэктеста"""
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
                         self.trade_df[self.trade_df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
        
        # Медианная продолжительность сделок
        if 'entry_date' in self.trade_df.columns and 'exit_date' in self.trade_df.columns:
            self.trade_df['duration'] = pd.to_datetime(self.trade_df['exit_date']) - pd.to_datetime(self.trade_df['entry_date'])
            median_duration_hours = self.trade_df['duration'].median().total_seconds() / 3600
        else:
            median_duration_hours = 0
        
        # Статистика по типам позиций
        long_trades = len(self.trade_df[self.trade_df['position'] == 'LONG'])
        short_trades = len(self.trade_df[self.trade_df['position'] == 'SHORT'])
        
        long_profit = self.trade_df[self.trade_df['position'] == 'LONG']['pnl'].sum() if long_trades > 0 else 0
        short_profit = self.trade_df[self.trade_df['position'] == 'SHORT']['pnl'].sum() if short_trades > 0 else 0
        
        # Анализ по причинам выхода
        if 'reason' in self.trade_df.columns:
            exit_reasons = self.trade_df['reason'].str.extract(r'(?:, )([^,]+)$', expand=False)
            exit_reasons_counts = exit_reasons.value_counts()
        else:
            exit_reasons_counts = pd.Series()
        
        # Получаем общий P&L и его составляющие из последней строки результатов
        last_row = self.backtest_results.iloc[-1]
        
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
        
        if median_duration_hours > 0:
            print(f"Медианная продолжительность сделки: {median_duration_hours:.2f} часов")
        
        print(f"Лонг сделок: {long_trades} (P&L: ${long_profit:.2f})")
        print(f"Шорт сделок: {short_trades} (P&L: ${short_profit:.2f})")
        
        if not exit_reasons_counts.empty:
            print("\n===== РАСПРЕДЕЛЕНИЕ ПРИЧИН ВЫХОДА =====")
            for reason, count in exit_reasons_counts.items():
                print(f"{reason}: {count} ({count/total_trades*100:.2f}%)")
        
        # Годовая эффективность
        if 'date' in self.backtest_results.columns:
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
        
        # Эффективность в разных режимах рынка
        if 'trend_entry' in self.trade_df.columns:
            trend_trades = self.trade_df[self.trade_df['trend_entry'] == True]
            range_trades = self.trade_df[self.trade_df['trend_entry'] == False]
            
            trend_win_rate = len(trend_trades[trend_trades['pnl'] > 0]) / len(trend_trades) * 100 if len(trend_trades) > 0 else 0
            range_win_rate = len(range_trades[range_trades['pnl'] > 0]) / len(range_trades) * 100 if len(range_trades) > 0 else 0
            
            trend_profit = trend_trades['pnl'].sum() if len(trend_trades) > 0 else 0
            range_profit = range_trades['pnl'].sum() if len(range_trades) > 0 else 0
            
            print("\n===== ЭФФЕКТИВНОСТЬ ПО ТИПАМ РЫНКА =====")
            print(f"Трендовые сделки: {len(trend_trades)} (Win Rate: {trend_win_rate:.2f}%, P&L: ${trend_profit:.2f})")
            print(f"Флетовые сделки: {len(range_trades)} (Win Rate: {range_win_rate:.2f}%, P&L: ${range_profit:.2f})")
        
        # Анализ влияния пирамидинга
        if 'pyramid_entries' in self.trade_df.columns:
            pyramid_trades = self.trade_df[self.trade_df['pyramid_entries'] > 0]
            non_pyramid_trades = self.trade_df[self.trade_df['pyramid_entries'] == 0]
            
            pyramid_win_rate = len(pyramid_trades[pyramid_trades['pnl'] > 0]) / len(pyramid_trades) * 100 if len(pyramid_trades) > 0 else 0
            non_pyramid_win_rate = len(non_pyramid_trades[non_pyramid_trades['pnl'] > 0]) / len(non_pyramid_trades) * 100 if len(non_pyramid_trades) > 0 else 0
            
            pyramid_profit = pyramid_trades['pnl'].sum() if len(pyramid_trades) > 0 else 0
            non_pyramid_profit = non_pyramid_trades['pnl'].sum() if len(non_pyramid_trades) > 0 else 0
            
            print("\n===== ВЛИЯНИЕ ПИРАМИДИНГА =====")
            print(f"Сделки с пирамидингом: {len(pyramid_trades)} (Win Rate: {pyramid_win_rate:.2f}%, P&L: ${pyramid_profit:.2f})")
            print(f"Сделки без пирамидинга: {len(non_pyramid_trades)} (Win Rate: {non_pyramid_win_rate:.2f}%, P&L: ${non_pyramid_profit:.2f})")
        
        # Сводная статистика
        stats = {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'monthly_return': monthly_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
        if 'date' in self.backtest_results.columns:
            stats['yearly_performance'] = yearly_performance
        
        if 'trend_entry' in self.trade_df.columns:
            stats['trend_win_rate'] = trend_win_rate
            stats['range_win_rate'] = range_win_rate
        
        return stats
    
    def plot_equity_curve(self, save_path=None):
        """Построение улучшенного графика доходности"""
        if self.backtest_results is None:
            print("Нет данных для построения графика. Сначала запустите бэктест.")
            return
        
        plt.figure(figsize=(16, 12))
        
        # График баланса и эквити
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(self.backtest_results['date'], self.backtest_results['equity'], label='Equity', color='blue', linewidth=2)
        ax1.plot(self.backtest_results['date'], self.backtest_results['balance'], label='Balance', color='green', linewidth=1.5)
        
        # Добавление затененных областей для режимов рынка
        for i in range(1, len(self.backtest_results)):
            if self.backtest_results['is_trending'].iloc[i]:
                # Затенение трендовых периодов
                ax1.axvspan(self.backtest_results['date'].iloc[i-1], self.backtest_results['date'].iloc[i], 
                            alpha=0.2, color='green', label='_nolegend_')
            elif self.backtest_results['is_ranging'].iloc[i]:
                # Затенение флетовых периодов
                ax1.axvspan(self.backtest_results['date'].iloc[i-1], self.backtest_results['date'].iloc[i], 
                            alpha=0.2, color='blue', label='_nolegend_')
        
        # Маркеры сделок
        for i, trade in enumerate(self.trade_history):
            if 'position' in trade and 'entry_date' in trade:
                if trade['position'] == 'LONG':
                    # Вход в лонг
                    entry_idx = self.backtest_results[self.backtest_results['date'] == trade['entry_date']].index
                    if len(entry_idx) > 0:
                        entry_price = trade.get('entry_price', self.backtest_results.loc[entry_idx[0], 'close'])
                        ax1.scatter(trade['entry_date'], entry_price, color='green', marker='^', s=100)
                    
                    # Выход из лонга
                    if 'exit_date' in trade and trade['exit_date'] is not None:
                        ax1.scatter(trade['exit_date'], trade.get('exit_price', 0), color='black', marker='o', s=100)
                
                elif trade['position'] == 'SHORT':
                    # Вход в шорт
                    entry_idx = self.backtest_results[self.backtest_results['date'] == trade['entry_date']].index
                    if len(entry_idx) > 0:
                        entry_price = trade.get('entry_price', self.backtest_results.loc[entry_idx[0], 'close'])
                        ax1.scatter(trade['entry_date'], entry_price, color='red', marker='v', s=100)
                    
                    # Выход из шорта
                    if 'exit_date' in trade and trade['exit_date'] is not None:
                        ax1.scatter(trade['exit_date'], trade.get('exit_price', 0), color='black', marker='o', s=100)
        
        # Форматирование первого графика
        ax1.set_title('Equity Curve & Trades', fontsize=16)
        ax1.set_ylabel('USD', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Добавление легенды
        leg_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Long Entry'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Short Entry'),
            plt.Line2D([0], [0], marker='o', color='black', markersize=10, label='Exit')
        ]
        ax1.legend(handles=[
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Equity'),
            plt.Line2D([0], [0], color='green', linewidth=1.5, label='Balance'),
            *leg_elements
        ], loc='upper left', fontsize=10)
        
        # График просадки
        ax2 = plt.subplot(3, 1, 2)
        equity_curve = self.backtest_results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max * 100
        ax2.fill_between(self.backtest_results['date'], 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_title('Drawdown', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # График весов режимов рынка
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(self.backtest_results['date'], self.backtest_results['trend_weight'], label='Trend Weight', color='green', linewidth=1.5)
        ax3.plot(self.backtest_results['date'], self.backtest_results['range_weight'], label='Range Weight', color='blue', linewidth=1.5)
        ax3.set_ylabel('Weight', fontsize=12)
        ax3.set_ylim(0, 1)
        ax3.set_title('Market Regime Weights', fontsize=16)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # Форматирование осей X для всех графиков
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Дополнительные графики для анализа
        plt.figure(figsize=(16, 16))
        
        # График P&L по категориям
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(self.backtest_results['date'], self.backtest_results['total_pnl'], label='Total P&L', color='black', linewidth=2)
        ax1.plot(self.backtest_results['date'], self.backtest_results['trend_pnl'], label='Trend P&L', color='green', linewidth=1.5)
        ax1.plot(self.backtest_results['date'], self.backtest_results['range_pnl'], label='Range P&L', color='blue', linewidth=1.5)
        ax1.set_title('P&L by Market Regime', fontsize=16)
        ax1.set_ylabel('USD', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # График P&L по типам позиций
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(self.backtest_results['date'], self.backtest_results['long_pnl'], label='Long P&L', color='green', linewidth=1.5)
        ax2.plot(self.backtest_results['date'], self.backtest_results['short_pnl'], label='Short P&L', color='red', linewidth=1.5)
        ax2.set_title('P&L by Position Type', fontsize=16)
        ax2.set_ylabel('USD', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Гистограмма P&L по сделкам
        if 'pnl' in self.trade_df.columns:
            ax3 = plt.subplot(3, 1, 3)
            self.trade_df['pnl'].hist(bins=50, ax=ax3, color='blue', alpha=0.7)
            ax3.axvline(0, color='red', linestyle='--', linewidth=1.5)
            ax3.set_title('Distribution of Trade P&L', fontsize=16)
            ax3.set_xlabel('P&L ($)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # Форматирование осей X для временных графиков
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_analysis.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def optimize_parameters(self, param_ranges, n_trials=50, scoring='monthly_return'):
        """
        Оптимизация параметров стратегии с использованием случайного поиска
        
        Args:
            param_ranges: словарь с диапазонами параметров для оптимизации
            n_trials: количество итераций оптимизации
            scoring: метрика для оптимизации ('monthly_return', 'sharpe', 'sortino')
            
        Returns:
            Лучшие параметры и их результаты
        """
        import random
        
        print(f"Начинаем оптимизацию с {n_trials} случайными комбинациями параметров...")
        
        results = []
        best_score = float('-inf') if scoring != 'max_drawdown' else float('inf')
        best_params = None
        
        # Сохраняем текущие параметры
        original_params = self.params.copy()
        
        for trial in range(n_trials):
            # Выбираем случайные значения для всех параметров
            random_params = {}
            for param, param_range in param_ranges.items():
                if isinstance(param_range[0], int):
                    random_params[param] = random.randint(param_range[0], param_range[1])
                else:
                    random_params[param] = random.uniform(param_range[0], param_range[1])
            
            # Устанавливаем новые параметры
            for param, value in random_params.items():
                self.params[param] = value
            
            # Пересчитываем индикаторы с новыми параметрами
            self.calculate_indicators()
            
            # Запускаем бэктест
            self.run_backtest()
            
            # Анализируем результаты
            stats = self.analyze_results()
            
            # Выбираем метрику для оптимизации
            if scoring == 'monthly_return':
                score = stats['monthly_return']
            elif scoring == 'sharpe':
                # Расчет коэффициента Шарпа
                returns = self.backtest_results['equity'].pct_change().dropna()
                sharpe = (returns.mean() * 252**0.5) / (returns.std() + 1e-10) if returns.std() > 0 else 0
                score = sharpe
            elif scoring == 'sortino':
                # Расчет коэффициента Сортино
                returns = self.backtest_results['equity'].pct_change().dropna()
                negative_returns = returns[returns < 0]
                downside_risk = negative_returns.std() if len(negative_returns) > 0 else 1e-10
                sortino = (returns.mean() * 252**0.5) / (downside_risk + 1e-10)
                score = sortino
            elif scoring == 'max_drawdown':
                score = -stats['max_drawdown']  # Минимизация просадки (с отрицательным знаком)
            elif scoring == 'combined':
                # Комбинированная метрика: доходность / просадка
                if stats['max_drawdown'] == 0:
                    score = stats['monthly_return'] * 10  # Особый случай - нет просадки
                else:
                    score = stats['monthly_return'] / (stats['max_drawdown'] + 1e-10)
            else:
                score = stats['monthly_return']  # По умолчанию
            
            # Сохраняем параметры и результаты
            result = {
                'parameters': random_params,
                'monthly_return': stats['monthly_return'],
                'max_drawdown': stats['max_drawdown'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'score': score
            }
            results.append(result)
            
            # Обновляем лучший результат
            if (scoring != 'max_drawdown' and score > best_score) or (scoring == 'max_drawdown' and score < best_score):
                best_score = score
                best_params = random_params.copy()
            
            print(f"Итерация {trial+1}/{n_trials}: Monthly Return = {stats['monthly_return']:.2f}%, Max Drawdown = {stats['max_drawdown']:.2f}%, Score = {score:.4f}")
        
        # Возвращаем исходные параметры
        self.params = original_params.copy()
        
        # Сортируем результаты
        results_df = pd.DataFrame(results)
        results_df.sort_values('score', ascending=(scoring == 'max_drawdown'), inplace=True)
        
        # Получаем топ-5 комбинаций параметров
        top_results = results_df.head(5)
        
        print("\n===== ТОП-5 ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ =====")
        for i, (_, row) in enumerate(top_results.iterrows()):
            print(f"Комбинация {i+1}:")
            print(f"  Параметры: {row['parameters']}")
            print(f"  Месячная доходность: {row['monthly_return']:.2f}%")
            print(f"  Максимальная просадка: {row['max_drawdown']:.2f}%")
            print(f"  Win Rate: {row['win_rate']:.2f}%")
            print(f"  Профит-фактор: {row['profit_factor']:.2f}")
            print(f"  Оценка: {row['score']:.4f}\n")
        
        # Сохраняем лучшие параметры
        self.optimized_params = best_params
        
        return best_params, results_df