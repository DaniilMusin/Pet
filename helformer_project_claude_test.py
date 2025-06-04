import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import glob
from datetime import datetime
import re
from scipy.signal import argrelextrema

# Установка случайного зерна для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# Конфигурация устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

##########################
# ОПРЕДЕЛЕНИЕ МОДЕЛИ И ВСПОМОГАТЕЛЬНЫХ КЛАССОВ
##########################

class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для трансформера
    """
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        
        # Создание матрицы позиционного кодирования
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Применение синуса к четным индексам и косинуса к нечетным
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Добавление размерности батча
        pe = pe.unsqueeze(0)
        
        # Регистрация буфера (постоянное состояние, которое не является параметром)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Добавление позиционного кодирования к входному тензору
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Трансформер модель с компонентом Holt-Winters (Helformer)
    """
    def __init__(
        self, 
        input_dim, 
        d_model=64, 
        nhead=4, 
        num_encoder_layers=2,
        dim_feedforward=256, 
        dropout=0.1, 
        activation='gelu',
        use_holt_winters=True
    ):
        super().__init__()
        
        self.use_holt_winters = use_holt_winters
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Входной проекционный слой
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Позиционное кодирование
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Энкодер трансформера
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Слой LSTM для моделирования временных рядов в сочетании с выходами трансформера
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )
        
        # Выходной слой (предсказывает следующую цену)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Предсказание следующей цены закрытия
        )
        
    def forward(self, x, hw_components=None):
        # x форма: [batch_size, seq_len, input_dim]
        
        # Проекция входных данных в размерность модели
        x = self.input_projection(x)
        
        # Добавление позиционного кодирования
        x = self.positional_encoding(x)
        
        # Применение энкодера трансформера
        # Маски внимания не нужны, так как мы используем весь прошлый контекст
        x = self.transformer_encoder(x)
        
        # Применение слоя LSTM (объединяет последовательную информацию с трансформером)
        x, _ = self.lstm(x)
        
        # Если используется декомпозиция Holt-Winters, добавляем компоненты
        if self.use_holt_winters and hw_components is not None:
            # Предполагаем, что hw_components содержит уровень, тренд и сезонные компоненты
            # Форма должна быть [batch_size, seq_len, 3], где 3 представляет уровень, тренд, сезон
            hw_projection = nn.Linear(hw_components.shape[-1], self.d_model).to(device)
            hw_features = hw_projection(hw_components)
            x = x + hw_features
            
        # Получаем последний временной шаг для прогнозирования
        last_time_step = x[:, -1, :]
        
        # Проекция в выход
        output = self.output_projection(last_time_step)
        
        return output

class CryptoDataset(Dataset):
    """
    Класс датасета для криптовалютных данных
    """
    def __init__(self, data, seq_length=60, target_horizon=1, hw_decomposition=True, use_features=None):
        """
        Инициализация датасета
        
        Args:
            data: DataFrame с данными криптовалюты
            seq_length: Длина входных последовательностей
            target_horizon: На сколько шагов вперед предсказывать
            hw_decomposition: Использовать ли декомпозицию Holt-Winters
            use_features: Список признаков для использования (если None, будут использованы все)
        """
        self.data = data
        self.seq_length = seq_length
        self.target_horizon = target_horizon
        self.hw_decomposition = hw_decomposition
        self.use_features = use_features
        
        # Нормализация признаков
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Извлечение признаков
        self.features = self._preprocess_features()
        
        # Расчет доходностей (процентное изменение цены закрытия)
        self.returns = data['Close'].pct_change().fillna(0).values
        
        # Создание декомпозиции Holt-Winters, если включено
        self.hw_components = None
        if hw_decomposition:
            self.hw_components = self._create_hw_decomposition()
            
    def _preprocess_features(self):
        """Предобработка признаков для модели"""
        # Выбор релевантных признаков
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Добавление технических индикаторов
        df = self.data.copy()
        
        # 1. Добавление скользящих средних
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # 2. Добавление MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 3. Добавление RSI (14-периодный)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. Добавление волатильности (скользящее стандартное отклонение)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 5. Добавление импульса цены
        df['Momentum'] = df['Close'] / df['Close'].shift(5) - 1
        
        # Заполнение значений NaN, которые появляются из расчетов
        df = df.ffill().bfill()  # Используем более надежные методы вместо fillna(method=...)
        
        # Определение, какие признаки использовать
        all_features = feature_columns + ['MA_5', 'MA_20', 'MACD', 'MACD_Signal', 'RSI', 'Volatility', 'Momentum']
        
        if self.use_features is not None:
            # Проверяем, что все указанные признаки существуют
            for feature in self.use_features:
                if feature not in all_features:
                    raise ValueError(f"Признак {feature} не найден в данных")
            feature_set = df[self.use_features]
        else:
            feature_set = df[all_features]
            
        print(f"Используемые признаки: {feature_set.columns.tolist()}")
        print(f"Размерность входных данных: {len(feature_set.columns)}")
        
        # Масштабирование признаков
        scaled_features = self.feature_scaler.fit_transform(feature_set)
        
        return scaled_features
        
    def _create_hw_decomposition(self):
        """Создание декомпозиции Holt-Winters для временного ряда"""
        # Мы будем использовать упрощенный подход:
        # Для каждого окна мы будем подгонять HW и извлекать компоненты
        
        # Это вычислительно затратно во время обучения, поэтому мы предвычисляем:
        print("Генерация декомпозиции Holt-Winters...")
        
        # Используем ExponentialSmoothing из statsmodels для Holt-Winters
        # Мы будем просто декомпозировать ряд цен закрытия для простоты
        close_series = self.data['Close'].values
        
        # Инициализация хранилища для компонентов
        level = np.zeros_like(close_series)
        trend = np.zeros_like(close_series)
        seasonal = np.zeros_like(close_series)
        
        # Для упрощения и надежности используем только простую декомпозицию
        # Уровень: 20-периодное скользящее среднее
        for i in range(len(close_series)):
            if i < 20:
                level[i] = np.mean(close_series[:i+1])
            else:
                level[i] = np.mean(close_series[i-19:i+1])
        
        # Тренд: разница в уровнях
        trend[1:] = level[1:] - level[:-1]
        
        # Сезонный: исходный - уровень
        seasonal = close_series - level
        
        # Объединение компонентов в один массив
        components = np.column_stack((level, trend, seasonal))
        
        # Масштабирование компонентов
        hw_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_components = hw_scaler.fit_transform(components)
        
        print("Декомпозиция Holt-Winters завершена.")
        return scaled_components
        
    def __len__(self):
        """Возвращает количество доступных последовательностей"""
        return len(self.data) - self.seq_length - self.target_horizon
        
    def __getitem__(self, idx):
        """Получить одну последовательность"""
        # Извлечение последовательности признаков
        features_seq = self.features[idx:idx+self.seq_length]
        
        # Цель - будущая цена закрытия
        target_idx = idx + self.seq_length + self.target_horizon - 1
        target = self.data['Close'].iloc[target_idx]
        
        # Масштабирование цели
        target_scaled = self.target_scaler.fit_transform(
            np.array(target).reshape(-1, 1)
        ).flatten()[0]
        
        # Подготовка компонентов HW, если доступны
        hw_seq = None
        if self.hw_decomposition and self.hw_components is not None:
            hw_seq = self.hw_components[idx:idx+self.seq_length]
            
        return {
            'features': torch.FloatTensor(features_seq),
            'target': torch.FloatTensor([target_scaled]),
            'hw_components': torch.FloatTensor(hw_seq) if hw_seq is not None else None,
            'raw_target': target,
            'idx': idx + self.seq_length
        }

def load_and_prepare_data(file_path):
    """
    Загрузка и подготовка данных криптовалюты из CSV
    """
    # Загрузка данных
    df = pd.read_csv(file_path)
    
    # Переименование столбцов на основе заголовка из снимка экрана
    df.columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    
    # Преобразование временных меток
    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Close time'] = pd.to_datetime(df['Close time'])
    
    # Установка индекса на datetime
    df.set_index('Open time', inplace=True)
    
    # Обеспечение числовых значений
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'Quote asset volume', 'Number of trades',
                      'Taker buy base asset volume', 'Taker buy quote asset volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    # Для простоты удаляем столбец 'Ignore'
    df = df.drop(columns=['Ignore'])
    
    return df

##########################
# ФУНКЦИИ ДЛЯ ТЕСТИРОВАНИЯ МОДЕЛИ
##########################

def find_latest_model_checkpoint():
    """
    Поиск последней директории с сохраненной моделью
    """
    # Ищем все директории model_checkpoints_*
    checkpoint_dirs = glob.glob('model_checkpoints_*')
    
    if not checkpoint_dirs:
        return None
    
    # Сортируем по дате создания (последний будет самым новым)
    latest_dir = max(checkpoint_dirs, key=os.path.getctime)
    
    # Проверяем наличие файла модели
    model_path = os.path.join(latest_dir, "best_model.pth")
    
    if os.path.exists(model_path):
        return model_path
    else:
        return None

def get_input_dim_from_model(model_path):
    """
    Получение размерности входных данных из обученной модели
    """
    # Загружаем state_dict модели
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Находим входной проекционный слой
    input_projection_weight = state_dict['input_projection.weight']
    
    # Размерность входных данных - это второе значение в размере весов (первое - это размерность выхода)
    input_dim = input_projection_weight.shape[1]
    
    print(f"Определена размерность входных данных: {input_dim}")
    
    return input_dim

def get_feature_list_for_model(input_dim):
    """
    Создает список признаков на основе размерности входных данных
    """
    # Базовые признаки, которые всегда используются
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Дополнительные признаки в порядке их создания в методе _preprocess_features
    additional_features = ['MA_5', 'MA_20', 'MACD', 'MACD_Signal', 'RSI', 'Volatility', 'Momentum']
    
    # Определяем, сколько дополнительных признаков нужно использовать
    num_additional_features = input_dim - len(base_features)
    
    if num_additional_features < 0:
        # Если размерность меньше базовых признаков, используем только часть базовых
        return base_features[:input_dim]
    elif num_additional_features <= len(additional_features):
        # Используем все базовые признаки и часть дополнительных
        return base_features + additional_features[:num_additional_features]
    else:
        # Размерность больше, чем у нас признаков - это странная ситуация
        print("ПРЕДУПРЕЖДЕНИЕ: Размерность входных данных модели больше, чем доступное количество признаков")
        return base_features + additional_features
    
def load_trained_model(model_path):
    """
    Загрузка обученной модели из файла
    """
    # Определяем входную размерность из модели
    input_dim = get_input_dim_from_model(model_path)
    
    # Создаем модель с правильной размерностью
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation='gelu',
        use_holt_winters=True
    ).to(device)
    
    # Загрузка весов модели
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Переключение в режим оценки
    model.eval()
    
    return model, input_dim

def detect_market_regime(price_data, window=20):
    """
    Определение режима рынка (тренд или боковик)
    
    Args:
        price_data: Series с ценами закрытия
        window: Размер окна для анализа
        
    Returns:
        Series с режимами рынка (1: восходящий тренд, -1: нисходящий тренд, 0: боковик)
    """
    # Рассчитываем ADX (Average Directional Index) для определения силы тренда
    # Сначала рассчитываем истинный диапазон (True Range)
    high = price_data['High']
    low = price_data['Low']
    close = price_data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    # Рассчитываем направленное движение (Directional Movement)
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    # Обнуляем неподходящие значения
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
    minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
    
    # Рассчитываем индикаторы направления (Directional Indicators)
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    
    # Рассчитываем разницу и сумму индикаторов
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    
    # Рассчитываем ADX
    dx = 100 * (di_diff / di_sum)
    adx = dx.rolling(window=window).mean()
    
    # Определяем режим рынка
    # ADX > 25 указывает на наличие тренда
    # plus_di > minus_di указывает на восходящий тренд
    # plus_di < minus_di указывает на нисходящий тренд
    regime = pd.Series(0, index=price_data.index)  # По умолчанию: боковик
    
    # Восходящий тренд
    regime[(adx > 25) & (plus_di > minus_di)] = 1
    
    # Нисходящий тренд
    regime[(adx > 25) & (plus_di < minus_di)] = -1
    
    return regime

def calculate_volatility_ratio(price_data, short_window=5, long_window=20):
    """
    Расчет соотношения краткосрочной и долгосрочной волатильности
    
    Args:
        price_data: Series с ценами закрытия
        short_window: Короткое окно для расчета волатильности
        long_window: Длинное окно для расчета волатильности
        
    Returns:
        Series с соотношением волатильности
    """
    # Рассчитываем доходность
    returns = price_data['Close'].pct_change()
    
    # Рассчитываем волатильность на коротком и длинном окнах
    short_vol = returns.rolling(window=short_window).std()
    long_vol = returns.rolling(window=long_window).std()
    
    # Рассчитываем соотношение волатильности
    vol_ratio = short_vol / long_vol
    
    return vol_ratio

def generate_improved_trading_signals(predictions, df, min_holding_period=8, 
                                     entry_threshold=0.004, exit_threshold=0.002,
                                     stop_loss=0.015, take_profit=0.03):
    """
    Генерация улучшенных торговых сигналов с учетом минимального периода удержания
    и контроля рисков
    
    Args:
        predictions: прогнозируемые цены
        df: DataFrame с рыночными данными
        min_holding_period: Минимальный период удержания позиции (в барах)
        entry_threshold: Порог для входа в позицию (% изменения)
        exit_threshold: Порог для выхода из позиции (% изменения)
        stop_loss: Уровень стоп-лосса (% от цены входа)
        take_profit: Уровень тейк-профита (% от цены входа)
        
    Returns:
        DataFrame с сигналами
    """
    # Создаем копию DataFrame для работы
    market_data = df.copy()
    
    # Добавляем прогнозы
    market_data['Predicted_Price'] = predictions
    
    # Рассчитываем режим рынка
    market_data['Market_Regime'] = detect_market_regime(market_data)
    
    # Рассчитываем соотношение волатильности
    market_data['Volatility_Ratio'] = calculate_volatility_ratio(market_data)
    
    # Рассчитываем прогнозируемую доходность
    market_data['Predicted_Return'] = market_data['Predicted_Price'].pct_change()
    
    # Вычисляем фактическую доходность
    market_data['Actual_Return'] = market_data['Close'].pct_change()
    
    # Инициализируем сигналы
    market_data['Signal'] = 0  # 0: нет позиции, 1: длинная, -1: короткая
    market_data['Position'] = 0
    
    # Инициализируем переменные для отслеживания позиций
    current_position = 0  # 0: нет позиции, 1: длинная, -1: короткая
    position_entry_price = 0
    position_entry_time = 0
    bars_in_position = 0
    trailing_stop = 0
    
    # Проходим по данным для генерации сигналов с учетом минимального периода удержания
    for i in range(1, len(market_data)):
        # Обновляем счетчик баров в позиции, если есть открытая позиция
        if current_position != 0:
            bars_in_position += 1
        
        # Текущая цена
        current_price = market_data['Close'].iloc[i]
        
        # Проверяем стоп-лосс и тейк-профит, если есть открытая позиция
        if current_position == 1:  # Длинная позиция
            # Расчет процентного изменения от цены входа
            price_change_pct = (current_price - position_entry_price) / position_entry_price
            
            # Обновляем трейлинг-стоп, если цена растет
            if price_change_pct > trailing_stop:
                # Устанавливаем трейлинг-стоп на уровень 50% от текущей прибыли, но не ниже безубытка после min_holding_period
                if bars_in_position > min_holding_period:
                    trailing_stop = max(0, price_change_pct * 0.5)
            
            # Проверяем условия выхода из позиции
            if bars_in_position >= min_holding_period:
                # Стоп-лосс сработал
                if price_change_pct < -stop_loss:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
                # Тейк-профит сработал
                elif price_change_pct > take_profit:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
                # Трейлинг-стоп сработал
                elif trailing_stop > 0 and price_change_pct < trailing_stop:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
                # Сигнал на разворот вниз
                elif market_data['Predicted_Return'].iloc[i] < -exit_threshold:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
        
        elif current_position == -1:  # Короткая позиция
            # Расчет процентного изменения от цены входа (для короткой позиции отрицательное значение - это прибыль)
            price_change_pct = (position_entry_price - current_price) / position_entry_price
            
            # Обновляем трейлинг-стоп, если цена падает (т.е. короткая позиция прибыльна)
            if price_change_pct > trailing_stop:
                # Устанавливаем трейлинг-стоп на уровень 50% от текущей прибыли, но не ниже безубытка после min_holding_period
                if bars_in_position > min_holding_period:
                    trailing_stop = max(0, price_change_pct * 0.5)
            
            # Проверяем условия выхода из позиции
            if bars_in_position >= min_holding_period:
                # Стоп-лосс сработал
                if price_change_pct < -stop_loss:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
                # Тейк-профит сработал
                elif price_change_pct > take_profit:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
                # Трейлинг-стоп сработал
                elif trailing_stop > 0 and price_change_pct < trailing_stop:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
                # Сигнал на разворот вверх
                elif market_data['Predicted_Return'].iloc[i] > exit_threshold:
                    current_position = 0
                    market_data['Signal'].iloc[i] = 0
                    bars_in_position = 0
                    trailing_stop = 0
        
        # Генерация новых сигналов, если нет открытой позиции
        if current_position == 0:
            # Проверяем условия для входа в длинную позицию
            # Прогнозируемая доходность выше порога и предпочтительно в тренде
            if (market_data['Predicted_Return'].iloc[i] > entry_threshold and 
                (market_data['Market_Regime'].iloc[i] >= 0 or market_data['Volatility_Ratio'].iloc[i] < 1.2)):
                current_position = 1
                market_data['Signal'].iloc[i] = 1
                position_entry_price = current_price
                position_entry_time = i
                bars_in_position = 0
                trailing_stop = 0
            
            # Проверяем условия для входа в короткую позицию
            # Прогнозируемая доходность ниже отрицательного порога и предпочтительно в тренде
            elif (market_data['Predicted_Return'].iloc[i] < -entry_threshold and 
                 (market_data['Market_Regime'].iloc[i] <= 0 or market_data['Volatility_Ratio'].iloc[i] < 1.2)):
                current_position = -1
                market_data['Signal'].iloc[i] = -1
                position_entry_price = current_price
                position_entry_time = i
                bars_in_position = 0
                trailing_stop = 0
        
        # Устанавливаем текущую позицию
        market_data['Position'].iloc[i] = current_position
    
    return market_data

def calculate_position_size(capital, volatility, risk_per_trade=0.02):
    """
    Расчет размера позиции на основе волатильности и риска
    
    Args:
        capital: Размер капитала
        volatility: Волатильность (стандартное отклонение)
        risk_per_trade: Риск на сделку (% от капитала)
        
    Returns:
        Размер позиции
    """
    # Рассчитываем размер риска
    risk_amount = capital * risk_per_trade
    
    # Рассчитываем размер позиции (капитал под риском / волатильность)
    # Используем 2 * волатильность как ожидаемый диапазон движения цены
    position_size = risk_amount / (2 * volatility)
    
    return position_size

def calculate_enhanced_trading_metrics(signals_df, initial_capital=10000, risk_per_trade=0.02, commission=0.0005):
    """
    Расчет улучшенных метрик эффективности торговой стратегии с позиционированием на основе волатильности
    
    Args:
        signals_df: DataFrame с сигналами
        initial_capital: Начальный капитал
        risk_per_trade: Риск на сделку (% от капитала)
        commission: Комиссия за сделку (одна сторона)
        
    Returns:
        dict со статистиками
    """
    # Копия DataFrame для расчетов
    df = signals_df.copy()
    
    # Рассчитываем 20-дневную волатильность
    df['20d_Volatility'] = df['Close'].rolling(window=20).std()
    
    # Заполняем пропущенные значения волатильности
    df['20d_Volatility'] = df['20d_Volatility'].fillna(method='bfill').fillna(df['Close'].std())
    
    # Инициализируем массивы для хранения размеров позиций и капитала
    df['Position_Size'] = 0.0
    df['Capital'] = initial_capital
    df['Equity'] = initial_capital
    
    # Для первой строки устанавливаем начальный капитал
    df['Capital'].iloc[0] = initial_capital
    df['Equity'].iloc[0] = initial_capital
    
    # Проходим по данным для расчета размера позиций и капитала
    for i in range(1, len(df)):
        prev_capital = df['Capital'].iloc[i-1]
        
        # Рассчитываем размер позиции на основе волатильности и риска
        position_size = calculate_position_size(
            capital=prev_capital,
            volatility=df['20d_Volatility'].iloc[i-1],
            risk_per_trade=risk_per_trade
        )
        
        # Если есть изменение позиции, учитываем комиссию
        position_change = df['Position'].iloc[i] - df['Position'].iloc[i-1]
        
        # Сохраняем размер позиции
        df['Position_Size'].iloc[i] = position_size * abs(df['Position'].iloc[i])
        
        # Рассчитываем P&L для текущего бара
        pnl = position_size * df['Position'].iloc[i-1] * df['Actual_Return'].iloc[i]
        
        # Учитываем комиссию при изменении позиции
        commission_cost = position_size * abs(position_change) * commission
        
        # Обновляем капитал
        df['Capital'].iloc[i] = prev_capital + pnl - commission_cost
        
        # Рассчитываем equity (капитал + стоимость открытой позиции)
        df['Equity'].iloc[i] = df['Capital'].iloc[i]
    
    # Рассчитываем дневную доходность
    df['Daily_Return'] = df['Capital'].pct_change()
    
    # Рассчитываем просадку
    df['Equity_Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Equity_Peak']) / df['Equity_Peak']
    
    # Получаем количество сделок
    position_changes = df['Position'].diff()
    # Подсчитываем изменения позиции (не учитываем нулевые изменения)
    trades = position_changes[position_changes != 0]
    total_trades = len(trades) / 2  # Делим на 2, так как каждая сделка состоит из входа и выхода
    
    # Подсчитываем выигрышные сделки (наивное приближение)
    profitable_days = df[df['Daily_Return'] > 0]
    win_rate = len(profitable_days) / len(df[df['Daily_Return'] != 0])
    
    # Годовая доходность (предполагаем, что данные 15-минутные)
    # 4 периода в час * 24 часа * 365 дней = 35040 периодов в год
    annual_factor = 35040 / len(df)
    annual_return = (df['Capital'].iloc[-1] / initial_capital) ** annual_factor - 1
    
    # Волатильность (годовая)
    daily_std = df['Daily_Return'].std() * np.sqrt(4 * 24)  # Дневная волатильность
    annual_std = daily_std * np.sqrt(365)  # Годовая волатильность
    
    # Коэффициент Шарпа (предполагаем безрисковую ставку 0)
    sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0
    
    # Рассчитываем коэффициент Сортино (использует только отрицательную волатильность)
    downside_returns = df['Daily_Return'][df['Daily_Return'] < 0]
    downside_std = downside_returns.std() * np.sqrt(4 * 24 * 365)  # Годовая отрицательная волатильность
    sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
    
    # Максимальная просадка
    max_drawdown = df['Drawdown'].min()
    
    # Создание словаря с метриками
    metrics = {
        'Initial_Capital': initial_capital,
        'Final_Capital': df['Capital'].iloc[-1],
        'Total_Return': df['Capital'].iloc[-1] / initial_capital - 1,
        'Annual_Return': annual_return,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Max_Drawdown': max_drawdown,
        'Total_Trades': total_trades,
        'Win_Rate': win_rate,
        'Profit_Factor': abs(df[df['Daily_Return'] > 0]['Daily_Return'].sum() / 
                           df[df['Daily_Return'] < 0]['Daily_Return'].sum()) 
                      if df[df['Daily_Return'] < 0]['Daily_Return'].sum() != 0 else float('inf'),
        'Recovery_Factor': abs(df['Capital'].iloc[-1] / initial_capital - 1) / abs(max_drawdown) 
                        if max_drawdown != 0 else float('inf'),
    }
    
    return metrics, df

def plot_enhanced_trading_results(results_df, save_path=None):
    """
    Визуализация улучшенных результатов торговли
    """
    # Создание многопанельного графика
    fig, axs = plt.subplots(4, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # 1. График цены, прогнозов и позиций
    axs[0].plot(results_df.index, results_df['Close'], label='Фактическая цена', color='blue')
    axs[0].plot(results_df.index, results_df['Predicted_Price'], label='Прогноз', color='red', alpha=0.7)
    
    # Отметить длинные и короткие позиции
    longs = results_df[results_df['Position'] == 1].index
    shorts = results_df[results_df['Position'] == -1].index
    
    axs[0].scatter(longs, results_df.loc[longs, 'Close'], 
                  marker='^', color='green', s=100, label='Длинная позиция')
    axs[0].scatter(shorts, results_df.loc[shorts, 'Close'], 
                  marker='v', color='red', s=100, label='Короткая позиция')
    
    # Добавляем линии MA
    if 'MA_20' in results_df.columns:
        axs[0].plot(results_df.index, results_df['MA_20'], label='MA 20', color='purple', linestyle='--', alpha=0.7)
    
    axs[0].set_title('BTC Цена и Торговые Сигналы')
    axs[0].set_ylabel('Цена')
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. График капитала
    axs[1].plot(results_df.index, results_df['Capital'], label='Капитал', color='green')
    axs[1].set_title('Динамика Капитала')
    axs[1].set_ylabel('Капитал')
    axs[1].legend()
    axs[1].grid(True)
    
    # 3. График просадки
    axs[2].fill_between(results_df.index, results_df['Drawdown'] * 100, 0, 
                      color='red', alpha=0.3, label='Просадка')
    axs[2].set_title('Просадка Стратегии')
    axs[2].set_ylabel('Просадка (%)')
    axs[2].set_ylim(min(results_df['Drawdown'] * 100) * 1.1, 0)
    axs[2].grid(True)
    
    # 4. График размера позиции
    axs[3].plot(results_df.index, results_df['Position_Size'], label='Размер позиции', color='blue')
    axs[3].set_title('Размер Позиции')
    axs[3].set_ylabel('Размер позиции')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

##########################
# ОСНОВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
##########################

def main():
    # Параметры
    file_path = 'btc_15m_data_2018_to_2025.csv'  # Обновленный путь к файлу с данными
    seq_length = 96  # 24 часа 15-минутных данных
    target_horizon = 1
    
    # Параметры стратегии
    min_holding_period = 8  # Минимальный период удержания позиции (в барах)
    entry_threshold = 0.004  # Порог для входа в позицию (0.4%)
    exit_threshold = 0.002  # Порог для выхода из позиции (0.2%)
    stop_loss = 0.015  # Стоп-лосс (1.5%)
    take_profit = 0.03  # Тейк-профит (3%)
    risk_per_trade = 0.02  # Риск на сделку (2% от капитала)
    initial_capital = 10000  # Начальный капитал
    commission = 0.0005  # Комиссия 0.05% (типично для спотовой торговли)
    
    # Поиск последней обученной модели
    model_path = find_latest_model_checkpoint()
    
    if model_path is None:
        print("ОШИБКА: Не найдена обученная модель. Убедитесь, что вы обучили модель и сохранили ее.")
        return
    
    print(f"Найдена модель: {model_path}")
    
    # Загрузка модели и определение размерности входа
    model, input_dim = load_trained_model(model_path)
    
    # Определение списка признаков на основе размерности входа
    feature_list = get_feature_list_for_model(input_dim)
    print(f"Используемые признаки: {feature_list}")
    
    # Загрузка и подготовка данных
    print("Загрузка данных...")
    df = load_and_prepare_data(file_path)
    print(f"Загружены данные размером: {df.shape}")
    
    # Используем последние 20% данных для тестирования
    test_size = int(len(df) * 0.2)
    test_data = df.iloc[-test_size:]
    print(f"Размер тестового набора: {test_data.shape}")
    
    # Создание датасета для теста с правильным набором признаков
    test_dataset = CryptoDataset(test_data, seq_length, target_horizon, use_features=feature_list)
    
    # Получение прогнозов
    predictions = []
    actuals = []
    dates = []
    
    print("Генерация прогнозов...")
    
    # Пройдем по тестовым данным
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        
        # Перевод в тензоры и на нужное устройство
        features = sample['features'].unsqueeze(0).to(device)  # Добавляем размерность батча
        hw_components = sample.get('hw_components')
        
        if hw_components is not None:
            hw_components = hw_components.unsqueeze(0).to(device)
        
        # Прогнозирование
        with torch.no_grad():
            output = model(features, hw_components)
        
        # Преобразование в исходный масштаб
        pred = output.cpu().numpy().reshape(-1, 1)
        pred_original = test_dataset.target_scaler.inverse_transform(pred).flatten()[0]
        
        # Сохранение результатов
        predictions.append(pred_original)
        actuals.append(sample['raw_target'])
        
        # Получение даты/времени этой точки
        idx = sample['idx']
        if idx < len(test_data.index):
            dates.append(test_data.index[idx])
            
        # Показать прогресс
        if i % 1000 == 0:
            print(f"Обработано {i}/{len(test_dataset)} точек данных")
    
    print("Прогнозы сгенерированы.")
    
    # Создание DataFrame с результатами
    predictions_array = np.array(predictions)
    
    # Генерация улучшенных торговых сигналов
    print("Генерация торговых сигналов...")
    trading_results = generate_improved_trading_signals(
        predictions=predictions_array,
        df=test_data.iloc[seq_length:seq_length+len(predictions)],
        min_holding_period=min_holding_period,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    
    # Расчет улучшенных торговых метрик
    print("Расчет метрик стратегии...")
    metrics, enhanced_results = calculate_enhanced_trading_metrics(
        signals_df=trading_results,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        commission=commission
    )
    
    # Вывод метрик
    print("\n===== Метрики Торговой Стратегии =====")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Создание директории для сохранения результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"enhanced_trading_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Визуализация результатов
    plot_path = os.path.join(results_dir, "enhanced_trading_performance.png")
    plot_enhanced_trading_results(enhanced_results, save_path=plot_path)
    
    # Сохранение результатов в CSV
    csv_path = os.path.join(results_dir, "enhanced_trading_signals.csv")
    enhanced_results.to_csv(csv_path)
    
    # Сохранение метрик
    metrics_path = os.path.join(results_dir, "enhanced_trading_metrics.txt")
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nРезультаты сохранены в {results_dir}")
    
    # Дополнительно: средняя месячная доходность
    monthly_return = (1 + metrics['Total_Return']) ** (30 / len(enhanced_results)) - 1
    print(f"\nСредняя месячная доходность: {monthly_return:.2%}")
    print(f"Годовая доходность: {metrics['Annual_Return']:.2%}")
    print(f"Коэффициент Шарпа: {metrics['Sharpe_Ratio']:.2f}")
    print(f"Коэффициент Сортино: {metrics['Sortino_Ratio']:.2f}")
    print(f"Максимальная просадка: {metrics['Max_Drawdown']:.2%}")
    print(f"Процент выигрышных сделок: {metrics['Win_Rate']:.2%}")
    
    # Информация о целевой доходности
    if monthly_return > 0.3:
        print("\nЦель по ежемесячной доходности >30% ДОСТИГНУТА!")
    else:
        print("\nТекущая ежемесячная доходность: {:.2%}".format(monthly_return))
        print("\nДля достижения цели >30% рекомендуется:")
        print("1. Оптимизировать параметры стратегии (порог входа/выхода, стоп-лосс, тейк-профит)")
        print("2. Рассмотреть дообучение модели на недельной/месячной основе")
        print("3. Добавить дополнительные источники данных (сентимент, ончейн-метрики)")

if __name__ == "__main__":
    main()