#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helformer (Holt-Winters + Transformer) - Финальная версия
- Убрана попытка изменить use_boxcox в fit(), теперь задаём его только в конструкторе
- batch_first=True
- fillna без FutureWarning
- Без maxiter
- CSV_FILE="btc_15m_data_2018_to_2025.csv"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

# Тех. индикаторы (rsi, ema) - опционально
try:
    import ta
except ImportError:
    print("Модуль 'ta' не установлен (pip install ta)")

# ====================== Глобальные параметры =======================
CSV_FILE = "btc_15m_data_2018_to_2025.csv"
DATE_COL = "Open time"

SEASON_LENGTH = 96       
HW_TREND = 'add'         
HW_SEASONAL = 'add'      
ROLLING_WINDOW_DAYS = 30 

SEQ_LEN = 64
BATCH_SIZE = 32
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
DROPOUT = 0.1
LR = 1e-4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints_final"
DO_CONTINUE_TRAINING = False

COMMISSION_RATE = 0.001  # 0.1%
SPREAD = 0.0005
STOP_LOSS_PCT = 0.02
TRAILING_STOP_PCT = 0.03

FORECAST_STEPS = 5  
VOL_WINDOW = 50     
VOL_FACTOR = 1.0

OUTLIER_THRESHOLD = 5  

# =============== 1) Dataset (batch_first) ===============
class MultiStepResidualDataset(Dataset):
    """
    Мультишаговый датасет для обучающей выборки:
      X: (seq_len, features),
      y: (forecast_steps,)
    """
    def __init__(self, X, y_residual, seq_len, forecast_steps):
        super().__init__()
        self.X = X
        self.y = y_residual
        self.seq_len = seq_len
        self.f_steps = forecast_steps
    
    def __len__(self):
        return len(self.X) - (self.seq_len + self.f_steps) + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx+self.seq_len]
        y_seq = self.y[idx+self.seq_len : idx+self.seq_len + self.f_steps]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32)
        )


# =============== 2) Трансформер (batch_first=True) ===============
class PositionalEncodingBatchFirst(nn.Module):
    """
    Позиционное кодирование для batch_first=True
    Формат входа: (B, seq_len, d_model)
    """
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # => (max_len, d_model)
        pe = pe.unsqueeze(0)   # => (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class MultiStepHelformerBatchFirst(nn.Module):
    """
    Трансформер с batch_first=True. 
    Вход: (B, seq_len, input_dim)
    Выход: (B, forecast_steps)
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1, forecast_steps=5):
        super().__init__()
        self.d_model = d_model
        self.forecast_steps = forecast_steps

        self.input_fc = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncodingBatchFirst(d_model)

        self.fc_out = nn.Linear(d_model, forecast_steps)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.input_fc(x)  
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)  # (B, seq_len, d_model)
        last_step = encoded[:, -1, :]          # (B, d_model)
        out = self.fc_out(last_step)           # (B, forecast_steps)
        return out


# =============== TRAIN / EVAL ===============
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


# =============== Торговая логика ===============
def apply_commission_spread(ret):
    # Учитываем 2 комиссии (вход + выход) + спред
    total_cost = (2 * COMMISSION_RATE) + SPREAD
    return ret - total_cost

def stop_loss_check(entry_price, current_price, direction):
    if direction>0:
        drop_pct = (entry_price - current_price)/entry_price
        if drop_pct>STOP_LOSS_PCT:
            return True
    else:
        rise_pct = (current_price - entry_price)/entry_price
        if rise_pct>STOP_LOSS_PCT:
            return True
    return False

def trailing_stop_check(entry_price, highest_price, lowest_price, current_price, direction):
    if direction>0:
        fall_pct = (highest_price - current_price)/highest_price
        if fall_pct>TRAILING_STOP_PCT:
            return True
    else:
        rise_pct = (current_price - lowest_price)/lowest_price
        if rise_pct>TRAILING_STOP_PCT:
            return True
    return False


# =============== MAIN ===============
def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    df = pd.read_csv(CSV_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df.set_index(DATE_COL, inplace=True)
    df.sort_index(inplace=True)

    # Удаляем NaN (важно иметь Close)
    df.dropna(subset=["Close"], inplace=True)

    # Фильтруем выбросы +/- OUTLIER_THRESHOLD * std
    c_mean = df["Close"].mean()
    c_std  = df["Close"].std()
    upper_bound = c_mean + OUTLIER_THRESHOLD*c_std
    lower_bound = c_mean - OUTLIER_THRESHOLD*c_std
    df = df[(df["Close"]>=lower_bound) & (df["Close"]<=upper_bound)]

    # Тех. индикаторы (rsi, ema20), если доступен 'ta'
    if 'ta' in globals():
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['ema20'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
    else:
        df['rsi'] = np.nan
        df['ema20'] = np.nan

    # Заполнение пропусков, без FutureWarning
    df = df.bfill()
    df = df.ffill()

    # Holt-Winters (задаём use_boxcox=False здесь, не меняем потом в fit)
    def fit_hw(ts):
        # Если method='L-BFGS-B' всё ещё даёт ошибку, замените на method='holts' или уберите
        hw_model = ExponentialSmoothing(
            ts,
            trend=HW_TREND,
            seasonal=HW_SEASONAL,
            seasonal_periods=SEASON_LENGTH,
            initialization_method='estimated',
            use_boxcox=False
        )
        hw_fit = hw_model.fit(
            optimized=True,
            method='L-BFGS-B'  # Уберите, если вызывает ошибку
        )
        return hw_fit

    def create_or_load_model(input_dim, forecast_steps, ckpt_file):
        model_ = MultiStepHelformerBatchFirst(
            input_dim=input_dim,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            forecast_steps=forecast_steps
        ).to(DEVICE)
        optimizer_ = optim.Adam(model_.parameters(), lr=LR)
        if DO_CONTINUE_TRAINING and os.path.exists(os.path.join(CHECKPOINT_DIR, ckpt_file)):
            st = torch.load(os.path.join(CHECKPOINT_DIR, ckpt_file), map_location=DEVICE)
            model_.load_state_dict(st["model_state"])
            optimizer_.load_state_dict(st["opt_state"])
        return model_, optimizer_

    window_size = ROLLING_WINDOW_DAYS*24*4
    start_idx = 0
    step_count= 0

    trades_records = []
    predictions_all = []
    actual_all = []

    capital = 1.0
    position=0
    entry_price=None
    highest_price=None
    lowest_price=None

    preds_eval=[]
    actual_eval=[]

    while True:
        train_end = start_idx + window_size
        if train_end>=len(df):
            break

        df_train = df.iloc[start_idx:train_end].copy()
        train_close = df_train["Close"].values

        hw_fit = fit_hw(train_close)  # обучаем HW
        fitted_hw = hw_fit.fittedvalues
        residual  = train_close - fitted_hw

        # Масштаб residual
        scaler_res = MinMaxScaler(feature_range=(-1,1))
        residual_scaled = scaler_res.fit_transform(residual.reshape(-1,1)).flatten()

        # Доп. фичи
        vol_scaled    = MinMaxScaler().fit_transform(df_train[['Volume']]).flatten()
        trades_scaled = MinMaxScaler().fit_transform(df_train[['Number of trades']]).flatten()
        rsi_scaled    = MinMaxScaler().fit_transform(df_train[['rsi']]).flatten()
        ema_scaled    = MinMaxScaler().fit_transform(df_train[['ema20']]).flatten()

        train_features = np.column_stack([
            residual_scaled,
            vol_scaled,
            trades_scaled,
            rsi_scaled,
            ema_scaled
        ])

        ds = MultiStepResidualDataset(train_features, residual_scaled, SEQ_LEN, FORECAST_STEPS)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        ckpt_name = f"helformer_{step_count}.pth"
        model, optimizer_ = create_or_load_model(5, FORECAST_STEPS, ckpt_name)

        # Обучение
        for ep in range(EPOCHS):
            loss_ = train_one_epoch(model, dl, optimizer_, DEVICE)
        final_loss = evaluate(model, dl, DEVICE)
        print(f"[Step {step_count}] Train range: {df_train.index[0]} -> {df_train.index[-1]}, MSE={final_loss:.6f}")

        # Сохраняем
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        torch.save({
            "model_state": model.state_dict(),
            "opt_state": optimizer_.state_dict()
        }, ckpt_path)

        # Тест
        test_start = train_end
        test_end   = test_start + window_size
        if test_end>len(df):
            test_end = len(df)
        df_test = df.iloc[test_start:test_end].copy()
        if len(df_test)<(SEQ_LEN+FORECAST_STEPS):
            print("Недостаточно данных для прогноза.")
            break

        hw_forecasts = hw_fit.forecast(len(df_test))

        # Масштаб test
        vol_t    = MinMaxScaler().fit_transform(df_test[['Volume']]).flatten()
        trades_t = MinMaxScaler().fit_transform(df_test[['Number of trades']]).flatten()
        rsi_t    = MinMaxScaler().fit_transform(df_test[['rsi']]).flatten()
        ema_t    = MinMaxScaler().fit_transform(df_test[['ema20']]).flatten()

        i=0
        c_std_ = df_train['Close'].rolling(VOL_WINDOW).std().fillna(0).iloc[-1]
        if np.isnan(c_std_) or c_std_==0:
            c_std_ = df_train['Close'].std()
        last_close_train = df_train['Close'].iloc[-1] if len(df_train)>0 else 1
        if last_close_train>0:
            adaptive_threshold = (c_std_/last_close_train)*VOL_FACTOR
        else:
            adaptive_threshold = 0.001

        # последние SEQ_LEN из train_features
        rolling_features = train_features[-SEQ_LEN:].copy()

        while i<(len(df_test) - SEQ_LEN - FORECAST_STEPS + 1):
            X_inp = torch.tensor(rolling_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_inp).cpu().numpy()[0]
            # Обратное масштабирование residual
            pred_res = scaler_res.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

            hw_seg   = hw_forecasts[i : i+FORECAST_STEPS]
            final_close_pred = hw_seg + pred_res

            ts_segment = df_test.index[i : i+FORECAST_STEPS]
            for k, ts_ in enumerate(ts_segment):
                predictions_all.append((ts_, final_close_pred[k]))
                actual_all.append((ts_, df_test.loc[ts_,'Close']))
                preds_eval.append(final_close_pred[k])
                actual_eval.append(df_test.loc[ts_,'Close'])

            # Торговая логика
            prev_pred=None
            for k, ts_ in enumerate(ts_segment):
                pred_close= final_close_pred[k]
                real_close= df_test.loc[ts_,'Close']

                if position==0:
                    if prev_pred is not None:
                        ratio= (pred_close - prev_pred)/prev_pred
                        if ratio>adaptive_threshold:
                            position=1
                            entry_price=real_close
                            highest_price=real_close
                            lowest_price=real_close
                        elif ratio< -adaptive_threshold:
                            position=-1
                            entry_price=real_close
                            highest_price=real_close
                            lowest_price=real_close
                else:
                    if real_close>highest_price: highest_price=real_close
                    if real_close<lowest_price:  lowest_price=real_close

                    if stop_loss_check(entry_price, real_close, position):
                        exit_p= real_close
                        ret= (exit_p - entry_price)/entry_price*position
                        ret= apply_commission_spread(ret)
                        capital*=(1+ret)
                        trades_records.append((ts_,position,entry_price,exit_p,ret,capital))
                        position=0
                        entry_price=None
                        highest_price=None
                        lowest_price=None
                    else:
                        if trailing_stop_check(entry_price, highest_price, lowest_price, real_close, position):
                            exit_p= real_close
                            ret= (exit_p - entry_price)/entry_price*position
                            ret= apply_commission_spread(ret)
                            capital*=(1+ret)
                            trades_records.append((ts_,position,entry_price,exit_p,ret,capital))
                            position=0
                            entry_price=None
                            highest_price=None
                            lowest_price=None
                        else:
                            ratio2= (pred_close - real_close)/real_close
                            if position>0 and ratio2< -adaptive_threshold:
                                exit_p= real_close
                                ret= (exit_p - entry_price)/entry_price
                                ret= apply_commission_spread(ret)
                                capital*=(1+ret)
                                trades_records.append((ts_,1,entry_price,exit_p,ret,capital))
                                position=0
                                entry_price=None
                                highest_price=None
                                lowest_price=None
                            elif position<0 and ratio2> adaptive_threshold:
                                exit_p= real_close
                                ret= (entry_price - exit_p)/entry_price
                                ret= apply_commission_spread(ret)
                                capital*=(1+ret)
                                trades_records.append((ts_,-1,entry_price,exit_p,ret,capital))
                                position=0
                                entry_price=None
                                highest_price=None
                                lowest_price=None

                prev_pred= pred_close

            i+=FORECAST_STEPS

        start_idx= test_start
        step_count+=1
        if start_idx>=len(df):
            break

    # Закрываем позицию, если осталась
    if position!=0 and entry_price is not None:
        final_close= df['Close'].iloc[-1]
        ret= (final_close - entry_price)/entry_price*position
        ret= apply_commission_spread(ret)
        capital*=(1+ret)
        trades_records.append((df.index[-1], position, entry_price, final_close, ret, capital))

    # Метрики прогноза
    pred_df = pd.DataFrame(predictions_all, columns=["timestamp","pred_close"]).set_index("timestamp")
    act_df  = pd.DataFrame(actual_all, columns=["timestamp","actual_close"]).set_index("timestamp")
    merged_df = pred_df.join(act_df, how="inner").sort_index()

    preds_arr  = np.array([x[1] for x in predictions_all])
    actual_arr = np.array([x[1] for x in actual_all])
    if len(preds_arr)>0:
        mae  = np.mean(np.abs(preds_arr - actual_arr))
        mape = np.mean(np.abs((preds_arr - actual_arr)/(actual_arr+1e-9)))*100
    else:
        mae  = 0
        mape = 0
    if len(preds_arr)>1:
        direction_acc = np.mean(
            np.sign(preds_arr[1:] - preds_arr[:-1]) == np.sign(actual_arr[1:] - actual_arr[:-1])
        )*100
    else:
        direction_acc=0

    print("\n=== Forecast Metrics ===")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Directional Accuracy: {direction_acc:.2f}%")

    if len(merged_df)>0:
        start_price= merged_df["actual_close"].iloc[0]
        merged_df["buy_hold_equity"] = merged_df["actual_close"]/start_price
    else:
        merged_df["buy_hold_equity"] = 1

    # Кривая капитала
    trades_df = pd.DataFrame(trades_records, columns=["timestamp","pos","entry","exit","ret","capital"]).set_index("timestamp")
    eq_curve= []
    for row in trades_df.itertuples():
        eq_curve.append((row.Index, row.capital))
    eq_df= pd.DataFrame(eq_curve, columns=["timestamp","capital"]).set_index("timestamp")
    eq_df= eq_df.reindex(merged_df.index, method='ffill').fillna(method='ffill')

    final_cap= eq_df["capital"].iloc[-1] if len(eq_df)>0 else capital
    bh_val= merged_df["buy_hold_equity"].iloc[-1] if "buy_hold_equity" in merged_df else 1
    print(f"\n=== Итоговый капитал стратегии: {final_cap:.2f}")
    print(f"Buy & Hold: {bh_val:.2f}")

    # Анализ сделок
    n_trades= len(trades_df)
    if n_trades>0:
        wins= sum(1 for x in trades_records if x[4]>0)
        losses= n_trades - wins
        win_rate= wins/n_trades*100
        avg_win= np.mean([x[4] for x in trades_records if x[4]>0]) if wins>0 else 0
        avg_loss= np.mean([x[4] for x in trades_records if x[4]<0]) if losses>0 else 0
        print(f"Сделок: {n_trades}, Win rate: {win_rate:.2f}%")
        print(f"Средняя прибыль (win): {avg_win:.4f}, Средний убыток (loss): {avg_loss:.4f}")
    else:
        print("Сделок не было.")

    # График
    plt.figure(figsize=(10,5))
    if len(eq_df)>0:
        plt.plot(eq_df.index, eq_df["capital"], label="Helformer Strategy")
    if "buy_hold_equity" in merged_df:
        plt.plot(merged_df.index, merged_df["buy_hold_equity"], label="Buy & Hold")
    plt.title("Helformer (use_boxcox=False in constructor), CSV_FILE=btc_15m_data_2018_to_2025.csv")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("""
Исправлено 'use_boxcox was set at model initialization and cannot be changed':
 - Мы задаём use_boxcox=False в ExponentialSmoothing(...) 
 - Не переопределяем в fit(...)

Если всё ещё ошибка с method='L-BFGS-B', замените 'L-BFGS-B' на 'holts' или уберите method.
Удачи!
    """)

if __name__ == "__main__":
    main()
