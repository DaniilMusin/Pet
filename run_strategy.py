#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуска улучшенной адаптивной стратегии для торговли BTC фьючерсами
"""

from enhanced_adaptive_strategy import EnhancedAdaptiveStrategy
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def main():
    # Путь к файлу с данными - проверяем текущий директорий
    data_path = 'btc_15m_data_2018_to_2025.csv'
    
    print(f"Проверка наличия файла с данными: {data_path}")
    if not os.path.exists(data_path):
        print(f"Ошибка: Файл {data_path} не найден")
        return
    
    print(f"Файл найден. Начинаем работу...")
    start_time = time.time()
    
    # Создаем экземпляр улучшенной стратегии
    strategy = EnhancedAdaptiveStrategy(
        data_path=data_path,
        initial_balance=10000,
        max_leverage=3,
        base_risk_per_trade=0.02,
        min_trades_interval=8,  # Уменьшаем интервал между сделками для большей активности
        timeframe='15m'  # Для 15-минутного таймфрейма
    )
    
    # Загружаем данные
    strategy.load_data()
    
    # Используем только последние 2 года данных для ускорения
    print("Ограничиваем данные последними 2 годами...")
    strategy.data = strategy.data.tail(17520)  # ~24*365*2 (2 года часовых данных)
    
    # Настраиваем параметры для лучшей работы на 15-минутных данных
    strategy.params.update({
        'short_ema': 8,        # Сокращаем периоды для 15m таймфрейма
        'long_ema': 24,
        'rsi_period': 10,
        'adx_period': 12,
        'bb_period': 15,
        'atr_period': 10,
        'rsi_oversold': 25,    # Более строгие условия для входа при перепроданности
        'rsi_overbought': 75,  # Более строгие условия для входа при перекупленности
        'atr_multiplier_sl': 2.0,  # Уменьшаем стоп-лосс для более короткого таймфрейма
        'atr_multiplier_tp': 4.0,  # Соответствующее значение тейк-профита
        'volume_threshold': 1.8,   # Увеличиваем требование к объему
        'trailing_stop_activation': 0.02,  # Активация трейлинг-стопа раньше
        'trailing_stop_distance': 0.01,    # Более близкий трейлинг-стоп
        'max_daily_trades': 8,      # Больше сделок на малом таймфрейме
        'max_consecutive_losses': 4, # Позволяем большему количеству убыточных сделок подряд
    })
    
    # Рассчитываем индикаторы
    strategy.calculate_indicators()
    
    # Запускаем бэктест с начальными параметрами
    strategy.run_backtest()
    
    # Анализируем результаты
    stats = strategy.analyze_results()
    
    # Строим график и сохраняем его
    strategy.plot_equity_curve("equity_curve_initial.png")
    
    # Оптимизация параметров с фокусом на более короткие периоды для 15m
    param_ranges = {
        'short_ema': (5, 15),
        'long_ema': (20, 40),
        'rsi_period': (8, 16),
        'rsi_oversold': (20, 30),
        'rsi_overbought': (70, 80),
        'adx_period': (8, 16),
        'adx_strong_trend': (20, 30),
        'adx_weak_trend': (10, 20),
        'atr_multiplier_sl': (1.5, 2.5),
        'atr_multiplier_tp': (3.0, 5.0),
        'volume_threshold': (1.5, 2.5),
        'trailing_stop_activation': (0.01, 0.03),
        'trailing_stop_distance': (0.005, 0.015)
    }
    
    # Уменьшаем количество итераций для тестирования
    n_trials = 5
    print(f"Запуск оптимизации с {n_trials} итерациями...")
    best_params, results = strategy.optimize_parameters(param_ranges, n_trials=n_trials, scoring='combined')
    
    # Применяем оптимизированные параметры
    optimized_stats = strategy.apply_optimized_parameters()
    
    # Строим график и сохраняем его
    strategy.plot_equity_curve("equity_curve_optimized.png")
    
    # Выводим сравнение результатов до и после оптимизации
    print("\n===== СРАВНЕНИЕ РЕЗУЛЬТАТОВ =====")
    print(f"                      | До оптимизации | После оптимизации")
    print(f"Месячная доходность   | {stats['monthly_return']:.2f}%        | {optimized_stats['monthly_return']:.2f}%")
    print(f"Максимальная просадка | {stats['max_drawdown']:.2f}%        | {optimized_stats['max_drawdown']:.2f}%")
    print(f"Win Rate              | {stats['win_rate']:.2f}%        | {optimized_stats['win_rate']:.2f}%")
    print(f"Профит-фактор         | {stats['profit_factor']:.2f}         | {optimized_stats['profit_factor']:.2f}")
    
    # Сохраняем оптимизированные параметры в файл
    with open('optimized_params.txt', 'w') as f:
        f.write("Оптимизированные параметры:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write("\nРезультаты:\n")
        f.write(f"Месячная доходность: {optimized_stats['monthly_return']:.2f}%\n")
        f.write(f"Максимальная просадка: {optimized_stats['max_drawdown']:.2f}%\n")
        f.write(f"Win Rate: {optimized_stats['win_rate']:.2f}%\n")
        f.write(f"Профит-фактор: {optimized_stats['profit_factor']:.2f}\n")
    
    # Время выполнения
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nВыполнение скрипта заняло {elapsed_time/60:.2f} минут")

if __name__ == "__main__":
    main()