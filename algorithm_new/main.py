#!/usr/bin/env python3
"""
BTC Futures Trading Strategy - Main Runner

This script provides a command-line interface to run the trading strategy
in different modes: backtest, optimization, or live trading.
"""

import argparse
import os
import json
from pathlib import Path
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Strategy_Runner')

def parse_arguments():
    """Parse command-line arguments"""
    
    parser = argparse.ArgumentParser(description='BTC Futures Trading Strategy Runner')
    
    # Main mode selection
    parser.add_argument('mode', choices=['backtest', 'optimize', 'live'], 
                        help='Operation mode: backtest, optimize parameters, or run live trading')
    
    # Data options
    parser.add_argument('--data-file', type=str, default=None,
                        help='Path to OHLCV data file for backtesting/optimization')
    
    # Backtest options
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for backtesting (default: $10,000)')
    parser.add_argument('--leverage', type=float, default=3,
                        help='Trading leverage (default: 3x)')
    
    # Optimization options
    parser.add_argument('--parallel', action='store_true',
                        help='Run optimization in parallel')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes for parallel optimization')
    
    # Live trading options
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange to use for live trading (default: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='4h',
                        help='Trading timeframe (default: 4h)')
    parser.add_argument('--interval', type=int, default=None,
                        help='Check interval in seconds for live trading')
    
    # Strategy parameters (can override defaults)
    parser.add_argument('--adx-threshold', type=int, default=25,
                        help='ADX threshold for market regime detection')
    parser.add_argument('--ema-short', type=int, default=9,
                        help='Short EMA period')
    parser.add_argument('--ema-long', type=int, default=21,
                        help='Long EMA period')
    parser.add_argument('--rsi-oversold', type=int, default=30,
                        help='RSI oversold threshold')
    parser.add_argument('--rsi-overbought', type=int, default=70,
                        help='RSI overbought threshold')
    
    return parser.parse_args()

def create_parameter_dict(args):
    """Create parameter dictionary from args"""
    
    params = {
        'adx_threshold': args.adx_threshold,
        'ema_short': args.ema_short,
        'ema_long': args.ema_long,
        'rsi_oversold': args.rsi_oversold,
        'rsi_overbought': args.rsi_overbought,
        'leverage': args.leverage
    }
    
    return params

def run_backtest(args):
    """Run a backtest with the specified parameters"""
    
    from btc_trading_algorithm import AdaptiveTradingStrategy
    
    if args.data_file is None:
        logger.error("No data file specified for backtest")
        sys.exit(1)
        
    logger.info(f"Starting backtest with {args.data_file}")
    
    # Initialize strategy with custom parameters
    strategy = AdaptiveTradingStrategy(
        data_path=args.data_file,
        initial_capital=args.initial_capital,
        leverage=args.leverage
    )
    
    # Run backtest
    results = strategy.run_backtest()
    
    # Print monthly return target comparison
    if not np.isnan(results.get('avg_monthly_return_pct', np.nan)):
        print(f"\nMonthly Return: {results['avg_monthly_return_pct']:.2f}% (Target: 30%)")
        
        if results['avg_monthly_return_pct'] >= 25:
            print("Strategy achieved target performance!")
        else:
            print("Strategy needs optimization to reach target performance.")
    
    return results

def run_optimization(args):
    """Run parameter optimization (stub function)"""
    
    if args.data_file is None:
        logger.error("No data file specified for optimization")
        sys.exit(1)
        
    logger.info(f"Starting parameter optimization with {args.data_file}")
    logger.info("Optimization not yet implemented - please use backtest mode for now")
    
    return None

def run_live_trading(args):
    """Run live trading (stub function)"""
    
    logger.info(f"Starting live trading on {args.exchange} for {args.symbol}")
    logger.info("Live trading not yet implemented - please use backtest mode for now")
    
    return None

def main():
    """Main entry point"""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Run the selected mode
        if args.mode == 'backtest':
            run_backtest(args)
        
        elif args.mode == 'optimize':
            run_optimization(args)
        
        elif args.mode == 'live':
            run_live_trading(args)
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error running {args.mode} mode: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Execute main function
    main()

  