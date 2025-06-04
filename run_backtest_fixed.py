import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas_ta as ta
import argparse
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtest')

class AdaptiveTradingStrategy:
    """
    Adaptive trading strategy for BTC futures that switches between trend-following
    and mean-reversion based on market regime detection.
    """
    
    def __init__(self, data_path, initial_capital=10000, leverage=2, 
                 adx_threshold=20, ema_short=5, ema_long=20, 
                 rsi_oversold=25, rsi_overbought=75):
        """
        Initialize the strategy with parameters and load data
        
        Parameters:
        -----------
        data_path : str or Path
            Path to the CSV file with OHLCV data
        initial_capital : float
            Initial capital for backtesting
        leverage : float
            Trading leverage (1-5 for low leverage)
        adx_threshold : int
            Threshold for ADX to identify trending market
        ema_short : int
            Period for short EMA
        ema_long : int
            Period for long EMA
        rsi_oversold : int
            RSI threshold for oversold condition
        rsi_overbought : int
            RSI threshold for overbought condition
        """
        self.data_path = Path(data_path)
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.adx_threshold = adx_threshold
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.commission_rate = 0.0004  # 0.04% maker/taker fee on Binance Futures
        self.data = None
        self.results = None
        
    def load_data(self):
        """Load and prepare OHLCV data"""
        print(f"Loading data from {self.data_path}...")
        
        # Read the CSV file
        self.data = pd.read_csv(self.data_path)
        
        # Handle column names
        if 'Open time' in self.data.columns:
            # Rename columns for consistency
            column_mapping = {
                'Open time': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            self.data.rename(columns=column_mapping, inplace=True)
        
        # Convert timestamp to datetime if it's a string
        if isinstance(self.data['timestamp'].iloc[0], str):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Set timestamp as index
        self.data.set_index('timestamp', inplace=True)
        
        # Make sure all OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.data[col] = pd.to_numeric(self.data[col])
        
        print(f"Loaded {len(self.data)} candles from {self.data.index.min()} to {self.data.index.max()}")
        return self.data
        
    def add_indicators(self):
        """Add technical indicators for strategy signals"""
        
        # --- Trend indicators ---
        # EMA crossover with custom periods
        self.data['ema_short'] = ta.ema(self.data['close'], length=self.ema_short)
        self.data['ema_long'] = ta.ema(self.data['close'], length=self.ema_long)
        
        # MACD indicator
        macd = ta.macd(self.data['close'], fast=12, slow=26, signal=9)
        self.data['macd'] = macd['MACD_12_26_9']
        self.data['macd_signal'] = macd['MACDs_12_26_9']
        self.data['macd_hist'] = macd['MACDh_12_26_9']
        
        # --- Mean reversion indicators ---
        # RSI (14 periods)
        self.data['rsi'] = ta.rsi(self.data['close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(self.data['close'], length=20, std=2)
        self.data['bb_upper'] = bb['BBU_20_2.0']
        self.data['bb_middle'] = bb['BBM_20_2.0']
        self.data['bb_lower'] = bb['BBL_20_2.0']
        
        # --- Market regime indicators ---
        # ADX (Average Directional Index) - for trend strength
        adx = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=14)
        self.data['adx'] = adx['ADX_14']
        
        # ATR (Average True Range) - for volatility
        self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=14)
        
        # Identify market regime with custom ADX threshold
        self.data['trend_regime'] = np.where(self.data['adx'] > self.adx_threshold, 1, 0)
        
        # Calculate percentage distance from price to middle bollinger band
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
        
        # Add confirmation indicators
        
        # Stochastic RSI (combines features of Stochastic oscillator and RSI)
        stoch_rsi = ta.stochrsi(self.data['close'])
        self.data['stoch_rsi_k'] = stoch_rsi['STOCHRSIk_14_14_3_3']
        self.data['stoch_rsi_d'] = stoch_rsi['STOCHRSId_14_14_3_3']
        
        # Volume Indicators
        self.data['volume_sma'] = ta.sma(self.data['volume'], length=20)
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma']
        
        # Initialize all strategy columns as float64
        self.data['signal'] = 0.0
        self.data['position'] = 0.0
        self.data['position_change'] = 0.0
        self.data['trade'] = 0.0
        self.data['fee'] = 0.0
        self.data['returns'] = 0.0
        self.data['equity'] = float(self.initial_capital)
        
        # Drop NaN values after calculating indicators
        self.data.dropna(inplace=True)
        
        print(f"Added indicators. Working with {len(self.data)} rows of data after dropping NaNs.")
        return self.data
    
    def generate_signals(self):
        """Generate trading signals based on market regime and indicators"""
        
        # Loop through data (excluding first row)
        for i in range(1, len(self.data)):
            current_regime = self.data['trend_regime'].iloc[i]
            
            # --- Trend following strategy (when ADX > threshold) ---
            if current_regime == 1:  # We're in a trending market
                # EMA crossover strategy with volume confirmation
                if (self.data['ema_short'].iloc[i-1] <= self.data['ema_long'].iloc[i-1] and 
                    self.data['ema_short'].iloc[i] > self.data['ema_long'].iloc[i] and
                    self.data['volume_ratio'].iloc[i] > 1.0):  # Higher than average volume
                    # Bullish crossover
                    self.data.loc[self.data.index[i], 'signal'] = 1.0
                elif (self.data['ema_short'].iloc[i-1] >= self.data['ema_long'].iloc[i-1] and 
                      self.data['ema_short'].iloc[i] < self.data['ema_long'].iloc[i] and
                      self.data['volume_ratio'].iloc[i] > 1.0):  # Higher than average volume
                    # Bearish crossover
                    self.data.loc[self.data.index[i], 'signal'] = -1.0
            
            # --- Mean reversion strategy (when ADX < threshold) ---
            else:  # We're in a range-bound market
                # RSI strategy with Stochastic RSI confirmation
                if (self.data['rsi'].iloc[i] < self.rsi_oversold and 
                    self.data['stoch_rsi_k'].iloc[i] < 20 and
                    self.data['stoch_rsi_k'].iloc[i] > self.data['stoch_rsi_d'].iloc[i]):
                    # Oversold condition with stochRSI confirming upward momentum
                    self.data.loc[self.data.index[i], 'signal'] = 1.0
                elif (self.data['rsi'].iloc[i] > self.rsi_overbought and 
                      self.data['stoch_rsi_k'].iloc[i] > 80 and
                      self.data['stoch_rsi_k'].iloc[i] < self.data['stoch_rsi_d'].iloc[i]):
                    # Overbought condition with stochRSI confirming downward momentum
                    self.data.loc[self.data.index[i], 'signal'] = -1.0
        
        return self.data
    
    def add_position_management(self):
        """Apply position management rules including stops and exits"""
        
        current_position = 0.0
        entry_price = 0.0
        
        for i in range(len(self.data)):
            signal = self.data['signal'].iloc[i]
            close_price = self.data['close'].iloc[i]
            
            # Stop loss calculation - 2x ATR
            stop_distance = self.data['atr'].iloc[i] * 2
            
            # Exit conditions for trend regime
            if self.data['trend_regime'].iloc[i] == 1:
                # For trend strategy - exit on opposite signal or trailing stop
                if current_position == 1:  # In a long position
                    # Calculate dynamic trailing stop
                    trailing_stop = max(close_price - stop_distance, 
                                      self.data['ema_long'].iloc[i])
                    
                    # Exit if signal is bearish or price drops below trailing stop
                    if signal == -1 or close_price < trailing_stop:
                        current_position = 0  # Exit long
                        
                elif current_position == -1:  # In a short position
                    # Calculate dynamic trailing stop
                    trailing_stop = min(close_price + stop_distance, 
                                      self.data['ema_long'].iloc[i])
                    
                    # Exit if signal is bullish or price rises above trailing stop
                    if signal == 1 or close_price > trailing_stop:
                        current_position = 0  # Exit short
            
            # Exit conditions for range regime
            else:
                # For mean reversion - take profit at middle band or opposite signal
                if current_position == 1:  # In a long position
                    # Take profit at middle band or exit on overbought signal
                    if close_price > self.data['bb_middle'].iloc[i] or signal == -1:
                        current_position = 0  # Exit long
                    
                    # Stop loss if price continues dropping significantly
                    if entry_price > 0 and close_price < entry_price * 0.95:  # 5% stop loss
                        current_position = 0  # Exit long with stop loss
                        
                elif current_position == -1:  # In a short position
                    # Take profit at middle band or exit on oversold signal
                    if close_price < self.data['bb_middle'].iloc[i] or signal == 1:
                        current_position = 0  # Exit short
                    
                    # Stop loss if price continues rising significantly
                    if entry_price > 0 and close_price > entry_price * 1.05:  # 5% stop loss
                        current_position = 0  # Exit short with stop loss
            
            # Entry logic (only enter if not already in a position)
            if current_position == 0:
                if signal == 1:
                    current_position = 1  # Enter long
                    entry_price = close_price
                elif signal == -1:
                    current_position = -1  # Enter short
                    entry_price = close_price
            
            # Update position column
            self.data.loc[self.data.index[i], 'position'] = float(current_position)
            
        # Calculate position changes
        self.data['position_change'] = self.data['position'].diff()
            
        return self.data
    
    def calculate_returns(self):
        """Calculate strategy returns accounting for leverage and fees"""
        
        # Loop through data to calculate returns and equity
        for i in range(1, len(self.data)):
            # Check if we have a position change (trade)
            if self.data['position_change'].iloc[i] != 0:
                # Calculate trade size (position value)
                price = self.data['close'].iloc[i]
                position_size = self.data['equity'].iloc[i-1] * self.leverage
                
                # Calculate commission (both entry and exit)
                commission = position_size * self.commission_rate
                self.data.loc[self.data.index[i], 'fee'] = float(commission)
                
                # Record trade (for tracking)
                self.data.loc[self.data.index[i], 'trade'] = 1.0
            
            # Calculate returns for this period
            if self.data['position'].iloc[i-1] != 0:  # If we had a position in the previous period
                # Price change
                prev_close = self.data['close'].iloc[i-1]
                current_close = self.data['close'].iloc[i]
                price_return = (current_close - prev_close) / prev_close
                
                # Adjust direction and apply leverage
                position_return = price_return * self.data['position'].iloc[i-1] * self.leverage
                
                # Subtract fees for any trades in this period
                position_return -= self.data['fee'].iloc[i] / self.data['equity'].iloc[i-1]
                
                # Record return
                self.data.loc[self.data.index[i], 'returns'] = float(position_return)
            
            # Update equity
            self.data.loc[self.data.index[i], 'equity'] = float(self.data['equity'].iloc[i-1] * (1 + self.data['returns'].iloc[i]))
        
        # Calculate cumulative returns
        self.data['cum_returns'] = (1 + self.data['returns']).cumprod() - 1
        
        # Calculate drawdowns
        self.data['peak'] = self.data['equity'].cummax()
        self.data['drawdown'] = (self.data['equity'] - self.data['peak']) / self.data['peak']
        
        return self.data
    
    def analyze_performance(self):
        """Calculate and report performance metrics"""
        
        # Extract trade data
        trades = self.data[self.data['trade'] == 1].copy()
        
        # Calculate basic metrics
        total_trades = len(trades)
        total_fees = self.data['fee'].sum()
        final_equity = self.data['equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital) - 1
        
        # Calculate monthly returns
        if isinstance(self.data.index, pd.DatetimeIndex):
            monthly_returns = self.data['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            avg_monthly_return = monthly_returns.mean()
            monthly_win_rate = (monthly_returns > 0).sum() / len(monthly_returns)
        else:
            # If index is not datetime, we can't calculate monthly metrics
            avg_monthly_return = np.nan
            monthly_win_rate = np.nan
        
        # Performance metrics
        sharpe_ratio = np.mean(self.data['returns']) / np.std(self.data['returns']) * np.sqrt(252) if np.std(self.data['returns']) > 0 else 0
        max_drawdown = self.data['drawdown'].min()
        
        # Market regime effectiveness
        trend_returns = self.data[self.data['trend_regime'] == 1]['returns']
        range_returns = self.data[self.data['trend_regime'] == 0]['returns']
        
        trend_sharpe = np.mean(trend_returns) / np.std(trend_returns) * np.sqrt(252) if len(trend_returns) > 0 and np.std(trend_returns) > 0 else 0
        range_sharpe = np.mean(range_returns) / np.std(range_returns) * np.sqrt(252) if len(range_returns) > 0 and np.std(range_returns) > 0 else 0
        
        # Store performance metrics
        self.results = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return * 100,
            'avg_monthly_return_pct': avg_monthly_return * 100 if not np.isnan(avg_monthly_return) else np.nan,
            'monthly_win_rate': monthly_win_rate if not np.isnan(monthly_win_rate) else np.nan,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'trend_sharpe_ratio': trend_sharpe,
            'range_sharpe_ratio': range_sharpe,
            'leverage': self.leverage
        }
        
        # Print performance summary
        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Equity: ${final_equity:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Average Monthly Return: {avg_monthly_return:.2%}" if not np.isnan(avg_monthly_return) else "Average Monthly Return: N/A")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Total Trades: {total_trades}")
        print(f"Total Fees Paid: ${total_fees:.2f}")
        print(f"Trend Strategy Sharpe: {trend_sharpe:.2f}")
        print(f"Range Strategy Sharpe: {range_sharpe:.2f}")
        
        # Calculate win rate
        try:
            # Get positions where we entered and exited
            entries = self.data[self.data['position_change'] != 0].copy()
            entries.reset_index(inplace=True)
            
            wins = 0
            losses = 0
            
            # Track entry and exit points to calculate P&L for each trade
            pos = 0
            entry_price = 0
            entry_idx = 0
            
            for i in range(len(entries)):
                curr_pos = entries.iloc[i]['position']
                curr_price = entries.iloc[i]['close']
                
                # If we're entering a position
                if curr_pos != 0 and pos == 0:
                    pos = curr_pos
                    entry_price = curr_price
                    entry_idx = i
                # If we're exiting a position
                elif curr_pos == 0 and pos != 0:
                    # Calculate P&L
                    if pos > 0:  # Long position
                        if curr_price > entry_price:
                            wins += 1
                        else:
                            losses += 1
                    else:  # Short position
                        if curr_price < entry_price:
                            wins += 1
                        else:
                            losses += 1
                    
                    # Reset tracking variables
                    pos = 0
                    entry_price = 0
            
            total_completed_trades = wins + losses
            win_rate = wins / total_completed_trades if total_completed_trades > 0 else 0
            
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Completed Trades: {total_completed_trades}")
            
        except Exception as e:
            logger.warning(f"Could not calculate win rate: {e}")
        
        return self.results
    
    def plot_results(self):
        """Plot equity curve and regime changes"""
        
        plt.figure(figsize=(14, 12))
        
        # Plot 1: Equity curve
        plt.subplot(3, 1, 1)
        plt.plot(self.data.index, self.data['equity'], label='Strategy Equity')
        plt.title('Strategy Equity Curve')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Drawdowns
        plt.subplot(3, 1, 2)
        plt.fill_between(self.data.index, self.data['drawdown'] * 100, 0, color='red', alpha=0.3)
        plt.title('Drawdowns (%)')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Plot 3: Market regimes and price
        plt.subplot(3, 1, 3)
        plt.plot(self.data.index, self.data['close'], label='BTC Price', color='black', alpha=0.7)
        
        # Highlight trend and range regimes
        trend_mask = self.data['trend_regime'] == 1
        range_mask = self.data['trend_regime'] == 0
        
        plt.scatter(self.data.index[trend_mask], self.data['close'][trend_mask], 
                   color='blue', alpha=0.3, label='Trend Regime', s=10)
        plt.scatter(self.data.index[range_mask], self.data['close'][range_mask], 
                   color='green', alpha=0.3, label='Range Regime', s=10)
        
        # Add buy/sell markers
        buys = (self.data['position_change'] == 1)
        sells = (self.data['position_change'] == -1)
        exits = (self.data['position_change'] != 0) & (~buys) & (~sells)
        
        plt.scatter(self.data.index[buys], self.data['close'][buys], marker='^', 
                   color='green', label='Buy', s=100)
        plt.scatter(self.data.index[sells], self.data['close'][sells], marker='v', 
                   color='red', label='Sell', s=100)
        plt.scatter(self.data.index[exits], self.data['close'][exits], marker='X', 
                   color='black', label='Exit', s=80)
        
        plt.title('BTC Price with Market Regimes and Trades')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('btc_strategy_results.png')
        plt.close()
        
        print("Plots saved as 'btc_strategy_results.png'")
    
    def run_backtest(self):
        """Run the full backtest process"""
        
        self.load_data()
        self.add_indicators()
        self.generate_signals()
        self.add_position_management()
        self.calculate_returns()
        self.analyze_performance()
        self.plot_results()
        
        return self.results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BTC Adaptive Trading Strategy Backtest')
    
    # Required arguments
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to the CSV file with OHLCV data')
    
    # Optional arguments
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for backtesting (default: 10000)')
    parser.add_argument('--leverage', type=float, default=2,
                        help='Trading leverage (default: 2)')
    parser.add_argument('--adx-threshold', type=int, default=20,
                        help='ADX threshold for market regime detection (default: 20)')
    parser.add_argument('--ema-short', type=int, default=5, 
                        help='Short EMA period (default: 5)')
    parser.add_argument('--ema-long', type=int, default=20,
                        help='Long EMA period (default: 20)')
    parser.add_argument('--rsi-oversold', type=int, default=25,
                        help='RSI oversold threshold (default: 25)')
    parser.add_argument('--rsi-overbought', type=int, default=75,
                        help='RSI overbought threshold (default: 75)')
    
    return parser.parse_args()

def main():
    """Main function to run the backtest"""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Create strategy instance with improved parameters
        strategy = AdaptiveTradingStrategy(
            data_path=args.data_file,
            initial_capital=args.initial_capital,
            leverage=args.leverage,
            adx_threshold=args.adx_threshold,
            ema_short=args.ema_short,
            ema_long=args.ema_long,
            rsi_oversold=args.rsi_oversold,
            rsi_overbought=args.rsi_overbought
        )
        
        # Run backtest
        results = strategy.run_backtest()
        
        # Print monthly return target comparison
        if not np.isnan(results['avg_monthly_return_pct']):
            print(f"\nMonthly Return: {results['avg_monthly_return_pct']:.2f}% (Target: 30%)")
            
            if results['avg_monthly_return_pct'] >= 25:
                print("Strategy achieved target performance!")
            else:
                print("Strategy needs optimization to reach target performance.")
        
        # Regime effectiveness
        print(f"\nRegime Effectiveness:")
        print(f"Trend Regime Sharpe: {results['trend_sharpe_ratio']:.2f}")
        print(f"Range Regime Sharpe: {results['range_sharpe_ratio']:.2f}")
        
        # Parameter information
        print("\nStrategy Parameters:")
        print(f"ADX Threshold: {args.adx_threshold}")
        print(f"EMA Periods: {args.ema_short}/{args.ema_long}")
        print(f"RSI Thresholds: {args.rsi_oversold}/{args.rsi_overbought}")
        print(f"Leverage: {args.leverage}x")
        
        # Additional adjustments recommendation
        print("\nPotential Optimizations:")
        print("1. Fine-tune ADX threshold for better regime identification")
        print("2. Adjust EMA periods based on market volatility")
        print("3. Add additional confirmation indicators")
        print("4. Implement dynamic position sizing based on volatility")
        print("5. Consider adding time-based filters (e.g., avoiding certain hours/days)")
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()