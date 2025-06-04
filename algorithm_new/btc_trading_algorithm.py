import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import talib
from datetime import datetime

class AdaptiveTradingStrategy:
    """
    Adaptive trading strategy for BTC futures that switches between trend-following
    and mean-reversion based on market regime detection.
    """
    
    def __init__(self, data_path, initial_capital=10000, leverage=3):
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
        """
        self.data_path = Path(data_path)
        self.initial_capital = initial_capital
        self.leverage = leverage
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
        # EMA crossover (9 and 21 periods)
        self.data['ema_short'] = talib.EMA(self.data['close'], timeperiod=9)
        self.data['ema_long'] = talib.EMA(self.data['close'], timeperiod=21)
        
        # MACD indicator
        self.data['macd'], self.data['macd_signal'], self.data['macd_hist'] = talib.MACD(
            self.data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # --- Mean reversion indicators ---
        # RSI (14 periods)
        self.data['rsi'] = talib.RSI(self.data['close'], timeperiod=14)
        
        # Bollinger Bands
        self.data['bb_upper'], self.data['bb_middle'], self.data['bb_lower'] = talib.BBANDS(
            self.data['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # --- Market regime indicators ---
        # ADX (Average Directional Index) - for trend strength
        self.data['adx'] = talib.ADX(
            self.data['high'], self.data['low'], self.data['close'], timeperiod=14
        )
        
        # ATR (Average True Range) - for volatility
        self.data['atr'] = talib.ATR(
            self.data['high'], self.data['low'], self.data['close'], timeperiod=14
        )
        
        # Identify market regime (1 for trend, 0 for range/flat)
        self.data['trend_regime'] = np.where(self.data['adx'] > 25, 1, 0)
        
        # Calculate percentage distance from price to middle bollinger band
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
        
        # Drop NaN values after calculating indicators
        self.data.dropna(inplace=True)
        
        print(f"Added indicators. Working with {len(self.data)} rows of data after dropping NaNs.")
        return self.data
    
    def generate_signals(self):
        """Generate trading signals based on market regime and indicators"""
        
        # Initialize signal column
        self.data['signal'] = 0
        
        # Loop through data (excluding first row)
        for i in range(1, len(self.data)):
            current_regime = self.data['trend_regime'].iloc[i]
            
            # --- Trend following strategy (when ADX > 25) ---
            if current_regime == 1:  # We're in a trending market
                # EMA crossover strategy
                if (self.data['ema_short'].iloc[i-1] <= self.data['ema_long'].iloc[i-1] and 
                    self.data['ema_short'].iloc[i] > self.data['ema_long'].iloc[i]):
                    # Bullish crossover
                    self.data.loc[self.data.index[i], 'signal'] = 1
                elif (self.data['ema_short'].iloc[i-1] >= self.data['ema_long'].iloc[i-1] and 
                      self.data['ema_short'].iloc[i] < self.data['ema_long'].iloc[i]):
                    # Bearish crossover
                    self.data.loc[self.data.index[i], 'signal'] = -1
            
            # --- Mean reversion strategy (when ADX < 25) ---
            else:  # We're in a range-bound market
                # RSI strategy for mean reversion
                if self.data['rsi'].iloc[i] < 30:
                    # Oversold condition
                    self.data.loc[self.data.index[i], 'signal'] = 1
                elif self.data['rsi'].iloc[i] > 70:
                    # Overbought condition
                    self.data.loc[self.data.index[i], 'signal'] = -1
        
        return self.data
    
    def add_position_management(self):
        """Apply position management rules including stops and exits"""
        
        # Initialize position column (0 = no position, 1 = long, -1 = short)
        self.data['position'] = 0
        
        current_position = 0
        entry_price = 0
        
        for i in range(len(self.data)):
            signal = self.data['signal'].iloc[i]
            close_price = self.data['close'].iloc[i]
            
            # Exit conditions for trend regime
            if self.data['trend_regime'].iloc[i] == 1:
                # For trend strategy - exit on opposite signal or when price crosses EMA
                if current_position == 1:  # In a long position
                    if signal == -1 or close_price < self.data['ema_long'].iloc[i]:
                        current_position = 0  # Exit long
                        
                elif current_position == -1:  # In a short position
                    if signal == 1 or close_price > self.data['ema_long'].iloc[i]:
                        current_position = 0  # Exit short
            
            # Exit conditions for range regime
            else:
                # For mean reversion - take profit at middle band or opposite signal
                if current_position == 1:  # In a long position
                    if close_price > self.data['bb_middle'].iloc[i] or signal == -1:
                        current_position = 0  # Exit long
                        
                elif current_position == -1:  # In a short position
                    if close_price < self.data['bb_middle'].iloc[i] or signal == 1:
                        current_position = 0  # Exit short
            
            # Entry logic (only enter if not already in a position)
            if current_position == 0:
                if signal == 1:
                    current_position = 1  # Enter long
                    entry_price = close_price
                elif signal == -1:
                    current_position = -1  # Enter short
                    entry_price = close_price
            
            # Update position column
            self.data.loc[self.data.index[i], 'position'] = current_position
            
        return self.data
    
    def calculate_returns(self):
        """Calculate strategy returns accounting for leverage and fees"""
        
        # Calculate position changes
        self.data['position_change'] = self.data['position'].diff()
        
        # Initialize columns for returns and equity
        self.data['trade'] = 0
        self.data['fee'] = 0
        self.data['returns'] = 0
        self.data['equity'] = self.initial_capital
        
        # Loop through data to calculate returns and equity
        for i in range(1, len(self.data)):
            # Check if we have a position change (trade)
            if self.data['position_change'].iloc[i] != 0:
                # Calculate trade size (position value)
                price = self.data['close'].iloc[i]
                position_size = self.data['equity'].iloc[i-1] * self.leverage
                
                # Calculate commission (both entry and exit)
                commission = position_size * self.commission_rate
                self.data.loc[self.data.index[i], 'fee'] = commission
                
                # Record trade (for tracking)
                self.data.loc[self.data.index[i], 'trade'] = 1
            
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
                self.data.loc[self.data.index[i], 'returns'] = position_return
            
            # Update equity
            self.data.loc[self.data.index[i], 'equity'] = self.data['equity'].iloc[i-1] * (1 + self.data['returns'].iloc[i])
        
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
        sharpe_ratio = np.mean(self.data['returns']) / np.std(self.data['returns']) * np.sqrt(252)  # Annualized
        max_drawdown = self.data['drawdown'].min()
        
        # Market regime effectiveness
        trend_returns = self.data[self.data['trend_regime'] == 1]['returns']
        range_returns = self.data[self.data['trend_regime'] == 0]['returns']
        
        trend_sharpe = np.mean(trend_returns) / np.std(trend_returns) * np.sqrt(252) if len(trend_returns) > 0 else 0
        range_sharpe = np.mean(range_returns) / np.std(range_returns) * np.sqrt(252) if len(range_returns) > 0 else 0
        
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


# Main execution code
if __name__ == "__main__":
    # Path to data file - replace with your actual path
    file_path = Path('btc_4h_data_2018_to_2025.csv')
    
    # Initialize strategy
    strategy = AdaptiveTradingStrategy(
        data_path=file_path,
        initial_capital=10000,  # $10,000 starting capital
        leverage=3  # Using 3x leverage
    )
    
    # Run backtest
    results = strategy.run_backtest()
    
    # Output monthly return (target is ~30% per month)
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
    
    # Additional adjustments recommendation
    print("\nPotential Optimizations:")
    print("1. Adjust ADX threshold for regime identification")
    print("2. Fine-tune EMA periods for trend following")
    print("3. Modify RSI thresholds for mean reversion")
    print("4. Test different leverage levels")
    print("5. Implement dynamic position sizing based on volatility")



pip install TA_Lib-0.4.29-cp310-cp310-win_amd64.whl
    