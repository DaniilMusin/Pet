import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import json
from datetime import datetime

# Import our strategy class
from btc_trading_algorithm import AdaptiveTradingStrategy

class StrategyOptimizer:
    """
    Optimize the trading strategy parameters to achieve target performance
    """
    
    def __init__(self, data_path, initial_capital=10000):
        self.data_path = Path(data_path)
        self.initial_capital = initial_capital
        self.results = []
        
    def evaluate_parameters(self, params):
        """Evaluate a single set of parameters"""
        
        # Unpack parameters
        adx_threshold = params['adx_threshold']
        ema_short = params['ema_short']
        ema_long = params['ema_long']
        rsi_oversold = params['rsi_oversold']
        rsi_overbought = params['rsi_overbought']
        leverage = params['leverage']
        
        # Create custom strategy class with modified indicator calculations
        class CustomStrategy(AdaptiveTradingStrategy):
            def add_indicators(self):
                """Override to use custom parameters"""
                
                # --- Trend indicators ---
                # EMA crossover with custom periods
                self.data['ema_short'] = talib.EMA(self.data['close'], timeperiod=ema_short)
                self.data['ema_long'] = talib.EMA(self.data['close'], timeperiod=ema_long)
                
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
                
                # Identify market regime with custom ADX threshold
                self.data['trend_regime'] = np.where(self.data['adx'] > adx_threshold, 1, 0)
                
                # Calculate percentage distance from price to middle bollinger band
                self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
                
                # Drop NaN values after calculating indicators
                self.data.dropna(inplace=True)
                
                return self.data
                
            def generate_signals(self):
                """Override to use custom RSI parameters"""
                
                # Initialize signal column
                self.data['signal'] = 0
                
                # Loop through data (excluding first row)
                for i in range(1, len(self.data)):
                    current_regime = self.data['trend_regime'].iloc[i]
                    
                    # --- Trend following strategy (when ADX > threshold) ---
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
                    
                    # --- Mean reversion strategy (when ADX < threshold) ---
                    else:  # We're in a range-bound market
                        # RSI strategy with custom thresholds
                        if self.data['rsi'].iloc[i] < rsi_oversold:
                            # Oversold condition
                            self.data.loc[self.data.index[i], 'signal'] = 1
                        elif self.data['rsi'].iloc[i] > rsi_overbought:
                            # Overbought condition
                            self.data.loc[self.data.index[i], 'signal'] = -1
                
                return self.data
        
        # Initialize strategy with custom parameters
        strategy = CustomStrategy(
            data_path=self.data_path,
            initial_capital=self.initial_capital,
            leverage=leverage
        )
        
        # Run backtest with custom parameters
        try:
            results = strategy.run_backtest()
            
            # Add parameters to results
            results.update(params)
            
            return results
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            return None
    
    def generate_parameter_grid(self):
        """Generate a grid of parameters to test"""
        
        param_grid = {
            'adx_threshold': [20, 25, 30],
            'ema_short': [7, 9, 12],
            'ema_long': [21, 25, 30],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'leverage': [2, 3, 4, 5]
        }
        
        # Create all combinations of parameters
        param_combinations = []
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)
        
        print(f"Created {len(param_combinations)} parameter combinations to test")
        return param_combinations
    
    def run_optimization(self, parallel=True, max_workers=None):
        """Run the optimization process with parallel execution"""
        
        # Generate parameter grid
        param_combinations = self.generate_parameter_grid()
        
        if parallel:
            print("Running parallel optimization...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.evaluate_parameters, param_combinations))
                self.results = [r for r in results if r is not None]
        else:
            print("Running sequential optimization...")
            self.results = []
            for params in param_combinations:
                result = self.evaluate_parameters(params)
                if result is not None:
                    self.results.append(result)
        
        # Sort results by monthly return
        self.results.sort(key=lambda x: x.get('avg_monthly_return_pct', 0), reverse=True)
        
        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'optimization_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"Optimization complete. Tested {len(self.results)} parameter combinations.")
        print(f"Results saved to optimization_results_{timestamp}.json")
        
        return self.results
    
    def print_top_results(self, n=10):
        """Print top n parameter combinations by monthly return"""
        
        if not self.results:
            print("No results available. Run optimization first.")
            return
        
        print(f"\n--- TOP {n} PARAMETER COMBINATIONS ---")
        print(f"{'ADX':^5} | {'EMA-S':^5} | {'EMA-L':^5} | {'RSI-OS':^6} | {'RSI-OB':^6} | {'LEV':^3} | {'Monthly %':^9} | {'Sharpe':^6} | {'Max DD %':^8}")
        print("-" * 70)
        
        for i in range(min(n, len(self.results))):
            result = self.results[i]
            print(f"{result['adx_threshold']:5} | "
                  f"{result['ema_short']:5} | "
                  f"{result['ema_long']:5} | "
                  f"{result['rsi_oversold']:6} | "
                  f"{result['rsi_overbought']:6} | "
                  f"{result['leverage']:3} | "
                  f"{result.get('avg_monthly_return_pct', 0):9.2f} | "
                  f"{result.get('sharpe_ratio', 0):6.2f} | "
                  f"{result.get('max_drawdown_pct', 0):8.2f}")
    
    def plot_parameter_impact(self):
        """Plot the impact of different parameters on monthly returns"""
        
        if not self.results:
            print("No results available. Run optimization first.")
            return
        
        # Create a dataframe from results
        results_df = pd.DataFrame(self.results)
        
        # Create subplots for each parameter
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot for ADX threshold
        adx_groups = results_df.groupby('adx_threshold')['avg_monthly_return_pct'].mean()
        axes[0, 0].bar(adx_groups.index, adx_groups.values)
        axes[0, 0].set_title('Average Monthly Return by ADX Threshold')
        axes[0, 0].set_xlabel('ADX Threshold')
        axes[0, 0].set_ylabel('Avg Monthly Return (%)')
        
        # Plot for EMA short
        ema_short_groups = results_df.groupby('ema_short')['avg_monthly_return_pct'].mean()
        axes[0, 1].bar(ema_short_groups.index, ema_short_groups.values)
        axes[0, 1].set_title('Average Monthly Return by EMA Short Period')
        axes[0, 1].set_xlabel('EMA Short Period')
        axes[0, 1].set_ylabel('Avg Monthly Return (%)')
        
        # Plot for EMA long
        ema_long_groups = results_df.groupby('ema_long')['avg_monthly_return_pct'].mean()
        axes[0, 2].bar(ema_long_groups.index, ema_long_groups.values)
        axes[0, 2].set_title('Average Monthly Return by EMA Long Period')
        axes[0, 2].set_xlabel('EMA Long Period')
        axes[0, 2].set_ylabel('Avg Monthly Return (%)')
        
        # Plot for RSI oversold
        rsi_os_groups = results_df.groupby('rsi_oversold')['avg_monthly_return_pct'].mean()
        axes[1, 0].bar(rsi_os_groups.index, rsi_os_groups.values)
        axes[1, 0].set_title('Average Monthly Return by RSI Oversold Threshold')
        axes[1, 0].set_xlabel('RSI Oversold')
        axes[1, 0].set_ylabel('Avg Monthly Return (%)')
        
        # Plot for RSI overbought
        rsi_ob_groups = results_df.groupby('rsi_overbought')['avg_monthly_return_pct'].mean()
        axes[1, 1].bar(rsi_ob_groups.index, rsi_ob_groups.values)
        axes[1, 1].set_title('Average Monthly Return by RSI Overbought Threshold')
        axes[1, 1].set_xlabel('RSI Overbought')
        axes[1, 1].set_ylabel('Avg Monthly Return (%)')
        
        # Plot for Leverage
        leverage_groups = results_df.groupby('leverage')['avg_monthly_return_pct'].mean()
        axes[1, 2].bar(leverage_groups.index, leverage_groups.values)
        axes[1, 2].set_title('Average Monthly Return by Leverage')
        axes[1, 2].set_xlabel('Leverage')
        axes[1, 2].set_ylabel('Avg Monthly Return (%)')
        
        plt.tight_layout()
        plt.savefig('parameter_impact.png')
        plt.close()
        
        print("Parameter impact analysis saved to 'parameter_impact.png'")
    
    def get_best_parameters(self):
        """Get the best parameter set based on monthly return"""
        
        if not self.results:
            print("No results available. Run optimization first.")
            return None
        
        # Find the best result by monthly return
        best_result = max(self.results, key=lambda x: x.get('avg_monthly_return_pct', 0))
        
        # Extract parameters from the best result
        best_params = {
            'adx_threshold': best_result['adx_threshold'],
            'ema_short': best_result['ema_short'],
            'ema_long': best_result['ema_long'],
            'rsi_oversold': best_result['rsi_oversold'],
            'rsi_overbought': best_result['rsi_overbought'],
            'leverage': best_result['leverage']
        }
        
        return best_params


# Main execution code
if __name__ == "__main__":
    # Path to data file
    file_path = Path('btc_4h_data_2018_to_2025.csv')
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(data_path=file_path, initial_capital=10000)
    
    # Run optimization
    optimizer.run_optimization(parallel=True, max_workers=4)  # Use 4 cores
    
    # Print top results
    optimizer.print_top_results(10)
    
    # Plot parameter impact
    optimizer.plot_parameter_impact()
    
    # Get best parameters
    best_params = optimizer.get_best_parameters()
    
    print("\n--- BEST PARAMETERS ---")
    for param, value in best_params.items():
        print(f"{param}: {value}")