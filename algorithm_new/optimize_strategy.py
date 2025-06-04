import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas_ta as ta
import argparse
import sys
import logging
import json
from datetime import datetime
from itertools import product
import concurrent.futures
import copy

# Import the strategy class
from run_backtest_fixed import AdaptiveTradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Optimizer')

class StrategyOptimizer:
    """
    Optimize the BTC trading strategy parameters
    """
    
    def __init__(self, data_path, initial_capital=10000):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        data_path : str or Path
            Path to the CSV file with OHLCV data
        initial_capital : float
            Initial capital for backtesting
        """
        self.data_path = Path(data_path)
        self.initial_capital = initial_capital
        self.results = []
        
    def generate_parameter_grid(self, param_grid=None):
        """
        Generate a grid of parameters to test
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary of parameter values to test
        
        Returns:
        --------
        list
            List of parameter combinations to test
        """
        if param_grid is None:
            # Default parameter grid
            param_grid = {
                'adx_threshold': [20, 25, 30],
                'ema_short': [7, 9, 12],
                'ema_long': [21, 26, 30],
                'rsi_oversold': [25, 30, 35],
                'rsi_overbought': [65, 70, 75],
                'leverage': [2, 3, 4]
            }
        
        # Create all combinations of parameters
        param_combinations = []
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)
        
        logger.info(f"Created {len(param_combinations)} parameter combinations to test")
        return param_combinations
    
    def evaluate_parameters(self, params, verbose=False):
        """
        Evaluate a single set of parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to test
        verbose : bool
            Print detailed information if True
        
        Returns:
        --------
        dict
            Results of the backtest with these parameters
        """
        try:
            # Create a custom strategy class
            class CustomStrategy(AdaptiveTradingStrategy):
                def add_indicators(self):
                    """Override with custom parameters"""
                    
                    # --- Trend indicators ---
                    # EMA crossover with custom periods
                    self.data['ema_short'] = ta.ema(self.data['close'], length=params['ema_short'])
                    self.data['ema_long'] = ta.ema(self.data['close'], length=params['ema_long'])
                    
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
                    self.data['trend_regime'] = np.where(self.data['adx'] > params['adx_threshold'], 1, 0)
                    
                    # Calculate percentage distance from price to middle bollinger band
                    self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
                    
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
                    
                    if verbose:
                        print(f"Added indicators with params: {params}")
                        print(f"Working with {len(self.data)} rows of data after dropping NaNs.")
                    
                    return self.data
                
                def generate_signals(self):
                    """Override to use custom RSI parameters"""
                    
                    # Loop through data (excluding first row)
                    for i in range(1, len(self.data)):
                        current_regime = self.data['trend_regime'].iloc[i]
                        
                        # --- Trend following strategy (when ADX > threshold) ---
                        if current_regime == 1:  # We're in a trending market
                            # EMA crossover strategy
                            if (self.data['ema_short'].iloc[i-1] <= self.data['ema_long'].iloc[i-1] and 
                                self.data['ema_short'].iloc[i] > self.data['ema_long'].iloc[i]):
                                # Bullish crossover
                                self.data.loc[self.data.index[i], 'signal'] = 1.0
                            elif (self.data['ema_short'].iloc[i-1] >= self.data['ema_long'].iloc[i-1] and 
                                self.data['ema_short'].iloc[i] < self.data['ema_long'].iloc[i]):
                                # Bearish crossover
                                self.data.loc[self.data.index[i], 'signal'] = -1.0
                        
                        # --- Mean reversion strategy (when ADX < threshold) ---
                        else:  # We're in a range-bound market
                            # RSI strategy with custom thresholds
                            if self.data['rsi'].iloc[i] < params['rsi_oversold']:
                                # Oversold condition
                                self.data.loc[self.data.index[i], 'signal'] = 1.0
                            elif self.data['rsi'].iloc[i] > params['rsi_overbought']:
                                # Overbought condition
                                self.data.loc[self.data.index[i], 'signal'] = -1.0
                    
                    return self.data
            
            # Initialize strategy with custom parameters
            strategy = CustomStrategy(
                data_path=self.data_path,
                initial_capital=self.initial_capital,
                leverage=params['leverage']
            )
            
            # Run backtest (but skip plotting to save time)
            strategy.load_data()
            strategy.add_indicators()
            strategy.generate_signals()
            strategy.add_position_management()
            strategy.calculate_returns()
            results = strategy.analyze_performance()
            
            # Add parameters to results
            results.update(params)
            
            if verbose:
                print(f"Tested parameters: {params}")
                print(f"Total Return: {results['total_return_pct']:.2f}%")
                print(f"Monthly Return: {results['avg_monthly_return_pct']:.2f}%")
                print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
                print(f"Total Trades: {results['total_trades']}")
                print("-" * 40)
            
            return results
        
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {e}")
            return None
    
    def run_optimization(self, param_grid=None, parallel=False, max_workers=None, verbose=False):
        """
        Run the optimization process
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary of parameter values to test
        parallel : bool
            Run optimization in parallel if True
        max_workers : int
            Number of worker processes for parallel optimization
        verbose : bool
            Print detailed information if True
        
        Returns:
        --------
        list
            List of results for each parameter combination
        """
        # Generate parameter grid
        param_combinations = self.generate_parameter_grid(param_grid)
        
        if parallel:
            logger.info(f"Running parallel optimization with {max_workers} workers")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map parameters to executor
                future_to_params = {executor.submit(self.evaluate_parameters, params, False): params 
                                   for params in param_combinations}
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(future_to_params)):
                    params = future_to_params[future]
                    try:
                        result = future.result()
                        if result is not None:
                            self.results.append(result)
                            
                            if verbose:
                                print(f"Completed {i+1}/{len(param_combinations)}: {params}")
                                print(f"Total Return: {result['total_return_pct']:.2f}%")
                    except Exception as e:
                        logger.error(f"Error processing result for {params}: {e}")
        else:
            logger.info("Running sequential optimization")
            for i, params in enumerate(param_combinations):
                if verbose:
                    print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
                
                result = self.evaluate_parameters(params, verbose)
                if result is not None:
                    self.results.append(result)
        
        # Sort results by monthly return
        self.results.sort(key=lambda x: x.get('avg_monthly_return_pct', float('-inf')), reverse=True)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"optimization_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        
        logger.info(f"Optimization complete. Tested {len(self.results)} parameter combinations.")
        logger.info(f"Results saved to {results_file}")
        
        return self.results
    
    def print_top_results(self, n=10):
        """
        Print the top n parameter combinations
        
        Parameters:
        -----------
        n : int
            Number of top results to print
        """
        if not self.results:
            logger.warning("No results available. Run optimization first.")
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
        """
        Create plots showing the impact of each parameter on performance
        """
        if not self.results:
            logger.warning("No results available. Run optimization first.")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Create figure for plotting
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot impact of ADX threshold
        adx_df = df.groupby('adx_threshold')['avg_monthly_return_pct'].mean().reset_index()
        axes[0, 0].bar(adx_df['adx_threshold'], adx_df['avg_monthly_return_pct'], color='skyblue')
        axes[0, 0].set_title('Impact of ADX Threshold on Monthly Return')
        axes[0, 0].set_xlabel('ADX Threshold')
        axes[0, 0].set_ylabel('Average Monthly Return (%)')
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot impact of EMA short period
        ema_short_df = df.groupby('ema_short')['avg_monthly_return_pct'].mean().reset_index()
        axes[0, 1].bar(ema_short_df['ema_short'], ema_short_df['avg_monthly_return_pct'], color='skyblue')
        axes[0, 1].set_title('Impact of Short EMA Period on Monthly Return')
        axes[0, 1].set_xlabel('Short EMA Period')
        axes[0, 1].set_ylabel('Average Monthly Return (%)')
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot impact of EMA long period
        ema_long_df = df.groupby('ema_long')['avg_monthly_return_pct'].mean().reset_index()
        axes[1, 0].bar(ema_long_df['ema_long'], ema_long_df['avg_monthly_return_pct'], color='skyblue')
        axes[1, 0].set_title('Impact of Long EMA Period on Monthly Return')
        axes[1, 0].set_xlabel('Long EMA Period')
        axes[1, 0].set_ylabel('Average Monthly Return (%)')
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot impact of RSI oversold
        rsi_os_df = df.groupby('rsi_oversold')['avg_monthly_return_pct'].mean().reset_index()
        axes[1, 1].bar(rsi_os_df['rsi_oversold'], rsi_os_df['avg_monthly_return_pct'], color='skyblue')
        axes[1, 1].set_title('Impact of RSI Oversold Threshold on Monthly Return')
        axes[1, 1].set_xlabel('RSI Oversold Threshold')
        axes[1, 1].set_ylabel('Average Monthly Return (%)')
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot impact of RSI overbought
        rsi_ob_df = df.groupby('rsi_overbought')['avg_monthly_return_pct'].mean().reset_index()
        axes[2, 0].bar(rsi_ob_df['rsi_overbought'], rsi_ob_df['avg_monthly_return_pct'], color='skyblue')
        axes[2, 0].set_title('Impact of RSI Overbought Threshold on Monthly Return')
        axes[2, 0].set_xlabel('RSI Overbought Threshold')
        axes[2, 0].set_ylabel('Average Monthly Return (%)')
        axes[2, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot impact of leverage
        leverage_df = df.groupby('leverage')['avg_monthly_return_pct'].mean().reset_index()
        axes[2, 1].bar(leverage_df['leverage'], leverage_df['avg_monthly_return_pct'], color='skyblue')
        axes[2, 1].set_title('Impact of Leverage on Monthly Return')
        axes[2, 1].set_xlabel('Leverage')
        axes[2, 1].set_ylabel('Average Monthly Return (%)')
        axes[2, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('parameter_impact.png')
        plt.close()
        
        logger.info("Parameter impact plots saved to parameter_impact.png")
    
    def save_best_parameters(self, filename='best_parameters.json'):
        """
        Save the best parameters to a file
        
        Parameters:
        -----------
        filename : str
            Name of the file to save parameters to
        """
        if not self.results:
            logger.warning("No results available. Run optimization first.")
            return
        
        # Get best parameters (by monthly return)
        best_result = self.results[0]
        best_params = {
            'adx_threshold': best_result['adx_threshold'],
            'ema_short': best_result['ema_short'],
            'ema_long': best_result['ema_long'],
            'rsi_oversold': best_result['rsi_oversold'],
            'rsi_overbought': best_result['rsi_overbought'],
            'leverage': best_result['leverage']
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        logger.info(f"Best parameters saved to {filename}")
        return best_params


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BTC Trading Strategy Optimizer')
    
    # Required arguments
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to the CSV file with OHLCV data')
    
    # Optional arguments
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for backtesting (default: 10000)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run optimization in parallel')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes for parallel optimization')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during optimization')
    
    return parser.parse_args()

def main():
    """Main function to run the optimization"""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Initialize optimizer
        optimizer = StrategyOptimizer(
            data_path=args.data_file,
            initial_capital=args.initial_capital
        )
        
        # Define custom parameter grid (optional)
        # Change these values to test different parameters
        param_grid = {
            'adx_threshold': [15, 20, 25, 30],
            'ema_short': [5, 7, 9, 12],
            'ema_long': [20, 25, 30, 35],
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],
            'leverage': [1, 2, 3]
        }
        
        # Run optimization
        optimizer.run_optimization(
            param_grid=param_grid,
            parallel=args.parallel,
            max_workers=args.workers,
            verbose=args.verbose
        )
        
        # Print top results
        optimizer.print_top_results(10)
        
        # Plot parameter impact
        optimizer.plot_parameter_impact()
        
        # Save best parameters
        best_params = optimizer.save_best_parameters()
        
        print("\n--- BEST PARAMETERS ---")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
    except Exception as e:
        logger.error(f"Error running optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()