#!/usr/bin/env python3
"""
BTC Historical Data Downloader

This script downloads historical OHLCV data for BTC/USDT from various exchanges
and saves it in the format required by the trading algorithm.
"""

import ccxt
import pandas as pd
import argparse
import os
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_downloader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Data_Downloader')

def parse_arguments():
    """Parse command-line arguments"""
    
    parser = argparse.ArgumentParser(description='BTC Historical Data Downloader')
    
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange to download data from (default: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='4h',
                       help='Candlestick timeframe (default: 4h)')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                       help='Start date for data (default: 2018-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data (default: today)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: [symbol]_[timeframe]_[start]_[end].csv)')
    
    return parser.parse_args()

def download_historical_data(exchange_id, symbol, timeframe, since, until=None):
    """
    Download historical OHLCV data from the specified exchange
    
    Parameters:
    -----------
    exchange_id : str
        ID of the exchange to use (e.g., 'binance', 'bybit')
    symbol : str
        Trading symbol (e.g., 'BTC/USDT')
    timeframe : str
        Candlestick timeframe (e.g., '1h', '4h', '1d')
    since : str or int
        Start date as string ('YYYY-MM-DD') or timestamp in milliseconds
    until : str or int, optional
        End date as string ('YYYY-MM-DD') or timestamp in milliseconds
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLCV data
    """
    
    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        
        logger.info(f"Connected to {exchange_id}")
        
        # Convert date strings to timestamps if needed
        if isinstance(since, str):
            since = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
        
        if until is None:
            until = int(datetime.now().timestamp() * 1000)
        elif isinstance(until, str):
            until = int(datetime.strptime(until, '%Y-%m-%d').timestamp() * 1000)
        
        # Prepare variables for data collection
        all_candles = []
        current_since = since
        
        # Fetch data in chunks to handle exchange limits
        while current_since < until:
            logger.info(f"Fetching data for {symbol} from {datetime.fromtimestamp(current_since/1000)}")
            
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update the since parameter for the next iteration
                current_since = candles[-1][0] + 1
                
                # Respect exchange rate limits
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                # Wait and retry
                time.sleep(10)
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Downloaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

def format_data_for_strategy(df):
    """Format the data for use with the trading strategy"""
    
    # Create a copy of the DataFrame
    formatted_df = df.copy()
    
    # Rename columns to match the strategy's expected format
    formatted_df.rename(columns={
        'timestamp': 'Open time',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Add additional columns required by the strategy
    formatted_df['Close time'] = formatted_df['Open time'] + pd.Timedelta(hours=4)
    formatted_df['Quote asset volume'] = formatted_df['Volume'] * formatted_df['Close']
    formatted_df['Number of trades'] = 0  # Placeholder, not used by the strategy
    formatted_df['Taker buy base asset volume'] = formatted_df['Volume'] * 0.4  # Approximate
    formatted_df['Taker buy quote asset volume'] = formatted_df['Quote asset volume'] * 0.4  # Approximate
    formatted_df['Ignore'] = 0
    
    # Reset index to ensure Open time is a column
    formatted_df.reset_index(drop=True, inplace=True)
    
    # Format datetime columns as strings
    formatted_df['Open time'] = formatted_df['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    formatted_df['Close time'] = formatted_df['Close time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return formatted_df

def main():
    """Main entry point"""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Download historical data
        df = download_historical_data(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            since=args.start_date,
            until=args.end_date
        )
        
        if df.empty:
            logger.error("No data downloaded")
            return
        
        # Format data for the trading strategy
        formatted_df = format_data_for_strategy(df)
        
        # Generate output filename if not provided
        if args.output is None:
            symbol_clean = args.symbol.replace('/', '_')
            start_date = df['timestamp'].min().strftime('%Y%m%d')
            end_date = df['timestamp'].max().strftime('%Y%m%d')
            output_filename = f"{symbol_clean}_{args.timeframe}_{start_date}_{end_date}.csv"
        else:
            output_filename = args.output
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        
        # Save the data to CSV
        formatted_df.to_csv(output_filename, index=False)
        
        logger.info(f"Data saved to {output_filename}")
        
    except Exception as e:
        logger.error(f"Error in main routine: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()