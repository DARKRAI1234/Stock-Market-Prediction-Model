#!/usr/bin/env python3
"""
Enhanced Stock Data Scraper for Indian Stock Market.
Fetches historical data from Yahoo Finance with robust error handling.
"""

import yfinance as yf
import pandas as pd
import os
import logging
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import sys
from pathlib import Path


def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, "data_scraper.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('data_scraper')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_symbols_list(symbols_file):
    """Load symbols from CSV file."""
    df = pd.read_csv(symbols_file)
    
    # Add .NS suffix for NSE stocks
    if 'Symbol' in df.columns:
        symbols = df['Symbol'].tolist()
        yahoo_tickers = [f"{symbol}.NS" for symbol in symbols]
        return list(zip(symbols, yahoo_tickers)), df
    else:
        raise ValueError("CSV file must contain 'Symbol' column")


def fetch_stock_data(ticker, start_date, end_date, retry_count=3, delay=1):
    """Fetch stock data with retry mechanism."""
    for attempt in range(retry_count):
        try:
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                progress=False,
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if not data.empty and len(data) > 10:  # Minimum data check
                return data
            else:
                if attempt < retry_count - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                return None
                
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise e
    
    return None


def process_stock_data(data, symbol):
    """Process and clean stock data."""
    if data is None or data.empty:
        return None
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Ensure we have the required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        return None
    
    # Clean data
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Add symbol column
    data['Symbol'] = symbol
    
    # Sort by date
    data = data.sort_values('Date')
    
    return data


def save_progress(completed_symbols, output_dir):
    """Save progress to resume later if needed."""
    progress_file = os.path.join(output_dir, "scraping_progress.txt")
    with open(progress_file, 'w') as f:
        for symbol in completed_symbols:
            f.write(f"{symbol}\n")


def load_progress(output_dir):
    """Load previously completed symbols."""
    progress_file = os.path.join(output_dir, "scraping_progress.txt")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()


def main():
    parser = argparse.ArgumentParser(description='Enhanced Stock Data Scraper')
    
    # Input arguments
    parser.add_argument('--symbols_file', type=str, required=True,
                       help='CSV file containing stock symbols')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Directory to save stock data')
    
    # Date arguments
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (YYYY-MM-DD), default is today')
    
    # Scraping arguments
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Number of stocks to process in each batch')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between requests (seconds)')
    parser.add_argument('--retry_count', type=int, default=3,
                       help='Number of retry attempts for failed requests')
    
    # Resume functionality
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    
    # Data options
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual CSV files for each stock')
    parser.add_argument('--save_combined', action='store_true',
                       help='Save combined CSV file with all stocks')
    
    args = parser.parse_args()
    
    # Set default end date to today
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting stock data scraping...")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Load symbols
    try:
        symbols_data, symbols_df = load_symbols_list(args.symbols_file)
        logger.info(f"Loaded {len(symbols_data)} symbols from {args.symbols_file}")
    except Exception as e:
        logger.error(f"Failed to load symbols: {e}")
        return
    
    # Load progress if resuming
    completed_symbols = set()
    if args.resume:
        completed_symbols = load_progress(args.output_dir)
        logger.info(f"Resuming: {len(completed_symbols)} symbols already completed")
    
    # Filter out completed symbols
    remaining_symbols = [(sym, ticker) for sym, ticker in symbols_data 
                        if sym not in completed_symbols]
    
    logger.info(f"Processing {len(remaining_symbols)} remaining symbols...")
    
    # Data storage
    all_data = []
    successful_downloads = 0
    failed_downloads = 0
    
    # Process symbols in batches
    for i in tqdm(range(0, len(remaining_symbols), args.batch_size), 
                  desc="Processing batches"):
        batch = remaining_symbols[i:i + args.batch_size]
        
        for symbol, yahoo_ticker in batch:
            try:
                logger.info(f"Fetching data for {symbol} ({yahoo_ticker})...")
                
                # Fetch data
                raw_data = fetch_stock_data(
                    yahoo_ticker, 
                    args.start_date, 
                    args.end_date,
                    retry_count=args.retry_count,
                    delay=args.delay
                )
                
                # Process data
                processed_data = process_stock_data(raw_data, symbol)
                
                if processed_data is not None and len(processed_data) > 0:
                    # Save individual file if requested
                    if args.save_individual:
                        file_path = os.path.join(args.output_dir, f"{symbol}.csv")
                        processed_data.to_csv(file_path, index=False)
                    
                    # Add to combined data
                    if args.save_combined:
                        all_data.append(processed_data)
                    
                    successful_downloads += 1
                    completed_symbols.add(symbol)
                    logger.info(f"✓ Successfully downloaded {symbol}: {len(processed_data)} records")
                    
                else:
                    failed_downloads += 1
                    logger.warning(f"✗ No valid data found for {symbol}")
                
            except Exception as e:
                failed_downloads += 1
                logger.error(f"✗ Error downloading {symbol}: {e}")
            
            # Add delay between requests
            time.sleep(args.delay)
        
        # Save progress after each batch
        save_progress(completed_symbols, args.output_dir)
        logger.info(f"Batch completed. Progress: {len(completed_symbols)}/{len(symbols_data)}")
    
    # Save combined data if requested
    if args.save_combined and all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(args.output_dir, "all_stocks_data.csv")
        combined_data.to_csv(combined_file, index=False)
        logger.info(f"Saved combined data to {combined_file}")
    
    # Save summary
    summary = {
        'total_symbols': len(symbols_data),
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads,
        'success_rate': successful_downloads / len(symbols_data) * 100,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(args.output_dir, "scraping_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Final report
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETED")
    logger.info(f"Total symbols: {len(symbols_data)}")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Success rate: {successful_downloads / len(symbols_data) * 100:.1f}%")
    logger.info(f"Data saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
