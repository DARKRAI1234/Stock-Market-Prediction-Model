#!/usr/bin/env python3
"""
Testing script for Stock Prediction Transformer.
Run with: python scripts/test.py --help for full options
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.runner import StockPredictionRunner
from src.data.dataset import TestStockDataset
from src.utils.collate import test_collate_stocks


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sector_mapping(symbols_file):
    """Load sector mapping from CSV file."""
    df = pd.read_csv(symbols_file)
    
    # Create sector to ID mapping
    unique_sectors = df['Industry'].unique()
    sector_to_id = {sector: idx for idx, sector in enumerate(unique_sectors)}
    
    # Create symbol to sector ID mapping
    sector_map = {}
    for _, row in df.iterrows():
        sector_map[str(row['Symbol'])] = sector_to_id[row['Industry']]
    
    return sector_map, list(df['Symbol']), df


def save_predictions_to_csv(predictions, symbols, dates, output_dir):
    """Save daily predictions to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for pred_returns, syms, date in zip(predictions, symbols, dates):
        # Create ranked predictions
        stock_predictions = list(zip(syms, pred_returns))
        ranked_stocks = sorted(stock_predictions, key=lambda x: x[1], reverse=True)
        
        # Format date string
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y_%m_%d')
        else:
            date_str = str(date).replace('-', '_').replace(' ', '_').replace(':', '_')
        
        # Save to CSV
        filename = f"predictions_{date_str}.csv"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("Rank,Symbol,PredictedReturn\n")
            for rank, (symbol, pred_return) in enumerate(ranked_stocks, 1):
                f.write(f"{rank},{symbol},{pred_return:.6f}\n")
        
        print(f"Saved predictions to {filepath}")


def calculate_metrics(predictions, symbols, actual_returns):
    """Calculate evaluation metrics."""
    metrics = {k: {'precision': [], 'irr': [], 'mrr': []} for k in [5, 10, 20]}
    
    for pred_returns, syms, actual_rets in zip(predictions, symbols, actual_returns):
        # Create rankings
        stock_predictions = list(zip(syms, pred_returns))
        ranked_stocks = sorted(stock_predictions, key=lambda x: x[1], reverse=True)
        pred_symbols = [sym for sym, _ in ranked_stocks]
        
        actual_dict = {sym: ret for sym, ret in zip(syms, actual_rets)}
        true_ranked = sorted(actual_dict.items(), key=lambda x: x[1], reverse=True)
        true_symbols = [sym for sym, _ in true_ranked]
        
        # Calculate metrics for different K values
        for k in [5, 10, 20]:
            if len(pred_symbols) >= k:
                pred_top_k = pred_symbols[:k]
                true_top_k = true_symbols[:k]
                
                # Precision@K
                precision = len(set(pred_top_k) & set(true_top_k)) / k
                metrics[k]['precision'].append(precision)
                
                # IRR@K (only if no NaN values)
                if not np.isnan(actual_rets).any():
                    true_top_k_sum = sum([actual_dict[sym] for sym in true_top_k])
                    pred_top_k_sum = sum([actual_dict[sym] for sym in pred_top_k])
                    irr = true_top_k_sum - pred_top_k_sum
                    metrics[k]['irr'].append(irr)
                
                # MRR@K
                mrr = 0
                for idx, sym in enumerate(true_symbols[:k]):
                    if sym in pred_symbols:
                        pred_rank = pred_symbols.index(sym) + 1
                        mrr += 1.0 / pred_rank
                mrr /= k
                metrics[k]['mrr'].append(mrr)
    
    return metrics


def plot_attention_heatmap(attn_weights, symbols, sector_map, date, output_dir, sector_df):
    """Plot attention heatmap with sector boundaries."""
    # Get sectors for symbols
    sectors = [sector_map[sym] for sym in symbols]
    
    # Sort by sector
    symbol_sector = list(zip(symbols, sectors))
    symbol_sector_sorted = sorted(symbol_sector, key=lambda x: x[1])
    sorted_symbols = [x[0] for x in symbol_sector_sorted]
    sorted_indices = [symbols.index(sym) for sym in sorted_symbols]
    
    # Reorder attention matrix
    if len(attn_weights.shape) == 3:  # (num_heads, num_stocks, num_stocks)
        attn_weights_avg = np.mean(attn_weights, axis=0)
    else:
        attn_weights_avg = attn_weights
    
    attn_weights_sorted = attn_weights_avg[np.ix_(sorted_indices, sorted_indices)]
    
    # Plot
    plt.figure(figsize=(20, 20))
    sns.heatmap(attn_weights_sorted, cmap='viridis', 
                xticklabels=sorted_symbols, yticklabels=sorted_symbols)
    plt.title(f"Attention Weights Heatmap for {date}", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    
    # Add sector boundaries
    sorted_sectors = [x[1] for x in symbol_sector_sorted]
    sector_counts = []
    current_sector = sorted_sectors[0]
    count = 1
    for i in range(1, len(sorted_sectors)):
        if sorted_sectors[i] == current_sector:
            count += 1
        else:
            sector_counts.append(count)
            current_sector = sorted_sectors[i]
            count = 1
    sector_counts.append(count)
    
    cum_counts = np.cumsum(sector_counts)
    for boundary in cum_counts[:-1]:
        plt.axhline(boundary, color='white', linewidth=2)
        plt.axvline(boundary, color='white', linewidth=2)
    
    plt.tight_layout()
    date_str = date.strftime('%Y%m%d') if hasattr(date, 'strftime') else str(date)
    plt.savefig(os.path.join(output_dir, f"attention_heatmap_{date_str}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test Stock Prediction Transformer')
    
    # Data arguments
    parser.add_argument('--test_data_dir', type=str, required=True,
                       help='Directory containing test data CSV files')
    parser.add_argument('--symbols_file', type=str, required=True,
                       help='CSV file containing symbols and sector information')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    
    # Test arguments
    parser.add_argument('--window_size', type=int, default=None,
                       help='Window size (overrides config)')
    parser.add_argument('--prediction_horizon', type=int, default=None,
                       help='Prediction horizon (overrides config)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                       help='Directory to save predictions and visualizations')
    parser.add_argument('--save_attention', action='store_true',
                       help='Save attention visualizations')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.window_size is not None:
        config['data']['window_size'] = args.window_size
    if args.prediction_horizon is not None:
        config['data']['prediction_horizon'] = args.prediction_horizon
    
    # Load sector mapping and symbols
    print("Loading sector mapping and symbols...")
    sector_map, valid_symbols, sector_df = load_sector_mapping(args.symbols_file)
    print(f"Loaded {len(valid_symbols)} symbols across {len(set(sector_map.values()))} sectors")
    
    # Initialize runner for testing
    print("Initializing test runner...")
    runner = StockPredictionRunner(
        valid_symbols=valid_symbols,
        sector_map=sector_map,
        window_size=config['data']['window_size'],
        prediction_horizon=config['data']['prediction_horizon'],
        device=args.device,
        mode='test'
    )
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    runner.load_best_model(args.model_path)
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = TestStockDataset(
        input_dir=args.test_data_dir,
        valid_symbols=valid_symbols,
        sector_map=sector_map,
        window_size=config['data']['window_size'],
        prediction_horizon=config['data']['prediction_horizon'],
        min_history=1
    )
    
    # Run predictions
    print("Running predictions...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_collate_stocks,
        num_workers=0,
        pin_memory=True
    )
    
    predictions, symbols, dates, actual_returns, attn_weights = runner.trainer.predict(test_loader)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions to CSV files
    print("Saving predictions...")
    save_predictions_to_csv(predictions, symbols, dates, args.output_dir)
    
    # Calculate and save metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, symbols, actual_returns)
    
    # Print average metrics
    print("\nEvaluation Metrics:")
    metrics_data = []
    for k in [5, 10, 20]:
        precision_avg = np.mean(metrics[k]['precision']) if metrics[k]['precision'] else 0
        irr_avg = np.mean(metrics[k]['irr']) if metrics[k]['irr'] else np.nan
        mrr_avg = np.mean(metrics[k]['mrr']) if metrics[k]['mrr'] else 0
        
        print(f"K={k}:")
        print(f"  Precision@K: {precision_avg:.4f}")
        print(f"  IRR@K: {irr_avg:.4f}")
        print(f"  MRR@K: {mrr_avg:.4f}")
        
        metrics_data.append({
            'K': k,
            'Precision': precision_avg,
            'IRR': irr_avg,
            'MRR': mrr_avg
        })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    # Save attention visualizations
    if args.save_attention and len(attn_weights) > 0:
        print("Creating attention visualizations...")
        # Visualize first day as example
        plot_attention_heatmap(
            attn_weights[0], symbols[0], sector_map, dates[0], 
            args.output_dir, sector_df
        )
    
    print(f"\nTesting completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
