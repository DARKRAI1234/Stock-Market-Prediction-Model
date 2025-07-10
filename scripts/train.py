#!/usr/bin/env python3
"""
Training script for Stock Prediction Transformer.
Run with: python scripts/train.py --help for full options
"""

import argparse
import os
import sys
import yaml
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.runner import StockPredictionRunner


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
    
    return sector_map, list(df['Symbol'])


def main():
    parser = argparse.ArgumentParser(description='Train Stock Prediction Transformer')
    
    # Data arguments
    parser.add_argument('--train_data_dir', type=str, required=True,
                       help='Directory containing training data CSV files')
    parser.add_argument('--symbols_file', type=str, required=True,
                       help='CSV file containing symbols and sector information')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--window_size', type=int, default=None,
                       help='Window size (overrides config)')
    parser.add_argument('--prediction_horizon', type=int, default=None,
                       help='Prediction horizon (overrides config)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/models',
                       help='Directory to save trained models')
    parser.add_argument('--model_name', type=str, default='best_model.pt',
                       help='Name of the saved model file')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.window_size is not None:
        config['data']['window_size'] = args.window_size
    if args.prediction_horizon is not None:
        config['data']['prediction_horizon'] = args.prediction_horizon
    
    # Load sector mapping and symbols
    print("Loading sector mapping and symbols...")
    sector_map, valid_symbols = load_sector_mapping(args.symbols_file)
    print(f"Loaded {len(valid_symbols)} symbols across {len(set(sector_map.values()))} sectors")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, args.model_name)
    
    # Initialize runner
    print("Initializing training runner...")
    runner = StockPredictionRunner(
        train_data_dir=args.train_data_dir,
        valid_symbols=valid_symbols,
        sector_map=sector_map,
        window_size=config['data']['window_size'],
        prediction_horizon=config['data']['prediction_horizon'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        device=args.device,
        mode='train'
    )
    
    # Start training
    print("Starting training...")
    print(f"Configuration:")
    print(f"  Window size: {config['data']['window_size']}")
    print(f"  Prediction horizon: {config['data']['prediction_horizon']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Output: {model_path}")
    print("-" * 60)
    
    runner.train(save_path=model_path)
    
    print(f"\nTraining completed! Model saved to: {model_path}")


if __name__ == '__main__':
    main()
