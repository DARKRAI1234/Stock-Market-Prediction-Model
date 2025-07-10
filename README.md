# Stock Prediction Transformer

A modular, production-ready, and open-source deep learning framework for **stock return prediction** using advanced Transformer architectures, sector-aware modeling, and robust data pipelines. This repository is designed for research, competition, and real-world deployment with Indian stock market data (NIFTY), but can be adapted to other markets.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Configuration](#configuration)
- [Data Scraping](#data-scraping)
- [Results & Metrics](#results--metrics)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Transformer-based Model:** Multi-head attention, sector embedding, and cross-stock context.
- **Comprehensive Data Pipeline:** Technical indicators, normalization, sector-aware features.
- **Flexible Training:** Configurable loss (Huber + Ranking), data augmentation, and evaluation metrics.
- **Attention Visualization:** Heatmaps for interpretability at stock and sector level.
- **Easy Data Scraping:** Robust script for Yahoo Finance with resume and batch support.
- **Production-Ready:** Modular code, YAML configs, CLI scripts, and logging.

## Project Structure

```
stock_prediction_transformer/
├── src/
│   ├── models/           # Model architectures (transformer, losses)
│   ├── data/             # Data processing (dataset, preprocessing)
│   ├── training/         # Training infrastructure (trainer, runner)
│   └── utils/            # Utilities (collate, metrics)
├── scripts/              # Executable scripts (train, test, data_scraper)
├── configs/              # Configuration files (config.yaml)
├── data/                 # Data directories (raw, processed, test)
├── outputs/              # Model outputs (models, predictions, plots)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/stock_prediction_transformer.git
   cd stock_prediction_transformer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```



## Quick Start

### 1. **Prepare Data**

- Use the provided data scraping script `scripts/data_scraper.py` to fetch historical stock data.
- **Create a folder named `data/raw` in the project root.**
- Save all scraped raw stock CSV files into the `data/raw` folder.

### 2. **Configure**

- Edit `configs/config.yaml` to set hyperparameters, features, and paths.

### 3. **Train Model**

```bash
python scripts/train.py \
    --train_data_dir ./data/processed \
    --symbols_file ./data/nifty_symbols.csv \
    --output_dir ./outputs/models
```

### 4. **Test & Evaluate**

```bash
python scripts/test.py \
    --test_data_dir ./data/test \
    --symbols_file ./data/nifty_symbols.csv \
    --model_path ./outputs/models/best_model.pt \
    --output_dir ./outputs/predictions \
    --save_attention
```

## Data Preparation

- **Raw Data:** Use `scripts/data_scraper.py` to fetch historical OHLCV data from Yahoo Finance.
- **Processed Data:** The pipeline computes 25+ technical indicators, sector averages, and normalizes features.
- **Symbols File:** CSV file with columns: `Symbol`, `Industry` (sector).

Example:
```csv
Symbol,Industry
RELIANCE,Energy
TCS,IT
HDFCBANK,Financials
...
```

## Training

- **Script:** `scripts/train.py`
- **Arguments:** See `--help` for all options (batch size, epochs, learning rate, etc.)
- **Config:** YAML file (`configs/config.yaml`) controls model/data/training parameters.
- **Model Saving:** Best model (by R²) is saved to `outputs/models/best_model.pt`.

## Testing & Evaluation

- **Script:** `scripts/test.py`
- **Outputs:**
  - Daily predictions as ranked CSVs.
  - Attention heatmaps (per day, sector).
  - Metrics: Precision@K, IRR@K, MRR@K, MAE, R².
- **Visualization:** Attention weights are visualized to interpret cross-stock and cross-sector dependencies.

## Configuration

All settings are managed in `configs/config.yaml`:

- **Model:** Architecture, embedding sizes, dropout, etc.
- **Data:** Window size, prediction horizon, features.
- **Training:** Batch size, epochs, optimizer, scheduler, loss weights.
- **Evaluation:** Metrics to compute and save.
- **Paths:** Data and output directories.

## Data Scraping

**Script:** `scripts/data_scraper.py`

- **Fetches data** from Yahoo Finance for all symbols in your CSV.
- **Features:**
  - Batch downloading with retries and delays.
  - Resume support (no duplicate downloads).
  - Saves individual and/or combined CSVs.
  - Logs errors and progress.
- **Usage Example:**
  ```bash
  python scripts/data_scraper.py \
      --symbols_file ./data/nifty_symbols.csv \
      --output_dir ./data/raw \
      --start_date 2020-01-01 \
      --end_date 2024-12-31 \
      --save_individual \
      --save_combined \
      --resume
  ```

## Results & Metrics

- **Ranking Metrics:** Precision@K, IRR@K, MRR@K for K = 5, 10, 20.
- **Return Metrics:** Mean Absolute Error (MAE), R² score.
- **Visualization:** Stock-level and sector-level attention heatmaps for interpretability.

## Customization

- **Add Features:** Edit `src/data/preprocessing.py` to include new technical indicators.
- **Change Model:** Modify `src/models/transformer.py` for architecture tweaks.
- **Adjust Training:** Tune hyperparameters in `configs/config.yaml` or via CLI.
- **New Markets:** Update symbol/sector files and retrain.

## Contributing

Contributions are welcome! Please:
- Fork the repo and create a feature branch.
- Follow the code style and modular structure.
- Add tests and update documentation as needed.
- Submit a pull request with a clear description.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for historical data.
- PyTorch, NumPy, Pandas, and the open-source ML community.
- Inspired by research in cross-asset attention and ranking-based financial modeling.

