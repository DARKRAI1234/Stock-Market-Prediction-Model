import torch
from torch.utils.data import DataLoader

from ..data.dataset import EnhancedStockDataset
from ..models.transformer import StockTransformer
from ..utils.collate import collate_stocks
from .trainer import StockPredictionTrainer


class StockPredictionRunner:
    def __init__(
        self,
        train_data_dir=None,  # Optional for training
        valid_symbols=None,   # Required for both train and test
        sector_map=None,      # Required for both train and test
        window_size=50,
        prediction_horizon=5,
        batch_size=16,
        num_epochs=50,
        learning_rate=1e-4,
        device=None,
        mode='train'         # New parameter to specify 'train' or 'test'
    ):
        self.train_data_dir = train_data_dir
        self.valid_symbols = valid_symbols
        self.sector_map = sector_map
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.mode = mode
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize model (required for both modes)
        if valid_symbols is not None and sector_map is not None:
            num_sectors = max(sector_map.values()) + 1
            # Default input_dim; will be updated for training mode if needed
            input_dim = 31  # 25 features + 6 sector averages, as per original code
        else:
            num_sectors = 20  # Default fallback
            input_dim = 31
            
        self.model = StockTransformer(
            input_dim=input_dim,
            d_model=128,
            nhead=4,
            num_encoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            num_sectors=num_sectors,
            sector_embedding_dim=16
        ).to(self.device)
        
        if mode == 'train':
            if train_data_dir is None or valid_symbols is None or sector_map is None:
                raise ValueError("train_data_dir, valid_symbols, and sector_map are required for training mode")
                
            # Create datasets
            self.train_dataset = EnhancedStockDataset(
                input_dir=train_data_dir,
                valid_symbols=valid_symbols,
                sector_map=sector_map,
                window_size=window_size,
                prediction_horizon=prediction_horizon,
                split='train',
                augment=True
            )
            
            self.val_dataset = EnhancedStockDataset(
                input_dir=train_data_dir,
                valid_symbols=valid_symbols,
                sector_map=sector_map,
                window_size=window_size,
                prediction_horizon=prediction_horizon,
                split='valid',
                augment=False
            )
            
            # Update input_dim based on actual data
            if self.train_dataset.stock_data:
                input_dim = self.train_dataset.stock_data[list(self.train_dataset.stock_data.keys())[0]]['features'].shape[1]
                # Reinitialize model with correct input_dim
                self.model = StockTransformer(
                    input_dim=input_dim,
                    d_model=128,
                    nhead=4,
                    num_encoder_layers=3,
                    dim_feedforward=256,
                    dropout=0.1,
                    num_sectors=num_sectors,
                    sector_embedding_dim=16
                ).to(self.device)
            
            # Create dataloaders
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_stocks,
                num_workers=4,
                pin_memory=True
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_stocks,
                num_workers=4,
                pin_memory=True
            )
            
            # Create optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
            
            # Create scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs
            )
            
            # Create trainer
            self.trainer = StockPredictionTrainer(
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                return_weight=0.3,
                ranking_weight=0.7,
                scheduler=self.scheduler
            )
        else:  # mode == 'test'
            # Initialize trainer for prediction only
            self.trainer = StockPredictionTrainer(
                model=self.model,
                device=self.device,
                optimizer=None,
                scheduler=None
            )
    
    def train(self, save_path='best_model.pt'):
        if self.mode != 'train':
            raise ValueError("Cannot call train() in test mode")
        self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=self.num_epochs,
            save_path=save_path
        )
    
    def load_best_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with metrics:")
        for k, v in checkpoint['metrics'].items():
            print(f"  {k}: {v:.4f}")
    
    def predict_and_rank(self, test_dataset):
        from ..utils.collate import test_collate_stocks
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=test_collate_stocks,
            num_workers=4,
            pin_memory=True
        )
        predictions, symbols, dates, actual_returns = self.trainer.predict(test_loader)
        ranking_results = []
        
        for pred, sym, date, actual in zip(predictions, symbols, dates, actual_returns):
            stock_predictions = {s: p for s, p in zip(sym, pred)}
            ranked_stocks = sorted(stock_predictions.items(), key=lambda x: x[1], reverse=True)
            actual_dict = {s: a for s, a in zip(sym, actual)}
            ranking_results.append({
                'date': date,
                'ranked_stocks': ranked_stocks,
                'actual_returns': actual_dict
            })

        return ranking_results
