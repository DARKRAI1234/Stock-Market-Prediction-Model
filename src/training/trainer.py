import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error

from ..models.losses import RankingLoss


class StockPredictionTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        return_weight=0.3,
        ranking_weight=0.7,
        scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.return_weight = return_weight
        self.ranking_weight = ranking_weight
        
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=0.1)
        self.ranking_loss = RankingLoss(temperature=0.1)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_return_loss = 0
        total_ranking_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            if batch is None:
                continue
                
            # Move batch to device
            features = batch['features'].to(self.device)
            returns = batch['returns'].to(self.device)
            sectors = batch['sectors'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_returns, _ = self.model(features, sectors)
            
            # Apply mask for padding
            pred_returns = pred_returns * mask.to(torch.float32)
            
            valid_pred = torch.masked_select(pred_returns, mask)
            valid_true = torch.masked_select(returns, mask)
            
            # Calculate losses
            return_loss = self.huber_loss(valid_pred, valid_true)
            ranking_loss = self.ranking_loss(pred_returns, returns, mask.float())
            
            # Combine losses
            loss = self.return_weight * return_loss + self.ranking_weight * ranking_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_return_loss += return_loss.item()
            total_ranking_loss += ranking_loss.item()
            num_batches += 1
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Calculate average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_return_loss = total_return_loss / num_batches if num_batches > 0 else float('inf')
        avg_ranking_loss = total_ranking_loss / num_batches if num_batches > 0 else float('inf')
        
        return avg_loss, avg_return_loss, avg_ranking_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_return_loss = 0
        total_ranking_loss = 0
        num_batches = 0
        
        all_true_returns = []
        all_pred_returns = []
        all_precision_at_k = {5: [], 10: [], 20: []}
        all_irr_at_k = {5: [], 10: [], 20: []}
        all_mrr_at_k = {5: [], 10: [], 20: []}
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue
                
            # Move batch to device
            features = batch['features'].to(self.device)
            returns = batch['returns'].to(self.device)
            sectors = batch['sectors'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            pred_returns, attn_weights = self.model(features, sectors)
            
            # Apply mask for padding
            pred_returns = pred_returns * mask.float()
            
            # Calculate losses
            return_loss = self.huber_loss(pred_returns[mask], returns[mask])
            ranking_loss = self.ranking_loss(pred_returns, returns, mask.float())
            
            # Combine losses
            loss = self.return_weight * return_loss + self.ranking_weight * ranking_loss
            
            # Track metrics
            total_loss += loss.item()
            total_return_loss += return_loss.item()
            total_ranking_loss += ranking_loss.item()
            num_batches += 1
            
            # Collect return predictions for metrics
            for i in range(features.size(0)):
                batch_mask = mask[i]
                batch_true_returns = returns[i][batch_mask].cpu().numpy()
                batch_pred_returns = pred_returns[i][batch_mask].cpu().numpy()
                
                if len(batch_true_returns) > 0:
                    all_true_returns.append(batch_true_returns)
                    all_pred_returns.append(batch_pred_returns)
                    
                    # Calculate ranking metrics
                    for k in [5, 10, 20]:
                        if len(batch_true_returns) >= k:
                            # Precision@k
                            true_top_k_indices = np.argsort(-batch_true_returns)[:k]
                            pred_top_k_indices = np.argsort(-batch_pred_returns)[:k]
                            precision_at_k = len(np.intersect1d(true_top_k_indices, pred_top_k_indices)) / k
                            all_precision_at_k[k].append(precision_at_k)
                            
                            # IRR@k
                            true_top_k_sum = np.sum(batch_true_returns[true_top_k_indices])
                            pred_top_k_sum = np.sum(batch_true_returns[pred_top_k_indices])
                            irr_at_k = true_top_k_sum - pred_top_k_sum
                            all_irr_at_k[k].append(irr_at_k)
                            
                            # MRR@k
                            mrr = 0
                            for idx in true_top_k_indices:
                                pred_rank = np.where(np.argsort(-batch_pred_returns) == idx)[0][0] + 1
                                if pred_rank <= len(batch_pred_returns):
                                    mrr += 1.0 / pred_rank
                            mrr /= k
                            all_mrr_at_k[k].append(mrr)
        
        # Calculate average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_return_loss = total_return_loss / num_batches if num_batches > 0 else float('inf')
        avg_ranking_loss = total_ranking_loss / num_batches if num_batches > 0 else float('inf')
        
        # Calculate return prediction metrics
        all_true_returns_flat = np.concatenate(all_true_returns) if all_true_returns else np.array([])
        all_pred_returns_flat = np.concatenate(all_pred_returns) if all_pred_returns else np.array([])
        
        if len(all_true_returns_flat) > 0:
            mae = mean_absolute_error(all_true_returns_flat, all_pred_returns_flat)
            r2 = r2_score(all_true_returns_flat, all_pred_returns_flat)
        else:
            mae = float('inf')
            r2 = float('-inf')
        
        # Calculate average ranking metrics
        avg_precision_at_k = {k: np.mean(v) if v else 0 for k, v in all_precision_at_k.items()}
        avg_irr_at_k = {k: np.mean(v) if v else 0 for k, v in all_irr_at_k.items()}
        avg_mrr_at_k = {k: np.mean(v) if v else 0 for k, v in all_mrr_at_k.items()}
        
        return {
            'loss': avg_loss,
            'return_loss': avg_return_loss,
            'ranking_loss': avg_ranking_loss,
            'mae': mae,
            'r2': r2,
            'precision@5': avg_precision_at_k[5],
            'precision@10': avg_precision_at_k[10],
            'precision@20': avg_precision_at_k[20],
            'irr@5': avg_irr_at_k[5],
            'irr@10': avg_irr_at_k[10],
            'irr@20': avg_irr_at_k[20],
            'mrr@5': avg_mrr_at_k[5],
            'mrr@10': avg_mrr_at_k[10],
            'mrr@20': avg_mrr_at_k[20]
        }
    
    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()
        all_predictions, all_symbols, all_dates, all_actual_returns, all_attn_weights = [], [], [], [], []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            if batch is None:
                continue
            
            dates = batch['date']
            features = batch['features'].to(self.device)
            sectors = batch['sectors'].to(self.device)
            masks = batch['masks'].to(self.device) if 'masks' in batch else None
            batch_symbols = batch['symbols']
            actual_returns = batch['actual_returns'].to(self.device)
            
            pred_returns, attn_weights = self.model(features, sectors, masks)
            
            pred_returns = pred_returns.cpu().numpy()
            attn_weights = attn_weights.cpu().numpy()  # Shape: (1, num_heads, num_stocks, num_stocks)
            actual_returns = actual_returns.cpu().numpy()
            
            all_predictions.append(pred_returns[0])  # Batch size is 1
            all_symbols.append(batch_symbols)
            all_dates.append(dates)
            all_actual_returns.append(actual_returns[0])
            all_attn_weights.append(attn_weights[0])  # Shape: (num_heads, num_stocks, num_stocks)
        
        return all_predictions, all_symbols, all_dates, all_actual_returns, all_attn_weights
    
    def train(self, train_loader, val_loader, num_epochs, save_path='best_model.pt'):
        best_r2 = float('-inf')  # Initialize to negative infinity
        for epoch in range(num_epochs):
            # Train epoch
            train_loss, train_return_loss, train_ranking_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Return Loss: {train_return_loss:.4f}, Ranking Loss: {train_ranking_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Return Loss: {val_metrics['return_loss']:.4f}, Ranking Loss: {val_metrics['ranking_loss']:.4f}")
            print(f"Return MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
            print(f"Precision@5: {val_metrics['precision@5']:.4f}, Precision@10: {val_metrics['precision@10']:.4f}, Precision@20: {val_metrics['precision@20']:.4f}")
            print(f"IRR@5: {val_metrics['irr@5']:.4f}, IRR@10: {val_metrics['irr@10']:.4f}, IRR@20: {val_metrics['irr@20']:.4f}")
            print(f"MRR@5: {val_metrics['mrr@5']:.4f}, MRR@10: {val_metrics['mrr@10']:.4f}, MRR@20: {val_metrics['mrr@20']:.4f}")
            print("-" * 80)
            
            # Save best model based on R-squared
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': val_metrics
                }, save_path)
                print(f"New best model saved with R²: {best_r2:.4f}")
