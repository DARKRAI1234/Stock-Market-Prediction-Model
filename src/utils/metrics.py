import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def calculate_ranking_metrics(pred_returns, true_returns, k_values=[5, 10, 20]):
    """
    Calculate ranking metrics: Precision@K, IRR@K, MRR@K
    
    Args:
        pred_returns: Predicted returns array
        true_returns: True returns array
        k_values: List of K values to calculate metrics for
    
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    # Get ranked indices
    pred_ranked_indices = np.argsort(-pred_returns)
    true_ranked_indices = np.argsort(-true_returns)
    
    for k in k_values:
        if len(pred_returns) >= k:
            # Precision@K
            pred_top_k = pred_ranked_indices[:k]
            true_top_k = true_ranked_indices[:k]
            precision_at_k = len(np.intersect1d(pred_top_k, true_top_k)) / k
            metrics[f'precision@{k}'] = precision_at_k
            
            # IRR@K (Information Retrieval Rate)
            true_top_k_sum = np.sum(true_returns[true_top_k])
            pred_top_k_sum = np.sum(true_returns[pred_top_k])
            irr_at_k = true_top_k_sum - pred_top_k_sum
            metrics[f'irr@{k}'] = irr_at_k
            
            # MRR@K (Mean Reciprocal Rank)
            mrr = 0
            for idx in true_top_k:
                pred_rank = np.where(pred_ranked_indices == idx)[0][0] + 1
                if pred_rank <= len(pred_returns):
                    mrr += 1.0 / pred_rank
            mrr /= k
            metrics[f'mrr@{k}'] = mrr
    
    return metrics


def calculate_return_metrics(pred_returns, true_returns):
    """
    Calculate return prediction metrics: MAE, R²
    
    Args:
        pred_returns: Predicted returns array
        true_returns: True returns array
    
    Returns:
        Dictionary with calculated metrics
    """
    if len(pred_returns) == 0 or len(true_returns) == 0:
        return {'mae': float('inf'), 'r2': float('-inf')}
    
    mae = mean_absolute_error(true_returns, pred_returns)
    r2 = r2_score(true_returns, pred_returns)
    
    return {'mae': mae, 'r2': r2}


def calculate_comprehensive_metrics(pred_returns, true_returns, k_values=[5, 10, 20]):
    """
    Calculate both ranking and return metrics
    
    Args:
        pred_returns: Predicted returns array
        true_returns: True returns array
        k_values: List of K values to calculate ranking metrics for
    
    Returns:
        Dictionary with all calculated metrics
    """
    ranking_metrics = calculate_ranking_metrics(pred_returns, true_returns, k_values)
    return_metrics = calculate_return_metrics(pred_returns, true_returns)
    
    return {**ranking_metrics, **return_metrics}
