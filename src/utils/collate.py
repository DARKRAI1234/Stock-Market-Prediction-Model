import torch


def collate_stocks(batch):
    """
    Collate function for training batches with padding and masking.
    """
    if not batch:
        return None
    
    # Find max number of stocks in any sample
    max_stocks = max(len(item['features']) for item in batch)
    batch_size = len(batch)
    
    # Initialize tensors with padding
    seq_len = batch[0]['features'].size(1)
    feature_dim = batch[0]['features'].size(2)
    features = torch.zeros(batch_size, max_stocks, seq_len, feature_dim, dtype=torch.float32)
    returns = torch.zeros(batch_size, max_stocks, dtype=torch.float32)
    sectors = torch.full((batch_size, max_stocks), fill_value=-1, dtype=torch.long)
    
    # Also collect symbols for reference
    all_symbols = []
    
    # Fill tensors with data
    for i, item in enumerate(batch):
        num_stocks = len(item['features'])
        features[i, :num_stocks] = item['features']
        returns[i, :num_stocks] = item['returns']
        sectors[i, :num_stocks] = item['sectors']
        
        # Pad symbols list with None
        symbols = item['symbols'] + [None] * (max_stocks - len(item['symbols']))
        all_symbols.append(symbols)
    
    mask = (sectors != -1)  # Assume -1 is padding
    
    return {
        'features': features,
        'returns': returns,
        'sectors': sectors,
        'mask': mask,
        'symbols': all_symbols
    }


def test_collate_stocks(batch):
    """
    Collate function for test batches (batch_size=1).
    """
    item = batch[0]  # batch_size=1
    return {
        'date': item['date'],
        'features': item['features'].unsqueeze(0),  # Shape: (1, num_stocks, window_size, feature_dim)
        'masks': item['masks'].unsqueeze(0),        # Shape: (1, num_stocks, window_size)
        'sectors': item['sectors'].unsqueeze(0),    # Shape: (1, num_stocks)
        'symbols': item['symbols'],
        'actual_returns': item['actual_returns'].unsqueeze(0)  # Shape: (1, num_stocks)
    }
