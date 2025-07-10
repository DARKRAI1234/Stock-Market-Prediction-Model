import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss that encourages the predicted ordering of returns
    to match the ground-truth ordering.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_scores, true_scores, stock_mask):
        """
        Args:
            pred_scores : [batch, stocks]
            true_scores : [batch, stocks]
            stock_mask  : [batch, stocks] – 1 valid, 0 pad
        Returns:
            scalar ranking loss
        """
        pred = pred_scores / self.temperature
        b, n = pred.shape

        # Pairwise differences
        s1 = pred.unsqueeze(2).expand(b, n, n)
        s2 = pred.unsqueeze(1).expand(b, n, n)
        pred_pref = torch.sigmoid(s1 - s2)

        # Ground truth preferences
        t1 = true_scores.unsqueeze(2).expand(b, n, n)
        t2 = true_scores.unsqueeze(1).expand(b, n, n)
        true_pref = (t1 > t2).float()

        # Valid pairs & exclude self-pairs
        mask = stock_mask.unsqueeze(2) * stock_mask.unsqueeze(1)
        mask = mask * (1 - torch.eye(n, device=mask.device).unsqueeze(0))

        # Binary cross-entropy
        logp = (
            torch.log(pred_pref + 1e-8) * true_pref
            + torch.log(1 - pred_pref + 1e-8) * (1 - true_pref)
        )
        loss = -torch.sum(logp * mask) / (mask.sum() + 1e-8)
        return loss
