import torch
import torch.nn as nn
import numpy as np


class StockTransformer(nn.Module):
    """
    Transformer model for predicting stock returns with cross-stock attention.
    """

    def __init__(
        self,
        input_dim: int = 31,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_sectors: int = 19,
        sector_embedding_dim: int = 16,
    ):
        super().__init__()

        # Projection & embeddings
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout)
        self.sector_embedding = nn.Embedding(num_sectors, sector_embedding_dim, padding_idx=-1)
        self.sector_projection = nn.Linear(sector_embedding_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Cross-stock attention
        self.cross_attention = CrossStockAttention(d_model)

        # Prediction head
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.apply(self._init_weights)

    # ----------------------------------------
    # Forward
    # ----------------------------------------
    def forward(self, features, sectors, mask=None):
        """
        Args:
            features: [batch, stocks, seq, feat]
            sectors : [batch, stocks]
            mask    : [batch, stocks, seq] – 1 valid, 0 pad
        Returns:
            returns_pred: [batch, stocks]
            attn_weights: cross-stock attention weights
        """
        b, n, t, f = features.shape
        x = features.reshape(b * n, t, f)
        s = sectors.reshape(b * n)

        # Sequence encoding
        x = self.input_projection(x)
        x = self.position_encoding(x)

        # Sector embedding
        sector_emb = self.sector_projection(self.sector_embedding(s)).unsqueeze(1)
        x = x + sector_emb

        # Optional padding mask
        if mask is not None:
            m = mask.reshape(b * n, t)
            x = self.transformer_encoder(x, src_key_padding_mask=~m.bool())
        else:
            x = self.transformer_encoder(x)

        seq_repr = x[:, -1, :].reshape(b, n, -1)

        # Cross-stock attention
        ctx_repr, attn = self.cross_attention(seq_repr)

        # Predict returns
        returns_pred = self.return_head(ctx_repr.reshape(b * n, -1)).squeeze(-1)
        return returns_pred.reshape(b, n), attn

    # ----------------------------------------
    # Utils
    # ----------------------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class CrossStockAttention(nn.Module):
    """Multi-head attention over the stock dimension."""

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        ctx, w = self.attn(x, x, x, need_weights=True)
        return self.norm(res + ctx), w
