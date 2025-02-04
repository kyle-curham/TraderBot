import torch
import torch.nn as nn

# NEW CODE: Positional encoding for temporal context
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


# NEW CODE: Temporal transformer encoder for the critic with input normalization
class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 63,  # Instantiate with the appropriate context length (e.g., 63) later.
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Add input normalization: Normalize the raw inputs before projection.
        self.input_norm = nn.LayerNorm(input_dim)
        # Project input to transformer hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        # Updated to use batch_first=True so that the transformer encoder works with (batch, seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        #x = self.input_norm(x)  # Normalize input tensor
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x = self.pos_encoder(x)
        # With batch_first=True, no need to transpose
        x = self.transformer_encoder(x)
        return x 