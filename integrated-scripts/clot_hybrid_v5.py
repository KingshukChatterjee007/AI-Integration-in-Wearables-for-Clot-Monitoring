import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. MULTI-SCALE 1D-CNN ENCODER
# =============================================================================

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        # Parallel branches with different kernel sizes
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.branch3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.branch4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=11, padding=5)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # x shape: (Batch, Features, Time)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate along channel dimension
        x = torch.cat([b1, b2, b3, b4], dim=1)
        return self.relu(self.bn(x))

# =============================================================================
# 2. TRANSFORMER POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq, D)
        return x + self.pe[:x.size(1), :]

# =============================================================================
# 3. HYBRID CLOT-STRESS V5 MODEL
# =============================================================================

class ClotHybridV5(nn.Module):
    def __init__(self, n_features, n_classes=5, seq_length=30, d_model=256, n_heads=8, n_layers=4, dropout=0.3):
        super().__init__()
        
        # Block 1: Multi-Scale CNN
        self.cnn = MultiScaleCNN(in_channels=n_features, out_channels=128)
        
        # Block 2: Bi-LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Block 3: Transformer Core
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Block 4: Bayesian Head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, x, return_uncertainty=False):
        # x: (Batch, Time, Features)
        # CNN expects (Batch, Features, Time)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        
        # LSTM expects (Batch, Time, Hidden)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x) # Output (Batch, Time, d_model)
        
        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Bayesian Gating: Aggregation via Mean Pooling
        x = x.mean(dim=1)
        logits = self.classifier(self.norm(x))
        
        if return_uncertainty:
            # Simple Mutual Information approx via MC Dropout would be implemented in inference loop
            return logits
            
        return logits

def check_architecture():
    # Mock data: (Batch, Time, Features)
    batch_size = 8
    seq_len = 30
    n_feat = 188
    x = torch.randn(batch_size, seq_len, n_feat)
    
    model = ClotHybridV5(n_features=n_feat)
    logits = model(x)
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {logits.shape}")
    logger.info("Architecture verification: SUCCESS")

if __name__ == "__main__":
    check_architecture()
