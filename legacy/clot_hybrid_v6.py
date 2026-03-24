import torch
import torch.nn as nn
from pathlib import Path

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        # Pruned stack: 32, 64, 128
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (B, C, L) where L=30
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x # (B, 128, 30)

class ClotHybridV6(nn.Module):
    def __init__(self, n_features, n_classes=3, dropout=0.5):
        super().__init__()
        self.noise = GaussianNoise(sigma=0.01)
        self.cnn = MultiScaleCNN(n_features, out_channels=128)
        
        # Pruned Bi-LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Pruned Transformer: 2 Layers, 4 Heads
        self.pos_encoder = nn.Parameter(torch.zeros(1, 30, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        # x: (Batch, Sequence=30, Features=188)
        x = self.noise(x)
        x = x.transpose(1, 2) # (B, 188, 30) for CNN
        feat = self.cnn(x) # (B, 128, 30)
        
        feat = feat.transpose(1, 2) # (B, 30, 128) for LSTM
        lstm_out, _ = self.lstm(feat) # (B, 30, 128)
        
        t_in = lstm_out + self.pos_encoder
        t_out = self.transformer(t_in) # (B, 30, 128)
        
        # Global Average Pooling
        pooled = torch.mean(t_out, dim=1)
        return self.head(pooled)

if __name__ == "__main__":
    # Internal test
    model = ClotHybridV6(n_features=188)
    dummy_x = torch.randn(8, 30, 188)
    out = model(dummy_x)
    print(f"Refactored V6 Output shape (Batch size 8): {out.shape}")
    print("SUCCESS: 3-Class 'Honest' Architecture initialized.")
