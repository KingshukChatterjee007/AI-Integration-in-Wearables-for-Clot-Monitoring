import torch
import torch.nn as nn

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
        self.conv3 = nn.Conv1d(64, out_channels, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

class ClotHybrid5Class(nn.Module):
    def __init__(self, n_features=159, seq_len=15, n_classes=5, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = n_features // seq_len
        self.padded_features = self.seq_len * self.input_size
        
        # Adjust if n_features is not divisible by seq_len
        if self.padded_features < n_features:
            self.input_size = (n_features + seq_len - 1) // seq_len
            self.padded_features = self.seq_len * self.input_size
            
        self.noise = GaussianNoise(sigma=0.01)
        
        # Spatial Encoder (CNN) expects input of shape (B, in_channels, seq_len)
        self.cnn = MultiScaleCNN(in_channels=self.input_size, out_channels=128)
        
        # Temporal Encoder (Bi-LSTM)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Relational Encoder (Transformer)
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_len, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.dropout = nn.Dropout(dropout)
        
        # 5-Class Linear Head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Pad features for pseudo-sequence
        if x.size(1) < self.padded_features:
            padding = torch.zeros(batch_size, self.padded_features - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
            
        # Reshape to pseudo-sequence: (Batch, seq_len, input_size)
        x = x.view(batch_size, self.seq_len, self.input_size)
        
        # Add noise
        x = self.noise(x)
        
        # 1D-CNN expects (Batch, Channels, Length)
        # We treat input_size as Channels and seq_len as Length
        x = x.transpose(1, 2) # (B, input_size, seq_len)
        feat = self.cnn(x) # (B, 128, seq_len)
        
        # LSTM expects (Batch, Length, Channels)
        feat = feat.transpose(1, 2) # (B, seq_len, 128)
        lstm_out, _ = self.lstm(feat) # (B, seq_len, 128)
        
        # Transformer
        t_in = lstm_out + self.pos_encoder
        t_out = self.transformer(t_in) # (B, seq_len, 128)
        
        # Global Average Pooling
        pooled = torch.mean(t_out, dim=1) # (B, 128)
        
        # Final Classification
        return self.head(pooled)

if __name__ == "__main__":
    # Test tensor
    model = ClotHybrid5Class(n_features=159)
    dummy_x = torch.randn(8, 159)
    out = model(dummy_x)
    print(f"5-Class Hybrid Tabular Output shape: {out.shape}")
    print("SUCCESS: Tabular-adapted Hybrid Architecture initialized.")
