import torch
import torch.nn as nn
import os
from pathlib import Path

# =============================================================================
# 1. MODEL ARCHITECTURE (v4 Optimized 8-Feature)
# =============================================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.randn(n_features, d_token))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x * self.weight
        x = x + self.bias
        return x

class ClotTransformer(nn.Module):
    def __init__(self, n_features=8, n_classes=5, d_token=96, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.fc1 = nn.Linear(d_token, d_token // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_token // 2, n_classes)
        
    def forward(self, x):
        tokens = self.tokenizer(x)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = self.transformer(x)
        
        cls_out = x[:, 0, :]
        x = self.norm(cls_out)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# =============================================================================
# 2. QUANTIZATION PIPELINE
# =============================================================================

def quantize_production_model():
    model_path = Path('trained_models/clot_transformer_balanced_best.pth')
    quant_path = Path('trained_models/clot_transformer_v4_quantized.pth')
    
    print(f"--- Production Edge Optimization (PyTorch Native INT8) ---")
    
    # 2.1 Load Model
    print(f"1. Loading v4 weights from {model_path}...")
    model = ClotTransformer()
    # Note: Using the sequential keys since that's how they were saved
    # Re-mapping handled automatically if we use the original Sequential structure, 
    # but for simplicity let's just use the Sequential structure here for load.
    
    # Actually, let's just Use the Sequential one for loading parity
    class ClotTransformerSequential(nn.Module):
        def __init__(self, n_features=8, n_classes=5, d_token=96, n_heads=4, n_layers=2, dropout=0.2):
            super().__init__()
            self.tokenizer = FeatureTokenizer(n_features, d_token)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                dropout=dropout, activation='gelu', batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_token)
            self.classifier = nn.Sequential(
                nn.Linear(d_token, d_token // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_token // 2, n_classes)
            )
        def forward(self, x):
            tokens = self.tokenizer(x)
            b = x.shape[0]
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls_tokens, tokens], dim=1)
            x = self.transformer(x)
            return self.classifier(self.norm(x[:, 0, :]))

    model = ClotTransformerSequential()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 2.2 Apply Dynamic Quantization
    print(f"2. Quantizing Linear Layers to INT8 (Dynamic)...")
    # This targets Linear and Transformer modules for weight quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.TransformerEncoderLayer}, 
        dtype=torch.qint8
    )
    
    # 2.3 Save
    print(f"3. Saving compressed model to {quant_path}...")
    torch.save(quantized_model.state_dict(), quant_path)
    
    # 2.4 Verify
    size_orig = os.path.getsize(model_path) / 1024
    size_quant = os.path.getsize(quant_path) / 1024
    print(f"\nOptimization Summary:")
    print(f"   - Original Model size:  {size_orig:.2f} KB")
    print(f"   - Quantized Model size: {size_quant:.2f} KB")
    print(f"   - Compression ratio:    {size_orig / size_quant:.2f}x")
    print(f"   - Reduction:            {(1 - size_quant/size_orig)*100:.1f}%")

if __name__ == "__main__":
    quantize_production_model()
