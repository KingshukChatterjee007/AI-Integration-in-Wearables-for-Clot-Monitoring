import torch
import torch.nn as nn
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

# =============================================================================
# 1. MODEL ARCHITECTURE (Self-Contained for Export)
# =============================================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.randn(n_features, d_token))

    def forward(self, x):
        # x: (B, N) -> (B, N, 1)
        x = x.unsqueeze(-1)
        # x: (B, N, 1) * (N, D) -> (B, N, D)
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
        
        # Explicit layers for clearer ONNX graph
        self.fc1 = nn.Linear(d_token, d_token // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_token // 2, n_classes)
        
    def forward(self, x):
        tokens = self.tokenizer(x)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = self.transformer(x)
        
        # Explicitly extract CLS token for ONNX
        cls_out = x.narrow(1, 0, 1).reshape(-1, 96)
        x = self.norm(cls_out)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# =============================================================================
# 2. EXPORT & QUANTIZATION PIPELINE
# =============================================================================

def export_and_quantize():
    model_path = Path('trained_models/clot_transformer_balanced_best.pth')
    onnx_path = "clot_transformer_v4.onnx"
    quant_path = "clot_transformer_v4_quantized.onnx"
    
    print(f"--- Edge Optimization Pipeline (v4) ---")
    
    # 2.1 Load PyTorch Model
    print(f"1. Loading PyTorch weights from {model_path}...")
    model = ClotTransformer()
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Map Sequential keys to Explicit layer keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('classifier.0.'):
            new_state_dict[k.replace('classifier.0.', 'fc1.')] = v
        elif k.startswith('classifier.3.'):
            new_state_dict[k.replace('classifier.3.', 'fc2.')] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 2.2 Trace Test: JIT Trace + ONNX
    print(f"2. [JIT TRACE TEST] Tracing and exporting Linear(96, 48)...")
    traced_model = torch.jit.trace(vanilla_model, vanilla_input)
    torch.onnx.export(
        traced_model, vanilla_input, "traced_test.onnx",
        export_params=True, opset_version=14, do_constant_folding=True
    )
    print("   - JIT Trace Test: SUCCESS.")
    
    # 2.3 Apply INT8 Dynamic Quantization
    print(f"3. Applying INT8 Dynamic Quantization (Target: CPU)...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QUInt8
    )
    
    # 2.4 Verification
    print(f"\n4. Final Size Verification:")
    size_pth = os.path.getsize(model_path) / (1024 * 1024)
    size_onnx = os.path.getsize(onnx_path) / (1024 * 1024)
    size_quant = os.path.getsize(quant_path) / (1024 * 1024)
    
    print(f"   - Original .pth Model:   {size_pth:.3f} MB")
    print(f"   - Standard ONNX Model:   {size_onnx:.3f} MB")
    print(f"   - Quantized INT8 Model:  {size_quant:.3f} MB")
    print(f"   - Compression Ratio:     {size_pth / size_quant:.2f}x")
    
    print(f"\nSuccess! Optimized model saved to {quant_path}")

if __name__ == "__main__":
    export_and_quantize()
