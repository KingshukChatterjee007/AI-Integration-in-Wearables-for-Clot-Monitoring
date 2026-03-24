import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
from clot_hybrid_v5 import ClotHybridV5
from sklearn.metrics import classification_report
import logging

from sklearn.model_selection import GroupShuffleSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_v5(model_path, data_dir):
    logger.info("--- Phase 5.1 Hybrid Verification (Subject-Wise) ---")
    
    # Load data
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt")
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    # 80/20 Subject-Wise Split (same as training seed 42)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=s))
    
    X_test, y_test = X[test_idx], y[test_idx]
    logger.info(f"Auditing on {len(np.unique(s[test_idx]))} unseen subjects ({len(X_test)} samples)")
    
    n_features = X_test.shape[2]
    model = ClotHybridV5(n_features=n_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 1. Latency Test
    start_time = time.time()
    with torch.no_grad():
        for i in range(100):
            _ = model(X_test[0:1])
    avg_latency = (time.time() - start_time) / 100 * 1000 # ms
    logger.info(f"Average Inference Latency (CPU): {avg_latency:.2f} ms")
    
    # 2. Accuracy & Report
    all_preds = []
    with torch.no_grad():
        logits = model(X_test)
        all_preds = torch.argmax(logits, dim=-1).numpy()
        
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    report = classification_report(y_test.numpy(), all_preds, target_names=class_names, labels=[0,1,2,3,4])
    logger.info(f"\nPhase 5 Hybrid Classification Report:\n{report}")
    
    # Final Output save
    report_path = Path("model_comparison_plots_CLEAN/HYBRID_V5_REPORT.txt")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(f"Phase 5 Hybrid Temporal Fusion Report\nLatency: {avg_latency:.2f} ms/sample\n\n{report}")
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    from logging import INFO
    model_file = Path("trained_models/clot_hybrid_v5_best.pth")
    data_path = Path("processed_data/v5_tensors")
    if model_file.exists() and data_path.exists():
        verify_v5(model_file, data_path)
    else:
        logger.error("Model or data not found. Run training first.")
