import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/legacy")
from clot_hybrid_v6 import ClotHybridV6

def inspect():
    model_path = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/trained_models/clot_hybrid_v6_1_recovered.pth")
    data_dir = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/processed_data/v6_honest")
    
    X_test = torch.load(data_dir / "X_test_v6.pt")
    y_test = torch.load(data_dir / "y_test_v6.pt").numpy()
    
    model = ClotHybridV6(n_features=X_test.shape[2], n_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1).numpy()
    
    print(f"Total samples: {len(y_test)}")
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"Actual counts: {dict(zip(unique, counts))}")

    for cls in [0, 1, 2]:
        idxs = np.where(y_test == cls)[0]
        if len(idxs) > 0:
            avg_p = np.mean(probs[idxs], axis=0)
            print(f"\nClass {cls} (Avg Probs): {avg_p}")
            print(f"Sample 0 from Class {cls}: {probs[idxs[0]]} (Argmax: {np.argmax(probs[idxs[0]])})")

if __name__ == "__main__":
    inspect()
