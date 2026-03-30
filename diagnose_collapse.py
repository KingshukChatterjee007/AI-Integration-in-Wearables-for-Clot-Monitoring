import torch
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add path to import ClotHybridV6 if needed, but we probably just need the data first
# data_dir = Path("processed_data/v6_honest")
data_dir = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/processed_data/v6_honest")

def diagnose():
    try:
        y_train = torch.load(data_dir / "y_train_v11_augmented.pt")
        y_test = torch.load(data_dir / "y_test_v6.pt")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("--- Class Distribution ---")
    train_counts = Counter(y_train.numpy())
    test_counts = Counter(y_test.numpy())
    
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    
    class_names = ['SAFE', 'WARNING', 'EMERGENCY']
    
    print(f"Train: { {class_names[i]: (train_counts[i], f'{train_counts[i]/total_train:.2%}') for i in range(3)} }")
    print(f"Test:  { {class_names[i]: (test_counts[i], f'{test_counts[i]/total_test:.2%}') for i in range(3)} }")

    # Load model and check logits if weights exist
    model_path = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/trained_models/clot_hybrid_v6_1_recovered.pth")
    if model_path.exists():
        print("\n--- Model Predictions Analysis ---")
        # Need to import ClotHybridV6. It's in the root or core? 
        # FILE_MAP says core/clot_hybrid_v6.py
        sys.path.append("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/legacy")
        try:
            from clot_hybrid_v6 import ClotHybridV6
            X_test = torch.load(data_dir / "X_test_v6.pt")
            model = ClotHybridV6(n_features=X_test.shape[2], n_classes=3)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            with torch.no_grad():
                logits = model(X_test)
                probs = torch.softmax(logits, dim=1).numpy()
            
            y_test_np = y_test.numpy()
            
            # RAW Predictions (argmax)
            raw_preds = np.argmax(probs, axis=1)
            print("\nRaw Argmax Confusion Matrix:")
            print(confusion_matrix(y_test_np, raw_preds))
            print(classification_report(y_test_np, raw_preds, target_names=class_names))
            
            # Probabilities for SAFE samples
            safe_idx = np.where(y_test_np == 0)[0]
            if len(safe_idx) > 0:
                avg_probs_for_safe = np.mean(probs[safe_idx], axis=0)
                print(f"\nAverage probabilities for actual SAFE samples: {avg_probs_for_safe}")
                print(f"SAFE samples with p(SAFE) > p(WARNING): {np.sum(probs[safe_idx, 0] > probs[safe_idx, 1])} / {len(safe_idx)}")
                print(f"SAFE samples with p(WARNING) > 0.35: {np.sum(probs[safe_idx, 1] > 0.35)} / {len(safe_idx)}")

        except Exception as e:
            print(f"Error analyzing model: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    diagnose()
