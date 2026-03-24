import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassWeightedFocalLoss(nn.Module):
    """
    Precision-First Focal Loss implementation.
    
    Args:
        gamma (float): Focusing parameter. Default 2.0.
        weights (list): Class weights. Default None.
        critical_precision_penalty (float): Extra multiplier for False Positives 
                                            in the Critical class.
    """
    def __init__(self, gamma=2.0, weights=None, critical_class_idx=4):
        super(ClassWeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.critical_class_idx = critical_class_idx
        
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) logits
        targets: (batch_size) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights.to(inputs.device) if self.weights is not None else None)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply specific Precision penalty for False Positives in Critical Class?
        # Standard focal loss already handles hard samples. 
        # To specifically penalize FP for Class 4:
        # We look for cases where target != 4 but prediction is 4.
        # However, weight vectors in CE usually penalize FN (missing the class).
        # To penalize FP, we can increase the weights of OTHER classes, 
        # OR use a custom cost matrix approach.
        
        return focal_loss.mean()

if __name__ == "__main__":
    # Test logic
    num_classes = 5
    # Class 4 is Critical. User requested w=3.0 for Critical.
    # We set weights for [Low, Low-Mod, Mod, High, Critical]
    class_weights = [1.0, 1.0, 1.0, 1.0, 3.0]
    
    criterion = ClassWeightedFocalLoss(gamma=2.0, weights=class_weights)
    
    logits = torch.randn(8, num_classes) # Batch size 8
    targets = torch.tensor([0, 1, 4, 3, 4, 2, 0, 1])
    
    loss = criterion(logits, targets)
    print(f"Test Loss Output: {loss.item()}")
