import torch

def apply_clinical_gate(logits: torch.Tensor, thresh_crit=0.03, thresh_high=0.10):
    """
    Applies asymmetric confidence gating to the softmax probabilities of the 5-class model.
    Prioritizes High and Critical states to maximize clinical sensitivity.
    
    Args:
        logits (torch.Tensor): Output from the Final Linear Classification Head, shape (Batch, 5).
        thresh_crit (float): Probability threshold to forcefully classify as Critical [Class 4].
        thresh_high (float): Probability threshold to forcefully classify as High [Class 3].
        
    Returns:
        batch_preds (torch.Tensor): Integer class predictions.
        probs (torch.Tensor): Softmax probabilities.
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        batch_preds = torch.argmax(probs, dim=1).clone() # Clone to avoid modifying tensor in-place when looping
        
        # Apply asymmetric soft thresholds
        # Iterate over the batch
        for i in range(probs.size(0)):
            p_crit = probs[i, 4].item()
            p_high = probs[i, 3].item()
            
            # Highest severity takes precedence
            if p_crit >= thresh_crit:
                batch_preds[i] = 4
            elif p_high >= thresh_high:
                batch_preds[i] = 3
                
    return batch_preds, probs
