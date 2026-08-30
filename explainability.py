import torch
import numpy as np

def compute_feature_importance(model, X_tensor, station_index, feature_names, numeric_cols):
    """
    Computes local sensitivity (gradients) of the anomaly prediction at a specific station 
    with respect to the input features at that same station.
    
    Returns influences focused on numeric features.
    """
    model.eval()
    
    # We need gradients with respect to the input
    X_tensor.requires_grad_(True)
    
    # Disable CuDNN temporarily to allow RNN backward pass in eval mode
    with torch.backends.cudnn.flags(enabled=False):
        # Forward pass
        out = model(X_tensor) # shape: [1, seq_len]
        
        # Get the logit (raw score) and probability for the target station
        logit = out[0, station_index]
        prob = torch.sigmoid(logit)
        
        # Zero gradients just in case
        model.zero_grad()
        
        # Backward pass on the LOGIT (prevents vanishing gradients when prob is 100%)
        logit.backward()
    
    # Gradients for the specific station's features
    grads = X_tensor.grad[0, station_index].cpu().numpy()
    
    feature_influences = []
    
    for i, name in enumerate(feature_names):
        # Only consider numeric columns for the top influences, as requested
        if name in numeric_cols:
            grad_val = float(grads[i])
            abs_importance = abs(grad_val)
            
            # Determine direction text based on gradient sign
            if abs_importance < 1e-7:
                direction = "negligible influence"
            elif grad_val > 0:
                direction = "increased anomaly risk"
            else:
                direction = "decreased anomaly risk"
                
            feature_influences.append({
                'feature': name, 
                'importance': abs_importance,
                'raw_gradient': grad_val,
                'direction': direction
            })
            
    # Sort by absolute importance descending
    feature_influences.sort(key=lambda x: x['importance'], reverse=True)
    
    return feature_influences[:3]
