"""
Evaluation metrics computation functions
"""
import torch
import numpy as np
from typing import Dict, List, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(predictions: Union[torch.Tensor, np.ndarray], 
                   labels: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """Compute binary classification evaluation metrics"""
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Binary classification predictions
    pred_labels = (predictions > 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    
    # Compute AUC
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def compute_batch_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute batch-level metrics"""
    # Apply sigmoid
    predictions = torch.sigmoid(outputs)
    
    return compute_metrics(predictions, labels) 