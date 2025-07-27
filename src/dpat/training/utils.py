"""
Training utility functions
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, Any


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Get optimizer"""
    # Different learning rates: BERT parameters use 1e-5, other parameters use 1e-3
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'bert' in name.lower():
            bert_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': config.get('bert_lr', 1e-5)},
        {'params': other_params, 'lr': config.get('lr', 1e-3)}
    ], weight_decay=config.get('weight_decay', 0.01))
    
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """Get learning rate scheduler"""
    total_steps = config.get('total_steps', 1000)
    warmup_steps = config.get('warmup_steps', 100)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics"""
    # Binary classification metrics
    pred_labels = (predictions > 0.5).float()
    
    # Calculate accuracy
    accuracy = (pred_labels == labels).float().mean().item()
    
    # Calculate precision and recall
    tp = ((pred_labels == 1) & (labels == 1)).float().sum().item()
    fp = ((pred_labels == 1) & (labels == 0)).float().sum().item()
    fn = ((pred_labels == 0) & (labels == 1)).float().sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 