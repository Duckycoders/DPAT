"""
评估指标计算函数
"""
import torch
import numpy as np
from typing import Dict, List, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(predictions: Union[torch.Tensor, np.ndarray], 
                   labels: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """计算二分类评估指标"""
    # 转换为numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 二分类预测
    pred_labels = (predictions > 0.5).astype(int)
    
    # 计算指标
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    
    # 计算AUC
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
    """计算batch级别的指标"""
    # 应用sigmoid
    predictions = torch.sigmoid(outputs)
    
    return compute_metrics(predictions, labels) 