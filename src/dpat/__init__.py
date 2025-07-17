"""
Dual-Path Alignment & Transformer (DPAT) for miRNA target prediction.

This package implements a novel dual-path architecture combining:
1. Alignment Path: Processes sequence alignment matrices with multi-scale convolutions
2. Semantic Path: Leverages RNA-BERT embeddings with BiLSTM for semantic understanding
3. Cross-attention fusion for optimal feature integration
"""

__version__ = "1.0.0"
__author__ = "DPAT Team"

from .models.dpat import DPAT
from .data.dataset import DPATDataset
from .training.trainer import DPATTrainer
from .config.config import DPATConfig

__all__ = [
    "DPAT",
    "DPATDataset", 
    "DPATTrainer",
    "DPATConfig",
] 