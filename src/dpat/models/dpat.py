"""DPAT main model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .modules.alignment import AlignmentPath
from .modules.semantic import SemanticPath, SimpleSemanticPath
from .modules.fusion import CrossAttentionFusion, GatedFusion
from .modules.utils import init_weights


class DPAT(nn.Module):
    """
    Dual-Path Alignment & Transformer (DPAT) model for miRNA target prediction.
    
    Architecture:
    1. Alignment Path: Processes 10×50 alignment matrices with multi-scale convolutions
    2. Semantic Path: Processes RNA-BERT embeddings with BiLSTM
    3. Cross-Attention Fusion: Combines both paths with bidirectional attention
    4. Classification Head: Outputs CTS functionality probability
    
    Args:
        alignment_config: Configuration for alignment path
        semantic_config: Configuration for semantic path
        fusion_config: Configuration for fusion module
        num_classes: Number of output classes (default: 1 for binary classification)
        dropout: Dropout probability
        use_simple_semantic: Whether to use simplified semantic path
    """
    
    def __init__(self,
                 alignment_config: Optional[Dict] = None,
                 semantic_config: Optional[Dict] = None,
                 fusion_config: Optional[Dict] = None,
                 num_classes: int = 1,
                 dropout: float = 0.1,
                 use_simple_semantic: bool = False):
        super(DPAT, self).__init__()
        
        # Default configurations
        if alignment_config is None:
            alignment_config = {
                'input_channels': 10,
                'output_channels': 256,
                'hidden_channels': 128,
                'kernel_sizes': [1, 3, 5, 7],
                'dropout': dropout,
                'use_se': True,
                'use_cbam': True
            }
        
        if semantic_config is None:
            semantic_config = {
                'bert_model_name': 'multimolecule/rnafm',
                'bert_hidden_size': 768,
                'conv_hidden_size': 1024,
                'lstm_hidden_size': 128,
                'proj_dim': 256,
                'dropout': dropout,
                'freeze_bert': False
            }
        
        if fusion_config is None:
            fusion_config = {
                'embed_dim': 256,
                'num_heads': 8,
                'dropout': dropout,
                'use_gate': True
            }
        
        self.alignment_config = alignment_config
        self.semantic_config = semantic_config
        self.fusion_config = fusion_config
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_simple_semantic = use_simple_semantic
        
        # Alignment path
        self.alignment_path = AlignmentPath(**alignment_config)
        
        # Semantic path
        if use_simple_semantic:
            self.semantic_path = SimpleSemanticPath(**semantic_config)
        else:
            self.semantic_path = SemanticPath(**semantic_config)
        
        # Cross-attention fusion
        self.cross_attention_fusion = CrossAttentionFusion(**fusion_config)
        
        # Classification head
        embed_dim = fusion_config['embed_dim']
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # Global pooling for sequence-level prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, 
                align_matrix: torch.Tensor,
                bert_input_ids: torch.Tensor,
                bert_attention_mask: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through DPAT model.
        
        Args:
            align_matrix: Alignment matrix of shape (batch_size, 10, 50)
            bert_input_ids: BERT input IDs of shape (batch_size, seq_len)
            bert_attention_mask: BERT attention mask of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Alignment path processing
        # Input: (batch_size, 10, 50) -> Output: (batch_size, 256, 50)
        align_features = self.alignment_path(align_matrix)
        
        # Transpose for sequence processing: (batch_size, 50, 256)
        align_features = align_features.transpose(1, 2)
        
        # Semantic path processing
        # Input: BERT tokens -> Output: (batch_size, seq_len, 256) and (batch_size, 256)
        semantic_features, semantic_global = self.semantic_path(
            bert_input_ids, bert_attention_mask
        )
        
        # Cross-attention fusion
        # Combine alignment and semantic features
        fused_features = self.cross_attention_fusion(
            align_features=align_features,
            semantic_features=semantic_features,
            semantic_mask=~bert_attention_mask.bool()  # Invert mask for padding
        )
        
        # Global pooling for sequence-level representation
        # (batch_size, 50, 256) -> (batch_size, 256, 50) -> (batch_size, 256, 1)
        fused_features_transposed = fused_features.transpose(1, 2)
        global_features = self.global_pool(fused_features_transposed)
        global_features = global_features.squeeze(-1)  # (batch_size, 256)
        
        # Classification
        logits = self.classifier(global_features)
        
        if return_attention:
            return logits, fused_features
        else:
            return logits
    
    def predict_proba(self, 
                     align_matrix: torch.Tensor,
                     bert_input_ids: torch.Tensor,
                     bert_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for CTS functionality.
        
        Args:
            align_matrix: Alignment matrix
            bert_input_ids: BERT input IDs
            bert_attention_mask: BERT attention mask
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(align_matrix, bert_input_ids, bert_attention_mask)
        
        if self.num_classes == 1:
            # Binary classification
            probs = torch.sigmoid(logits)
        else:
            # Multi-class classification
            probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def get_attention_weights(self,
                             align_matrix: torch.Tensor,
                             bert_input_ids: torch.Tensor,
                             bert_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights from cross-attention fusion.
        
        Args:
            align_matrix: Alignment matrix
            bert_input_ids: BERT input IDs
            bert_attention_mask: BERT attention mask
            
        Returns:
            Attention weights and fused features
        """
        _, attention_features = self.forward(
            align_matrix, bert_input_ids, bert_attention_mask, return_attention=True
        )
        
        return attention_features
    
    def get_feature_importance(self,
                              align_matrix: torch.Tensor,
                              bert_input_ids: torch.Tensor,
                              bert_attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get feature importance for interpretability.
        
        Args:
            align_matrix: Alignment matrix
            bert_input_ids: BERT input IDs
            bert_attention_mask: BERT attention mask
            
        Returns:
            Dictionary with feature importance scores
        """
        # Get intermediate features
        align_features = self.alignment_path(align_matrix)
        align_features = align_features.transpose(1, 2)
        
        semantic_features, semantic_global = self.semantic_path(
            bert_input_ids, bert_attention_mask
        )
        
        # Compute feature norms as importance scores
        align_importance = torch.norm(align_features, dim=-1)  # (batch_size, 50)
        semantic_importance = torch.norm(semantic_features, dim=-1)  # (batch_size, seq_len)
        
        return {
            'alignment_importance': align_importance,
            'semantic_importance': semantic_importance,
            'semantic_global': semantic_global
        }


class DPATWithMaxPooling(DPAT):
    """
    DPAT model with max pooling for handling multiple CTS per miRNA-mRNA pair.
    
    This variant applies max pooling across CTS candidates for the same miRNA-mRNA pair.
    """
    
    def __init__(self, **kwargs):
        super(DPATWithMaxPooling, self).__init__(**kwargs)
    
    def forward_with_grouping(self,
                             align_matrix: torch.Tensor,
                             bert_input_ids: torch.Tensor,
                             bert_attention_mask: torch.Tensor,
                             sample_keys: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with max pooling across CTS candidates.
        
        Args:
            align_matrix: Alignment matrix
            bert_input_ids: BERT input IDs
            bert_attention_mask: BERT attention mask
            sample_keys: List of sample keys for grouping
            
        Returns:
            Tuple of (grouped_logits, group_indices)
        """
        # Get CTS-level predictions
        cts_logits = self.forward(align_matrix, bert_input_ids, bert_attention_mask)
        
        # Group by sample key and apply max pooling
        unique_keys = list(set(sample_keys))
        grouped_logits = []
        group_indices = []
        
        for key in unique_keys:
            # Find indices for this key
            indices = [i for i, k in enumerate(sample_keys) if k == key]
            group_indices.append(indices)
            
            # Get logits for this group
            group_logits = cts_logits[indices]
            
            # Apply max pooling
            max_logits = torch.max(group_logits, dim=0)[0]
            grouped_logits.append(max_logits)
        
        # Stack grouped logits
        grouped_logits = torch.stack(grouped_logits, dim=0)
        
        return grouped_logits, group_indices


class DPATEnsemble(nn.Module):
    """
    Ensemble of DPAT models for improved performance.
    
    Args:
        model_configs: List of model configurations
        ensemble_method: Method for combining predictions ('mean', 'weighted', 'learned')
        dropout: Dropout probability
    """
    
    def __init__(self,
                 model_configs: list,
                 ensemble_method: str = 'mean',
                 dropout: float = 0.1):
        super(DPATEnsemble, self).__init__()
        
        self.ensemble_method = ensemble_method
        self.num_models = len(model_configs)
        
        # Create ensemble models
        self.models = nn.ModuleList([
            DPAT(**config) for config in model_configs
        ])
        
        # Ensemble combination
        if ensemble_method == 'learned':
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
            self.ensemble_linear = nn.Linear(self.num_models, 1)
        elif ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
    
    def forward(self,
                align_matrix: torch.Tensor,
                bert_input_ids: torch.Tensor,
                bert_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            align_matrix: Alignment matrix
            bert_input_ids: BERT input IDs
            bert_attention_mask: BERT attention mask
            
        Returns:
            Ensemble predictions
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(align_matrix, bert_input_ids, bert_attention_mask)
            predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=-1)  # (batch_size, num_classes, num_models)
        
        # Combine predictions
        if self.ensemble_method == 'mean':
            ensemble_pred = torch.mean(predictions, dim=-1)
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_pred = torch.sum(predictions * weights, dim=-1)
        elif self.ensemble_method == 'learned':
            # Learned combination
            ensemble_pred = self.ensemble_linear(predictions.squeeze(1))
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred 