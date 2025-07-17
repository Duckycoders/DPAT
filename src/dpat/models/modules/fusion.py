"""Cross-attention fusion module for DPAT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .utils import init_weights


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module for combining alignment and semantic features.
    
    Architecture:
    1. Bidirectional cross-attention (alignment -> semantic, semantic -> alignment)
    2. Layer normalization and residual connections
    3. Feature concatenation and gated fusion
    
    Args:
        embed_dim: Embedding dimension (should match proj_dim)
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_gate: Whether to use gated fusion
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_gate: bool = True):
        super(CrossAttentionFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gate = use_gate
        
        # Cross-attention: alignment -> semantic
        self.align_to_semantic = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: semantic -> alignment
        self.semantic_to_align = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Gated fusion
        if self.use_gate:
            # Gate network for weighted combination
            self.gate_network = nn.Sequential(
                nn.Linear(3 * embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
        
        # Final fusion layer
        self.fusion_layer = nn.Linear(3 * embed_dim, embed_dim)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self,
                align_features: torch.Tensor,
                semantic_features: torch.Tensor,
                align_mask: Optional[torch.Tensor] = None,
                semantic_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through cross-attention fusion.
        
        Args:
            align_features: Alignment features of shape (batch_size, align_len, embed_dim)
            semantic_features: Semantic features of shape (batch_size, semantic_len, embed_dim)
            align_mask: Mask for alignment features (optional)
            semantic_mask: Mask for semantic features (optional)
            
        Returns:
            Fused features of shape (batch_size, align_len, embed_dim)
        """
        # Store original features for residual connections
        align_residual = align_features
        semantic_residual = semantic_features
        
        # Cross-attention 1: Query=alignment, Key/Value=semantic
        align_attended, _ = self.align_to_semantic(
            query=align_features,
            key=semantic_features,
            value=semantic_features,
            key_padding_mask=semantic_mask
        )
        
        # Add residual connection and layer norm
        align_attended = self.layer_norm1(
            align_residual + self.dropout1(align_attended)
        )
        
        # Cross-attention 2: Query=semantic, Key/Value=alignment
        semantic_attended, _ = self.semantic_to_align(
            query=semantic_features,
            key=align_features,
            value=align_features,
            key_padding_mask=align_mask
        )
        
        # Add residual connection and layer norm
        semantic_attended = self.layer_norm2(
            semantic_residual + self.dropout2(semantic_attended)
        )
        
        # Ensure same sequence length for fusion (pad or truncate semantic to align length)
        align_len = align_features.size(1)
        semantic_len = semantic_attended.size(1)
        
        if semantic_len > align_len:
            # Truncate semantic features
            semantic_attended = semantic_attended[:, :align_len, :]
        elif semantic_len < align_len:
            # Pad semantic features
            padding = torch.zeros(
                semantic_attended.size(0), 
                align_len - semantic_len, 
                semantic_attended.size(2),
                device=semantic_attended.device
            )
            semantic_attended = torch.cat([semantic_attended, padding], dim=1)
        
        # Concatenate features for fusion
        concatenated = torch.cat([
            align_attended,
            semantic_attended,
            align_features  # Original alignment features
        ], dim=2)  # (batch_size, align_len, 3 * embed_dim)
        
        # Gated fusion
        if self.use_gate:
            # Compute gate weights
            gate_weights = self.gate_network(concatenated)
            
            # Apply gated combination
            fused_features = (
                gate_weights * align_attended + 
                (1 - gate_weights) * semantic_attended
            )
            
            # Add original features
            fused_features = fused_features + align_features
        else:
            # Simple linear fusion
            fused_features = self.fusion_layer(concatenated)
        
        return fused_features


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with learnable position embeddings.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(init_weights)
    
    def add_position_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embedding to input features."""
        seq_len = x.size(1)
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
        return x + pos_emb
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head cross-attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Padding mask for keys
            
        Returns:
            Attention output of shape (batch_size, seq_len, embed_dim)
        """
        # Add position embeddings
        query = self.add_position_embedding(query)
        key = self.add_position_embedding(key)
        value = self.add_position_embedding(value)
        
        # Multi-head attention
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout_layer(attn_output))
        
        return output


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module that learns to weight different features.
    
    Args:
        embed_dim: Embedding dimension
        num_features: Number of feature types to fuse
        hidden_dim: Hidden dimension for fusion network
        dropout: Dropout probability
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_features: int = 3,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super(AdaptiveFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Attention network for feature weighting
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
            nn.Softmax(dim=-1)
        )
        
        # Feature projection layers
        self.feature_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_features)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptive fusion.
        
        Args:
            *features: Variable number of feature tensors
            
        Returns:
            Fused features
        """
        assert len(features) == self.num_features, \
            f"Expected {self.num_features} features, got {len(features)}"
        
        batch_size, seq_len, embed_dim = features[0].shape
        
        # Stack features
        stacked_features = torch.stack(features, dim=2)  # (batch, seq_len, num_features, embed_dim)
        
        # Compute attention weights for each position
        # Average pooling over sequence dimension for global context
        global_context = torch.mean(stacked_features, dim=1)  # (batch, num_features, embed_dim)
        global_context = torch.mean(global_context, dim=1)  # (batch, embed_dim)
        
        # Compute attention weights
        attention_weights = self.attention_net(global_context)  # (batch, num_features)
        attention_weights = attention_weights.unsqueeze(1).unsqueeze(-1)  # (batch, 1, num_features, 1)
        
        # Apply attention weights
        weighted_features = stacked_features * attention_weights
        
        # Sum over features
        fused_features = torch.sum(weighted_features, dim=2)  # (batch, seq_len, embed_dim)
        
        # Project each feature individually and sum
        projected_features = []
        for i, feature in enumerate(features):
            projected = self.feature_projections[i](feature)
            projected_features.append(projected)
        
        # Weighted sum of projected features
        weights_expanded = attention_weights.expand(-1, seq_len, -1, embed_dim)
        weighted_projected = sum(
            weight * feature for weight, feature in zip(
                weights_expanded.unbind(dim=2), projected_features
            )
        )
        
        # Final projection
        output = self.output_projection(weighted_projected)
        
        return output


class GatedFusion(nn.Module):
    """
    Gated fusion module for combining two feature streams.
    
    Args:
        embed_dim: Embedding dimension
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super(GatedFusion, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Gate network
        self.gate_linear = nn.Linear(2 * embed_dim, embed_dim)
        self.gate_activation = nn.Sigmoid()
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gated fusion.
        
        Args:
            feature1: First feature tensor of shape (batch_size, seq_len, embed_dim)
            feature2: Second feature tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Fused features of shape (batch_size, seq_len, embed_dim)
        """
        # Concatenate features
        concat_features = torch.cat([feature1, feature2], dim=-1)
        
        # Compute gate
        gate = self.gate_activation(self.gate_linear(concat_features))
        
        # Gated combination
        gated_output = gate * feature1 + (1 - gate) * feature2
        
        # Feature transformation
        transformed = self.feature_transform(concat_features)
        
        # Residual connection and layer norm
        output = self.layer_norm(gated_output + transformed)
        
        return output 