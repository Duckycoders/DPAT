"""Attention mechanisms for DPAT: SE and CBAM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for hidden layer
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Squeeze and excitation layers
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE block.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Attention-weighted tensor of same shape
        """
        batch_size, channels, length = x.size()
        
        # Squeeze: global average pooling
        # (batch_size, channels, length) -> (batch_size, channels, 1)
        squeeze = self.avg_pool(x).view(batch_size, channels)
        
        # Excitation: FC layers
        # (batch_size, channels) -> (batch_size, channels)
        excitation = self.fc(squeeze)
        
        # Reshape for broadcasting
        # (batch_size, channels) -> (batch_size, channels, 1)
        excitation = excitation.view(batch_size, channels, 1)
        
        # Apply attention weights
        return x * excitation


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global pooling layers
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of channel attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Channel attention weights of shape (batch_size, channels, 1)
        """
        batch_size, channels, length = x.size()
        
        # Average pooling branch
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.mlp(avg_pool)
        
        # Max pooling branch
        max_pool = self.max_pool(x).view(batch_size, channels)
        max_out = self.mlp(max_pool)
        
        # Combine and apply sigmoid
        channel_attention = torch.sigmoid(avg_out + max_out)
        
        return channel_attention.view(batch_size, channels, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        
        # Convolution for spatial attention
        self.conv = nn.Conv1d(
            in_channels=2, 
            out_channels=1, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Spatial attention weights of shape (batch_size, 1, length)
        """
        # Channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, length)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, length)
        
        # Concatenate along channel dimension
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, 2, length)
        
        # Apply convolution and sigmoid
        spatial_attention = torch.sigmoid(self.conv(spatial_input))
        
        return spatial_attention


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Applies channel attention followed by spatial attention.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAMBlock, self).__init__()
        self.channels = channels
        
        # Channel and spatial attention modules
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBAM.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Attention-weighted tensor of same shape
        """
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(x_channel)
        x_spatial = x_channel * spatial_weights
        
        return x_spatial


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention combining SE and CBAM.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for attention
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(MultiScaleAttention, self).__init__()
        self.channels = channels
        
        # SE and CBAM blocks
        self.se_block = SEBlock(channels, reduction)
        self.cbam_block = CBAMBlock(channels, reduction, kernel_size)
        
        # Fusion weights
        self.fusion_conv = nn.Conv1d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )
        self.fusion_norm = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-scale attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Multi-scale attention weighted tensor
        """
        # Apply SE and CBAM attention
        se_out = self.se_block(x)
        cbam_out = self.cbam_block(x)
        
        # Concatenate and fuse
        combined = torch.cat([se_out, cbam_out], dim=1)
        fused = self.fusion_conv(combined)
        fused = self.fusion_norm(fused)
        
        # Add residual connection
        return x + fused


class SelfAttention(nn.Module):
    """
    Self-attention module for sequence modeling.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Padding mask for attention
            
        Returns:
            Self-attention output of same shape
        """
        # Self-attention
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        
        # Add residual connection and layer norm
        output = self.layer_norm(x + self.dropout(attn_output))
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-like models.
    
    Args:
        embed_dim: Embedding dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, embed_dim: int, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Position-encoded tensor of same shape
        """
        return x + self.pe[:, :x.size(1)] 