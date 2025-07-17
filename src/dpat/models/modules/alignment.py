"""Alignment Path module for DPAT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .attention import SEBlock, CBAMBlock
from .utils import init_weights


class InceptionBlock(nn.Module):
    """
    Multi-scale Inception block for alignment features.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_sizes: List of kernel sizes for parallel branches
        dropout: Dropout probability
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: List[int] = [1, 3, 5, 7], 
                 dropout: float = 0.1):
        super(InceptionBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        
        # Output channels per branch
        branch_channels = out_channels // len(kernel_sizes)
        
        # Create parallel branches
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                ),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)
        
        # Adjust output channels if not evenly divisible
        remaining_channels = out_channels - branch_channels * len(kernel_sizes)
        if remaining_channels > 0:
            # Add an extra 1x1 conv for remaining channels
            extra_branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=remaining_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm1d(remaining_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.branches.append(extra_branch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Inception block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, length)
        """
        # Process through all branches
        branch_outputs = []
        for branch in self.branches:
            branch_output = branch(x)
            branch_outputs.append(branch_output)
        
        # Concatenate along channel dimension
        output = torch.cat(branch_outputs, dim=1)
        
        return output


class AlignmentPath(nn.Module):
    """
    Alignment path for processing sequence alignment matrices.
    
    Architecture:
    1. 1×1 convolution for channel adjustment
    2. Multi-scale Inception convolutions (k=1,3,5,7)
    3. SE channel attention
    4. CBAM spatial attention
    
    Args:
        input_channels: Number of input channels (default: 10 for alignment matrix)
        output_channels: Number of output channels
        hidden_channels: Number of hidden channels
        kernel_sizes: Kernel sizes for Inception block
        dropout: Dropout probability
        use_se: Whether to use SE attention
        use_cbam: Whether to use CBAM attention
    """
    
    def __init__(self, 
                 input_channels: int = 10,
                 output_channels: int = 256,
                 hidden_channels: int = 128,
                 kernel_sizes: List[int] = [1, 3, 5, 7],
                 dropout: float = 0.1,
                 use_se: bool = True,
                 use_cbam: bool = True):
        super(AlignmentPath, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.use_se = use_se
        self.use_cbam = use_cbam
        
        # 1×1 convolution for initial channel adjustment
        self.stem_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Multi-scale Inception convolution
        self.inception_block = InceptionBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # SE channel attention
        if self.use_se:
            self.se_block = SEBlock(
                channels=hidden_channels,
                reduction=16
            )
        
        # CBAM spatial attention
        if self.use_cbam:
            self.cbam_block = CBAMBlock(
                channels=hidden_channels,
                reduction=16,
                kernel_size=7
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Residual connection if dimensions match
        self.use_residual = (input_channels == output_channels)
        if not self.use_residual:
            self.residual_projection = nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                bias=False
            )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through alignment path.
        
        Args:
            x: Input alignment matrix of shape (batch_size, 10, 50)
            
        Returns:
            Processed alignment features of shape (batch_size, output_channels, 50)
        """
        # Store input for residual connection
        residual = x
        
        # 1×1 convolution for channel adjustment
        x = self.stem_conv(x)
        
        # Multi-scale Inception convolution
        x = self.inception_block(x)
        
        # SE channel attention
        if self.use_se:
            x = self.se_block(x)
        
        # CBAM spatial attention
        if self.use_cbam:
            x = self.cbam_block(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
        else:
            x = x + self.residual_projection(residual)
        
        return x


class MultiScaleAlignmentPath(nn.Module):
    """
    Multi-scale alignment path with multiple Inception blocks.
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        hidden_channels: Number of hidden channels
        num_blocks: Number of Inception blocks
        kernel_sizes: Kernel sizes for each block
        dropout: Dropout probability
    """
    
    def __init__(self,
                 input_channels: int = 10,
                 output_channels: int = 256,
                 hidden_channels: int = 128,
                 num_blocks: int = 3,
                 kernel_sizes: List[int] = [1, 3, 5, 7],
                 dropout: float = 0.1):
        super(MultiScaleAlignmentPath, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Multiple Inception blocks
        self.inception_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                InceptionBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_sizes=kernel_sizes,
                    dropout=dropout
                ),
                SEBlock(hidden_channels),
                CBAMBlock(hidden_channels)
            )
            self.inception_blocks.append(block)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale alignment path.
        
        Args:
            x: Input alignment matrix of shape (batch_size, 10, 50)
            
        Returns:
            Multi-scale alignment features of shape (batch_size, output_channels, 50)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Process through multiple Inception blocks
        for block in self.inception_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class AdaptiveAlignmentPath(nn.Module):
    """
    Adaptive alignment path that adjusts to different sequence lengths.
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        hidden_channels: Number of hidden channels
        max_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self,
                 input_channels: int = 10,
                 output_channels: int = 256,
                 hidden_channels: int = 128,
                 max_length: int = 50,
                 dropout: float = 0.1):
        super(AdaptiveAlignmentPath, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.max_length = max_length
        
        # Adaptive convolution layers
        self.adaptive_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Multi-scale processing
        self.multi_scale = InceptionBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_sizes=[1, 3, 5, 7],
            dropout=dropout
        )
        
        # Attention mechanisms
        self.se_attention = SEBlock(hidden_channels)
        self.cbam_attention = CBAMBlock(hidden_channels)
        
        # Adaptive pooling for different lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_length)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptive alignment path.
        
        Args:
            x: Input alignment matrix of shape (batch_size, input_channels, length)
            
        Returns:
            Adaptive alignment features of shape (batch_size, output_channels, max_length)
        """
        # Adaptive convolution
        x = self.adaptive_conv(x)
        
        # Multi-scale processing
        x = self.multi_scale(x)
        
        # Apply attention mechanisms
        x = self.se_attention(x)
        x = self.cbam_attention(x)
        
        # Adaptive pooling to fixed length
        x = self.adaptive_pool(x)
        
        # Output projection
        x = self.output_layers(x)
        
        return x 