"""Semantic Path module for DPAT including BiLSTMBlock."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional
import multimolecule  # 必不可少，否则类注册不到
from transformers import AutoModel

from .utils import init_weights


class BiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM block for processing sequential features.
    
    Args:
        input_size: Size of input features (default: 768 for BERT)
        hidden_size: Hidden size for LSTM (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.3)
        proj_dim: Projection dimension for global features
    """
    
    def __init__(self, 
                 input_size: int = 768,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 proj_dim: int = 256):
        super(BiLSTMBlock, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.proj_dim = proj_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Global pooling projection
        # 4 * hidden_size: max_pool + mean_pool for both directions
        self.global_projection = nn.Linear(
            4 * hidden_size, proj_dim
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BiLSTM block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of:
            - timestep_features: (batch_size, seq_len, 2*hidden_size)
            - global_features: (batch_size, proj_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Calculate actual sequence lengths
        lens = attention_mask.sum(dim=1).cpu()
        
        # Pack padded sequence for efficient processing
        packed_input = pack_padded_sequence(
            x, lens, batch_first=True, enforce_sorted=False
        )
        
        # Forward through LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        
        # Unpack to get padded output
        lstm_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        
        # Apply layer normalization
        lstm_output = self.layer_norm(lstm_output)
        
        # Global pooling for position-independent features
        # Apply attention mask to ignore padded positions
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(lstm_output)
        masked_output = lstm_output * mask_expanded
        
        # Max pooling (ignore padded positions)
        max_pooled = torch.max(masked_output, dim=1)[0]  # (batch_size, 2*hidden_size)
        
        # Mean pooling (ignore padded positions)
        sum_pooled = torch.sum(masked_output, dim=1)  # (batch_size, 2*hidden_size)
        valid_lengths = lens.float().unsqueeze(1).to(sum_pooled.device)
        mean_pooled = sum_pooled / valid_lengths  # (batch_size, 2*hidden_size)
        
        # Concatenate max and mean pooled features
        global_features = torch.cat([max_pooled, mean_pooled], dim=1)  # (batch_size, 4*hidden_size)
        
        # Project to target dimension
        global_features = self.global_projection(global_features)  # (batch_size, proj_dim)
        
        return lstm_output, global_features


class ConvBlock(nn.Module):
    """
    1D Convolutional block for processing BERT embeddings.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolution
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class SemanticPath(nn.Module):
    """
    Semantic path for processing RNA-BERT embeddings.
    
    Architecture:
    1. RNA-BERT embedding extraction
    2. Two-level 1D convolution (kernel_size=3, then 5)
    3. BiLSTM block for sequence modeling
    4. Global pooling and projection
    
    Args:
        bert_model_name: Name of pre-trained BERT model
        bert_hidden_size: Hidden size of BERT model (default: 768)
        conv_hidden_size: Hidden size for convolution layers (default: 1024)
        lstm_hidden_size: Hidden size for LSTM (default: 128)
        proj_dim: Projection dimension (default: 256)
        dropout: Dropout probability (default: 0.1)
        freeze_bert: Whether to freeze BERT parameters
    """
    
    def __init__(self,
                 bert_model_name: str = "multimolecule/mrnafm",
                 bert_hidden_size: int = 768,
                 conv_hidden_size: int = 1024,
                 lstm_hidden_size: int = 128,
                 proj_dim: int = 256,
                 dropout: float = 0.1,
                 freeze_bert: bool = False):
        super(SemanticPath, self).__init__()
        
        self.bert_model_name = bert_model_name
        self.bert_hidden_size = bert_hidden_size
        self.conv_hidden_size = conv_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.freeze_bert = freeze_bert
        
        # Load pre-trained RNA-BERT model
        try:
            self.bert_model = AutoModel.from_pretrained(
                bert_model_name,
                trust_remote_code=True  # 必须！允许自定义类
            )
            self.bert_hidden_size = self.bert_model.config.hidden_size
        except Exception as e:
            print(f"Error loading BERT model {bert_model_name}: {e}")
            print("Using simple embedding layer instead...")
            self.bert_model = nn.Embedding(
                num_embeddings=8192,  # Large vocab size
                embedding_dim=bert_hidden_size,
                padding_idx=0
            )
        
        # Freeze BERT parameters if requested
        if freeze_bert and hasattr(self.bert_model, 'parameters'):
            for param in self.bert_model.parameters():
                param.requires_grad = False
        
        # Two-level 1D convolution
        # First level: kernel_size=3, 768 -> 1024
        self.conv1 = ConvBlock(
            in_channels=self.bert_hidden_size,
            out_channels=conv_hidden_size,
            kernel_size=3,
            dropout=dropout
        )
        
        # Second level: kernel_size=5, 1024 -> 2*lstm_hidden_size
        self.conv2 = ConvBlock(
            in_channels=conv_hidden_size,
            out_channels=2 * lstm_hidden_size,
            kernel_size=5,
            dropout=dropout
        )
        
        # BiLSTM block
        self.bilstm_block = BiLSTMBlock(
            input_size=2 * lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=0.3,
            proj_dim=proj_dim
        )
        
        # Initialize weights for new layers
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
    
    def forward(self, 
                bert_input_ids: torch.Tensor,
                bert_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through semantic path.
        
        Args:
            bert_input_ids: BERT input IDs of shape (batch_size, seq_len)
            bert_attention_mask: BERT attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of:
            - timestep_features: (batch_size, seq_len, proj_dim)
            - global_features: (batch_size, proj_dim)
        """
        # Extract BERT embeddings
        if hasattr(self.bert_model, 'config'):
            # Real BERT model
            bert_outputs = self.bert_model(
                input_ids=bert_input_ids,
                attention_mask=bert_attention_mask
            )
            bert_embeddings = bert_outputs.last_hidden_state
        else:
            # Simple embedding layer
            bert_embeddings = self.bert_model(bert_input_ids)
        
        # bert_embeddings: (batch_size, seq_len, bert_hidden_size)
        
        # Transpose for 1D convolution: (batch_size, bert_hidden_size, seq_len)
        conv_input = bert_embeddings.transpose(1, 2)
        
        # First convolution layer
        conv_output1 = self.conv1(conv_input)
        
        # Second convolution layer
        conv_output2 = self.conv2(conv_output1)
        
        # Transpose back for BiLSTM: (batch_size, seq_len, 2*lstm_hidden_size)
        lstm_input = conv_output2.transpose(1, 2)
        
        # BiLSTM processing
        timestep_features, global_features = self.bilstm_block(
            lstm_input, bert_attention_mask
        )
        
        # Project timestep features to target dimension
        timestep_proj = nn.Linear(
            2 * self.lstm_hidden_size, self.proj_dim
        ).to(timestep_features.device)
        timestep_features = timestep_proj(timestep_features)
        
        return timestep_features, global_features


class SimpleSemanticPath(nn.Module):
    """
    Simplified semantic path without external BERT dependency.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        conv_hidden_size: Hidden size for convolution layers
        lstm_hidden_size: Hidden size for LSTM
        proj_dim: Projection dimension
        dropout: Dropout probability
    """
    
    def __init__(self,
                 vocab_size: int = 8192,
                 embed_dim: int = 768,
                 conv_hidden_size: int = 1024,
                 lstm_hidden_size: int = 128,
                 proj_dim: int = 256,
                 dropout: float = 0.1):
        super(SimpleSemanticPath, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.conv_hidden_size = conv_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.proj_dim = proj_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        
        # Two-level convolution
        self.conv1 = ConvBlock(
            in_channels=embed_dim,
            out_channels=conv_hidden_size,
            kernel_size=3,
            dropout=dropout
        )
        
        self.conv2 = ConvBlock(
            in_channels=conv_hidden_size,
            out_channels=2 * lstm_hidden_size,
            kernel_size=5,
            dropout=dropout
        )
        
        # BiLSTM block
        self.bilstm_block = BiLSTMBlock(
            input_size=2 * lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=0.3,
            proj_dim=proj_dim
        )
        
        # Timestep projection
        self.timestep_proj = nn.Linear(2 * lstm_hidden_size, proj_dim)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through simple semantic path.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of:
            - timestep_features: (batch_size, seq_len, proj_dim)
            - global_features: (batch_size, proj_dim)
        """
        # Embedding
        embeddings = self.embedding(input_ids)
        
        # Convolution layers
        conv_input = embeddings.transpose(1, 2)
        conv_output1 = self.conv1(conv_input)
        conv_output2 = self.conv2(conv_output1)
        
        # BiLSTM
        lstm_input = conv_output2.transpose(1, 2)
        timestep_features, global_features = self.bilstm_block(
            lstm_input, attention_mask
        )
        
        # Project timestep features
        timestep_features = self.timestep_proj(timestep_features)
        
        return timestep_features, global_features 