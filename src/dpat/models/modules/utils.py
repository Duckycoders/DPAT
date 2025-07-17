"""Utility functions for DPAT model modules."""

import torch
import torch.nn as nn
import torch.nn.init as init
import math
from typing import Any


def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for different layer types.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        # Xavier normal initialization for linear layers
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        # Xavier normal initialization for convolutional layers
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    
    elif isinstance(module, nn.LSTM):
        # Orthogonal initialization for LSTM weights
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                init.xavier_normal_(param)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights
                init.orthogonal_(param)
            elif 'bias' in name:
                # Bias initialization
                init.constant_(param, 0)
                # Set forget gate bias to 1 for better gradient flow
                if 'bias_ih' in name:
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        # Batch normalization initialization
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    
    elif isinstance(module, nn.LayerNorm):
        # Layer normalization initialization
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    
    elif isinstance(module, nn.Embedding):
        # Embedding layer initialization
        init.normal_(module.weight, mean=0, std=0.1)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    
    elif isinstance(module, nn.MultiheadAttention):
        # Multi-head attention initialization
        for name, param in module.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0)


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_parameters(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze model parameters.
    
    Args:
        model: PyTorch model
        freeze: Whether to freeze parameters
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def get_model_size(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)  # Convert to MB


def apply_gradient_clipping(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask for variable-length sequences.
    
    Args:
        lengths: Tensor of sequence lengths
        max_len: Maximum sequence length
        
    Returns:
        Boolean mask tensor (True for padding positions)
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) >= lengths.unsqueeze(1)
    
    return mask


def create_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask for transformer models.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Causal attention mask
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def compute_receptive_field(kernel_sizes: list, strides: list = None, 
                           dilations: list = None) -> int:
    """
    Compute receptive field size for a stack of convolutional layers.
    
    Args:
        kernel_sizes: List of kernel sizes
        strides: List of strides (default: all 1s)
        dilations: List of dilations (default: all 1s)
        
    Returns:
        Receptive field size
    """
    if strides is None:
        strides = [1] * len(kernel_sizes)
    if dilations is None:
        dilations = [1] * len(kernel_sizes)
    
    rf = 1
    for k, s, d in zip(kernel_sizes, strides, dilations):
        rf += (k - 1) * d
    
    return rf


def get_activation_function(activation: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation: Activation function name
        
    Returns:
        Activation function module
    """
    activation_map = {
        'relu': nn.ReLU(inplace=True),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.1, inplace=True),
        'elu': nn.ELU(inplace=True),
        'selu': nn.SELU(inplace=True),
        'mish': nn.Mish(inplace=True),
        'identity': nn.Identity()
    }
    
    if activation.lower() not in activation_map:
        raise ValueError(f"Unknown activation function: {activation}")
    
    return activation_map[activation.lower()]


def warmup_cosine_schedule(step: int, warmup_steps: int, total_steps: int,
                          base_lr: float = 1e-3, min_lr: float = 1e-6) -> float:
    """
    Compute learning rate with warmup and cosine annealing.
    
    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        total_steps: Total number of steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        
    Returns:
        Device object
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         scheduler: Any, epoch: int, loss: float,
                         checkpoint_path: str, **kwargs) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Current loss value
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_model_checkpoint(model: nn.Module, checkpoint_path: str,
                         optimizer: torch.optim.Optimizer = None,
                         scheduler: Any = None, 
                         device: torch.device = None) -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint 