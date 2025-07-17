#!/usr/bin/env python3
"""Training script for DPAT model."""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchmetrics
from tqdm import tqdm
import logging
from datetime import datetime

from src.dpat import DPAT, DPATDataset, DPATConfig
from src.dpat.data.dataset import create_dataloaders
from src.dpat.models.modules.utils import apply_gradient_clipping
from src.dpat.utils.metrics import compute_metrics
from src.dpat.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DPAT model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum number of training steps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create DPAT model from configuration."""
    model = DPAT(
        alignment_config=config['model']['alignment_config'],
        semantic_config=config['model']['semantic_config'],
        fusion_config=config['model']['fusion_config'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        use_simple_semantic=config['model']['use_simple_semantic']
    )
    
    return model


def create_optimizer(model, config):
    """Create optimizer with different learning rates for BERT and other parameters."""
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'bert' in name.lower():
            bert_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': config['training']['bert_learning_rate']},
        {'params': other_params, 'lr': config['training']['learning_rate']}
    ])
    
    return optimizer


def create_scheduler(optimizer, config, total_steps):
    """Create learning rate scheduler."""
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config['training']['warmup_steps']
    )
    
    return scheduler


def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Metrics
    accuracy = torchmetrics.Accuracy(task='binary').to(device)
    precision = torchmetrics.Precision(task='binary').to(device)
    recall = torchmetrics.Recall(task='binary').to(device)
    f1 = torchmetrics.F1Score(task='binary').to(device)
    auroc = torchmetrics.AUROC(task='binary').to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Move to device
        align_matrix = batch['align_matrix'].to(device)
        bert_input_ids = batch['bert_input_ids'].to(device)
        bert_attention_mask = batch['bert_attention_mask'].to(device)
        labels = batch['label'].float().to(device)
        
        # Forward pass with AMP
        with autocast(enabled=config['training_settings']['use_amp']):
            logits = model(align_matrix, bert_input_ids, bert_attention_mask)
            loss = criterion(logits.squeeze(), labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['training']['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            apply_gradient_clipping(model, config['training']['max_grad_norm'])
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update metrics
        probs = torch.sigmoid(logits.squeeze())
        accuracy.update(probs, labels.int())
        precision.update(probs, labels.int())
        recall.update(probs, labels.int())
        f1.update(probs, labels.int())
        auroc.update(probs, labels.int())
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log every N steps
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Compute epoch metrics
    epoch_metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy.compute().item(),
        'precision': precision.compute().item(),
        'recall': recall.compute().item(),
        'f1': f1.compute().item(),
        'auroc': auroc.compute().item()
    }
    
    return epoch_metrics


def validate(model, val_loader, config, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Metrics
    accuracy = torchmetrics.Accuracy(task='binary').to(device)
    precision = torchmetrics.Precision(task='binary').to(device)
    recall = torchmetrics.Recall(task='binary').to(device)
    f1 = torchmetrics.F1Score(task='binary').to(device)
    auroc = torchmetrics.AUROC(task='binary').to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device
            align_matrix = batch['align_matrix'].to(device)
            bert_input_ids = batch['bert_input_ids'].to(device)
            bert_attention_mask = batch['bert_attention_mask'].to(device)
            labels = batch['label'].float().to(device)
            
            # Forward pass
            logits = model(align_matrix, bert_input_ids, bert_attention_mask)
            loss = criterion(logits.squeeze(), labels)
            
            # Update metrics
            probs = torch.sigmoid(logits.squeeze())
            accuracy.update(probs, labels.int())
            precision.update(probs, labels.int())
            recall.update(probs, labels.int())
            f1.update(probs, labels.int())
            auroc.update(probs, labels.int())
            
            total_loss += loss.item()
            num_batches += 1
    
    # Compute validation metrics
    val_metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy.compute().item(),
        'precision': precision.compute().item(),
        'recall': recall.compute().item(),
        'f1': f1.compute().item(),
        'auroc': auroc.compute().item()
    }
    
    return val_metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    torch.manual_seed(config['seed'])
    
    # Setup logging
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dpat_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DPATDataset(
        data_path=config['data']['train_data_path'],
        split='train',
        cache_dir=config['data']['cache_dir'],
        **config['data_processing']
    )
    
    val_dataset = DPATDataset(
        data_path=config['data']['train_data_path'],
        split='val',
        cache_dir=config['data']['cache_dir'],
        **config['data_processing']
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # Create scaler for mixed precision
    scaler = GradScaler(enabled=config['training_settings']['use_amp'])
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, config, device)
        
        # Log metrics
        logging.info(f"Epoch {epoch} - Train: {train_metrics}")
        logging.info(f"Epoch {epoch} - Val: {val_metrics}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_dir = config['paths']['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_metrics['f1'],
                'config': config
            }, checkpoint_path)
            
            logging.info(f"New best model saved with val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training_settings']['early_stopping_patience']:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        
        # Max steps check
        if args.max_steps and (epoch + 1) * len(train_loader) >= args.max_steps:
            logging.info(f"Reached max steps: {args.max_steps}")
            break
    
    logging.info("Training completed!")


if __name__ == "__main__":
    main() 