"""
DPAT训练器
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional

from ..models import DPAT
from ..data import DPATDataset
from ..config import DPATConfig

logger = logging.getLogger(__name__)


class DPATTrainer:
    """DPAT训练器"""
    
    def __init__(self, config: DPATConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 初始化模型
        self.model = DPAT(config.model)
        self.model.to(self.device)
        
        # 初始化优化器
        self._setup_optimizer()
        
        # 初始化损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 初始化数据加载器
        self._setup_data_loaders()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_score = 0.0
        
    def _setup_optimizer(self):
        """设置优化器"""
        # 不同学习率：BERT参数使用1e-5，其他参数使用1e-3
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name.lower():
                bert_params.append(param)
            else:
                other_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': self.config.training.bert_lr},
            {'params': other_params, 'lr': self.config.training.lr}
        ], weight_decay=self.config.training.weight_decay)
        
    def _setup_data_loaders(self):
        """设置数据加载器"""
        # 训练集
        train_dataset = DPATDataset(
            self.config.data.train_data_path,
            self.config.data,
            split='train'
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,  # 保持原始顺序
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        
        # 验证集
        val_dataset = DPATDataset(
            self.config.data.train_data_path,
            self.config.data,
            split='val'
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练一个batch"""
        self.model.train()
        
        # 移动数据到设备
        align_matrix = batch['align_matrix'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device).float()
        
        # 前向传播
        if self.config.training.mixed_precision:
            with autocast():
                outputs = self.model(align_matrix, input_ids, attention_mask)
                loss = self.criterion(outputs.squeeze(), labels)
        else:
            outputs = self.model(align_matrix, input_ids, attention_mask)
            loss = self.criterion(outputs.squeeze(), labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        
        if self.config.training.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                align_matrix = batch['align_matrix'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device).float()
                
                outputs = self.model(align_matrix, input_ids, attention_mask)
                loss = self.criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                
                # 收集预测和标签
                preds = torch.sigmoid(outputs.squeeze())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算指标
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        
        # 简单的二分类指标
        pred_labels = (all_preds > 0.5).float()
        accuracy = (pred_labels == all_labels).float().mean().item()
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(self, max_steps: Optional[int] = None):
        """训练主循环"""
        logger.info(f"Starting training on {self.device}")
        
        step_count = 0
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # 训练循环
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                # 训练步骤
                train_metrics = self.train_step(batch)
                
                # 更新进度条
                pbar.set_postfix(train_metrics)
                
                self.global_step += 1
                step_count += 1
                
                # 检查是否达到最大步数
                if max_steps and step_count >= max_steps:
                    logger.info(f"Reached max_steps ({max_steps}), stopping training")
                    return
            
            # 验证
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch}: {val_metrics}")
            
            # 早停检查
            if val_metrics['val_accuracy'] > self.best_score:
                self.best_score = val_metrics['val_accuracy']
                self.save_checkpoint('best.pt')
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved to {filename}") 