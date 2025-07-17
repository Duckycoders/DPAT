"""Configuration management for DPAT."""

import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DPATConfig:
    """Configuration class for DPAT training and inference."""
    
    # Data paths
    train_data_path: str = "data/miRAW_Train_Validation.txt"
    test_data_paths: Optional[Dict[str, str]] = None
    
    # Model parameters
    align_channels: int = 256
    semantic_channels: int = 256
    proj_dim: int = 256
    num_heads: int = 8
    lstm_dim: int = 128
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    bert_learning_rate: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Data processing
    window_size: int = 40
    seed_length: int = 12
    alignment_threshold: int = 6
    max_seq_length: int = 50
    
    # Training settings
    early_stopping_patience: int = 10
    save_top_k: int = 3
    use_amp: bool = True
    accumulate_grad_batches: int = 1
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    
    # RNA-BERT settings
    rna_bert_model: str = "multimolecule/rna_fm_t12u10_b512_v2"
    max_bert_length: int = 512
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_every_n_steps: int = 100
    val_check_interval: float = 1.0
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.test_data_paths is None:
            self.test_data_paths = {
                f"test{i}": f"data/miRAW_Test{i}.txt"
                for i in range(10)
            }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DPATConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "DPATConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2) 