"""Dataset class for DPAT training and inference."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import h5py
from transformers import AutoTokenizer

from .preprocessing import preprocess_dataset, process_mirna_mrna_pair
from .utils import (
    load_rna_bert_tokenizer, 
    prepare_bert_input, 
    save_processed_data,
    load_processed_data,
    validate_data_consistency
)


class DPATDataset(Dataset):
    """
    Dataset class for DPAT model training and inference.
    
    Returns align_matrix, bert_input_ids, bert_attention_mask, label for each sample.
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer = None,
                 split: str = 'train',
                 window_size: int = 40,
                 seed_length: int = 12,
                 alignment_threshold: int = 6,
                 max_length: int = 50,
                 max_bert_length: int = 512,
                 cache_dir: str = 'cache',
                 force_reprocess: bool = False):
        """
        Initialize DPAT dataset.
        
        Args:
            data_path: Path to the raw data file
            tokenizer: RNA-BERT tokenizer (if None, will load default)
            split: Data split ('train', 'val', or 'test')
            window_size: Sliding window size for alignment
            seed_length: miRNA seed region length
            alignment_threshold: Minimum alignment score
            max_length: Maximum length for alignment matrix
            max_bert_length: Maximum length for BERT input
            cache_dir: Directory for caching processed data
            force_reprocess: Force reprocessing even if cache exists
        """
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.seed_length = seed_length
        self.alignment_threshold = alignment_threshold
        self.max_length = max_length
        self.max_bert_length = max_bert_length
        self.cache_dir = cache_dir
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = load_rna_bert_tokenizer()
        else:
            self.tokenizer = tokenizer
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache file path
        data_name = os.path.basename(data_path).replace('.txt', '')
        cache_file = os.path.join(
            cache_dir, 
            f"{data_name}_processed_w{window_size}_s{seed_length}_t{alignment_threshold}.h5"
        )
        
        # Load or process data
        if os.path.exists(cache_file) and not force_reprocess:
            print(f"Loading cached data from {cache_file}")
            self.data = self._load_cached_data(cache_file)
        else:
            print(f"Processing data from {data_path}")
            self.data = self._process_data(data_path, cache_file)
        
        # Filter by split
        if split in ['train', 'val']:
            self.data = self._filter_by_split(self.data, split)
        
        # Validate data consistency
        if not force_reprocess:
            self._validate_data_consistency()
        
        print(f"Dataset loaded: {len(self.data['labels'])} samples for split '{split}'")
    
    def _process_data(self, data_path: str, cache_file: str) -> Dict[str, Any]:
        """Process raw data and cache results."""
        # Process data using preprocessing functions
        processed_samples = preprocess_dataset(
            data_path=data_path,
            window_size=self.window_size,
            seed_length=self.seed_length,
            alignment_threshold=self.alignment_threshold,
            max_length=self.max_length
        )
        
        # Prepare data for caching
        alignment_matrices = []
        bert_input_ids = []
        bert_attention_masks = []
        labels = []
        sample_keys = []
        mirna_ids = []
        mrna_ids = []
        splits = []
        alignment_scores = []
        
        print("Generating BERT encodings...")
        for sample in processed_samples:
            # Get alignment matrix
            alignment_matrices.append(sample['alignment_matrix'])
            
            # Prepare BERT input
            bert_input = prepare_bert_input(
                sample['mirna_seq'],
                sample['mrna_window'],
                self.tokenizer,
                self.max_bert_length
            )
            
            bert_input_ids.append(bert_input['input_ids'])
            bert_attention_masks.append(bert_input['attention_mask'])
            
            # Store other information
            labels.append(sample['label'])
            sample_keys.append(sample['sample_key'])
            mirna_ids.append(sample['mirna_id'])
            mrna_ids.append(sample['mrna_id'])
            splits.append(sample['split'])
            alignment_scores.append(sample['alignment_score'])
        
        # Convert to numpy arrays
        data = {
            'alignment_matrices': np.array(alignment_matrices),
            'bert_input_ids': np.array(bert_input_ids),
            'bert_attention_masks': np.array(bert_attention_masks),
            'labels': np.array(labels),
            'sample_keys': sample_keys,
            'mirna_ids': mirna_ids,
            'mrna_ids': mrna_ids,
            'splits': splits,
            'alignment_scores': np.array(alignment_scores)
        }
        
        # Cache processed data
        self._cache_data(data, cache_file)
        
        return data
    
    def _cache_data(self, data: Dict[str, Any], cache_file: str):
        """Cache processed data to HDF5 file."""
        with h5py.File(cache_file, 'w') as f:
            # Save numpy arrays
            f.create_dataset('alignment_matrices', data=data['alignment_matrices'])
            f.create_dataset('bert_input_ids', data=data['bert_input_ids'])
            f.create_dataset('bert_attention_masks', data=data['bert_attention_masks'])
            f.create_dataset('labels', data=data['labels'])
            f.create_dataset('alignment_scores', data=data['alignment_scores'])
            
            # Save string data
            sample_keys = [key.encode('utf-8') for key in data['sample_keys']]
            f.create_dataset('sample_keys', data=sample_keys)
            
            mirna_ids = [mid.encode('utf-8') for mid in data['mirna_ids']]
            f.create_dataset('mirna_ids', data=mirna_ids)
            
            mrna_ids = [mid.encode('utf-8') for mid in data['mrna_ids']]
            f.create_dataset('mrna_ids', data=mrna_ids)
            
            splits = [split.encode('utf-8') for split in data['splits']]
            f.create_dataset('splits', data=splits)
    
    def _load_cached_data(self, cache_file: str) -> Dict[str, Any]:
        """Load cached data from HDF5 file."""
        with h5py.File(cache_file, 'r') as f:
            data = {
                'alignment_matrices': f['alignment_matrices'][:],
                'bert_input_ids': f['bert_input_ids'][:],
                'bert_attention_masks': f['bert_attention_masks'][:],
                'labels': f['labels'][:],
                'alignment_scores': f['alignment_scores'][:],
                'sample_keys': [key.decode('utf-8') for key in f['sample_keys'][:]],
                'mirna_ids': [mid.decode('utf-8') for mid in f['mirna_ids'][:]],
                'mrna_ids': [mid.decode('utf-8') for mid in f['mrna_ids'][:]],
                'splits': [split.decode('utf-8') for split in f['splits'][:]],
            }
        
        return data
    
    def _filter_by_split(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        """Filter data by split."""
        split_indices = [i for i, s in enumerate(data['splits']) if s == split]
        
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                filtered_data[key] = value[split_indices]
            else:
                filtered_data[key] = [value[i] for i in split_indices]
        
        return filtered_data
    
    def _validate_data_consistency(self):
        """Validate data consistency with original file."""
        try:
            # Load original data for comparison
            original_df = pd.read_csv(self.data_path, sep='\t')
            
            # Check columns
            expected_columns = ['mirna_id', 'mirna_seq', 'mrna_id', 'mrna_seq', 'label', 'split']
            if original_df.columns.tolist() != expected_columns:
                raise ValueError(
                    f"Original data columns {original_df.columns.tolist()} "
                    f"do not match expected {expected_columns}"
                )
            
            # Check sample keys exist in original data
            original_keys = set(original_df['mirna_id'] + '|' + original_df['mrna_id'])
            processed_keys = set(self.data['sample_keys'])
            
            if not processed_keys.issubset(original_keys):
                raise ValueError("Processed data contains keys not in original data")
            
            print("Data consistency validation passed")
            
        except Exception as e:
            print(f"Data consistency validation failed: {e}")
            raise
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data['labels'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with align_matrix, bert_input_ids, bert_attention_mask, label
        """
        return {
            'align_matrix': torch.tensor(self.data['alignment_matrices'][idx], dtype=torch.float32),
            'bert_input_ids': torch.tensor(self.data['bert_input_ids'][idx], dtype=torch.long),
            'bert_attention_mask': torch.tensor(self.data['bert_attention_masks'][idx], dtype=torch.long),
            'label': torch.tensor(self.data['labels'][idx], dtype=torch.long),
            'sample_key': self.data['sample_keys'][idx],
            'mirna_id': self.data['mirna_ids'][idx],
            'mrna_id': self.data['mrna_ids'][idx],
            'alignment_score': torch.tensor(self.data['alignment_scores'][idx], dtype=torch.float32)
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get sample metadata."""
        return {
            'sample_key': self.data['sample_keys'][idx],
            'mirna_id': self.data['mirna_ids'][idx],
            'mrna_id': self.data['mrna_ids'][idx],
            'split': self.data['splits'][idx],
            'alignment_score': self.data['alignment_scores'][idx],
            'label': self.data['labels'][idx]
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        unique, counts = np.unique(self.data['labels'], return_counts=True)
        return dict(zip(unique, counts))
    
    def get_sample_keys(self) -> List[str]:
        """Get all sample keys."""
        return self.data['sample_keys']


def create_dataloaders(train_dataset: DPATDataset,
                      val_dataset: DPATDataset,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    def collate_fn(batch):
        """Custom collate function for padding BERT sequences."""
        # Extract fields
        align_matrices = torch.stack([item['align_matrix'] for item in batch])
        bert_input_ids = torch.stack([item['bert_input_ids'] for item in batch])
        bert_attention_masks = torch.stack([item['bert_attention_mask'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        # Extract metadata
        sample_keys = [item['sample_key'] for item in batch]
        mirna_ids = [item['mirna_id'] for item in batch]
        mrna_ids = [item['mrna_id'] for item in batch]
        alignment_scores = torch.stack([item['alignment_score'] for item in batch])
        
        return {
            'align_matrix': align_matrices,
            'bert_input_ids': bert_input_ids,
            'bert_attention_mask': bert_attention_masks,
            'label': labels,
            'sample_key': sample_keys,
            'mirna_id': mirna_ids,
            'mrna_id': mrna_ids,
            'alignment_score': alignment_scores
        }
    
    # Create dataloaders - DO NOT SHUFFLE to maintain order
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Explicitly no shuffling
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Explicitly no shuffling
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader 