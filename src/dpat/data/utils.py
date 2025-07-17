"""Data utilities for DPAT including RNA-BERT tokenizer."""

from typing import Dict, List, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import h5py
import os


def load_rna_bert_tokenizer(model_name: str = "multimolecule/rnabert"):
    """
    Load RNA-BERT tokenizer.
    
    Args:
        model_name: Name of the RNA-BERT model
        
    Returns:
        Tokenizer instance
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer {model_name}: {e}")
        print("Using backup tokenizer...")
        # Fallback to a more basic tokenizer if RNA-BERT is not available
        return create_simple_rna_tokenizer()


def create_simple_rna_tokenizer():
    """Create a simple RNA tokenizer if RNA-BERT is not available."""
    
    class SimpleRNATokenizer:
        """Simple RNA tokenizer for 6-mer tokens."""
        
        def __init__(self):
            self.vocab = self._build_vocab()
            self.vocab_size = len(self.vocab)
            
        def _build_vocab(self):
            """Build vocabulary for 6-mer tokens."""
            bases = ['A', 'U', 'G', 'C']
            vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
            
            # Add single nucleotides
            for base in bases:
                vocab[base] = len(vocab)
            
            # Add 6-mers
            for i in range(4**6):
                kmer = ''
                temp = i
                for _ in range(6):
                    kmer = bases[temp % 4] + kmer
                    temp //= 4
                vocab[kmer] = len(vocab)
            
            return vocab
        
        def tokenize(self, sequence: str) -> List[str]:
            """Tokenize sequence into 6-mers."""
            tokens = []
            for i in range(0, len(sequence), 6):
                kmer = sequence[i:i+6]
                if len(kmer) == 6:
                    tokens.append(kmer)
                else:
                    tokens.append(kmer.ljust(6, 'N'))
            return tokens
        
        def encode(self, sequence: str, add_special_tokens: bool = True,
                  max_length: int = 512, padding: str = 'max_length',
                  truncation: bool = True, return_tensors: str = 'pt'):
            """Encode sequence to token IDs."""
            tokens = self.tokenize(sequence)
            
            if add_special_tokens:
                tokens = ['[CLS]'] + tokens + ['[SEP]']
            
            # Convert to IDs
            input_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
            
            # Padding and truncation
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            
            if padding == 'max_length':
                attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
                input_ids = input_ids + [self.vocab['[PAD]']] * (max_length - len(input_ids))
            else:
                attention_mask = [1] * len(input_ids)
            
            if return_tensors == 'pt':
                return {
                    'input_ids': torch.tensor([input_ids], dtype=torch.long),
                    'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
                }
            else:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
    
    return SimpleRNATokenizer()


def prepare_bert_input(mirna_seq: str, mrna_window: str, tokenizer,
                      max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Prepare BERT input from miRNA and mRNA sequences.
    
    Args:
        mirna_seq: miRNA sequence
        mrna_window: mRNA window sequence
        tokenizer: RNA-BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    # Get reverse complement of mRNA window
    complement_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    mrna_rc = ''.join(complement_map.get(base, base) for base in mrna_window[::-1])
    
    # Combine sequences with [SEP] token
    combined_seq = mirna_seq + '[SEP]' + mrna_rc
    
    # Tokenize
    encoded = tokenizer.encode(
        combined_seq,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'].squeeze(0),
        'attention_mask': encoded['attention_mask'].squeeze(0)
    }


def create_kfold_splits(data_path: str, n_splits: int = 10, 
                       random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified k-fold splits for cross-validation.
    
    Args:
        data_path: Path to the dataset
        n_splits: Number of folds
        random_state: Random state for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Load data
    df = pd.read_csv(data_path, sep='\t')
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Use both label and mirna_id for stratification to ensure balanced splits
    # across both target labels and miRNA types
    stratify_labels = df['label'].astype(str) + '_' + df['mirna_id'].astype(str)
    
    splits = []
    for train_idx, val_idx in skf.split(df, stratify_labels):
        splits.append((train_idx, val_idx))
    
    return splits


def save_processed_data(data: List[Dict[str, Any]], output_path: str):
    """
    Save processed data to HDF5 format.
    
    Args:
        data: List of processed samples
        output_path: Path to save the data
    """
    with h5py.File(output_path, 'w') as f:
        # Save alignment matrices
        alignment_matrices = np.array([item['alignment_matrix'] for item in data])
        f.create_dataset('alignment_matrices', data=alignment_matrices)
        
        # Save labels
        labels = np.array([item['label'] for item in data])
        f.create_dataset('labels', data=labels)
        
        # Save sample keys for grouping
        sample_keys = [item['sample_key'].encode('utf-8') for item in data]
        f.create_dataset('sample_keys', data=sample_keys)
        
        # Save other metadata
        mirna_ids = [item['mirna_id'].encode('utf-8') for item in data]
        f.create_dataset('mirna_ids', data=mirna_ids)
        
        mrna_ids = [item['mrna_id'].encode('utf-8') for item in data]
        f.create_dataset('mrna_ids', data=mrna_ids)
        
        splits = [item['split'].encode('utf-8') for item in data]
        f.create_dataset('splits', data=splits)
        
        alignment_scores = np.array([item['alignment_score'] for item in data])
        f.create_dataset('alignment_scores', data=alignment_scores)


def load_processed_data(data_path: str) -> Dict[str, Any]:
    """
    Load processed data from HDF5 format.
    
    Args:
        data_path: Path to the HDF5 file
        
    Returns:
        Dictionary with loaded data
    """
    with h5py.File(data_path, 'r') as f:
        data = {
            'alignment_matrices': f['alignment_matrices'][:],
            'labels': f['labels'][:],
            'sample_keys': [key.decode('utf-8') for key in f['sample_keys'][:]],
            'mirna_ids': [mid.decode('utf-8') for mid in f['mirna_ids'][:]],
            'mrna_ids': [mid.decode('utf-8') for mid in f['mrna_ids'][:]],
            'splits': [split.decode('utf-8') for split in f['splits'][:]],
            'alignment_scores': f['alignment_scores'][:]
        }
    
    return data


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        labels: Array of binary labels
        
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced', classes=classes, y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float32)


def validate_data_consistency(original_data_path: str, 
                            processed_data_path: str) -> bool:
    """
    Validate that processed data maintains consistency with original data.
    
    Args:
        original_data_path: Path to original data
        processed_data_path: Path to processed data
        
    Returns:
        True if data is consistent, False otherwise
    """
    # Load original data
    original_df = pd.read_csv(original_data_path, sep='\t')
    
    # Load processed data
    processed_data = load_processed_data(processed_data_path)
    
    # Check if number of unique sample keys matches original rows
    unique_keys = set(processed_data['sample_keys'])
    original_keys = set(original_df['mirna_id'] + '|' + original_df['mrna_id'])
    
    if unique_keys != original_keys:
        print(f"Mismatch in sample keys: {len(unique_keys)} vs {len(original_keys)}")
        return False
    
    # Check label distribution
    original_labels = original_df['label'].value_counts()
    processed_labels = pd.Series(processed_data['labels']).value_counts()
    
    print(f"Original label distribution: {original_labels}")
    print(f"Processed label distribution: {processed_labels}")
    
    return True 