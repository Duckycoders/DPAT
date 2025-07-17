"""Data preprocessing for DPAT including sliding window and sequence alignment."""

import numpy as np
from typing import List, Tuple, Dict, Any
import torch
from Bio import Align
from Bio.Seq import Seq
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SequenceAligner:
    """Implements Needleman-Wunsch global alignment for miRNA-mRNA pairs."""
    
    def __init__(self, seed_length: int = 12, alignment_threshold: int = 6):
        """
        Initialize sequence aligner.
        
        Args:
            seed_length: Length of miRNA seed region to align (default: 12)
            alignment_threshold: Minimum score to keep alignment (default: 6)
        """
        self.seed_length = seed_length
        self.alignment_threshold = alignment_threshold
        self.aligner = Align.PairwiseAligner()
        
        # Set scoring parameters for Watson-Crick and G:U wobble pairs
        self.aligner.match_score = 1.0
        self.aligner.mismatch_score = 0.0
        self.aligner.open_gap_score = -1.0
        self.aligner.extend_gap_score = -0.5
        
        # Define complementary base pairs
        self.complement_pairs = {
            ('A', 'U'): 1, ('U', 'A'): 1,
            ('G', 'C'): 1, ('C', 'G'): 1,
            ('G', 'U'): 1, ('U', 'G'): 1,  # G:U wobble pair
        }
    
    def get_reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of RNA sequence."""
        complement_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(complement_map.get(base, base) for base in sequence[::-1])
    
    def score_alignment(self, seq1: str, seq2: str) -> float:
        """Score alignment between two sequences."""
        score = 0
        min_len = min(len(seq1), len(seq2))
        
        for i in range(min_len):
            if (seq1[i], seq2[i]) in self.complement_pairs:
                score += self.complement_pairs[(seq1[i], seq2[i])]
        
        return score
    
    def align_sequences(self, mirna_seed: str, mrna_window: str) -> Tuple[float, str, str]:
        """
        Perform global alignment between miRNA seed and mRNA window.
        
        Args:
            mirna_seed: First 12 nucleotides of miRNA
            mrna_window: 40nt window from mRNA
            
        Returns:
            Tuple of (alignment_score, aligned_mirna, aligned_mrna)
        """
        # Get reverse complement of mRNA window for alignment
        mrna_rc = self.get_reverse_complement(mrna_window)
        
        # Perform alignment
        alignments = self.aligner.align(mirna_seed, mrna_rc)
        
        if len(alignments) == 0:
            return 0.0, mirna_seed, mrna_rc
        
        # Get best alignment
        best_alignment = alignments[0]
        aligned_mirna = str(best_alignment[0])
        aligned_mrna = str(best_alignment[1])
        
        # Calculate custom score based on Watson-Crick and G:U pairs
        score = self.score_alignment(
            aligned_mirna.replace('-', ''),
            aligned_mrna.replace('-', '')
        )
        
        return score, aligned_mirna, aligned_mrna


def create_alignment_matrix(aligned_mirna: str, aligned_mrna: str, 
                          max_length: int = 50) -> np.ndarray:
    """
    Create 10 x max_length one-hot alignment matrix.
    
    Args:
        aligned_mirna: Aligned miRNA sequence
        aligned_mrna: Aligned mRNA sequence  
        max_length: Maximum length for padding/truncation
        
    Returns:
        One-hot matrix of shape (10, max_length)
    """
    # Nucleotide encoding: A=0, U=1, G=2, C=3, gap=4
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '-': 4}
    
    # Initialize matrix
    matrix = np.zeros((10, max_length), dtype=np.float32)
    
    # Align sequences to same length
    max_seq_len = max(len(aligned_mirna), len(aligned_mrna))
    aligned_mirna = aligned_mirna.ljust(max_seq_len, '-')
    aligned_mrna = aligned_mrna.ljust(max_seq_len, '-')
    
    # Truncate or pad to max_length
    seq_len = min(max_seq_len, max_length)
    
    for i in range(seq_len):
        # miRNA encoding (first 5 channels)
        mirna_nt = aligned_mirna[i]
        if mirna_nt in nucleotide_map:
            matrix[nucleotide_map[mirna_nt], i] = 1.0
        
        # mRNA encoding (last 5 channels)  
        mrna_nt = aligned_mrna[i]
        if mrna_nt in nucleotide_map:
            matrix[5 + nucleotide_map[mrna_nt], i] = 1.0
    
    return matrix


def sliding_window_alignment(mirna_seq: str, mrna_seq: str, 
                           window_size: int = 40,
                           step_size: int = 1,
                           seed_length: int = 12,
                           alignment_threshold: int = 6,
                           max_length: int = 50) -> List[Dict[str, Any]]:
    """
    Perform sliding window alignment analysis.
    
    Args:
        mirna_seq: Full miRNA sequence
        mrna_seq: Full mRNA sequence
        window_size: Size of sliding window
        step_size: Step size for sliding window
        seed_length: Length of miRNA seed region
        alignment_threshold: Minimum alignment score
        max_length: Maximum length for alignment matrix
        
    Returns:
        List of candidate target sites with alignment matrices
    """
    aligner = SequenceAligner(seed_length, alignment_threshold)
    mirna_seed = mirna_seq[:seed_length]
    candidate_sites = []
    
    # Slide window across mRNA sequence
    for i in range(0, len(mrna_seq) - window_size + 1, step_size):
        mrna_window = mrna_seq[i:i + window_size]
        
        # Perform alignment
        score, aligned_mirna, aligned_mrna = aligner.align_sequences(
            mirna_seed, mrna_window
        )
        
        # Keep sites with score >= threshold
        if score >= alignment_threshold:
            alignment_matrix = create_alignment_matrix(
                aligned_mirna, aligned_mrna, max_length
            )
            
            candidate_sites.append({
                'window_start': i,
                'window_end': i + window_size,
                'mrna_window': mrna_window,
                'alignment_score': score,
                'aligned_mirna': aligned_mirna,
                'aligned_mrna': aligned_mrna,
                'alignment_matrix': alignment_matrix
            })
    
    return candidate_sites


def process_mirna_mrna_pair(mirna_id: str, mirna_seq: str, 
                          mrna_id: str, mrna_seq: str,
                          label: int, split: str,
                          window_size: int = 40,
                          seed_length: int = 12,
                          alignment_threshold: int = 6,
                          max_length: int = 50) -> List[Dict[str, Any]]:
    """
    Process a single miRNA-mRNA pair to generate candidate target sites.
    
    Args:
        mirna_id: miRNA identifier
        mirna_seq: miRNA sequence
        mrna_id: mRNA identifier  
        mrna_seq: mRNA sequence
        label: Binary label (0 or 1)
        split: Data split ('train' or 'val')
        window_size: Sliding window size
        seed_length: miRNA seed length
        alignment_threshold: Minimum alignment score
        max_length: Maximum alignment matrix length
        
    Returns:
        List of processed candidate sites
    """
    candidate_sites = sliding_window_alignment(
        mirna_seq, mrna_seq, window_size, 1, 
        seed_length, alignment_threshold, max_length
    )
    
    # If no candidates found, create one with zero matrix
    if not candidate_sites:
        candidate_sites = [{
            'window_start': 0,
            'window_end': min(window_size, len(mrna_seq)),
            'mrna_window': mrna_seq[:window_size].ljust(window_size, 'N'),
            'alignment_score': 0.0,
            'aligned_mirna': mirna_seq[:seed_length],
            'aligned_mrna': 'N' * seed_length,
            'alignment_matrix': np.zeros((10, max_length), dtype=np.float32)
        }]
    
    # Add metadata to each candidate
    for site in candidate_sites:
        site.update({
            'mirna_id': mirna_id,
            'mirna_seq': mirna_seq,
            'mrna_id': mrna_id,
            'mrna_seq': mrna_seq,
            'label': label,
            'split': split,
            'sample_key': f"{mirna_id}|{mrna_id}"
        })
    
    return candidate_sites


def preprocess_dataset(data_path: str, 
                      window_size: int = 40,
                      seed_length: int = 12,
                      alignment_threshold: int = 6,
                      max_length: int = 50) -> List[Dict[str, Any]]:
    """
    Preprocess entire dataset with sliding window alignment.
    
    Args:
        data_path: Path to input data file
        window_size: Sliding window size
        seed_length: miRNA seed length
        alignment_threshold: Minimum alignment score
        max_length: Maximum alignment matrix length
        
    Returns:
        List of all candidate sites across the dataset
    """
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    
    # Verify columns match expected format
    expected_columns = ['mirna_id', 'mirna_seq', 'mrna_id', 'mrna_seq', 'label', 'split']
    if df.columns.tolist() != expected_columns:
        raise ValueError(
            f"Data columns {df.columns.tolist()} do not match expected {expected_columns}. "
            f"Please check data format."
        )
    
    print(f"Processing {len(df)} samples...")
    all_candidates = []
    
    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        candidates = process_mirna_mrna_pair(
            row['mirna_id'], row['mirna_seq'],
            row['mrna_id'], row['mrna_seq'],
            row['label'], row['split'],
            window_size, seed_length, alignment_threshold, max_length
        )
        all_candidates.extend(candidates)
    
    print(f"Generated {len(all_candidates)} candidate sites")
    return all_candidates 