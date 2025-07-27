# DPAT: Dual-Path Alignment & Transformer for miRNA Target Prediction

## Project Overview

DPAT is a deep learning model for miRNA target prediction implemented with dual-path architecture:

1. **Alignment Path**: Processes sequence alignment matrices using multi-scale Inception convolutions, SE and CBAM attention mechanisms
2. **Semantic Path**: Processes RNA-BERT embeddings using two-level 1D convolutions and BiLSTM
3. **Cross-Attention Fusion**: Bidirectional cross-attention mechanism with gated fusion

## Installation

### Python Environment
```bash
conda create -n dpat python=3.8
conda activate dpat
```

### Core Dependencies
```bash
# PyTorch (adjust according to CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Essential libraries
pip install transformers
pip install datasets
pip install tokenizers
pip install biopython
pip install scikit-learn
pip install pandas
pip install numpy
pip install tqdm
pip install h5py
pip install pyyaml
pip install torchmetrics
pip install torch-scatter

# Optional dependencies
pip install flake8  # Code linting
pip install pytest  # Unit testing
pip install matplotlib  # Visualization
pip install seaborn  # Visualization
```

### Complete Dependencies List
```bash
# Save to requirements.txt
torch>=1.12.0
transformers>=4.20.0
datasets>=2.0.0
tokenizers>=0.12.0
biopython>=1.79
scikit-learn>=1.1.0
pandas>=1.4.0
numpy>=1.21.0
tqdm>=4.64.0
h5py>=3.7.0
pyyaml>=6.0
torchmetrics>=0.9.0
torch-scatter>=2.0.9
flake8>=4.0.0
pytest>=7.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Data Preparation

Ensure data directory structure as follows:
```
data/
├── miRAW_Train_Validation.txt
├── miRAW_Test0.txt
├── miRAW_Test1.txt
├── ...
└── miRAW_Test9.txt
```

Data format validation:
```python
import pandas as pd
df = pd.read_csv('data/miRAW_Train_Validation.txt', sep='\t')
print("Data shape:", df.shape)
print("Column names:", df.columns.tolist())
print("First 3 rows:")
print(df.head(3))
```

## Model Training

### Basic Training
```bash
python train.py --config configs/dpat_run.yaml
```

### Quick Test (10 iterations)
```bash
python train.py --config configs/dpat_run.yaml --max_steps 10
```

### Debug Mode
```bash
python train.py --config configs/dpat_run.yaml --debug --max_steps 10
```

### Resume Training
```bash
python train.py --config configs/dpat_run.yaml --resume checkpoints/best_model_epoch_10.pt
```

## Model Evaluation

### Single Model Evaluation
```python
from src.dpat import DPAT, DPATDataset

# Load model
model = DPAT.load_from_checkpoint('checkpoints/best_model.pt')

# Create test dataset
test_dataset = DPATDataset(
    data_path='data/miRAW_Test0.txt',
    split='test'
)

# Evaluate
results = model.evaluate(test_dataset)
print(f"F1 Score: {results['f1']:.4f}")
print(f"AUROC: {results['auroc']:.4f}")
```

### Batch Evaluation
```python
import torch
from torch.utils.data import DataLoader

# Create data loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Predictions
predictions = []
labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            batch['align_matrix'],
            batch['bert_input_ids'],
            batch['bert_attention_mask']
        )
        probs = torch.sigmoid(logits)
        predictions.extend(probs.cpu().numpy())
        labels.extend(batch['label'].numpy())

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print(f"Accuracy: {accuracy_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"Precision: {precision_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"Recall: {recall_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"F1 Score: {f1_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"AUROC: {roc_auc_score(labels, predictions):.4f}")
```

## Model Architecture

### Core Components
- **AlignmentPath**: Processes 10×50 alignment matrices
- **SemanticPath**: Processes RNA-BERT embeddings
- **BiLSTMBlock**: Bidirectional LSTM with global pooling
- **CrossAttentionFusion**: Cross-attention fusion
- **SEBlock/CBAMBlock**: Attention mechanisms

### Model Configuration
```yaml
model:
  alignment_config:
    input_channels: 10
    output_channels: 256
    hidden_channels: 128
    kernel_sizes: [1, 3, 5, 7]
    use_se: true
    use_cbam: true
  
  semantic_config:
    bert_model_name: "multimolecule/rnafm"
    lstm_hidden_size: 128
    proj_dim: 256
    freeze_bert: false
  
  fusion_config:
    embed_dim: 256
    num_heads: 8
    use_gate: true
```

## Performance Optimization

### Memory Optimization
- Use AMP (Automatic Mixed Precision)
- Gradient accumulation: `accumulate_grad_batches: 4`
- Cache preprocessed data: `cache_dir: "cache"`

### Training Optimization
- Different learning rates: BERT (1e-5) vs other layers (1e-3)
- Gradient clipping: `max_grad_norm: 1.0`
- Early stopping: `early_stopping_patience: 10`

## Code Quality

### Run Code Linting
```bash
flake8 src/dpat/
```

### Run Unit Tests
```bash
pytest src/dpat/tests/
```

### Model Debugging
```python
from torchinfo import summary
model = DPAT()
summary(model, input_size=[(32, 10, 50), (32, 128), (32, 128)])
```

## Reproducing Results

### Complete Training Pipeline
1. Install dependencies: `pip install -r requirements.txt`
2. Data validation: Check data format and completeness
3. Start training: `python train.py --config configs/dpat_run.yaml`
4. Monitor training: Check `runs/dpat_*.log`
5. Evaluate results: Use best checkpoint

### Expected Results
- Training time: ~2-4 hours (A100 GPU)
- Memory usage: <20GB
- Best validation F1: >0.85 (depends on dataset)

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch_size or use gradient_checkpointing
2. **RNA-BERT loading failure**: Ensure network connection or use simplified version
3. **Data format error**: Check column names and data types

### Debugging Tips
- Use `--debug` mode for detailed logging
- Use `--max_steps 10` for quick validation
- Check preprocessed data in `cache/` directory

## Extensions

### Custom Data
```python
# Custom dataset
class CustomDataset(DPATDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        # Custom processing logic
```

### Model Variants
```python
# Use simplified semantic path
model = DPAT(use_simple_semantic=True)

# Ensemble learning
ensemble = DPATEnsemble(
    model_configs=[config1, config2, config3],
    ensemble_method='weighted'
)
```

## Citation

If you use this code, please cite:
```bibtex
@article{dpat2024,
  title={DPAT: Dual-Path Alignment \& Transformer for miRNA Target Prediction},
  author={shanruoxu},
  year={2024}
}
```

## Contact

For questions, please contact: xushanruo@gmail.com 