# DPAT: Dual-Path Alignment & Transformer for miRNA Target Prediction

## 项目概述

DPAT是一个用于miRNA目标预测的深度学习模型，基于TargetNet/miTDS数据集实现。该模型采用双路径架构：

1. **对齐路径 (Alignment Path)**: 处理序列对齐矩阵，使用多尺度Inception卷积、SE和CBAM注意力机制
2. **语义路径 (Semantic Path)**: 处理RNA-BERT嵌入，使用两级一维卷积和BiLSTM
3. **交叉注意力融合**: 双向交叉注意力机制结合门控融合

## 依赖安装

### Python环境
```bash
conda create -n dpat python=3.8
conda activate dpat
```

### 核心依赖
```bash
# PyTorch (根据CUDA版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 必要库
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

# 可选依赖
pip install flake8  # 代码检查
pip install pytest  # 单元测试
pip install matplotlib  # 可视化
pip install seaborn  # 可视化
```

### 完整依赖列表
```bash
# 保存到requirements.txt
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

## 数据准备

确保数据目录结构如下：
```
data/
├── miRAW_Train_Validation.txt
├── miRAW_Test0.txt
├── miRAW_Test1.txt
├── ...
└── miRAW_Test9.txt
```

数据格式验证：
```python
import pandas as pd
df = pd.read_csv('data/miRAW_Train_Validation.txt', sep='\t')
print("数据形状:", df.shape)
print("列名:", df.columns.tolist())
print("前3行:")
print(df.head(3))
```

## 训练模型

### 基本训练
```bash
python train.py --config configs/dpat_run.yaml
```

### 快速测试（10步迭代）
```bash
python train.py --config configs/dpat_run.yaml --max_steps 10
```

### 调试模式
```bash
python train.py --config configs/dpat_run.yaml --debug --max_steps 10
```

### 恢复训练
```bash
python train.py --config configs/dpat_run.yaml --resume checkpoints/best_model_epoch_10.pt
```

## 模型评估

### 单模型评估
```python
from src.dpat import DPAT, DPATDataset

# 加载模型
model = DPAT.load_from_checkpoint('checkpoints/best_model.pt')

# 创建测试数据
test_dataset = DPATDataset(
    data_path='data/miRAW_Test0.txt',
    split='test'
)

# 评估
results = model.evaluate(test_dataset)
print(f"F1 Score: {results['f1']:.4f}")
print(f"AUROC: {results['auroc']:.4f}")
```

### 批量评估
```python
import torch
from torch.utils.data import DataLoader

# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测
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

# 计算指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print(f"Accuracy: {accuracy_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"Precision: {precision_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"Recall: {recall_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"F1 Score: {f1_score(labels, [p > 0.5 for p in predictions]):.4f}")
print(f"AUROC: {roc_auc_score(labels, predictions):.4f}")
```

## 模型架构

### 核心组件
- **AlignmentPath**: 处理10×50对齐矩阵
- **SemanticPath**: 处理RNA-BERT嵌入
- **BiLSTMBlock**: 双向LSTM with 全局池化
- **CrossAttentionFusion**: 交叉注意力融合
- **SEBlock/CBAMBlock**: 注意力机制

### 模型配置
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

## 性能优化

### 内存优化
- 使用AMP (Automatic Mixed Precision)
- 梯度累积: `accumulate_grad_batches: 4`
- 缓存预处理数据: `cache_dir: "cache"`

### 训练优化
- 不同学习率: BERT (1e-5) vs 其他层 (1e-3)
- 梯度裁剪: `max_grad_norm: 1.0`
- 早停: `early_stopping_patience: 10`

## 代码质量

### 运行代码检查
```bash
flake8 src/dpat/
```

### 运行单元测试
```bash
pytest src/dpat/tests/
```

### 模型调试
```python
from torchinfo import summary
model = DPAT()
summary(model, input_size=[(32, 10, 50), (32, 128), (32, 128)])
```

## 结果复现

### 完整训练流程
1. 安装依赖: `pip install -r requirements.txt`
2. 数据验证: 检查数据格式和完整性
3. 开始训练: `python train.py --config configs/dpat_run.yaml`
4. 监控训练: 查看 `runs/dpat_*.log`
5. 评估结果: 使用最佳checkpoint

### 预期结果
- 训练时间: ~2-4小时 (A100 GPU)
- 内存使用: <20GB
- 最佳验证F1: >0.85 (取决于数据集)

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size或使用gradient_checkpointing
2. **RNA-BERT加载失败**: 确保网络连接或使用简化版本
3. **数据格式错误**: 检查列名和数据类型

### 调试技巧
- 使用 `--debug` 模式启用详细日志
- 使用 `--max_steps 10` 快速验证
- 检查 `cache/` 目录中的预处理数据

## 扩展功能

### 自定义数据
```python
# 自定义数据集
class CustomDataset(DPATDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        # 自定义处理逻辑
```

### 模型变体
```python
# 使用简化语义路径
model = DPAT(use_simple_semantic=True)

# 集成学习
ensemble = DPATEnsemble(
    model_configs=[config1, config2, config3],
    ensemble_method='weighted'
)
```

## 引用

如果使用本代码，请引用：
```bibtex
@article{dpat2024,
  title={DPAT: Dual-Path Alignment \& Transformer for miRNA Target Prediction},
  author={Your Name},
  year={2024}
}
```

## 联系方式

如有问题，请联系：your.email@example.com

---

**注意**: 本项目基于TargetNet官方实现，保持了原始数据格式和处理逻辑的完整性。 