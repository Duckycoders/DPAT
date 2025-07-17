# Google Colab 设置指南

## 在Google Colab中运行TargetNet DPAT

### 1. 克隆仓库

```python
# 克隆仓库
!git clone https://github.com/Duckycoders/DPAT.git
%cd DPAT
```

### 2. 安装依赖

```python
# 安装依赖包
!pip install -r requirements.txt
```

### 3. 检查GPU

```python
# 检查GPU可用性
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
```

### 4. 运行训练

```python
# 运行训练（限制步数用于测试）
!python train.py --config configs/dpat_run.yaml --max_steps 100
```

### 5. 完整训练

```python
# 完整训练（如果需要）
!python train.py --config configs/dpat_run.yaml
```

### 6. 监控训练

```python
# 如果需要监控训练进度，可以使用tensorboard
%load_ext tensorboard
%tensorboard --logdir runs/
```

### 7. 文件说明

- `train.py`: 主训练脚本
- `configs/dpat_run.yaml`: 训练配置文件
- `src/dpat/`: 核心DPAT实现
- `data/`: 数据文件（已包含在仓库中）

### 8. 重要提示

- 数据文件已包含在GitHub仓库中
- 缓存文件会自动生成在`cache/`目录下
- 训练过程会自动保存最佳模型
- 支持混合精度训练以提高速度

### 9. 修改配置

如果需要修改训练参数，可以编辑 `configs/dpat_run.yaml` 文件：

```python
# 查看当前配置
!cat configs/dpat_run.yaml

# 如果需要修改，可以创建新的配置文件
```

### 10. 故障排除

如果遇到问题：

1. **内存不足**: 减少batch_size
2. **训练太慢**: 确保使用GPU
3. **依赖问题**: 重新安装requirements.txt

```python
# 检查内存使用
!nvidia-smi

# 重新安装依赖
!pip install -r requirements.txt --force-reinstall
``` 