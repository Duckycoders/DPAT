# Google Colab 专用安装指南

## 重要：必须按照此顺序安装！

### 1. 首先安装升级版transformers和multimolecule

```python
# 升级transformers到最新版本
!pip install --upgrade git+https://github.com/huggingface/transformers.git

# 安装multimolecule（注册自定义类）
!pip install multimolecule

# 安装其他依赖
!pip install -r requirements.txt
```

### 2. 重启运行时

安装完成后，**必须重启运行时**：
- 点击 `Runtime` -> `Restart runtime`
- 或在代码中运行：

```python
import os
os.kill(os.getpid(), 9)
```

### 3. 验证安装

重启后，在新的cell中验证：

```python
# 必须按照此顺序导入！
import multimolecule  # 必不可少，否则类注册不到
from transformers import AutoTokenizer, AutoModel

# 测试加载
model_id = "multimolecule/rnafm"
print(f"Testing model: {model_id}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    print("✅ RNA-BERT模型加载成功！")
    print(f"Model config: {model.config}")
except Exception as e:
    print(f"❌ 加载失败: {e}")
```

### 4. 运行训练

```python
# 运行训练
!python train.py --config configs/dpat_run.yaml --max_steps 100
```

### 5. 预期日志

成功的话，应该看到类似的日志：

```
loading configuration file config.json from https://huggingface.co/multimolecule/rnafm
Model config: RnaFmConfig {
  ...
}
```

**不应该再看到**：
- "Using backup tokenizer..."
- "Using simple embedding layer instead..."

## 故障排除

如果仍然失败：

1. **清除缓存**：
```python
!rm -rf ~/.cache/huggingface/
```

2. **重新安装**：
```python
!pip uninstall transformers multimolecule -y
!pip install --upgrade git+https://github.com/huggingface/transformers.git
!pip install multimolecule
```

3. **检查版本**：
```python
import transformers
import multimolecule
print(f"transformers version: {transformers.__version__}")
print(f"multimolecule version: {multimolecule.__version__}")
``` 