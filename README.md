# TargetNet - DPAT Implementation

**Dual-Path Alignment & Transformer (DPAT)** for miRNA target prediction

## 项目概述

本项目实现了一个新颖的双路径架构，用于miRNA靶点预测：

1. **Alignment Path**: 使用多尺度卷积处理序列比对矩阵
2. **Semantic Path**: 利用RNA-BERT嵌入和BiLSTM进行语义理解
3. **Cross-attention融合**: 实现最优特征整合

## 项目结构

```
TargetNet/
├── src/dpat/                    # DPAT核心代码
│   ├── models/                  # 模型定义
│   │   ├── dpat.py             # 主模型
│   │   └── modules/            # 模型组件
│   ├── data/                   # 数据处理
│   │   ├── dataset.py          # 数据集类
│   │   ├── preprocessing.py    # 数据预处理
│   │   └── utils.py            # 工具函数
│   ├── training/               # 训练相关
│   │   ├── trainer.py          # 训练器
│   │   └── utils.py            # 训练工具
│   └── utils/                  # 通用工具
├── configs/                    # 配置文件
├── data/                       # 数据文件
├── train.py                    # 训练脚本
└── requirements.txt            # 依赖包

```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py --config configs/dpat_run.yaml --max_steps 10
```

### 3. Google Colab使用

可以直接在Google Colab中运行此项目：

1. 克隆仓库
2. 安装依赖
3. 运行训练脚本

## 核心特性

- **双路径架构**: 同时处理序列比对和语义信息
- **多尺度处理**: Inception模块处理不同尺度的特征
- **注意力机制**: SE和CBAM注意力增强特征表示
- **跨注意力融合**: 双向多头注意力融合两个路径的特征
- **混合精度训练**: 支持AMP加速训练
- **灵活配置**: 完整的配置文件系统

## 模型架构

### Alignment Path
- 1×1 卷积初始化
- 多尺度Inception模块 (核大小: 1,3,5,7)
- SE注意力机制
- CBAM空间注意力

### Semantic Path
- RNA-BERT编码 (带[SEP]标记连接)
- 双层1D卷积 (核大小: 3,5)
- BiLSTM块处理
- 全局池化 (max + mean)

### Cross-Attention Fusion
- 双向多头交叉注意力
- 门控机制
- 残差连接

## 技术规格

- **数据格式**: 保持原始CSV/HDF5行顺序和列名
- **滑动窗口**: 40nt窗口，1nt步长
- **全局比对**: Needleman-Wunsch算法
- **比对评分**: Watson-Crick和G:U wobble配对为1，其他为0
- **阈值**: 比对分数≥6
- **矩阵大小**: 10×50 one-hot编码

## 训练配置

- **不同学习率**: BERT层1e-5，其他层1e-3
- **优化器**: AdamW with 线性预热
- **混合精度**: AMP + GradScaler
- **早停**: 监控val_f1
- **梯度裁剪**: 防止梯度爆炸

## 许可证

MIT License

## 贡献

欢迎提交问题和拉取请求！ 