
---

# 超单元刚度矩阵预测模型V1.0

本项目使用 PyTorch 构建了一个深度学习模型，用于预测超单元的刚度矩阵分布。输入为 25 维的超单元密度分布，输出为 1600 维的刚度矩阵分布。通过残差神经网络（ResNet）结构，实现了从低维输入到高维输出的精确映射。

---

## 项目特点

- **模型架构**：基于残差网络（ResNet）的深度神经网络，支持自适应维度匹配
- **训练优化**：
  - 动态学习率调整（OneCycleLR）
  - 早停机制（Early Stopping）
  - 梯度裁剪（Gradient Clipping）
  - 混合精度训练（Mixed Precision）
- **可视化**：
  - 使用 Rich 库实现训练进度条
  - 自动保存训练损失曲线
- **兼容性**：支持 CUDA、MPS 和 CPU 后端

---

## 环境要求

### 硬件要求
- GPU：推荐 NVIDIA GPU（支持 CUDA）
- 内存：至少 16GB
- 存储：至少 1GB 可用空间

### 软件依赖
- Python 3.8+
- PyTorch 2.0+
- 其他依赖库：
  ```bash
  pip install numpy pandas scikit-learn matplotlib rich
  ```

---

## 快速开始

### 1. 准备数据
- 将 `input.csv` 和 `output.csv` 放置在 `data/` 目录下
- 数据格式：
  - `input.csv`：10000 行 × 25 列，每行代表一种超单元的密度分布
  - `output.csv`：10000 行 × 1600 列，每行代表一种超单元的刚度矩阵分布
  - 行数表示样本量

### 3. 训练模型
运行以下命令开始训练：
```bash
  python ResNet.py -use_gpu
```

### 4. 查看结果
- 训练日志：实时显示在终端
- 损失曲线：保存为 `loss_curve.png`
- 最佳模型：保存为 `best_model.pth`

### 5. 结果对比
运行以下命令进行对比操作：
```bash
  python predict.py

  python comparison.py
```

---

## 项目结构

```
mapping/
├── data/                   # 数据目录
│   ├── input.csv           # 输入数据集
│   └── output.csv          # 输出数据集
├── code/                   # 模型定义
│   ├── ResNet.py           # 残差网络与主训练脚本
│   ├── predict.py          # 预测函数
│   └── comparison.py       # 比对函数
└── README.md               # 项目文档
```

---

## 参数配置

在 `ResNet.py` 中可调整以下参数（可以利用k折交叉验证+随机搜索寻求最佳参数）：

| 参数名称            | 默认值  | 描述           |
|-----------------|------|--------------|
| `batch_size`    | 64   | 训练时的批次大小     |
| `epochs`        | 1000 | 最大训练轮次       |
| `learning_rate` | 1e-3 | 初始学习率        |
| `patience`      | 30   | 早停机制的等待轮次    |
| `dropout_rate`  | 0.3  | Dropout 概率   |
| `weight_decay`  | 1e-4 | 权重衰减（L2 正则化） |

## 常见问题

### 1. 如何修改模型结构？
- 编辑 `code/ResNet.py` 中的 `ResNet` 类
- 添加或修改残差块（ResidualBlock）

### 2. 如何调整训练参数？
- 修改 `ResNet.py` 中的超参数（如 `batch_size`、`learning_rate` 等）

### 3. 如何在不同设备上运行？
- 代码自动检测可用设备（CUDA > MPS > CPU）
- 如需强制使用 CPU，可在代码中添加：
```python
  device = torch.device('cpu')
```
