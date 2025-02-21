import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TextColumn,
)

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_data():
    # 读取CSV文件 里面放绝对路径
    input_data = pd.read_csv('/Users/chen/mapping/data/input.csv', header=None).values
    output_data = pd.read_csv('/Users/chen/mapping/data/output.csv', header=None).values

    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, output_data, test_size=0.2, random_state=42
    )

    # 数据标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # 保存xy坐标方便后续预测调用
    # 保存 x_scaler 到 x_scaler.pkl
    with open('x_scaler.pkl', 'wb') as f:
        pickle.dump(x_scaler, f)

    # 保存 y_scaler 到 y_scaler.pkl
    with open('y_scaler.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)

    print("Scaler 对象已成功保存！")

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test, y_scaler


# ---------------------------
# 2. 残差块定义
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.act = nn.GELU()

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc(x)
        out += identity
        return self.act(out)


# ---------------------------
# 3. 神经网络（引入残差块）
# ---------------------------
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始映射层，将25维输入映射到1024维
        self.initial = nn.Sequential(
            nn.Linear(25, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        # 后续残差块
        self.res_block1 = ResidualBlock(1024, 1024, dropout=0.2)
        self.res_block2 = ResidualBlock(1024, 2048, dropout=0.3)
        self.res_block3 = ResidualBlock(2048, 4096, dropout=0.4)
        # 最终映射层，将2048维映射到1600维输出
        self.fc_out = nn.Linear(4096, 1600)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.fc_out(x)
        return x


# ---------------------------
# 4. 训练函数（早停）
# ---------------------------
def train_model_with_early_stop(model, train_loader, test_loader, epochs=1000):
    # 设备选择：优先使用CUDA，其次MPS，最后CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    best_loss = float('inf')
    patience = 20
    counter = 0
    train_losses = []
    test_losses = []

    # rich进度条设置
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>6.2f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    # 进度条显示（rich库）
    with progress:
        # 外层任务：训练 epoch 进度
        epoch_task = progress.add_task("Training Epochs", total=epochs)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            # 内层任务：当前 epoch 中 batch 的训练进度
            batch_task = progress.add_task(
                f"Epoch {epoch + 1}/{epochs} Batches", total=len(train_loader)
            )
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # 更新内层进度条的描述和进度
                progress.update(
                    batch_task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/{epochs} (Loss: {loss.item():.6f})",
                )

            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)

            # 验证阶段
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item() * inputs.size(0)
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            scheduler.step()

            # 早停机制
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), 'best_model.pth')
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    progress.console.print(
                        f"[bold red]Early stopping triggered at epoch {epoch + 1}[/bold red]"
                    )
                    print("Early stopping!")
                    break

            progress.console.print(
                f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            progress.update(epoch_task, advance=1)
            progress.remove_task(batch_task)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, test_losses


# ---------------------------
# 5. 主程序
# ---------------------------
if __name__ == '__main__':
    # 加载数据
    X_train, X_test, y_train, y_test, y_scaler = load_data()

    # 创建 DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    model = ResNet()
    print("\nStarting training...")
    # 可以设定1000次，让早停自己停止
    train_losses, test_losses = train_model_with_early_stop(model, train_loader, test_loader, epochs=1000)

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Process with Early Stopping (Optimized Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()
