import torch
import torch.nn as nn
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. 定义模型（必须与训练时保持一致）
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
        self.res_block1 = ResidualBlock(1024, 1024, dropout=0.3)
        self.res_block2 = ResidualBlock(1024, 2048, dropout=0.4)
        self.res_block3 = ResidualBlock(2048, 4096, dropout=0.5)
        # 最终映射层，将4096维映射到1600维输出
        self.fc_out = nn.Linear(4096, 1600)

        # 权重初始化（保持与训练时一致）
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
# 2. 加载保存的 scaler 和模型参数
# ---------------------------
# 加载 x_scaler 和 y_scaler（放绝对路径）
with open('/Users/chen/mapping/code/x_scaler.pkl', 'rb') as f:
    x_scaler: StandardScaler = pickle.load(f)
with open('/Users/chen/mapping/code/y_scaler.pkl', 'rb') as f:
    y_scaler: StandardScaler = pickle.load(f)

# 创建模型实例，并加载保存的最佳模型参数
model = ResNet()
# 如果训练时在GPU上保存参数，加载时建议指定 map_location为cpu
model.load_state_dict(torch.load('/Users/chen/mapping/code/best_model.pth', map_location=torch.device('cpu')))
model.eval()  # 切换到评估模式

# ---------------------------
# 3. 加载输入数据并进行预测
# ---------------------------
# 读取 input.csv
input_data = pd.read_csv('/Users/chen/mapping/data/input.csv', header=None).values  # 形状: (样本数, 25)

# 对输入数据进行标准化（使用训练时拟合的 x_scaler）
input_scaled = x_scaler.transform(input_data)

# 转换为 PyTorch 张量
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():
    output_scaled = model(input_tensor).cpu().numpy()  # 形状: (样本数, 1600)

# 反标准化预测结果（恢复到原始数值尺度）
output_pred = y_scaler.inverse_transform(output_scaled)

# ---------------------------
# 4. 保存预测结果到 predict.csv
# ---------------------------
pd.DataFrame(output_pred).to_csv('/Users/chen/mapping/data/predict.csv', index=False, header=False)

print("预测结果已保存到 predict.csv。")
