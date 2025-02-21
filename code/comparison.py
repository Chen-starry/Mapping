import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------
# 1. 加载数据
# ---------------------------
# 读取数据
predict_df = pd.read_csv('/Users/chen/mapping/data/predict.csv', header=None)
output_df = pd.read_csv('/Users/chen/mapping/data/output.csv', header=None)

# ---------------------------
# 2. 计算误差指标
# ---------------------------
# 将 DataFrame 转换为 NumPy 数组
y_pred = predict_df.values
y_true = output_df.values

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("Mean Squared Error (MSE): {:.6f}".format(mse))
print("Mean Absolute Error (MAE): {:.6f}".format(mae))
print("R2 Score: {:.6f}".format(r2))

# ---------------------------
# 3. 绘制预测值与真实值之间的差值直方图
# ---------------------------
# 计算差值
diff = y_pred - y_true
# 将所有差值拉平成一维数组
diff_flat = diff.flatten()

plt.figure(figsize=(10, 6))
plt.hist(diff_flat, bins=50, edgecolor='black', alpha=0.75)
plt.title("Distribution of Differences (Prediction - Ground Truth)")
plt.xlabel("Difference")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ---------------------------
# 4. 绘制散点图：预测值 vs 真实值
# ---------------------------
# 将两者均拉平成一维数组（也就是有样本数*1600个点）
y_pred_flat = y_pred.flatten()
y_true_flat = y_true.flatten()

plt.figure(figsize=(8, 8))
plt.scatter(y_true_flat, y_pred_flat, alpha=0.2, s=5)
# 绘制 y=x 参考线
min_val = min(y_true_flat.min(), y_pred_flat.min())
max_val = max(y_true_flat.max(), y_pred_flat.max())
plt.plot([min_val, max_val], [min_val, max_val], color='#90EE90', linestyle='--', label='y = x')
plt.title("Predictions vs Truth")
plt.xlabel("Truth Value")
plt.ylabel("Predictions")
plt.grid(True)
plt.show()


