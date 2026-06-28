import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# =========================
# 1. 基本配置
# =========================

TRAIN_PATH = "数据集/train_data.csv"
TEST_PATH = "数据集/test_data.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("当前使用设备:", device)


# =========================
# 2. 读取数据
# =========================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("训练集前5行:")
print(train_df.head())

print("\n测试集前5行:")
print(test_df.head())


# =========================
# 3. 选择特征和目标变量
# =========================

target_col = "MedHouseVal"

feature_cols = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

X_all = train_df[feature_cols].values
y_all = train_df[[target_col]].values

# 在训练集内部再划分训练子集与验证子集（8:2）
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

X_test = test_df[feature_cols].values
y_test = test_df[[target_col]].values

print(f"\n训练子集样本数: {len(X_train)}")
print(f"验证子集样本数: {len(X_val)}")
print(f"测试集样本数: {len(X_test)}")


# =========================
# 4. 数据标准化
# =========================
# 注意：只能用训练集 fit scaler，测试集只能 transform
# 这样可以避免测试集信息泄露

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)


# =========================
# 5. 转换为 PyTorch Tensor
# =========================

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)


# =========================
# 6. 构建线性回归模型
# =========================

class RegressionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


input_dim = len(feature_cols)
model = RegressionMLP(input_dim).to(device)

print("\n模型结构:")
print(model)


# =========================
# 7. 定义损失函数和优化器
# =========================

criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20
)


# =========================
# 8. 训练模型（Early Stopping）
# =========================

max_epochs = 5000
patience = 80
min_delta = 1e-5

train_loss_history = []
val_loss_history = []

best_val_loss = float("inf")
best_state_dict = None
best_epoch = 0
wait = 0

for epoch in range(max_epochs):
    model.train()

    # 前向传播
    y_train_pred = model(X_train_tensor)

    # 计算损失
    train_loss = criterion(y_train_pred, y_train_tensor)

    # 清空梯度
    optimizer.zero_grad()

    # 反向传播
    train_loss.backward()

    # 更新参数
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = criterion(y_val_pred, y_val_tensor)

    train_loss_value = train_loss.item()
    val_loss_value = val_loss.item()

    train_loss_history.append(train_loss_value)
    val_loss_history.append(val_loss_value)
    scheduler.step(val_loss_value)

    if val_loss_value < best_val_loss - min_delta:
        best_val_loss = val_loss_value
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1
        wait = 0
    else:
        wait += 1

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{max_epochs}], "
            f"Train Loss: {train_loss_value:.6f}, Val Loss: {val_loss_value:.6f}"
        )

    if wait >= patience:
        print(f"\nEarly stopping 在第 {epoch + 1} 轮触发")
        break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

print(f"最佳验证损失: {best_val_loss:.6f} (第 {best_epoch} 轮)")


# =========================
# 9. 测试模型
# =========================

model.eval()

with torch.no_grad():
    y_train_pred_scaled = model(X_train_tensor)
    y_val_pred_scaled = model(X_val_tensor)
    y_test_pred_scaled = model(X_test_tensor)

# 转回 CPU 和 NumPy
y_train_pred_scaled = y_train_pred_scaled.cpu().numpy()
y_val_pred_scaled = y_val_pred_scaled.cpu().numpy()
y_test_pred_scaled = y_test_pred_scaled.cpu().numpy()

# 将标准化后的预测结果还原成真实房价数值
y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

# 真实值
y_train_true = y_train
y_val_true = y_val
y_test_true = y_test


# =========================
# 10. 计算评价指标
# =========================

mae = mean_absolute_error(y_test_true, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
r2 = r2_score(y_test_true, y_test_pred)

val_mae = mean_absolute_error(y_val_true, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
val_r2 = r2_score(y_val_true, y_val_pred)

print("\n验证集评价结果:")
print(f"MAE  平均绝对误差: {val_mae:.4f}")
print(f"RMSE 均方根误差: {val_rmse:.4f}")
print(f"R2   决定系数: {val_r2:.4f}")

print("\n测试集评价结果:")
print(f"MAE  平均绝对误差: {mae:.4f}")
print(f"RMSE 均方根误差: {rmse:.4f}")
print(f"R2   决定系数: {r2:.4f}")


# =========================
# 11. 保存预测结果
# =========================

result_df = test_df.copy()
result_df["Predicted_MedHouseVal"] = y_test_pred
result_df["Absolute_Error"] = np.abs(result_df[target_col] - result_df["Predicted_MedHouseVal"])

result_path = os.path.join(OUTPUT_DIR, "prediction_results.csv")
result_df.to_csv(result_path, index=False)

print(f"\n预测结果已保存到: {result_path}")


# =========================
# 12. 绘制训练集与验证集 Loss 曲线（分开）
# =========================

plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, color="#1f77b4")
plt.xlabel("Epoch")
plt.ylabel("SmoothL1 Loss")
plt.title("Training Loss Curve")
plt.grid(True)

train_loss_fig_path = os.path.join(OUTPUT_DIR, "train_loss_curve.png")
plt.savefig(train_loss_fig_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(val_loss_history, color="#ff7f0e")
plt.xlabel("Epoch")
plt.ylabel("SmoothL1 Loss")
plt.title("Validation Loss Curve")
plt.grid(True)

val_loss_fig_path = os.path.join(OUTPUT_DIR, "val_loss_curve.png")
plt.savefig(val_loss_fig_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"训练集 Loss 曲线已保存到: {train_loss_fig_path}")
print(f"验证集 Loss 曲线已保存到: {val_loss_fig_path}")

# 兼容旧文件名：同时保存一张合并 Loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("SmoothL1 Loss")
plt.title("Training & Validation Loss Curve")
plt.grid(True)
plt.legend()

legacy_loss_fig_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(legacy_loss_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"兼容版 Loss 曲线已保存到: {legacy_loss_fig_path}")


# =========================
# 13. 绘制真实值与预测值对比图
# =========================

def save_centered_true_pred_plot(y_true, y_pred, title, save_path):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    low = min(np.percentile(y_true, 1), np.percentile(y_pred, 1))
    high = max(np.percentile(y_true, 99), np.percentile(y_pred, 99))
    pad = (high - low) * 0.08
    lower = low - pad
    upper = high + pad

    # 房价主区间可视化：限制极端离群点对坐标轴的拉伸
    lower = max(lower, -1.0)

    plt.figure(figsize=(6.2, 6.0))
    plt.scatter(y_true, y_pred, s=18, alpha=0.35, c="#1f77b4", edgecolors="none")
    plt.plot([lower, upper], [lower, upper], linestyle="--", linewidth=2)
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.xlabel("True MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_test_style_plot(y_true, y_pred, save_path):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # 按测试集主分布设置坐标范围，保证点群居中且可读
    x_upper = float(np.ceil(np.percentile(y_true, 99.5) * 10) / 10)
    y_upper = float(np.ceil(np.percentile(y_pred, 99.7) * 10) / 10)
    upper = max(6.5, x_upper, y_upper)

    plt.figure(figsize=(8.0, 5.2))
    plt.scatter(
        y_true,
        y_pred,
        s=4,
        alpha=0.9,
        c="#3d45e0",
        edgecolors="none",
        label="Test samples",
    )
    plt.plot([0, upper], [0, upper], color="#e53935", linewidth=1.2, label="Ideal line")
    plt.xlim(0, upper)
    plt.ylim(-1, upper)
    plt.xlabel("True Value")
    plt.ylabel("Predicted House Value")
    plt.title("PyTorch Prediction Check: True vs Predicted")
    plt.grid(True, alpha=0.18)
    plt.legend(loc="upper right", fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


pred_fig_path = os.path.join(OUTPUT_DIR, "true_vs_predicted.png")
save_test_style_plot(y_test_true, y_test_pred, pred_fig_path)
print(f"预测对比图已保存到: {pred_fig_path}")

train_pred_fig_path = os.path.join(OUTPUT_DIR, "train_true_vs_pred.png")
save_centered_true_pred_plot(y_train_true, y_train_pred, "Train: True vs Predicted", train_pred_fig_path)
print(f"训练集对比图已保存到: {train_pred_fig_path}")

test_pred_fig_path = os.path.join(OUTPUT_DIR, "test_true_vs_pred.png")
save_test_style_plot(y_test_true, y_test_pred, test_pred_fig_path)
print(f"测试集对比图已保存到: {test_pred_fig_path}")

# 重绘 MedInc -> MedHouseVal 线性拟合图
x_medinc = train_df["MedInc"].values.reshape(-1, 1)
y_house = train_df[target_col].values
w, b = np.polyfit(x_medinc.flatten(), y_house, 1)
x_line = np.linspace(x_medinc.min(), x_medinc.max(), 300)
y_line = w * x_line + b

plt.figure(figsize=(10, 6.5))
plt.scatter(x_medinc, y_house, s=10, alpha=0.5, c="#7f8fa6", edgecolors="none", label="Training samples")
plt.plot(x_line, y_line, color="#16a34a", linewidth=2.5, label="Learned linear function")
plt.xlim(float(x_medinc.min()), float(x_medinc.max()))
plt.ylim(-0.2, 7.6)
plt.xlabel("MedInc", fontsize=16, labelpad=10)
plt.ylabel("MedHouseVal", fontsize=16, labelpad=10)
plt.title("Single-feature Linear Fit: MedInc -> MedHouseVal", fontsize=20, pad=18)
plt.grid(True, alpha=0.18)
plt.legend(loc="upper right", frameon=False, fontsize=11)
plt.tight_layout()
medinc_fig_path = os.path.join(OUTPUT_DIR, "train_medinc_linear_fit.png")
plt.savefig(medinc_fig_path, dpi=300)
plt.close()
print(f"MedInc线性拟合图已保存到: {medinc_fig_path}")


# =========================
# 14. 优化方案二：去掉封顶样本对比实验
# =========================

print("\n=========================")
print("对比实验 B：去掉 MedHouseVal >= 5 的封顶样本")
print("=========================")

train_df_no_cap = train_df[train_df[target_col] < 5].copy()
print(f"原始训练集样本数: {len(train_df)}")
print(f"去掉封顶样本后: {len(train_df_no_cap)}")

X_all_b = train_df_no_cap[feature_cols].values
y_all_b = train_df_no_cap[[target_col]].values

X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
    X_all_b, y_all_b, test_size=0.2, random_state=42
)

x_scaler_b = StandardScaler()
y_scaler_b = StandardScaler()

X_train_b_scaled = x_scaler_b.fit_transform(X_train_b)
X_val_b_scaled = x_scaler_b.transform(X_val_b)
X_test_b_scaled = x_scaler_b.transform(X_test)

y_train_b_scaled = y_scaler_b.fit_transform(y_train_b)
y_val_b_scaled = y_scaler_b.transform(y_val_b)

X_train_b_tensor = torch.tensor(X_train_b_scaled, dtype=torch.float32).to(device)
y_train_b_tensor = torch.tensor(y_train_b_scaled, dtype=torch.float32).to(device)
X_val_b_tensor = torch.tensor(X_val_b_scaled, dtype=torch.float32).to(device)
y_val_b_tensor = torch.tensor(y_val_b_scaled, dtype=torch.float32).to(device)
X_test_b_tensor = torch.tensor(X_test_b_scaled, dtype=torch.float32).to(device)

model_b = RegressionMLP(input_dim).to(device)
criterion_b = nn.SmoothL1Loss()
optimizer_b = optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_b, mode="min", factor=0.5, patience=20
)

train_loss_history_b = []
val_loss_history_b = []
best_val_loss_b = float("inf")
best_state_dict_b = None
best_epoch_b = 0
wait_b = 0

for epoch in range(max_epochs):
    model_b.train()
    y_train_b_pred = model_b(X_train_b_tensor)
    train_loss_b = criterion_b(y_train_b_pred, y_train_b_tensor)

    optimizer_b.zero_grad()
    train_loss_b.backward()
    optimizer_b.step()

    model_b.eval()
    with torch.no_grad():
        y_val_b_pred = model_b(X_val_b_tensor)
        val_loss_b = criterion_b(y_val_b_pred, y_val_b_tensor)

    train_loss_b_value = train_loss_b.item()
    val_loss_b_value = val_loss_b.item()
    train_loss_history_b.append(train_loss_b_value)
    val_loss_history_b.append(val_loss_b_value)
    scheduler_b.step(val_loss_b_value)

    if val_loss_b_value < best_val_loss_b - min_delta:
        best_val_loss_b = val_loss_b_value
        best_state_dict_b = {
            k: v.detach().cpu().clone() for k, v in model_b.state_dict().items()
        }
        best_epoch_b = epoch + 1
        wait_b = 0
    else:
        wait_b += 1

    if wait_b >= patience:
        break

if best_state_dict_b is not None:
    model_b.load_state_dict(best_state_dict_b)

with torch.no_grad():
    y_val_b_pred_scaled = model_b(X_val_b_tensor).cpu().numpy()
    y_test_b_pred_scaled = model_b(X_test_b_tensor).cpu().numpy()

y_val_b_pred = y_scaler_b.inverse_transform(y_val_b_pred_scaled)
y_test_b_pred = y_scaler_b.inverse_transform(y_test_b_pred_scaled)

val_mae_b = mean_absolute_error(y_val_b, y_val_b_pred)
val_rmse_b = np.sqrt(mean_squared_error(y_val_b, y_val_b_pred))
val_r2_b = r2_score(y_val_b, y_val_b_pred)

test_mae_b = mean_absolute_error(y_test, y_test_b_pred)
test_rmse_b = np.sqrt(mean_squared_error(y_test, y_test_b_pred))
test_r2_b = r2_score(y_test, y_test_b_pred)

print(f"实验 B 最佳验证损失: {best_val_loss_b:.6f} (第 {best_epoch_b} 轮)")
print(f"实验 B 验证集: MAE={val_mae_b:.4f}, RMSE={val_rmse_b:.4f}, R2={val_r2_b:.4f}")
print(f"实验 B 测试集: MAE={test_mae_b:.4f}, RMSE={test_rmse_b:.4f}, R2={test_r2_b:.4f}")

# 额外输出实验 B 的测试图，便于观察点群是否更集中
test_pred_fig_path_b = os.path.join(OUTPUT_DIR, "test_true_vs_pred_no_cap.png")
# 图像仅展示非封顶测试样本，避免 MedHouseVal=5 的竖线影响观感
test_mask_no_cap = y_test.reshape(-1) < 5
save_test_style_plot(
    y_test[test_mask_no_cap],
    y_test_b_pred[test_mask_no_cap],
    test_pred_fig_path_b,
)
print(f"实验 B 测试集对比图已保存到: {test_pred_fig_path_b}")

comparison_df = pd.DataFrame(
    [
        {
            "Experiment": "A_all_data",
            "Val_MAE": val_mae,
            "Val_RMSE": val_rmse,
            "Val_R2": val_r2,
            "Test_MAE": mae,
            "Test_RMSE": rmse,
            "Test_R2": r2,
        },
        {
            "Experiment": "B_remove_cap_samples",
            "Val_MAE": val_mae_b,
            "Val_RMSE": val_rmse_b,
            "Val_R2": val_r2_b,
            "Test_MAE": test_mae_b,
            "Test_RMSE": test_rmse_b,
            "Test_R2": test_r2_b,
        },
    ]
)
comparison_path = os.path.join(OUTPUT_DIR, "experiment_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)
print(f"对比结果已保存到: {comparison_path}")


# =========================
# 15. 保存模型
# =========================

model_path = os.path.join(OUTPUT_DIR, "mlp_regression_model.pth")
torch.save(model.state_dict(), model_path)
model_b_path = os.path.join(OUTPUT_DIR, "mlp_regression_model_no_cap.pth")
torch.save(model_b.state_dict(), model_b_path)

print(f"模型参数已保存到: {model_path}")
print(f"实验 B 模型参数已保存到: {model_b_path}")


# =========================
# 16. 输出部分预测样例
# =========================

print("\n前10个测试样本预测结果:")
print(result_df[[target_col, "Predicted_MedHouseVal", "Absolute_Error"]].head(10))