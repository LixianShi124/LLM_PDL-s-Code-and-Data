import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import wandb
from scipy import signal
from collections import Counter
import os
from vmdpy import VMD
from z1_amp import AmplitudeSignalProcessor,Amplitude_Only_Net,AmplitudeDataset
import matplotlib.pyplot as plt
# --- 1. 配置与随机种子 ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# # === 5. 评估与分析工具 ===
def analyze_bad_cases(loader, model, device):
    model.eval()
    bad_cases_list = []

    # 定义幅值类标签名称 (根据之前的索引 0, 1, 4)
    # 假设原数据集顺序为: [Sag, Swell, Harm, Fluc, Int, Impul, Oscill]
    # 我们只取 idx_amp 对应的名字
    target_names =["Normal", "Swell", "Sag", "Int"]

    print("\n" + "=" * 60)
    print(">>> 幅值类错误样本分析 (Bad Case Analysis)")
    print("=" * 60)

    error_pattern_counter = Counter()

    with torch.no_grad():
        for i, (x_amp, labels) in enumerate(loader):
            x_amp = x_amp.to(device)
            labels = labels.to(device)

            outputs = model(x_amp)
            preds = torch.argmax(outputs, dim=1)  # 多分类预测

            pred_np = preds.cpu().numpy()
            target_np = labels.cpu().numpy()

            for b in range(len(labels)):
                p = pred_np[b]
                t = target_np[b]

                if p != t:  # 预测错误
                    t_str = target_names[t]
                    p_str = target_names[p]
                    error_pattern_counter[(t_str, p_str)] += 1

    # 输出统计
    print(">>> 错误模式统计 (真实 -> 预测):")
    if len(error_pattern_counter) > 0:
        pattern_data = []
        for (t_str, p_str), count in error_pattern_counter.most_common():
            pattern_data.append({"True": t_str, "Pred": p_str, "Count": count})
        print(pd.DataFrame(pattern_data))
    else:
        print("完美！没有发现错误。")
    print("=" * 60 + "\n")

# def analyze_bad_cases(loader, model, device, num_plots=5):
#     """
#     分析错误样本并可视化波形
#     :param num_plots: 最多展示多少张错误波形图，避免弹窗过多
#     """
#     model.eval()
#
#     # 定义幅值类标签名称 (根据之前的索引)
#     target_names = ["Normal", "Swell", "Sag", "Int"]
#
#     print("\n" + "=" * 60)
#     print(">>> 幅值类错误样本分析与波形可视化 (Bad Case Analysis)")
#     print("=" * 60)
#
#     error_pattern_counter = Counter()
#     plot_count = 0  # 记录已画图的数量
#
#     with torch.no_grad():
#         for i, (x_amp, labels) in enumerate(loader):
#             x_amp = x_amp.to(device)
#             labels = labels.to(device)
#
#             outputs = model(x_amp)
#             preds = torch.argmax(outputs, dim=1)  # 多分类预测
#
#             pred_np = preds.cpu().numpy()
#             target_np = labels.cpu().numpy()
#
#             # 将输入数据转回 CPU 以便绘图
#             # 假设 x_amp 的形状为 (Batch, Channels, Seq_Len)
#             # 根据特征提取算法，通道 0 为基波，通道 1 为包络
#             x_np = x_amp.cpu().numpy()
#
#             for b in range(len(labels)):
#                 p = pred_np[b]
#                 t = target_np[b]
#
#                 if p != t:  # 发现预测错误
#                     t_str = target_names[t]
#                     p_str = target_names[p]
#                     error_pattern_counter[(t_str, p_str)] += 1
#
#                     # === 新增：画出被误判的波形 ===
#                     if plot_count < num_plots:
#                         plt.figure(figsize=(10, 4))
#
#                         # 提取当前错误样本的双通道数据
#                         signal_fund = x_np[b, 0, :]  # 通道 0：基波
#                         signal_env = x_np[b, 1, :]  # 通道 1：幅值包络
#
#                         # 绘制波形
#                         plt.plot(signal_fund, label='Fundamental Wave', color='royalblue', alpha=0.6, linestyle='--')
#                         plt.plot(signal_env, label='Extracted Envelope', color='crimson', linewidth=2)
#
#                         # 设置标题，高亮显示 True 和 Pred
#                         plt.title(f"Bad Case #{plot_count + 1} | True: [{t_str}] ❌ Pred: [{p_str}]",
#                                   fontsize=14, fontweight='bold', color='darkred')
#                         plt.xlabel("Sample Index", fontsize=12)
#                         plt.ylabel("Amplitude (p.u.)", fontsize=12)
#                         plt.ylim(0, 1.5)  # 根据标幺值范围可适当调整
#                         plt.grid(True, linestyle=':', alpha=0.7)
#                         plt.legend(loc='upper right', fontsize=11)
#                         plt.tight_layout()
#                         plt.show()
#
#                         plot_count += 1
#
#     # 输出统计信息
#     print(">>> 错误模式统计 (真实 -> 预测):")
#     if len(error_pattern_counter) > 0:
#         pattern_data = []
#         for (t_str, p_str), count in error_pattern_counter.most_common():
#             pattern_data.append({"True": t_str, "Pred": p_str, "Count": count})
#         print(pd.DataFrame(pattern_data).to_string(index=False))
#     else:
#         print("完美！没有发现错误。")
#     print("=" * 60 + "\n")


# === 6. 训练流程 ===
def train_amplitude_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128

    print("Loading Data (Amplitude Mode)...")
    # 请确保文件名正确
    Training_Data = AmplitudeDataset('zzPQD_training_dataset_6000Hz_unseen_0db.csv', mode='train')
    Testing_Data = AmplitudeDataset('zzPQD_testing_dataset_6000Hz_unseen_0db.csv', mode='test')

    train_loader = DataLoader(Training_Data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Testing_Data, batch_size=batch_size, shuffle=False)

    model = Amplitude_Only_Net(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()  # 多分类损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for x_amp, labels in train_loader:
            x_amp = x_amp.to(device)
            labels = labels.to(device)  # labels形状: [batch_size]

            optimizer.zero_grad()
            outputs = model(x_amp)  # 输出形状: [batch_size, 4]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算训练准确率
            preds = torch.argmax(outputs, dim=1)  # 多分类预测
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_acc = correct / total_samples if total_samples > 0 else 0

        # --- Test Evaluation ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x_amp, labels in test_loader:
                x_amp = x_amp.to(device)
                labels = labels.to(device)

                outputs = model(x_amp)
                preds = torch.argmax(outputs, dim=1)  # 多分类预测

                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total if test_total > 0 else 0

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        # WandB 日志
        wandb.log({
            "loss": total_loss / len(train_loader),
            "train_acc": train_acc,
            "test_acc": test_acc
        })
    # ================= 核心：保存模型代码 =================
    # 1. 创建 Model 文件夹 (如果不存在)
    save_dir = "Model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 2. 设定保存路径
    save_path = os.path.join(save_dir, "a1_Amp_Model.pth")

    # 3. 执行保存 (保存 state_dict)
    torch.save(model.state_dict(), save_path)
    print(f"\n>>> Model saved successfully at: {save_path}")
    # print(f">>> Final Test Accuracy: {test_acc:.4f} | Best Test Accuracy: {best_test_acc:.4f}")
    # ====================================================
        # 训练结束后分析错误
    analyze_bad_cases(test_loader, model, device)

if __name__ == "__main__":
    # wandb.init(project="Amplitude_encoder", entity="shilixian", name="a0_No_process")
    wandb.init(project="Amplitude_encoder", entity="shilixian", name="还是回到老思路，a5_TFM-ZHAPE_广义双正交分数阶构造 0db")
    # wandb.init(mode="disabled")  # 初始化时设置
    train_amplitude_only()