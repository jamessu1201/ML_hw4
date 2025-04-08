import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.dataloader import make_train_dataloader
from models.model import ExampleCNN

import os
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def get_loss_function_name(loss_type_index):
    # 將數字映射到損失函數名稱
    loss_mapping = {
        0: "cross_entropy",
        1: "mse",
        2: "kl",
        3: "hinge",
        4: "l1"
    }
    
    # 如果是整數且在映射表中，返回對應名稱，否則返回默認值
    if isinstance(loss_type_index, int) and loss_type_index in loss_mapping:
        return loss_mapping[loss_type_index]
    else:
        return "cross_entropy"  # 默認使用交叉熵損失

# 定義一個計算損失的函數，處理不同損失函數的輸入需求
def calculate_loss(output, target, criterion_type):
    if criterion_type == "mse":
        criterion = nn.MSELoss()
        # 將標籤轉換為 one-hot 編碼
        target_one_hot = F.one_hot(target, num_classes=output.size(1)).float()
        return criterion(output, target_one_hot)
    
    elif criterion_type == "kl":
        criterion = nn.KLDivLoss(reduction='batchmean')
        # KL 散度需要 log_softmax 輸出和歸一化目標
        output = F.log_softmax(output, dim=1)
        target_one_hot = F.one_hot(target, num_classes=output.size(1)).float()
        return criterion(output, target_one_hot)
    
    elif criterion_type == "hinge":
        criterion = nn.MultiMarginLoss()
        # MultiMarginLoss 可以直接處理分類問題
        return criterion(output, target)
    
    elif criterion_type == "l1":
        criterion = nn.L1Loss()
        target_one_hot = F.one_hot(target, num_classes=output.size(1)).float()
        return criterion(output, target_one_hot)
    
    else:  # 默認使用 CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        return criterion(output, target)

# 解析命令行參數
arg = sys.argv[1:]
for i in range(len(arg)):
    if '0.' in arg[i]:
        arg[i] = float(arg[i])
    else:
        arg[i] = int(arg[i])

# 設置 logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f'{arg[1:]}_output.txt'),
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)

# 訓練參數
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = arg[1]
batch_size = arg[2]
learning_rate = arg[3]

# 根據作業編號和參數選擇損失函數
if arg[0] == 2 and len(arg) > 4:
    loss_type_index = arg[4]
    loss_type = get_loss_function_name(loss_type_index)
    logging.info(f"HW4-2: 使用損失函數: {loss_type}")
else:
    loss_type = "cross_entropy"  # 默認使用交叉熵損失
    logging.info(f"HW4-1: 使用默認損失函數: {loss_type}")

# 數據路徑和權重路徑
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "data", "train")
weight_path = os.path.join(base_path, "weights", "weight.pth")

# 創建 dataloader
train_loader, valid_loader = make_train_dataloader(train_data_path, batch_size)

# 設置 CNN 模型
model = ExampleCNN()
model = model.to(device)

# 設置優化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 訓練循環
train_loss_list = list()
valid_loss_list = list()
train_accuracy_list = list()
valid_accuracy_list = list()
best = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    logging.info(f'\nEpoch: {epoch+1}/{epochs}')
    logging.info('-' * len(f'Epoch: {epoch+1}/{epochs}'))
    train_loss, valid_loss = 0.0, 0.0
    train_correct, valid_correct = 0, 0
    train_accuracy, valid_accuracy = 0.0, 0.0

    # 訓練階段
    model.train()
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)

        # 前向傳播 + 反向傳播 + 優化
        output = model(data)
        _, preds = torch.max(output.data, 1)
        
        # 使用我們的函數計算損失
        loss = calculate_loss(output, target, loss_type)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        train_correct += torch.sum(preds == target.data)
    
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = float(train_correct) / len(train_loader.dataset)
    train_accuracy_list.append(train_accuracy)

    # 驗證階段
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            # 使用我們的函數計算損失
            loss = calculate_loss(output, target, loss_type)
            _, preds = torch.max(output.data, 1)

            valid_loss += loss.item() * data.size(0)
            valid_correct += torch.sum(preds == target.data)
        
        valid_loss /= len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_accuracy = float(valid_correct) / len(valid_loader.dataset)
        valid_accuracy_list.append(valid_accuracy)
    
    # 打印損失和準確率
    logging.info(f'Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}')
    logging.info(f'Training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}')

    # 記錄最佳權重
    if valid_loss < best:
        best = valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())

# 保存最佳權重
torch.save(best_model_wts, weight_path)

# 繪製損失曲線
logging.info("\nFinished Training")

# 構建文件名：包含參數和損失函數類型
if arg[0] == 2:
    name = f"HW4-2_e{epochs}_b{batch_size}_lr{learning_rate}_{loss_type}"
else:
    name = f"HW4-1_e{epochs}_b{batch_size}_lr{learning_rate}"

pd.DataFrame({
    "train-loss": train_loss_list,
    "valid-loss": valid_loss_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1, epoch+1)
plt.xlabel("Epoch"), plt.ylabel("Loss")
l = name + "_Loss_curve.png"
plt.savefig(os.path.join(base_path, "result", l))

# 繪製準確率曲線
pd.DataFrame({
    "train-accuracy": train_accuracy_list,
    "valid-accuracy": valid_accuracy_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1, epoch+1)
plt.xlabel("Epoch"), plt.ylabel("Accuracy")
t = name + "_Training_accuracy.png"
plt.savefig(os.path.join(base_path, "result", t))