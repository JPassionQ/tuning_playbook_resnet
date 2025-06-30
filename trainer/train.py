import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from models.resnet import resnet18
from data.dataset_loader import get_dataset, get_transforms

# 配置
DATA_DIR = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
RESULTS_DIR = "/home/jingqi/DeepLearningWorkshop/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "train.log")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20
VAL_RATIO = 0.2
NUM_CLASSES = 10
LR = 0.1

def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

def main():
    # 日志初始化
    with open(LOG_PATH, "w") as f:
        f.write(f"Training started at {datetime.now()}\n")

    # 数据集与划分
    full_trainset = get_dataset(DATA_DIR, dataset_name="CIFAR10", train=True, transform=get_transforms(resize=(32,32)))
    val_len = int(len(full_trainset) * VAL_RATIO)
    train_len = len(full_trainset) - val_len
    trainset, valset = random_split(full_trainset, [train_len, val_len])
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    testset = get_dataset(DATA_DIR, dataset_name="CIFAR10", train=False, transform=get_transforms(resize=(32,32)))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 模型、损失、优化器
    model = resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        log(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step()

    # 测试集评估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100. * correct / total
    log(f"Test Accuracy: {acc:.2f}%")

    # 绘制loss曲线
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train/Val Loss Curve')
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()

    log(f"Training finished at {datetime.now()}")

if __name__ == "__main__":
    main()
