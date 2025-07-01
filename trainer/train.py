import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime

from models.resnet import resnet18
from data.dataset_loader import get_dataloader, get_dataset, get_transforms
from utils.logger import Logger
from utils.plotter import plot_loss_curve

# 配置
DATA_DIR = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
RESULTS_DIR = "/home/jingqi/DeepLearningWorkshop/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20
VAL_RATIO = 0.2
NUM_CLASSES = 10
LR = 0.1

def main():
    # 创建本次训练的唯一结果目录
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")
    loss_curve_path = os.path.join(run_dir, "loss_curve.png")
    best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
    last_ckpt_path = os.path.join(checkpoints_dir, "last_model.pth")

    # 日志初始化
    logger = Logger(log_path)
    logger.log(f"Training started at {datetime.now()}")

    # 数据集与划分
    full_trainset = get_dataset(DATA_DIR, split="train",dataset_name="CIFAR10", custom_transform=get_transforms(resize=(32,32)))
    val_len = int(len(full_trainset) * VAL_RATIO)
    train_len = len(full_trainset) - val_len
    trainset, valset = random_split(full_trainset, [train_len, val_len])
    train_loader = get_dataloader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = get_dataloader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    testset = get_dataset(DATA_DIR, split="test", dataset_name="CIFAR10", custom_transform=get_transforms(resize=(32,32)))
    test_loader = get_dataloader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 模型、损失、优化器
    model = resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

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

        logger.log(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            logger.log(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        scheduler.step()

    # 保存最后模型
    torch.save(model.state_dict(), last_ckpt_path)
    logger.log("Last model checkpoint saved.")

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
    logger.log(f"Test Accuracy: {acc:.2f}%")

    # 绘制loss曲线
    plot_loss_curve(
        range(1, EPOCHS+1),
        train_losses,
        val_losses,
        loss_curve_path
    )

    logger.log(f"Training finished at {datetime.now()}")

if __name__ == "__main__":
    main()
