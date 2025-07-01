import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
import torch.distributed as dist
import argparse

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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # DDP初始化，设置当前进程的分布式环境
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda", local_rank)

    # 只在主进程(rank==0)创建输出目录和日志，避免多进程冲突
    is_main = dist.get_rank() == 0
    if is_main:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RESULTS_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")
        loss_curve_path = os.path.join(run_dir, "loss_curve.png")
        best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
        last_ckpt_path = os.path.join(checkpoints_dir, "last_model.pth")
        logger = Logger(log_path)
        logger.log(f"Training started at {datetime.now()}")
    else:
        run_dir = checkpoints_dir = log_path = loss_curve_path = best_ckpt_path = last_ckpt_path = None
        logger = None

    # 数据集与划分
    full_trainset = get_dataset(DATA_DIR, split="train",dataset_name="CIFAR10", custom_transform=get_transforms(resize=(32,32)))
    val_len = int(len(full_trainset) * VAL_RATIO)
    train_len = len(full_trainset) - val_len
    trainset, valset = random_split(full_trainset, [train_len, val_len])

    # 使用DistributedSampler确保每个进程读取不同的数据子集
    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(valset, shuffle=False)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2, pin_memory=True)

    testset = get_dataset(DATA_DIR, split="test", dataset_name="CIFAR10", custom_transform=get_transforms(resize=(32,32)))
    test_sampler = DistributedSampler(testset, shuffle=False)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=2, pin_memory=True)

    # 模型、损失、优化器
    model = resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    # 使用SyncBatchNorm和DistributedDataParallel包装模型，实现多卡同步训练
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # 设置epoch，保证不同进程shuffle一致
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        # 所有进程同步loss，保证统计全局平均
        total_loss = torch.tensor(running_loss, device=DEVICE)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_loss = total_loss.item() / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        # 所有进程同步val loss
        total_val_loss = torch.tensor(val_loss, device=DEVICE)
        dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
        val_loss = total_val_loss.item() / len(val_loader.dataset)
        val_losses.append(val_loss)

        if is_main:
            logger.log(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 只在主进程保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), best_ckpt_path)
                logger.log(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        scheduler.step()

    # 只在主进程保存最后模型
    if is_main:
        torch.save(model.module.state_dict(), last_ckpt_path)
        logger.log("Last model checkpoint saved.")

    # 测试集评估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 所有进程同步测试集正确数和总数
    total_tensor = torch.tensor([correct, total], device=DEVICE)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * total_tensor[0].item() / total_tensor[1].item()
    if is_main:
        logger.log(f"Test Accuracy: {acc:.2f}%")

        # 只在主进程绘制loss曲线
        plot_loss_curve(
            range(1, EPOCHS+1),
            train_losses,
            val_losses,
            loss_curve_path
        )
        logger.log(f"Training finished at {datetime.now()}")

    # 训练结束后销毁进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
