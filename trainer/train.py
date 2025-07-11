import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import argparse
import yaml
import random
import numpy as np

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from data.dataset_loader import get_dataloader, get_dataset, get_transforms
from utils.logger import Logger
from utils.plotter import plot_loss_curve
from utils.plotter import plot_accuracy_curve

# 从yaml配置文件读取超参数
def load_config(config_path):
    with open(config_path, "r") as f:
        config =  yaml.safe_load(f)
    model_config = {}
    training_config = {}
    dataset_config = {}
    # model config
    model_config['num_classes'] = config.get('num_classes', 10)
    model_config['model_layer'] = config.get('model_layer', 50)
    model_config['activation'] = config.get('activation', 'relu')
    model_config['dropout'] = config.get('dropout', 0.0)

    # training config
    training_config['train_batch'] = config.get('train_batch', 128)
    training_config['eval_batch'] = config.get('eval_batch', 256)
    training_config['optimizer_type'] = config.get('optimizer_type', 'sgd')
    training_config['epochs'] = config.get('epochs', 10)
    training_config['lr'] = config.get('lr', 0.1)
    training_config['weight_decay'] = config.get('weight_decay', 0)
    training_config['momentum'] = config.get('momentum', 0.9)
    training_config['eval_steps'] = config.get('eval_steps', 50)
    training_config['log_steps'] = config.get('log_steps', 1)
    
    
    # dataset config
    dataset_config['dataset_path'] = config.get('dataset_path')
    dataset_config['res_path'] = config.get('res_path')
    dataset_config['data_augmentation'] = config.get('data_augmentation', None) # 字典

    return model_config, training_config, dataset_config

#设置全局种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Train a ResNet on CIFAR10")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config yaml file')
    args = parser.parse_args()
    config_path = args.config_path

    # 模型相关配置
    model_config, training_config, dataset_config = load_config(config_path)
    
    num_classes = model_config['num_classes']
    model_layer = model_config['model_layer']
    activation = model_config['activation']
    dropout = model_config['dropout']

    # 训练相关配置
    train_batch_size = training_config['train_batch']
    eval_batch_size = training_config['eval_batch']
    optimizer_type = training_config['optimizer_type']
    epochs = training_config['epochs']
    lr = training_config['lr']
    weight_decay = training_config['weight_decay']  
    momentum = training_config['momentum']
    eval_steps = training_config['eval_steps']
    log_steps = training_config['log_steps']

    # 数据集相关配置
    dataset_path = dataset_config['dataset_path']
    res_path = dataset_config['res_path']
    data_augmentation = dataset_config['data_augmentation']  # 字典

    DEVICE = torch.device("cuda", 0)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 运行结果文件路径
    run_dir = os.path.join(res_path, run_name)
    os.makedirs(run_dir, exist_ok=True)
    # 检查点目录
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    # 日志文件路径
    log_path = os.path.join(run_dir, "train.log")
    loss_curve_path = os.path.join(run_dir, "loss_curve.png")
    accuracy_curve_path = os.path.join(run_dir, "accuracy_curve.png")
    best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
    last_ckpt_path = os.path.join(checkpoints_dir, "last_model.pth")
    logger = Logger(log_path)
    logger.log(f"Training started at {datetime.now()}", verbose=True)

    # 数据集与划分
    trainset = get_dataset(dataset_path, split="train",dataset_name="CIFAR10", custom_transform=get_transforms(
        resize=(32,32),
        normalize=True,
        # random_crop=data_augmentation.get('random_crop', False),
        # random_horizontal_filp=data_augmentation.get('random_horizontal_filp', False),
        # random_rotate=data_augmentation.get('random_rotate',False),
        # color_jitter=data_augmentation.get('color_jitter', False),
        # gaussian_blur=data_augmentation.get('gaussian_blur', False)
        )
    )
    valset = get_dataset(dataset_path, split="val", dataset_name="CIFAR10", custom_transform=get_transforms(resize=(32,32),normalize=True))

    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    testset = get_dataset(dataset_path, split="test", dataset_name="CIFAR10", custom_transform=get_transforms(resize=(32,32),normalize=True))
    test_loader = DataLoader(testset, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 模型层数
    if model_layer == 18:
        model = resnet18(num_classes=num_classes, dropout_p=dropout).to(DEVICE)
    elif model_layer == 34:
        model = resnet34(num_classes=num_classes, dropout_p=dropout).to(DEVICE)
    elif model_layer == 50:
        model = resnet50(num_classes=num_classes, dropout_p=dropout).to(DEVICE)
    elif model_layer == 101:
        model = resnet101(num_classes=num_classes, dropout_p=dropout).to(DEVICE)
    elif model_layer == 152:
        model = resnet152(num_classes=num_classes, dropout_p=dropout).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd_with_momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "Nestrov":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "NAdam":
        optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    val_accuracies = []  # 新增
    best_val_acc = 0.0
    
    total_steps = len(train_loader.dataset) // train_batch_size * epochs
    step = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            step += 1
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 统计每个step的损失（取平均）
            step_loss = loss.item()
            logger.log(f"steps [{step}/{total_steps}] Train Loss: {step_loss:.2f}")
            if step % log_steps == 0:
                train_losses.append(step_loss)

            # 间隔一定的step数进行模型评估
            if step % eval_steps == 0:
                # 验证
                model.eval()
                correct = 0
                total = 0
                valid_loss = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        # 统计准确率
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        valid_loss += loss.item() * images.size(0)
                val_acc = 100. * correct / total
                val_accuracies.append(val_acc)
                val_losses.append(valid_loss / len(val_loader.dataset))
                logger.log(f"step [{step}/{total_steps}] | Val Acc: {val_acc:.2f}% | Val Loss: {val_losses[-1]:.2f}", verbose=True)
                # 保存最佳性能的模型权重
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_ckpt_path)
                    logger.log(f"Best model saved at step {step} with val acc {best_val_acc:.2f}%", verbose=True)

    torch.save(model.state_dict(), last_ckpt_path)
    logger.log("Last model checkpoint saved.", verbose=True)

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
    acc = 100. * correct / total
    logger.log(f"Test Accuracy: {acc:.2f}%",verbose=True)

    # 只绘制loss曲线和准确率曲线
    plot_loss_curve(
        total_steps,
        train_losses,
        val_losses,
        loss_curve_path
    )
    plot_accuracy_curve(
        range(1, len(val_accuracies)+1),
        val_accuracies,
        accuracy_curve_path
    )
    logger.log(f"Training finished at {datetime.now()}", verbose=True)

if __name__ == "__main__":
    main()
