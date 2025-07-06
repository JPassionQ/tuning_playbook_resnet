import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import argparse
import yaml
import re

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from data.dataset_loader import get_dataloader, get_dataset, get_transforms
from utils.logger import Logger
from utils.plotter import plot_loss_curve
from utils.plotter import plot_accuracy_curve, plot_hyperparam_curve, plot_hyperparam_2d_curve

DEVICE = torch.device("cuda", 0)
model = resnet50(num_classes=10).to(DEVICE)

dataset_path = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round1"
recipes_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_optimizer/round1"
n_trials = 16

testset = get_dataset(dataset_path, split="test", dataset_name="CIFAR10", custom_transform=get_transforms(resize=(32,32),normalize=True))
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

optimizer_type = [
    "sgd",
    "sgd_with_momentum",
    "Nestrov",
    "Adam",
    "NAdam",
    "AdamW",
    "RMSprop",
]

def run_eval():
    for type in optimizer_type:
        optimizer_res_path = os.path.join(res_path, type)
        optimizer_recipe_path = os.path.join(recipes_path, type)
        for i in range(n_trials):
            recipe_path = os.path.join(optimizer_recipe_path, f"config_{i}.yaml")
            with open(recipe_path, "r") as f:
                config =  yaml.safe_load(f)
            lr = config.get("lr")
            momentum = config.get("momentum", "none")
            ckpt_path = os.path.join(optimizer_res_path, f"config_{i}")
            best_ckpt_path = None
            for root, dirs, files in os.walk(ckpt_path):
                for file in files:
                    if file.startswith("best_model"):
                        best_ckpt_path = os.path.join(root, file)
                        break
            model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
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
            with open(os.path.join(optimizer_res_path, "acc.txt"), "a") as f:
                f.write(f"LR: {lr}, Momentum: {momentum}, Accuracy: {acc:.2f}%\n")

if __name__ == "__main__":
    # run_eval()
    lr_list = []
    acc_list = []
    momentum_list = []
    save_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round1/sgd_with_momentum/sgd_with_momentum_with_lr_and_momentum_acc.png"
    with open("/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round1/sgd_with_momentum/acc.txt", "r") as f:
        for line in f:
            match = re.search(r'LR:\s*([\d\.eE+-]+),\s*Momentum:\s*([\w\.]+),\s*Accuracy:\s*([\d\.]+)%', line)
            if match:
                lr = float(match.group(1))
                momentum = float(match.group(2))
                acc = float(match.group(3))
                lr_list.append(lr)
                momentum_list.append(momentum)
                acc_list.append(acc)
    # plot_hyperparam_curve(lr_list, acc_list, save_path)
    # plot_hyperparam_curve(momentum_list, acc_list, save_path)
    plot_hyperparam_2d_curve(lr_list, momentum_list, acc_list, save_path)
    