import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import argparse
import yaml
import re
import math

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from data.dataset_loader import get_dataloader, get_dataset, get_transforms
from utils.logger import Logger
from utils.plotter import plot_loss_curve
from utils.plotter import plot_accuracy_curve, plot_hyperparam_curve, plot_hyperparam_2d_curve

DEVICE = torch.device("cuda", 0)
model = resnet18(num_classes=10).to(DEVICE)

dataset_path = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_regularization/round1"
recipes_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_regularization/round1"

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

regularization_type = [
    "dropout",
    "weight_decay",
]

# def run_eval():
#     for type in optimizer_type:
#         optimizer_res_path = os.path.join(res_path, type)
#         optimizer_recipe_path = os.path.join(recipes_path, type)
#         for i in range(n_trials):
#             recipe_path = os.path.join(optimizer_recipe_path, f"config_{i}.yaml")
#             with open(recipe_path, "r") as f:
#                 config =  yaml.safe_load(f)
#             lr = config.get("lr")
#             momentum = config.get("momentum", "none")
#             ckpt_path = os.path.join(optimizer_res_path, f"config_{i}")
#             best_ckpt_path = None
#             for root, dirs, files in os.walk(ckpt_path):
#                 for file in files:
#                     if file.startswith("best_model"):
#                         best_ckpt_path = os.path.join(root, file)
#                         break
#             model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
#             model.eval()
#             correct, total = 0, 0
#             with torch.no_grad():
#                 for images, labels in test_loader:
#                     images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
#                     outputs = model(images)
#                     _, predicted = torch.max(outputs, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()
#             acc = 100. * correct / total
#             with open(os.path.join(optimizer_res_path, "acc.txt"), "a") as f:
#                 f.write(f"LR: {lr}, Momentum: {momentum}, Accuracy: {acc:.2f}%\n")

def run_eval():
    for type_ in regularization_type:
        recipe_path = os.path.join(recipes_path, type_)
        if type_ == "dropout":
            n_trials = 10
            for i in range(n_trials):
                config_path = os.path.join(recipe_path, f"config_{i}.yaml")
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                dropout = config.get("dropout")
                ckpt_path = os.path.join(res_path, type_, f"config_{i}")
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
                with open(os.path.join(res_path, type_, "acc.txt" ), "a") as f:
                    f.write(f"Dropout: {dropout}, Accuracy: {acc:.2f}%\n")
        elif type_ == "weight_decay":
            n_trials = 8
            for i in range(n_trials):
                config_path = os.path.join(recipe_path, f"config_{i}.yaml")
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                weight_decay = config.get("weight_decay")
                ckpt_path = os.path.join(res_path, type_, f"config_{i}")
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
                with open(os.path.join(res_path, type_, "acc.txt" ), "a") as f:
                    f.write(f"weight_decay: {weight_decay}, Accuracy: {acc:.2f}%\n")
        
        

if __name__ == "__main__":
    # run_eval()
    # dropout_range = [0.0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # dropout_acc = [81.6,79.4,79.0,79.8,80.5,78.5,78.7,80.2,80.3,81.4]
    weight_decay_range = [math.log10(x) if x > 0 else float('-inf') for x in [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]]
    weight_decay_acc = [77.18, 76.88, 80.12, 81.65, 78.24, 73.29, 59.97, 52.31]

    # plot_hyperparam_curve(dropout_range, dropout_acc, os.path.join(res_path, "dropout", "dropout_acc.png"), x_name="dropout")
    plot_hyperparam_curve(weight_decay_range, weight_decay_acc, os.path.join(res_path, "weight_decay", "weight_decay_acc.png"), x_name="weight_decay(log)")
    
    # for type in optimizer_type:
    #     lr_list = []
    #     acc_list = []
    #     momentum_list = []
    #     lr_path = f"/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round2/{type}/{type}_with_lr_acc.png"
    #     momentum_path = f"/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round2/{type}/{type}_with_momentum_acc.png"
    #     both_path = f"/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round2/{type}/{type}_with_lr_and_momentum_acc.png"
    #     with open(f"/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round2/{type}/acc.txt", "r") as f:
    #         for line in f:
    #             match = re.search(r'LR:\s*([\d\.eE+-]+),\s*Momentum:\s*([\w\.]+),\s*Accuracy:\s*([\d\.]+)%', line)
    #             if match:
    #                 lr = float(match.group(1))
    #                 momentum = match.group(2)
    #                 if momentum != 'none':
    #                     momentum = float(momentum)
    #                     momentum_list.append(momentum)
    #                 acc = float(match.group(3))
    #                 lr_list.append(lr)
    #                 acc_list.append(acc)
    #     plot_hyperparam_curve(lr_list, acc_list, lr_path, x_name="lr")
    #     if len(momentum_list) != 0:
    #         plot_hyperparam_curve(momentum_list, acc_list, momentum_path, x_name="momentum")
    #         plot_hyperparam_2d_curve(lr_list, momentum_list, acc_list, both_path)
    