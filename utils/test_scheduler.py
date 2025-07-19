import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR, CosineAnnealingLR
from models.resnet import resnet18
import matplotlib.pyplot as plt
def get_scheduler(optimizer, lr_schedule, lr, total_steps, warmup_steps):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    if lr_schedule == "multistep":
        decay_scheduler = MultiStepLR(
            optimizer,
            milestones=[18000, 30000],
            gamma=0.1
        )
    elif lr_schedule == "linear":
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=total_steps - warmup_steps
        )
    elif lr_schedule == "cosine":
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(total_steps - warmup_steps) / 5,
            eta_min=lr * 0.01
        )
    else:
        raise ValueError("Unknown lr_schedule")
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler

def simulate_lr_curve(lr_schedule, lr=0.001):
    warmup_steps = 800
    total_steps = 39200
    model = resnet18(num_classes=10).to("cuda:0")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, lr_schedule, lr, total_steps, warmup_steps)
    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    return lrs

if __name__=="__main__":
    lr = 0.001
    schedules = ["multistep", "linear", "cosine"]
    plt.figure(figsize=(10,6))
    for sch in schedules:
        lrs = simulate_lr_curve(sch, lr)
        plt.plot(lrs, label=sch)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedules")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lr_schedules.png")  # 已保存到当前目录
    plt.show()