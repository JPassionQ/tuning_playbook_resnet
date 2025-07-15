import os
import random
import numpy as np

from plotter import plot_loss_curve, plot_accuracy_curve, plot_loss_compare_curve, plot_acc_compare_curve

import re

def main():
    train_pattern = re.compile(r"steps \[(\d+)/\d+\] Train Loss: ([\d\.]+)")
    val_pattern = re.compile(r"step \[(\d+)/\d+\] \| Val Acc: ([\d\.]+)% \| Val Loss: ([\d\.]+)")
    res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_augmentation/round4/quadruple_aug"
    train_loss_png_path = os.path.join(res_path, "train_loss_compare.png")
    val_loss_png_path = os.path.join(res_path, "val_loss_compare.png")
    val_acc_png_path = os.path.join(res_path, "val_acc_compare.png")
    aug_types = [
        "baseline",
        "crop+flip+color+gaussian",
        "crop+flip+rotate+color",
        "crop+flip+rotate+gaussian",
        "full"
    ]
    aug_to_train_loss = {}
    aug_to_valid_loss = {}
    aug_to_valid_acc = {}
    total_steps = 15640
    for type_ in aug_types:
        log_path = os.path.join(res_path, f"{type_}.log")
        train_loss = []
        valid_loss = []
        valid_acc = []
        with open(log_path, "r") as f:
            for line in f:
                m = train_pattern.match(line)
                if m:
                    step = int(m.group(1))
                    loss = float(m.group(2))
                    if step % 100 == 0 and step <= total_steps:
                        train_loss.append(loss)
                m = val_pattern.match(line)
                if m:
                    step = int(m.group(1))
                    val_acc = float(m.group(2))
                    val_loss = float(m.group(3))
                    if step <= total_steps:
                        valid_loss.append(val_loss)
                        valid_acc.append(val_acc)
        aug_to_train_loss[type_] = train_loss
        if type_ != "flip+color" and type_ != "crop+color":
            aug_to_valid_loss[type_] = valid_loss
        aug_to_valid_acc[type_] = valid_acc
    plot_loss_compare_curve(total_steps, aug_to_train_loss, save_path=train_loss_png_path, loss_type="Train")
    plot_loss_compare_curve(total_steps, aug_to_valid_loss, save_path=val_loss_png_path, loss_type="Valid")
    plot_acc_compare_curve(total_steps, aug_to_valid_acc, save_path=val_acc_png_path)

if __name__ == "__main__":
    main()
