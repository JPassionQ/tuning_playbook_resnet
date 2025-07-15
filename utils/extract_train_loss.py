import os
import random
import numpy as np

from plotter import plot_loss_curve
from plotter import plot_accuracy_curve

import re

def main():
    train_pattern = re.compile(r"steps \[(\d+)/\d+\] Train Loss: ([\d\.]+)")
    val_pattern = re.compile(r"step \[(\d+)/\d+\] \| Val Acc: ([\d\.]+)% \| Val Loss: ([\d\.]+)")
    res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_activation/round1"
    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith("log"):
                train_log_path = os.path.join(root, file)
                loss_curve_path = os.path.join(root, "loss_curve.png")
                accuracy_curve_path = os.path.join(root, "accuracy_curve.png")
                train_loss = []
                valid_loss = []
                valid_acc = []
                with open(train_log_path, "r") as f:
                    for line in f:
                        m = train_pattern.match(line)
                        if m:
                            step = int(m.group(1))
                            loss = float(m.group(2))
                            if step % 10 == 0:
                                train_loss.append(loss)
                        m = val_pattern.match(line)
                        if m:
                            step = int(m.group(1))
                            val_acc = float(m.group(2))
                            val_loss = float(m.group(3))
                            valid_loss.append(val_loss)
                            valid_acc.append(val_acc)
                plot_loss_curve(7800, train_loss,valid_loss, save_path=loss_curve_path)
                plot_accuracy_curve(range(1, len(valid_acc)+1), valid_acc, save_path=accuracy_curve_path)

if __name__ == "__main__":
    main()
