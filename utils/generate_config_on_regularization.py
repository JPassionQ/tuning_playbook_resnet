import yaml
import os
import numpy as np
from hyperparameter_search import quassi_random_search

n_trials = 24
lr_range = (7e-4, 1.3e-4)
research_on_regularization_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_regularization/round1"

model_layer = 18
train_batch = 128
eval_batch = 256
eval_steps = 50
log_steps = 10
epochs = 20
num_classes = 10
optimizer_type = 'Adam'
lr = 1e-3
dataset_path = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_regularization/round1"

dropout_range = [0.0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
weight_decay_range = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

def generate_config(idx: int=0, regularization_type=None, dropout=0, weight_decay=0):
    if regularization_type == "dropout":
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_steps,
            "log_steps": log_steps,
            "epochs": epochs,
            "optimizer_type": optimizer_type,
            "lr": lr,
            "dropout": dropout,
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, regularization_type, f"config_{idx}")
        }
    elif regularization_type == "weight_decay":
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_steps,
            "log_steps": log_steps,
            "epochs": epochs,
            "optimizer_type": optimizer_type,
            "lr": lr,
            "weight_decay": weight_decay,
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, regularization_type, f"config_{idx}")
        }
    with open(os.path.join(research_on_regularization_path,regularization_type, f"config_{idx}.yaml"), "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    for i, v in enumerate(dropout_range):
        generate_config(i, regularization_type="dropout", dropout=v)
    for i, v in enumerate(weight_decay_range):
        generate_config(i, regularization_type="weight_decay", weight_decay=v)

