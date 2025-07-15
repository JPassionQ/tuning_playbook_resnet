import yaml
import os
import numpy as np
from hyperparameter_search import quassi_random_search

research_on_activation_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_activation/round1"

model_layer = 18
train_batch = 128
eval_batch = 256
eval_steps = 50
log_steps = 10
epochs = 20
num_classes = 10
optimizer_type = 'Adam'
lr = 1e-3
weight_decay = 1e-4
dataset_path = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_activation/round1"

activation_type = [
    "relu",
    "leaky_relu",
    "gelu",
    "tanh",
    "silu",
]
# def generate_config(idx: int=0, regularization_type=None, dropout=0, weight_decay=0):
#     if regularization_type == "dropout":
#         config = {
#             "model_layer": model_layer,
#             "num_classes": num_classes,
#             "train_batch": train_batch,
#             "eval_batch": eval_batch,
#             "eval_steps": eval_steps,
#             "log_steps": log_steps,
#             "epochs": epochs,
#             "optimizer_type": optimizer_type,
#             "lr": lr,
#             "dropout": dropout,
#             "dataset_path": dataset_path,
#             "res_path": os.path.join(res_path, regularization_type, f"config_{idx}")
#         }
#     elif regularization_type == "weight_decay":
#         config = {
#             "model_layer": model_layer,
#             "num_classes": num_classes,
#             "train_batch": train_batch,
#             "eval_batch": eval_batch,
#             "eval_steps": eval_steps,
#             "log_steps": log_steps,
#             "epochs": epochs,
#             "optimizer_type": optimizer_type,
#             "lr": lr,
#             "weight_decay": weight_decay,
#             "dataset_path": dataset_path,
#             "res_path": os.path.join(res_path, regularization_type, f"config_{idx}")
#         }
#     with open(os.path.join(research_on_regularization_path,regularization_type, f"config_{idx}.yaml"), "w") as f:
#         yaml.dump(config, f)

if __name__ == "__main__":
    for idx, activation in enumerate(activation_type):
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
            "activation": activation,
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, f"config_{idx}")
        }
        with open(os.path.join(research_on_activation_path, f"config_{idx}.yaml"), "w") as f:
            yaml.dump(config, f)

