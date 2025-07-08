import yaml
import os
import numpy as np
from hyperparameter_search import quassi_random_search

# 每一组研究的试验次数
n_trails = 16
# 每一个超参数的取值范围
lr_sgd = (0.07,0.09)
lr_sgd_with_momentum = (0.02,0.03)
lr_nesterov = (0.015,0.025)
lr_adam = (0.0008,0.0012)
lr_nadam = (0.0018, 0.0022)
lr_adamW = (0.0008, 0.0012)
lr_rmsprop = (0.008,0.012)

momentum = (0.980, 0.999)

research_on_optimizer_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_optimizer/round3/"
optimizer_type = [
    "sgd",
    "sgd_with_momentum",
    "Nestrov",
    "Adam",
    "NAdam",
    "AdamW",
    "RMSprop",
]
model_layer = 18
train_batch = 128
eval_batch = 256
eval_step= 50
epochs = 2
num_classes = 10
dataset_path = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round3/"

def make_dirs(path_to_make):
    for opt in optimizer_type:
        path = os.path.join(path_to_make, opt)
        if not os.path.exists(path):
            os.makedirs(path)
# sgd配置文件
def sgd_config():
    param_ranges = [lr_sgd]
    lr_sample = quassi_random_search(param_ranges, n_trails)
    for i in range(n_trails):
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_step,
            "epochs": epochs,
            "optimizer_type": "sgd",
            "lr": float(lr_sample[i][0]),
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, "sgd", f"config_{i}")
        }
        with open(os.path.join(research_on_optimizer_path, "sgd", f"config_{i}.yaml"), "w") as f:
            yaml.dump(config, f)

# sgd with momentum配置文件
def sgd_with_momentum_config():
    param_ranges = [lr_sgd_with_momentum, momentum]
    lr_sample = quassi_random_search(param_ranges, n_trails)
    for i in range(n_trails):
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_step,
            "epochs": epochs,
            "optimizer_type": "sgd_with_momentum",
            "lr": float(lr_sample[i][0]),
            "momentum": float(lr_sample[i][1]),
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, "sgd_with_momentum", f"config_{i}")
        }
        with open(os.path.join(research_on_optimizer_path, "sgd_with_momentum", f"config_{i}.yaml"), "w") as f:
            yaml.dump(config, f)

# sgd with Nesterov配置文件
def sgd_with_nesterov_config():
    param_ranges = [lr_nesterov, momentum]
    lr_sample = quassi_random_search(param_ranges, n_trails)
    for i in range(n_trails):
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_step,
            "epochs": epochs,
            "optimizer_type": "Nestrov",
            "lr": float(lr_sample[i][0]),
            "momentum": float(lr_sample[i][1]),
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, "Nestrov", f"config_{i}")
        }
        with open(os.path.join(research_on_optimizer_path, "Nestrov", f"config_{i}.yaml"), "w") as f:
            yaml.dump(config, f)
# Adam配置文件
def Adam_config():
    param_ranges = [lr_adam]
    lr_sample = quassi_random_search(param_ranges, n_trails)
    for i in range(n_trails):
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_step,
            "epochs": epochs,
            "optimizer_type": "Adam",
            "lr": float(lr_sample[i][0]),
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, "Adam", f"config_{i}")
        }
        with open(os.path.join(research_on_optimizer_path, "Adam", f"config_{i}.yaml"), "w") as f:
            yaml.dump(config, f)
# NAdam配置文件
def NAdam_config():
    param_ranges = [lr_nadam]
    lr_sample = quassi_random_search(param_ranges, n_trails)
    for i in range(n_trails):
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_step,
            "epochs": epochs,
            "optimizer_type": "NAdam",
            "lr": float(lr_sample[i][0]),
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, "NAdam", f"config_{i}")
        }
        with open(os.path.join(research_on_optimizer_path, "NAdam", f"config_{i}.yaml"), "w") as f:
            yaml.dump(config, f)
# AdamW配置文件
def AdamW_config():
    param_ranges = [lr_adamW]
    lr_sample = quassi_random_search(param_ranges, n_trails)
    for i in range(n_trails):
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_step,
            "epochs": epochs,
            "optimizer_type": "AdamW",
            "lr": float(lr_sample[i][0]),
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, "AdamW", f"config_{i}")
        }
        with open(os.path.join(research_on_optimizer_path, "AdamW", f"config_{i}.yaml"), "w") as f:
            yaml.dump(config, f)
# RMSprop配置文件
def RMSprop_config():
    param_ranges = [lr_rmsprop, momentum]
    lr_sample = quassi_random_search(param_ranges, n_trails)
    for i in range(n_trails):
        config = {
            "model_layer": model_layer,
            "num_classes": num_classes,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "eval_steps": eval_step,
            "epochs": epochs,
            "optimizer_type": "RMSprop",
            "lr": float(lr_sample[i][0]),
            "momentum": float(lr_sample[i][1]),
            "dataset_path": dataset_path,
            "res_path": os.path.join(res_path, "RMSprop", f"config_{i}")
        }
        with open(os.path.join(research_on_optimizer_path, "RMSprop", f"config_{i}.yaml"), "w") as f:
            yaml.dump(config, f)

if __name__ == "__main__":
    make_dirs(path_to_make=research_on_optimizer_path)
    make_dirs(path_to_make=res_path)
    sgd_config()
    sgd_with_momentum_config()
    sgd_with_nesterov_config()
    Adam_config()
    NAdam_config()
    AdamW_config()
    RMSprop_config()
