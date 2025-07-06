import yaml
import os
import numpy as np
from hyperparameter_search import quassi_random_search

# 每一组研究的试验次数
n_trails = 16
# 每一个超参数的取值范围
lr = (0.001, 0.1)
lr_sgd_with_momentum = (0.005, 0.03)
lr_nesterov = (0.001, 0.03)
lr_adam = (0.001, 0.06)
lr_nadam = (0.001, 0.05)
lr_adamW = (0.0006, 0.01)
lr_rmsprop = (0.0005,0.007)

momentum = (0.8, 0.99)

research_on_optimizer_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_optimizer/round2/"
optimizer_type = [
    # "sgd",
    "sgd_with_momentum",
    "Nestrov",
    "Adam",
    "NAdam",
    "AdamW",
    "RMSprop",
]
model_layer = 50
train_batch = 128
eval_batch = 256
eval_step= 50
epochs = 10
num_classes = 10
dataset_path = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_optimizer/round2/"

def make_dirs():
    for opt in optimizer_type:
        path = os.path.join(research_on_optimizer_path, opt)
        if not os.path.exists(path):
            os.makedirs(path)
# sgd配置文件
def sgd_config():
    param_ranges = [lr]
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
    # sgd_config()
    make_dirs()
    sgd_with_momentum_config()
    sgd_with_nesterov_config()
    Adam_config()
    NAdam_config()
    AdamW_config()
    RMSprop_config()
