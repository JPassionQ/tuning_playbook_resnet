import yaml
import os
import numpy as np
from hyperparameter_search import quassi_random_search

# 每一组研究的试验次数
n_trails = 16

augmentation_type = [
    'random_crop',
    'random_horizontal_filp',
    'random_rotate',
    'color_jitter',
    'gaussian_blur',
]

research_on_augmentation_path = "/home/jingqi/DeepLearningWorkshop/recipes/research_on_augmentation/round1"

model_layer = 18
train_batch = 128
eval_batch = 256
eval_step= 50
epochs = 2
num_classes = 10
optimizer_type = 'Adam'
lr = 1e-3
dataset_path = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
res_path = "/home/jingqi/DeepLearningWorkshop/results/research_on_augmentation/round1"

data_augmentation_configs = [
    {
        'random_crop': True,
        'random_horizontal_filp': False,
        'random_rotate': False,
        'color_jitter': False,
        'gaussian_blur': False
    },
    {
        'random_crop': False,
        'random_horizontal_filp': True,
        'random_rotate': False,
        'color_jitter': False,
        'gaussian_blur': False
    },
    {
        'random_crop': False,
        'random_horizontal_filp': False,
        'random_rotate': True,
        'color_jitter': False,
        'gaussian_blur': False
    },
    {
        'random_crop': False,
        'random_horizontal_filp': False,
        'random_rotate': False,
        'color_jitter': True,
        'gaussian_blur': False
    },
    {
        'random_crop': False,
        'random_horizontal_filp': False,
        'random_rotate': False,
        'color_jitter': False,
        'gaussian_blur': True
    },
    {
        'random_crop': True,
        'random_horizontal_filp': True,
        'random_rotate': False,
        'color_jitter': False,
        'gaussian_blur': False
    },
    {
        'random_crop': True,
        'random_horizontal_filp': False,
        'random_rotate': True,
        'color_jitter': False,
        'gaussian_blur': False
    },
    {
        'random_crop': True,
        'random_horizontal_filp': False,
        'random_rotate': False,
        'color_jitter': True,
        'gaussian_blur': False
    },
    {
        'random_crop': True,
        'random_horizontal_filp': False,
        'random_rotate': False,
        'color_jitter': False,
        'gaussian_blur': True
    },
    {
        'random_crop': False,
        'random_horizontal_filp': True,
        'random_rotate': True,
        'color_jitter': False,
        'gaussian_blur': False
    },
    {
        'random_crop': False,
        'random_horizontal_filp': True,
        'random_rotate': False,
        'color_jitter': True,
        'gaussian_blur': False
    },
    {
        'random_crop': False,
        'random_horizontal_filp': True,
        'random_rotate': False,
        'color_jitter': False,
        'gaussian_blur': True
    },
    {
        'random_crop': True,
        'random_horizontal_filp': True,
        'random_rotate': True,
        'color_jitter': False,
        'gaussian_blur': False
    },
    {
        'random_crop': True,
        'random_horizontal_filp': True,
        'random_rotate': False,
        'color_jitter': True,
        'gaussian_blur': False
    },
    {
        'random_crop': True,
        'random_horizontal_filp': True,
        'random_rotate': False,
        'color_jitter': False,
        'gaussian_blur': True
    },
    {
        'random_crop': True,
        'random_horizontal_filp': True,
        'random_rotate': True,
        'color_jitter': True,
        'gaussian_blur': True
    },
]


def generate_config(data_augmentation: dict=None, idx: int=0):
    config = {
        "model_layer": model_layer,
        "num_classes": num_classes,
        "train_batch": train_batch,
        "eval_batch": eval_batch,
        "eval_steps": eval_step,
        "epochs": epochs,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "data_augmentation": data_augmentation,
        "dataset_path": dataset_path,
        "res_path": os.path.join(res_path, "AdamW", f"config_{idx}")
    }
    with open(os.path.join(research_on_augmentation_path, f"config_{idx}.yaml"), "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    for idx, data_augmentation in enumerate(data_augmentation_configs):
        generate_config(data_augmentation, idx=idx)

