import os
from typing import Callable, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from .cifar_dataset import CIFAR_10, CIFAR_100


def get_transforms(normalize: bool = True, resize: Optional[Tuple[int, int]] = None, to_tensor: bool = True):
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    if to_tensor:
        transform_list.append(transforms.ToTensor())
    if normalize:
        # 默认均值和方差，适用于ImageNet，如有需要可参数化
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def get_dataset(
    data_dir: str,
    dataset_name: str = "CIFAR10",
    train: bool = True,
    transform: Optional[Callable] = None,
) -> Dataset:
    dataset_name = dataset_name.upper()
    if dataset_name == "CIFAR10":
        return CIFAR_10(root=data_dir, train=train, transform=transform)
    elif dataset_name == "CIFAR100":
        return CIFAR_100(root=data_dir, train=train, transform=transform)
    elif dataset_name == "CINIC10":
        pass
    elif dataset_name == "TINYIMAGENET":
        pass
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_dataloader(
    data_dir: str,
    dataset_name: str = "CIFAR10",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    train: bool = True,
    resize: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    custom_transform: Optional[Callable] = None,
) -> DataLoader:
    if custom_transform is not None:
        transform = custom_transform
    else:
        transform = get_transforms(normalize=normalize, resize=resize)
    dataset = get_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        train=train,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# 调试
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    CIFAR10_ROOT_DIR = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
    CIFAR100_ROOT_DIR = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-100/raw/"
    train_loader = get_dataloader(
        data_dir=CIFAR100_ROOT_DIR,
        dataset_name="CIFAR100",
        batch_size=64,
        shuffle=True,
        train=True,
        resize=(32, 32),
        normalize=True
    )
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        # 可视化第一张图片
        img = images[0]
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img.numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Label: {labels[0].item()}")
        plt.axis('off')
        plt.show()
        break
