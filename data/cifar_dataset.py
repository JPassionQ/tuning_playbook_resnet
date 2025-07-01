import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class CIFAR_10(Dataset):
    base_folder = "cifar-10-batches-py"
    train_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    test_list = [
        "test_batch",
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        self.root_dir = root_dir
        self.split = split  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.classes = []
        if self.split == "train":
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root_dir, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"]) # cifar-10
                else:
                    self.targets.extend(entry["fine_labels"]) # cifar-100

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root_dir, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # 这样做是为了与所有其他数据集保持一致
        # 返回一个 PIL 图像
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

class CIFAR_100(CIFAR_10):

    base_folder = "cifar-100-python"
    train_list = [
        "train",
    ]

    test_list = [
        "test",
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names"
    }

if __name__ == "__main__":
    CIFAR10_root_dir_DIR = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-10/raw/"
    CIFAR100_root_dir_DIR = "/home/jingqi/DeepLearningWorkshop/dataset/CIFAR-100/raw/"
    import matplotlib.pyplot as plt

    # 测试 CIFAR10
    cifar10 = CIFAR_10(root_dir=CIFAR10_root_dir_DIR, split="train")
    img10, label10 = cifar10[0]
    plt.figure()
    plt.title(f"CIFAR10 Label: {cifar10.classes[label10]}")
    plt.imshow(img10)
    plt.axis('off')
    print("CIFAR10 label:", cifar10.classes[label10])

    # 测试 CIFAR100
    cifar100 = CIFAR_100(root_dir=CIFAR100_root_dir_DIR, split="test")
    img100, label100 = cifar100[0]
    plt.figure()
    plt.title(f"CIFAR100 Label: {cifar100.classes[label100]}")
    plt.imshow(img100)
    plt.axis('off')
    print("CIFAR100 label:", cifar100.classes[label100])

    plt.show()

