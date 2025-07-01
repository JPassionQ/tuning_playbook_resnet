import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class CINIC10Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): 数据集根目录，包含train/test/val子目录
            split (str): 'train', 'test', 或 'val'
            transform (callable, optional): 应用于样本的变换
        """
        assert split in ['train', 'valid', 'test'], "split 必须为 'train', 'val', 或 'test'"
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        # 读取类别
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                self.classes.append(class_name)
                for fname in os.listdir(class_path):
                    if fname.endswith('.png') or fname.endswith('.jpg'):
                        self.samples.append((os.path.join(class_path, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_classes(self):
        return self.classes

    def get_class_to_idx(self):
        return self.class_to_idx

if __name__ == '__main__':
    # 数据集测试:
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

    data_root = '/home/jingqi/DeepLearningWorkshop/dataset/CINIC-10'  # 修改为你的数据集路径
    transform = transforms.Compose([transforms.ToTensor()])

    for split in ['train', 'valid', 'test']:
        dataset = CINIC10Dataset(data_root, split=split, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        print(f"{split} set: {len(dataset)} samples, {len(dataset.get_classes())} classes")
        # 取一个batch测试
        images, labels = next(iter(dataloader))
        print(f"{split} batch shape: {images.shape}, labels: {labels[:5]}")

        # 随机可视化一张图片
        rand_idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[rand_idx]
        img_np = img.numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
        plt.figure()
        plt.imshow(img_np)
        plt.title(f"{split} label: {dataset.get_classes()[label]}")
        plt.axis('off')
        print(f"{split} sample label: {label} ({dataset.get_classes()[label]})")
    plt.show()

