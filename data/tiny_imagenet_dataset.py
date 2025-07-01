import os
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'val'
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.classes = []
        self.wnids = []
        self.wnid_to_classname = {}
        # 加载类别ID列表
        wnids_path = f"{root_dir}/wnids.txt"
        with open(wnids_path, "r") as f:
            self.wnids = [line.strip() for line in f if line.strip()]
        # 加载类别ID到类别名的映射
        words_path = f"{root_dir}/words.txt"
        with open(words_path, "r") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid, classname = parts
                    self.wnid_to_classname[wnid] = classname
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.idx_to_class = {idx: wnid for wnid, idx in self.class_to_idx.items()}
        self.classes = [self.wnid_to_classname[wnid] for wnid in self.wnids]
        self._load_dataset()
    def _load_dataset(self):
        if self.split == 'train':
            train_dir = os.path.join(self.root_dir, 'train')
            classes = sorted(os.listdir(train_dir))
            for cls_name in classes:
                img_dir = os.path.join(train_dir, cls_name, 'images')
                img_files = os.listdir(img_dir)
                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    self.data.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])
        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, 'val')
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            img_to_class = {}
            with open(val_annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_to_class[parts[0]] = parts[1]
            classes = sorted(set(img_to_class.values()))
            images_dir = os.path.join(val_dir, 'images')
            for img_file, cls_name in img_to_class.items():
                img_path = os.path.join(images_dir, img_file)
                self.data.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 假设数据集根目录为 '../tiny-imagenet-200'
    dataset_root = "/home/jingqi/DeepLearningWorkshop/dataset/Tiny_ImageNet/raw"
    dataset = TinyImageNetDataset(root_dir=dataset_root, split='train')

    # 读取第一张图片及其标签
    image, label = dataset[535]
    print("标签索引:", label)
    print("标签:", dataset.idx_to_class[label])
    print("类别名:", dataset.classes[label])

    # 可视化图片
    plt.imshow(image)
    plt.title(f"Label: {label} ({dataset.classes[label]})")
    plt.axis('off')
    plt.show()
