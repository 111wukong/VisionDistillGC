import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset
import warnings

def calculate_class_weights(labels):
    """
    计算类别权重，处理类别不平衡问题
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / np.sum(weights) * len(unique_labels)
    return paddle.to_tensor(weights, dtype='float32')

def load_and_preprocess_data(data_dir, categories, img_size):
    """
    加载并预处理数据
    """
    features = []
    labels = []

    for label, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        if not os.path.exists(folder_path):
            warnings.warn(f"警告: 目录 {folder_path} 不存在")
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            if not os.path.isfile(img_path):
                continue  # 跳过非文件项
            try:
                img = cv2.imread(img_path)
                if img is None:
                    warnings.warn(f"警告: 无法读取图像 {img_path}")
                    continue

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = cv2.resize(img, img_size)

                if img.shape[2] != 3:
                    warnings.warn(f"警告: 图像 {img_path} 不是3通道，跳过。")
                    continue

                features.append(img)
                labels.append(label)
            except Exception as e:
                warnings.warn(f"处理图像 {img_path} 时出错: {str(e)}")

    return np.array(features, dtype=np.uint8), np.array(labels, dtype=np.int64)

class GarbageDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        super(GarbageDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.features)