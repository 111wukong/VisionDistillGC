import os
import warnings
import numpy as np
import paddle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

from models import GarbageClassifier, get_teacher_model
from utils import (
    load_and_preprocess_data,
    calculate_class_weights,
    train_transform,
    test_transform,
    DistillationLoss,
    train_model,
    GarbageDataset
)

# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module='paddle.nn.layer.norm')

# 启用动态图模式
paddle.disable_static()

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    paddle.seed(seed)

set_seed(42)

# 定义数据路径和分类标签
data_dir = "./data/data101031"  # 请确保此路径下有对应的分类文件夹
categories = ["Harmful", "Kitchen", "Other", "Recyclable"]  # 实际分类
num_classes = len(categories)  # 4
img_size = (224, 224)

def main():
    # 检查是否有可用 GPU
    device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
    paddle.set_device(device)
    print(f'使用设备: {device}')

    # 加载数据
    print("加载数据...")
    X, y = load_and_preprocess_data(data_dir, categories, img_size)

    if len(X) == 0:
        raise ValueError("没有找到任何有效的图像数据！")

    print(f'数据集总样本数: {len(X)}')
    print(f'类别分布: {np.bincount(y)}')

    # 显示前5张图像及其标签（调试）
    print("\n显示前5张图像及其标签:")
    for i in range(min(5, len(X))):
        img = X[i]
        label = y[i]
        plt.imshow(img)
        plt.title(categories[label])
        plt.axis('off')
        plt.show()

    # 初始化教师模型
    print("加载教师模型...")
    teacher_model = get_teacher_model(num_classes=num_classes, pretrained=True)
    teacher_model.to(device)

    # K-Fold Cross Validation 使用交叉验证
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n开始训练第 {fold+1} fold...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 计算类别权重
        class_weights = calculate_class_weights(y_train)
        print("类别权重:", class_weights.cpu().numpy())

        # 创建数据集实例
        train_dataset = GarbageDataset(X_train, y_train, transform=train_transform)
        val_dataset = GarbageDataset(X_val, y_val, transform=test_transform)

        # 创建数据加载器
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0  # 设置为0以避免多线程问题
        )
        val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0  # 设置为0以避免多线程问题
        )

        # 创建学生模型实例
        model = GarbageClassifier(num_classes=num_classes)
        model = model.to(device)

        # 定义知识蒸馏损失函数
        criterion = DistillationLoss(temperature=4.0, alpha=0.5)

        # 训练模型
        try:
            train_model(model, teacher_model, train_loader, val_loader, criterion, num_epochs=50, fold=fold, device=device)
        except Exception as e:
            print(f"Fold {fold+1} 训练过程中出现错误: {e}")
            continue  # 继续下一个 fold

        # 加载最佳模型进行评估
        best_model_path = f'models_saved/best_model_fold{fold}.pdparams'
        if os.path.exists(best_model_path):
            model.set_state_dict(paddle.load(best_model_path))
            model.eval()
            # 评估最佳模型
            all_preds = []
            all_labels = []
            correct = 0
            total = 0

            with paddle.no_grad():
                for batch_id, (images, labels) in enumerate(val_loader):
                    images = images.astype('float32')
                    labels = labels.astype('int64')
                    # 将数据移动到设备
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    preds = paddle.argmax(logits, axis=1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    # 手动计算准确率
                    correct += (preds == labels).astype('float32').sum().item()
                    total += labels.shape[0]
            fold_acc = correct / total if total > 0 else 0
            fold_scores.append(fold_acc)

            print(f"\n第 {fold+1} fold最终评估:")
            print(f"准确率: {fold_acc:.4f}")
            print("\n分类报告:")
            print(classification_report(all_labels, all_preds, target_names=categories))
            print("\n混淆矩阵:")
            print(confusion_matrix(all_labels, all_preds))
        else:
            print(f"警告: 最佳模型文件 {best_model_path} 不存在。")

    # 打印平均性能
    if fold_scores:
        print("\n交叉验证结果:")
        print(f"平均准确率: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    else:
        print("\n没有成功完成任何 fold 的训练和评估。")

if __name__ == '__main__':
    main()