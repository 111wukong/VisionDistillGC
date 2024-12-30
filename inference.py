import os
import argparse
import numpy as np
import paddle
import cv2
import matplotlib.pyplot as plt

# 导入项目中的模型与预处理
from models import GarbageClassifier
from utils.transforms import test_transform
from utils.data_utils import load_and_preprocess_data

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="VisionDistillGC Inference Script")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved .pdparams file of the student model.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to a single image or a directory containing multiple images.")
    parser.add_argument("--categories", nargs="+", default=["Harmful", "Kitchen", "Other", "Recyclable"],
                        help="List of category names. Default matches the trained model categories.")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for inference if available.")
    parser.add_argument("--show", action="store_true",
                        help="Whether to display the image and prediction result.")
    args = parser.parse_args()
    return args

def load_model(model_path, num_classes=4, use_gpu=False):
    """
    加载学生模型权重并返回模型实例
    """
    device = "gpu" if (use_gpu and paddle.is_compiled_with_cuda()) else "cpu"
    paddle.set_device(device)

    # 创建学生模型实例（需与训练时保持一致）
    model = GarbageClassifier(num_classes=num_classes)
    
    # 加载指定的参数文件
    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)
    
    model.eval()
    return model

def preprocess_image(image_path):
    """
    对单张图像进行预处理，返回转换后的 Paddle Tensor
    """
    # 读取图像 (BGR 通道)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    # 如果是单通道，则转换为 3 通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用测试集同样的变换
    transformed_img = test_transform(img)  # 返回的是 Paddle Tensor (C, H, W)
    
    # 增加一个 batch 维度 (N, C, H, W)
    transformed_img = transformed_img.unsqueeze(0)
    return transformed_img

def predict_single_image(model, image_path, categories, device="cpu", show=False):
    """
    对单张图像进行推理，并可视化结果
    """
    # 预处理
    img_tensor = preprocess_image(image_path)
    
    # 将数据放到指定设备
    img_tensor = img_tensor.astype("float32").to(device)
    
    # 推理
    logits = model(img_tensor)
    probs = paddle.nn.functional.softmax(logits, axis=1)
    preds = paddle.argmax(probs, axis=1).numpy()[0]
    confidence = float(paddle.max(probs).numpy()[0])

    predicted_label = categories[preds]
    
    if show:
        # 可视化结果
        plt.figure()
        img_bgr = cv2.imread(image_path)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"Pred: {predicted_label}, Conf: {confidence:.2f}")
            plt.axis("off")
            plt.show()
    
    return predicted_label, confidence

def main():
    args = parse_args()
    
    # 设备选择
    device = "gpu" if (args.use_gpu and paddle.is_compiled_with_cuda()) else "cpu"
    paddle.set_device(device)
    print(f"Using device: {device}")

    # 加载模型
    num_classes = len(args.categories)
    model = load_model(args.model_path, num_classes=num_classes, use_gpu=args.use_gpu)
    print(f"Model loaded from: {args.model_path}")

    # 判断是单张图像还是文件夹
    if os.path.isdir(args.image_path):
        # 批量预测文件夹中的所有图像
        all_images = [f for f in os.listdir(args.image_path)
                      if os.path.splitext(f)[-1].lower() in [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]]
        
        if not all_images:
            print(f"目录 {args.image_path} 下未找到可用图像。")
            return
        
        for img_file in all_images:
            img_path = os.path.join(args.image_path, img_file)
            pred_label, conf = predict_single_image(
                model, 
                img_path, 
                categories=args.categories, 
                device=device,
                show=args.show
            )
            print(f"[{img_file}] --> Predicted: {pred_label}, Confidence: {conf:.2f}")
    
        print(f"\nProcessed {len(all_images)} images in directory: {args.image_path}")
    
    else:
        # 推断单张图像
        if not os.path.isfile(args.image_path):
            print(f"{args.image_path} 并非有效的文件路径。")
            return
        
        pred_label, conf = predict_single_image(
            model, 
            args.image_path, 
            categories=args.categories, 
            device=device,
            show=args.show
        )
        print(f"\n[{args.image_path}] --> Predicted: {pred_label}, Confidence: {conf:.2f}")

if __name__ == "__main__":
    main()
