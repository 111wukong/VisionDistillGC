# 基于 PaddlePaddle 的知识蒸馏多类别视觉分类模型

**VisionDistillGC** 是一个基于 PaddlePaddle 深度学习框架的多类别图像分类项目，旨在通过知识蒸馏的方式训练一个高效、轻量级的学生模型。在该项目中，ResNet50 作为教师模型，被预训练于 ImageNet 等大型数据集，并在训练过程中保持参数冻结，为学生模型提供稳定且具有丰富特征表示的软标签。学生模型采用 MobileNetV2 的网络结构，在保持较小模型规模的同时，具备良好的特征提取能力和较高的推理速度。

项目主要针对垃圾分类场景，可按照用户需求自定义若干类别标签。通过引入知识蒸馏技术，学生模型在训练阶段同时接收来自教师模型的软标签与真实数据的硬标签，避免了仅使用真实标签时可能出现的过拟合和梯度震荡问题，从而得到性能与效率兼备的模型。此外，为了进一步提升模型的泛化能力与鲁棒性，项目中使用了多种数据增强策略，包括随机翻转、随机裁剪、随机旋转、颜色扰动与随机擦除等，让模型在面对真实场景中的噪声与干扰时更具适应性。

在训练方面，项目采用 K-Fold 交叉验证，尽可能充分利用数据并减少随机划分带来的不稳定性。训练过程中还引入了 Mixup 数据增强、学习率预热与余弦退火等技巧，平衡了模型在早期训练阶段的收敛速度与后期微调阶段的性能提升。项目集成了 VisualDL 日志记录工具，用户可以通过实时可视化曲线观察训练与验证过程中损失和准确率的动态变化，并在训练结束后查看分类报告与混淆矩阵等详细评估指标。若验证集准确率达到新的峰值，系统会自动保存相应的模型参数，方便进一步推断或微调。

在使用 **VisionDistillGC** 前，用户需准备包含不同垃圾类别的图像数据集，并按照指定文件夹结构进行整理。项目默认使用图像文件（如 JPG、PNG）进行训练，并利用 OpenCV 进行图像读取与预处理。若需在部署阶段对模型进行推断，用户可结合 PaddlePaddle 提供的 Serving 工具或其它部署方案，将训练完成的学生模型应用到在线或离线推断服务中，实现高效的垃圾分类或更多类似的多类别识别任务。

**VisionDistillGC** 通过融合知识蒸馏与多种深度学习技巧，在保持较高准确率的同时大幅降低了模型推理的复杂度，适用于实时响应或计算资源受限的场景。无论是科研还是实际应用，用户都能在此项目的基础上，定制并扩展适用于自身需求的多类别视觉任务。

## 目录结构

``` 
VisionDistillGC/
│
├── data/
│   └── data101031/          # 数据集目录，包含各分类文件夹
│
├── models/
│   ├── __init__.py
│   ├── student_model.py     # 学生模型定义
│   └── teacher_model.py     # 教师模型定义
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py        # 数据加载和预处理
│   ├── loss.py              # 知识蒸馏损失函数
│   ├── train_utils.py       # 训练和验证函数
│   └── transforms.py        # 数据增强和变换
│
├── logs/                    # VisualDL 日志目录
│
├── models_saved/            # 保存的模型参数
│
├── notebooks/               # Jupyter Notebook 文件
│
├── main.py                  # 主训练脚本
├── requirements.txt         # 项目依赖
├── README.md                # 项目说明文档
└── .gitignore              

```

###   1. **教师模型（ResNet50）** 

教师模型使用 ResNet50 架构，预训练于 ImageNet 数据集。模型的参数被冻结，仅用于生成软标签，辅助学生模型的训练。

### 2. **学生模型（MobileNetV2）** 

学生模型采用轻量级的 MobileNetV2 架构，经过改进的分类头以适应特定的分类任务。部分特征提取层被冻结，以减少训练时间并提高性能。 

### 3. **知识蒸馏** 

通过知识蒸馏技术，结合教师模型的软标签和学生模型的预测，优化学生模型的训练过程，提高其分类性能。 

### 4. **数据增强** 

使用多种数据增强技术（如随机翻转、旋转、裁剪、颜色抖动、随机擦除等）提升模型的泛化能力。

### 5. **交叉验证** 

采用 K-Fold 交叉验证（默认 5 fold），确保模型的稳健性和泛化性能。 

### 6. **日志记录** 

利用 VisualDL 记录训练过程中的损失和准确率，便于后续分析和调试。 

## 快速开始 

### 环境配置 

确保您已安装 [PaddlePaddle](https://www.paddlepaddle.org.cn/) 和其他必要的依赖。推荐使用虚拟环境管理工具（如 `venv` 或 `conda`）来隔离项目依赖。

1. **克隆仓库**

``` 
git clone:https://github.com/111wukong/VisionDistillGC.git
cd VisionDistillGC
```

2. **创建并激活虚拟环境**

```
# 使用 conda
conda create -n env python=3.10
conda activate env
```

3. **安装依赖**

```
pip install -r requirements.txt
```

4.**数据准备**

```
将数据集按照类别放置在 `data` 目录下，确保每个类别有相应的子文件夹。
```

5.**运行训练**

```
python main.py
```
训练过程中，日志将保存在 logs/ 目录，最佳模型参数将保存在 models_saved/ 目录。

6.**监控训练过程**

```
visualdl --logdir=./logs --port=8040
```

然后在浏览器中访问 http://localhost:8040 查看训练过程中的损失和准确率变化。

7.**执行推理命令**

```
# 推理单张图像
python inference.py \
    --model_path ./models_saved/best_model_fold0.pdparams \
    --image_path ./test.jpg \
    --categories Harmful Kitchen Other Recyclable \
    --use_gpu \
    --show
```
- 参数说明：

--model_path：训练完成的模型权重文件路径。
--image_path：可以是单张图像的路径，也可以是目录。
--categories：指定类别列表，需与训练时的类别顺序一致。
--use_gpu：在有 GPU 环境的情况下使用 GPU 进行推理。
--show：显示预测图像和推理结果，可选。
- 批量推理

如果 --image_path 指向一个文件夹，则脚本会对目录下所有符合指定扩展名的图片进行推理，输出预测结果和置信度。

### 使用本项目您可能需要关注的文件

```
1.main.py

数据目录：
修改 data_dir = "./data/data101031" 为您实际的数据路径。
类别列表：修改 categories = ["Harmful", "Kitchen", "Other", "Recyclable"] 为实际数据中的类别名称。
同时更新 num_classes = len(categories)。
超参数：如需修改交叉验证折数、批量大小（batch_size）、训练轮数（num_epochs）等，可在此文件中进行调整。
调试与可视化：如果您想要在运行前查看更多或更少的图像，可以修改对应代码片段（例如“显示前5张图像及其标签”）。
日志与模型保存路径：该项目默认将日志写入 logs/、最佳模型存入 models_saved/ 文件夹，如果需要自定义，可以在 main.py 中更改相应路径。
2.models/student_model.py 和 models/teacher_model.py

模型类别输出数：在这两个文件中，确保输出层的维度与 num_classes 一致。如果使用自定义的模型或者需要修改输出层的大小，需要手动更改相关参数。
冻结层策略：在 student_model.py 中，默认冻结了 backbone.features[:14] 的参数。您可以根据实际需求或实验结果，调整冻结的层数。
3.utils/transforms.py

图像尺寸：该文件中设置的默认图像尺寸为 224 x 224。如果您的数据尺寸不同，或需要更改图像尺寸，可在此文件中进行调整。
数据增强策略：可以增删或更改增强操作，例如翻转、裁剪、颜色抖动、随机擦除等，以适配您的数据特点。
4.utils/data_utils.py

数据格式和加载：该文件默认使用 cv2 加载图像，并期望文件名不含其他特殊字符。若数据中包含视频文件或其他无效文件，需要在此文件中进一步筛选或跳过。
自定义警告或错误处理：如果您需要对缺失文件、文件读取错误等情况进行更强的错误处理或日志记录，可以在这里自定义对应的逻辑。
5.utils/train_utils.py

训练配置：调整学习率（base_lr）、余弦退火的策略、早停耐心值（patience）等超参数。
损失函数与 Mixup：若需更改 Mixup 强度、不同数据增强策略或修改蒸馏损失的温度系数、权重系数（如 alpha 值），可在此进行设置。
VisualDL 日志目录：如需改变日志存储位置，可在 log_writer = visualdl.LogWriter(logdir=...) 处进行调整。
6.utils/loss.py

知识蒸馏损失函数参数：如果您需要调节温度系数（temperature）或交叉熵与 KL 散度之间的权重系数（alpha），可在该文件中进行自定义或封装成可配置参数。
```

