import paddle
import paddle.nn as nn
from paddle.vision.models import mobilenet_v2

class GarbageClassifier(nn.Layer):
    def __init__(self, num_classes):
        super(GarbageClassifier, self).__init__()
        # 加载预训练的 MobileNetV2
        backbone = mobilenet_v2(pretrained=True)

        # 冻结部分特征提取层（根据需要调整冻结层数）
        for param in backbone.features[:14].parameters():  # 冻结前14层，根据需要调整
            param.trainable = False

        self.features = backbone.features

        # 改进的分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.last_channel, 1024),
            nn.BatchNorm1D(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)  # 确保输出为 num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
