import paddle
import paddle.nn as nn
from paddle.vision.models import resnet50

def get_teacher_model(num_classes, pretrained=True):
    """
    获取教师模型，并确保其输出为指定的类别数
    """
    try:
        teacher = resnet50(pretrained=pretrained, num_classes=num_classes)
    except TypeError:
        teacher = resnet50(pretrained=pretrained)
        in_features = teacher.fc.weight.shape[1]  # 通常为2048
        teacher.fc = nn.Linear(in_features, num_classes)
    
    # 冻结教师模型的参数
    for param in teacher.parameters():
        param.trainable = False

    teacher.eval()  # 设置为评估模式
    return teacher
