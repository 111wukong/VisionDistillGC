import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DistillationLoss(nn.Layer):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # Kullback-Leibler 散度

    def forward(self, student_logits, teacher_logits, labels):
        """
        计算综合损失，包括交叉熵损失和KL散度损失
        """
        # 交叉熵损失
        ce = self.ce_loss(student_logits, labels)

        # 蒸馏损失
        soft_student = F.log_softmax(student_logits / self.temperature, axis=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, axis=1)
        kl = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # 综合损失
        loss = self.alpha * ce + (1. - self.alpha) * kl
        return loss