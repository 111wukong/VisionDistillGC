import os
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.optimizer import Adam
from paddle.regularizer import L2Decay
from sklearn.metrics import classification_report, confusion_matrix
import visualdl

def train_model(model, teacher_model, train_loader, val_loader, criterion, num_epochs=50, fold=0, device='cpu'):
    """
    训练模型并进行验证，包含知识蒸馏
    """
    # 定义优化器
    base_lr = 0.0001
    optimizer = Adam(
        learning_rate=base_lr,  # 直接使用基准学习率
        parameters=model.parameters(),
        weight_decay=L2Decay(0.0001)
    )

    # 初始化VisualDL
    log_writer = visualdl.LogWriter(logdir=f'./logs/fold_{fold}')

    # 记录最佳验证准确率
    best_val_acc = 0.0
    patience = 10  # 提前停止的耐心值
    patience_counter = 0
    warmup_epochs = 5

    for epoch in range(num_epochs):
        # 重新初始化手动准确率变量
        train_correct = 0
        train_total = 0

        # 学习率预热和余弦退火
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            cosine_epoch = epoch - warmup_epochs
            total_cosine_epochs = num_epochs - warmup_epochs
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * cosine_epoch / total_cosine_epochs))

        optimizer.set_lr(lr)  # 手动设置学习率
        print(f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {lr:.6f}')

        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_id, data in enumerate(train_loader):
            images, labels = data
            images = images.astype('float32')
            labels = labels.astype('int64')

            # 将数据移动到设备
            images = images.to(device)
            labels = labels.to(device)

            # 添加 Mixup 数据增强
            if epoch >= warmup_epochs:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = paddle.randperm(images.shape[0])
                mixed_images = lam * images + (1 - lam) * images[index]
                labels_a = labels
                labels_b = labels[index]
            else:
                mixed_images = images
                lam = 1.0
                labels_a = labels
                labels_b = labels

            # 学生模型输出
            student_logits = model(mixed_images)

            # 教师模型输出（不需要梯度）
            with paddle.no_grad():
                teacher_logits = teacher_model(images)

            # 确保教师和学生输出维度匹配
            if student_logits.shape != teacher_logits.shape:
                raise ValueError(f"教师模型输出形状 {teacher_logits.shape} 与学生模型输出形状 {student_logits.shape} 不匹配。")

            # 计算蒸馏损失
            loss = criterion(student_logits, teacher_logits, labels_a)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            train_loss += loss.item()

            # 计算预测结果
            preds = paddle.argmax(student_logits, axis=1)

            # 更新手动准确率
            if epoch >= warmup_epochs:
                train_correct += (lam * (preds == labels_a).astype('float32') + (1 - lam) * (preds == labels_b).astype('float32')).sum().item()
            else:
                train_correct += (preds == labels_a).astype('float32').sum().item()
            train_total += labels_a.shape[0]

            if batch_id % 5 == 0:
                current_acc = train_correct / train_total if train_total > 0 else 0
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_id}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {current_acc:.4f}')

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total if train_total > 0 else 0

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with paddle.no_grad():
            for images, labels in val_loader:
                images = images.astype('float32')
                labels = labels.astype('int64')

                # 将数据移动到设备
                images = images.to(device)
                labels = labels.to(device)

                # 学生模型输出
                logits = model(images)

                # 教师模型输出
                teacher_logits = teacher_model(images)

                # 确保教师和学生输出维度匹配
                if logits.shape != teacher_logits.shape:
                    raise ValueError(f"教师模型输出形状 {teacher_logits.shape} 与学生模型输出形状 {logits.shape} 不匹配。")

                # 计算蒸馏损失
                loss = criterion(logits, teacher_logits, labels)

                preds = paddle.argmax(logits, axis=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                val_loss += loss.item()
                correct += (preds == labels).astype('float32').sum().item()
                total += labels.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        manual_acc = correct / total if total > 0 else 0  # 计算手动准确率

        # 记录训练日志
        log_writer.add_scalar(tag='train/loss', step=epoch, value=avg_train_loss)
        log_writer.add_scalar(tag='train/acc', step=epoch, value=avg_train_acc)
        log_writer.add_scalar(tag='val/loss', step=epoch, value=avg_val_loss)
        log_writer.add_scalar(tag='val/acc', step=epoch, value=manual_acc)

        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {manual_acc:.4f}')  # 使用手动准确率

        # 每个 epoch 打印分类报告
        print("\n验证集分类报告:")
        print(classification_report(all_labels, all_preds, target_names=["Harmful", "Kitchen", "Other", "Recyclable"]))

        # 保存最佳模型
        if manual_acc > best_val_acc:
            best_val_acc = manual_acc
            os.makedirs('models_saved', exist_ok=True)
            paddle.save(model.state_dict(), f'models_saved/best_model_fold{fold}.pdparams')
            print(f'最佳模型已保存 (Val Acc: {best_val_acc:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'验证准确率未提升 ({patience_counter}/{patience})')

        # 提前停止
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break