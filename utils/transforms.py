import paddle.vision.transforms as T

img_size = (224, 224)

train_transform = T.Compose([
    T.RandomHorizontalFlip(prob=0.5),
    T.RandomVerticalFlip(prob=0.3),
    T.RandomRotation(degrees=30),
    T.RandomCrop(size=img_size, padding=8),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomErasing(prob=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])