import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义数据预处理流程
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.permute(1, 0, 2))  # 修正EMNIST图像方向
])

# 加载训练集
train_dataset = datasets.EMNIST(
    root='./data',
    split='byclass',
    train=True,
    transform=transform,
    download=True  # 自动从PyTorch官方源下载
)

# 加载测试集
test_dataset = datasets.EMNIST(
    root='./data',
    split='byclass',
    train=False,
    transform=transform,
    download=True
)

