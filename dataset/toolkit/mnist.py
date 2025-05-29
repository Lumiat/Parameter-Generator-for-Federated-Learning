""" 
task.py 是联邦学习任务的 核心模块，负责：
管理数据加载：实现数据分区、预处理和客户端隔离。
封装训练评估：提供标准化的本地训练和测试接口。
支持参数同步：实现模型参数的序列化与反序列化。
"""


import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor
# from command import parser

fds = None  # Cache FederatedDataset

# 使用 FederatedDataset 加载 MNIST 数据集，并通过 IidPartitioner 进行 独立同分布（IID）分区，模拟联邦学习中的客户端数据分布。

# 数据预处理，转换为张量（ToTensor），mnist不需要特别的数据增强

# 将每个客户端的数据划分为 80% 训练集 和 20% 测试集，并生成 PyTorch 的 DataLoader。

def get_transforms():
    """Return a function that apply standard transformations to images."""
    transform_train = Compose([
        ToTensor(),
    ])

    transform_test = Compose([
        ToTensor(),
    ])

    def apply_train_transform(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [transform_train(image) for image in batch["image"]]
        return batch
    
    def apply_test_transform(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [transform_test(image) for image in batch["image"]]
        return batch

    return apply_train_transform, apply_test_transform

def load_data(partition_id: int, num_partitions: int, alpha: float, batch_size: int):
    """Load partition MNIST data. Can be iid or non-iid."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=alpha, partition_by="label")
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
            seed=None,
        )
    partition = fds.load_partition(partition_id)
    # fig, ax, df = plot_label_distributions(
    #     partitioner,
    #     label_name="label",
    #     plot_type="bar",
    #     size_unit="absolute",
    #     partition_id_axis="x",
    #     legend=True,
    #     verbose_labels=True,
    #     title="Per Partition Labels Distribution",
    # )
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2,seed=None)

    apply_train_transform, apply_test_transform = get_transforms()
    
    
    partition_train_test["train"] = partition_train_test["train"].with_transform(apply_train_transform)
    partition_train_test["test"] = partition_train_test["test"].with_transform(apply_test_transform)


    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

# 执行本地训练，使用SGD优化器，返回验证损失和准确率
def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(
    #     net.parameters(),
    #     lr=learning_rate,
    #     weight_decay=,
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = epochs
    )
    net.train()
    running_loss = 0.0
    for i in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        # print(f"Epoch {i+1}: validation loss {loss.item()}")

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    val_loss, val_acc = test(net, valloader, device)
    # print(f"Final test set performance:\n\tavg_trainloss {avg_trainloss}")
    # print("\n")
    results = {"val_loss": val_loss, "val_accuracy": val_acc}
    return results


# 评估本地模型在本地测试集上的性能，返回损失和准确率
def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total_samples += images.size(0)
    accuracy = correct / total_samples
    loss = loss / len(testloader)
    return loss, accuracy
