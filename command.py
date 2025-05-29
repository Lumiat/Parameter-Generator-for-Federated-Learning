import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='resnet18', help="模型选择")
parser.add_argument("--dataset", type=str, default='cifar10', help="数据集选择")
parser.add_argument("--iid", action='store_true', default=False, help="是否是独立同分布")
parser.add_argument("--dirichlet", action='store_true', default=False, help="是否选择迪利克雷分布")
parser.add_argument("--classes_per_client", type=int, default=2, help="每个客户端分到多少个类")
parser.add_argument("--alpha", type=float, default=0.1, help="迪利克雷分布的参数")
parser.add_argument("--num_partitions", type=int, default=10, help="数据集分区")

parser.add_argument("--lr", type=float, default=3e-3, help="客户端学习率")
parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW优化器weight_decay参数")
parser.add_argument("--local_epochs", type=int, default=10, help="本地训练轮数")
parser.add_argument("--minlr", type=float, default=1e-5, help="余弦退火最终达到的最小学习率")
parser.add_argument("--local_bs", type=int, default=32, help="本地训练的batch大小")