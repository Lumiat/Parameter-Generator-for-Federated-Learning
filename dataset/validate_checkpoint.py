import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from datasets import load_dataset
import importlib
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from .utils import dynamic_model_import
except (ImportError, ValueError):
    # relative import failed
    try:
        from utils import dynamic_model_import
    except ImportError:
        sys.path.append(os.path.join(current_dir, ".."))  # add parent_dir to path
        from dataset.utils import dynamic_model_import


# config datasets
DATASET_CONFIG = {
    "mnist": {
        "path": "ylecun/mnist",
        "size": 32,
        "channels": 1,
        "num_classes": 10,
        "test_set": "test",
        "normalize_mean": [0.1307],
        "normalize_std": [0.3081],
        "image_key": "image",
        "label_key": "label",
        "reshape": None  
    },
    "fmnist": {
        "path": "zalando-datasets/fashion_mnist", 
        "size": 32,
        "channels": 1,
        "num_classes": 10,
        "test_set": "test",
        "normalize_mean": [0.2860],
        "normalize_std": [0.3530],
        "image_key": "image",
        "label_key": "label",
        "reshape": None
    },
    "svhn": {
        "path": "ufldl-stanford/svhn",
        "size": 32,
        "channels": 3,
        "num_classes": 10,
        "test_set": "test",
        "normalize_mean": [0.4377, 0.4438, 0.4728],
        "normalize_std": [0.1980, 0.2010, 0.1970],
        "image_key": "image",
        "label_key": "label",
        "reshape": None,
        "subset": "cropped_digits"
    },
    "emnist": {
        "path": "randall-lab/emnist",
        "size": 32,
        "channels": 1,
        "num_classes": 62,  # byclass
        "test_set": "test",
        "normalize_mean": [0.171],
        "normalize_std": [0.333],
        "image_key": "image",
        "label_key": "label",
        "reshape": None,
        "subset": "byclass"  # EMNIST subset config
    },
    "cifar10": {
        "path": "uoft-cs/cifar10",
        "size": 32,
        "channels": 3,
        "num_classes": 10,
        "test_set": "test",
        "normalize_mean": [0.4914, 0.4822, 0.4465],
        "normalize_std": [0.2470, 0.2435, 0.2616],
        "image_key": "img",
        "label_key": "label",
        "reshape": None
    },
    "cifar100": {
        "path": "uoft-cs/cifar100",
        "size": 32,
        "channels": 3,
        "num_classes": 100,
        "test_set": "test",
        "normalize_mean": [0.5071, 0.4867, 0.4408],
        "normalize_std": [0.2675, 0.2565, 0.2761],
        "image_key": "img",
        "label_key": "fine_label",
        "reshape": None
    },
    "imagenet1k": {
        "path": "ILSVRC/imagenet-1k",
        "size": 224,
        "channels": 3,
        "num_classes": 1000,
        "test_set": "validation",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "image_key": "image",
        "label_key": "label",
        "reshape": None
    }
}

MODEL_PATH_CONFIG = {
    "cnn": {
        "model_path": "models.cnn.CNN"
    },
    "lenet5": {
        "model_path": "models.lenet5.LeNet5"
    },
    "resnet18": {
        "model_path": "models.resnet18.ResNet18"
    },
    "vit_tiny": {
        "model_path": "models.vit_tiny.VitTiny"
    }
}

def get_transform(config):
    """pre-processing"""
    transform_test = Compose(
        [ToTensor(),
         Normalize(
        mean=config["normalize_mean"], 
        std=config["normalize_std"]
        )]
    )

    def apply_test_transform(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[config["image_key"]] = [transform_test(img) for img in batch[config["image_key"]]]
        return batch

    return apply_test_transform

def test_model(model, test_loader, device):
    """test checkpoint model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch[DATASET_CONFIG[args.dataset]["image_key"]]
            labels = batch[DATASET_CONFIG[args.dataset]["label_key"]]
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def load_test_dataset(dataset_name):
    """load and preprocess test_dataset"""
    config = DATASET_CONFIG[dataset_name].copy()
    config["dataset_name"] = dataset_name
    transform = get_transform(config)
    
    dataset_args = {
        "path": config["path"],
        "split": config["test_set"]
    }
    
    # dealing with special dataset
    if dataset_name == "emnist" or dataset_name == "svhn":
        dataset_args["name"] = config["subset"]
        dataset_args["trust_remote_code"] = True
    if dataset_name == "imagenet1k":
        dataset_args["trust_remote_code"] = True
    
    # 加载数据集
    dataset = load_dataset(**dataset_args)

    # dataset = dataset.map(transform_examples, batched=False)
    return dataset.with_transform(transform)

def main(args):
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # validate the dataset name
    if args.dataset not in DATASET_CONFIG:
        print(f"error: unsupported dataset '{args.dataset}'")
        print("supported datasets:", list(DATASET_CONFIG.keys()))
        return
    
    # load configuration of dataset
    config = DATASET_CONFIG[args.dataset]
    
    try:
        # dynamically import model
        model_class = dynamic_model_import(MODEL_PATH_CONFIG[args.model_class]["model_path"])
        
        # create model
        model = model_class(
            image_size=config["size"],
            num_classes=config["num_classes"],
            in_channels=config["channels"]
        )
        model.to(device)
        
        # printf model information
        total_params = sum(p.numel() for p in model.parameters())
        print(f"loaded model: {args.model_class}")
        print(f"number of parameters: {total_params:,}")
    except Exception as e:
        print(f"load model failed: {e}")
        return
    
    # load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # deal with different types of checkpoint
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"successfully loaded checkpoint: {args.checkpoint_path}")
    except Exception as e:
        print(f"load checkpoint failed: {e}")
        return
    
    # load test dataset
    try:
        print(f"load test dataset: {args.dataset}")
        test_dataset = load_test_dataset(args.dataset)
        
        # create dataloader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=64,
        )
        print(f"volume of testset: {len(test_dataset)} samples")
    except Exception as e:
        print(f"load dataset failed: {e}")
        return
    
    # test accuracy
    accuracy = test_model(model, test_loader, device)
    print(f"testing acc: {accuracy:.2f}%")
    
    # 分类报告
    # if config["num_classes"] <= 20:  # 对于类别较少的数据集输出详细报告
    #     from sklearn.metrics import classification_report
    #     all_preds = []
    #     all_labels = []
        
    #     model.eval()
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             images = batch[config["image_key"]].to(device)
    #             labels = batch[config["label_key"]].numpy()
    #             outputs = model(images)
    #             _, preds = torch.max(outputs, 1)
    #             all_preds.extend(preds.cpu().numpy())
    #             all_labels.extend(labels)
        
    #     print("\n分类报告:")
    #     print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' testing checkpoint model acc')
    parser.add_argument('dataset', type=str, help='dataset_name (mnist, fmnist, svhn, emnist, cifar10, cifar100, imagenet1k)')
    parser.add_argument('model_class', type=str, help='model_name (resnet18, cnn, lenet5, vit_tiny)')
    parser.add_argument('checkpoint_path', type=str, help='complete checkpoint path')
    
    args = parser.parse_args()
    
    # run the main method
    main(args)
