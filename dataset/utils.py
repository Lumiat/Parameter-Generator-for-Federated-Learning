from collections import OrderedDict
import torch
import importlib

def dynamic_model_import(model_class_path: str):
    """dynamically import model, such as `cifar10_resnet18.model.ResNet18`"""
    module_path, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def dynamic_toolkit_import(dataset_name:str):
    """dynamically import dataset related toolkits,such as toolkit.cifar10）"""
    try:
        return importlib.import_module(f"dataset.toolkit.{dataset_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Dataset toolkit {dataset_name} not found")

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
