from typing import List, Tuple

import torch
import torchvision
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedProx
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


from .utils import get_weights, set_weights, dynamic_model_import, dynamic_toolkit_import
# from toolkit.cifar10 import test, get_transforms
from .custom_strategy import CustomFedProx

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy_avg = sum(accuracies) / sum(examples)
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": accuracy_avg}


def get_evaluate_fn(testloader, device, image_size, model_class, num_classes, num_channels, test):
    """Return a callback that evaluates the global model."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""
        # Instantiate model
        net = model_class(image_size = image_size, num_classes = num_classes, in_channels=num_channels)
        # Apply global_model parameters
        set_weights(net, parameters_ndarrays)
        net.to(device)
        # Run test
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate
    
def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """handle metrics from fit method in clients"""
    for _, m in metrics:
        print(m)
    return {}

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    dataset_name = context.run_config["dataset-name"]
    model_name = context.run_config["model-name"]
    model_class_path = context.run_config["model-path"]
    image_size = context.run_config["image-size"]
    num_classes = context.run_config["num-classes"]
    num_channels = context.run_config["num-channels"]
    num_rounds = context.run_config["num-server-rounds"]
    batch = context.run_config["batch-size"]

    # dynamically import
    toolkit_module = dynamic_toolkit_import(dataset_name)
    test = toolkit_module.test  # 获取动态的 test 函数
    get_transforms = toolkit_module.get_transforms  # 获取动态的数据增强函数
    Model = dynamic_model_import(model_class_path)

    # Initialize model parameters
    ndarrays = get_weights(Model(image_size=image_size, num_classes=num_classes, in_channels=num_channels))
    parameters = ndarrays_to_parameters(ndarrays)

    # load global test set
    
    _, apply_test_transform = get_transforms()

    if dataset_name == 'svhn':
        testset = load_dataset(path="svhn",name="cropped_digits",split='test')
    elif dataset_name == 'emnist':
        testset = load_dataset(path="randall-lab/emnist", name="byclass", split="test",trust_remote_code=True)
    elif dataset_name == 'imagenet-1k':
        testset = load_dataset(path="ILSVRC/imagenet-1k", split="test", trust_remote_code=True)
    else:
        testset = load_dataset(path=dataset_name,split='test')
    testloader = DataLoader(testset.with_transform(apply_test_transform), batch)

    # Define the strategy
    # strategy = FedProx(
    #     initial_parameters = parameters,
    #     fraction_fit=context.run_config["fraction-fit"],
    #     fraction_evaluate=context.run_config["fraction-evaluate"],
    #     min_available_clients=2,
    #     proximal_mu=0.1,  # Regularization strength for FedProx
    #     evaluate_metrics_aggregation_fn = weighted_average,
    #     fit_metrics_aggregation_fn = handle_fit_metrics,
    #     evaluate_fn = get_evaluate_fn(testloader, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # )

    strategy = CustomFedProx(
        dataset_name=dataset_name,
        model_name=model_name,
        model_class_path=model_class_path,
        image_size = image_size,
        num_classes=num_classes,
        num_channels=num_channels,
        server_rounds=context.run_config["num-server-rounds"],
        local_epochs=context.run_config["local-epochs"],
        learning_rate=context.run_config["learning-rate"],
        batch_size=context.run_config["batch-size"],
        alpha=context.run_config["alpha"],
        c=context.run_config["fraction-fit"],
        initial_parameters = parameters,
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        proximal_mu=0.1,  # Regularization strength for FedProx
        evaluate_metrics_aggregation_fn = weighted_average,
        fit_metrics_aggregation_fn = handle_fit_metrics,
        evaluate_fn = get_evaluate_fn(
            testloader, 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            model_class = Model,
            image_size=image_size,
            num_classes=num_classes,
            num_channels=num_channels,
            test = test
        )
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
