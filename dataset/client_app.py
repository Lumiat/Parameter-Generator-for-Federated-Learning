import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .utils import get_weights, set_weights, dynamic_model_import, dynamic_toolkit_import
# from dataset.toolkit.cifar10 import  load_data, test, train
# from dataset.cifar10_resnet18.model import Model

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, toolkit_module, model_class, image_size, num_classes, num_channels, trainloader, valloader, local_epochs, learning_rate):
        self.toolkit_module = toolkit_module
        self.net = model_class(image_size = image_size, num_classes=num_classes, in_channels=num_channels)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = self.toolkit_module.train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )

        # # Append to persistent state the `train_loss` just obtained
        # fit_metrics = self.client_state.config_records["fit_metrics"]
        # if "train_loss_hist" not in fit_metrics:
        #     # If first entry, create the list
        #     fit_metrics["train_loss_hist"] = [results["avg_trainloss"]]
        # else:
        #     # If it's not the first entry, append to the existing list
        #     fit_metrics["train_loss_hist"].append(results["avg_trainloss"])

        # A complex metric strcuture can be returned by a ClientApp if it is first
        # converted to a supported type by `flwr.common.Scalar`. Here we serialize it with
        # JSON and therefore representing it as a string (one of the supported types)
        # complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        # complex_metric_str = json.dumps(complex_metric)
        
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = self.toolkit_module.test(self.net, self.valloader, self.device)
        print(f"client local eval: loss = {loss}, acc = {accuracy}")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    
def client_fn(context:Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the configuration of model and dataset
    dataset_name = context.run_config["dataset-name"]
    model_path = context.run_config["model-path"]
    image_size = context.run_config["image-size"]
    num_classes = context.run_config["num-classes"]
    num_channels = context.run_config["num-channels"]

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    alpha = context.run_config["alpha"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # dynamic load model and toolkit
    toolkit_module = dynamic_toolkit_import(dataset_name)
    load_data = toolkit_module.load_data  
    model_class = dynamic_model_import(model_path)

    trainloader, valloader = load_data(
        partition_id, 
        num_partitions, 
        alpha, 
        batch_size
    )
    
    # Return Client instance
    return FlowerClient(
        toolkit_module=toolkit_module,
        model_class=model_class,
        image_size = image_size, 
        num_classes=num_classes,
        num_channels=num_channels,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
