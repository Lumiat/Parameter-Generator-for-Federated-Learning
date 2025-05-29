from flwr.server.strategy import FedProx
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, parameters_to_ndarrays

import os
import torch
import json
from datetime import datetime
import wandb

from .utils import set_weights, dynamic_model_import

class CustomFedProx(FedProx):
    """A strategy that keeps the core functionality of FedProx unchanged but enables
    additional features: 
    1. Saving global checkpoints
    2. saving metrics to the local file system as a JSON
    3. pushing metrics to Weight & Biases.
    """
    def __init__(self,
                 dataset_name: str,
                 model_name:str,
                 model_class_path: str,
                 image_size:int,
                 num_classes:int,
                 num_channels:int,
                 server_rounds: int,
                 local_epochs: int,
                 learning_rate: float,
                 batch_size: int,
                 alpha: float,
                 c:float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = dynamic_model_import(model_class_path)
        self.results_to_save = {}
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.hparams = {
            "server_rounds": server_rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "alpha": alpha,
            "fraction_fit": c
        }

        # file name id
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_id = (
            f"dataset_{dataset_name}_"
            f"model_{model_name}_"
            f"alpha_{alpha}_"
            f"{timestamp}"
        )

        # create save dir
        base_dir = os.path.join("dataset", f"{dataset_name}_{model_name}")
        self.save_dir = os.path.join(base_dir, self.exp_id)

        os.makedirs(self.save_dir, exist_ok=True)

        # initialize wanb
        wandb.init(
            project=f"fed-{dataset_name}-{model_name}",
            name=self.exp_id,
            config=self.hparams,
            dir=base_dir 
        )

    def aggregate_fit(self,
                      server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      )-> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        parameter_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        if parameter_aggregated is not None:
            # create model and set parameters
            model = self.model_class(image_size=self.image_size, num_classes=self.num_classes, in_channels=self.num_channels)
            ndarrays = parameters_to_ndarrays(parameter_aggregated)
            set_weights(model, ndarrays)
            
            if server_round >= 200:
                # create path
                ckpt_name = f"round_{server_round}.pth"
                ckpt_path = os.path.join(self.save_dir, ckpt_name)
                
                # save model
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")


        return super().aggregate_fit(server_round, results, failures)

    def evaluate(self, 
                 server_round: int, 
                 parameters: Parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        results = {"loss": loss, **metrics}

        self.results_to_save[server_round] = results

        # save metrics as json file
        json_path = os.path.join(self.save_dir, "results.json")
        with open(json_path,'w') as json_file:
            json.dump(self.results_to_save, json_file, indent=4)
        return super().evaluate(server_round, parameters)  
    
    def __del__(self):
        if wandb.run:
            wandb.finish()
