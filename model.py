from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = []
        self.create_layers()

    def create_layers(self) -> None:
        last_layer = self.layers[0]
        for i in range(1, len(self.layers)):
            self.model += [nn.Linear(last_layer, self.layers[i]), nn.Tanh()]
            last_layer = self.layers[i]
        self.model = nn.Sequential(self.model)


class Policy_MLP(MLP):
    def __init__(self, model_config: Dict, device: torch.device) -> None:
        self.in_dim = model_config['o_dim']
        self.out_dim = model_config['a_dim']
        self.hidden_layers = model_config['policy_hidden_layers']
        self.std = model_config['action_std']
        self.a_low, self.a_high = model_config['action_low'], model_config['action_high']
        self.device = device

        self.layers = [self.in_dim] + self.hidden_layers + [self.out_dim]
        
        super.__init__()

    def forward(self, obs: np.array) -> np.array:
        obs = torch.from_numpy(obs).float().to(self.device)
        mean = self.model(obs).numpy()
        action = np.clip(mean, self.a_low, self.a_high)
        return action

    def batch_forward(self, obs: torch.tensor) -> torch.tensor:
        return self.model(obs)
    

class QFunction(MLP):
    def __init__(self, model_config: Dict, device: torch.device) -> None:
        self.in_dim = model_config['o_dim'] + model_config['a_dim']
        self.out_dim = 1
        self.hidden_layers = model_config['value_hidden_layers']
        self.device = device

        self.layers = [self.in_dim] + self.hidden_layers + [self.out_dim]

        super().__init__()

    def forward(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        x = torch.concat([obs, action], dim=-1)
        q_value = self.model(x)
        return q_value
