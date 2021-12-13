from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


def hard_update(source_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
    target_model.load_state_dict(source_model.state_dict())


def soft_update(source_model: torch.nn.Module, target_model: torch.nn.Module, rho: float) -> None:
    for param_s, param_t in zip(source_model.parameters(), target_model.parameters()):
        param_t.data.copy_(rho * param_s.data + (1 - rho) * param_t.data)