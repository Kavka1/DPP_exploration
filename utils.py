from typing import Dict, List, Tuple
import numpy as np
import sys
import yaml
import os
import datetime
import torch
import torch.nn as nn
from env import Env_wrapper


def hard_update(source_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
    target_model.load_state_dict(source_model.state_dict())


def soft_update(source_model: torch.nn.Module, target_model: torch.nn.Module, rho: float) -> None:
    for param_s, param_t in zip(source_model.parameters(), target_model.parameters()):
        param_t.data.copy_(rho * param_s.data + (1 - rho) * param_t.data)


def check_path(path: str) -> None:
    if os.path.exists(path) is False:
        os.makedirs(path)


def refine_model_config(config: Dict) -> Dict:
    env = Env_wrapper(config['env_config'])
    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0],
        'action_low': float(env.action_space.low[0]),
        'action_high': float(env.action_space.high[0])
    })
    return config


def create_exp_path(config: Dict) -> Dict:
    exp_name = f"{config['env_config']['env_name']}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    exp_path = config['results_path'] + exp_name + '/'
    check_path(exp_path)
    config.update({'exp_path': exp_path})
    return config


def refine_config(config: Dict) -> Dict:
    config = refine_model_config(config)
    config = create_exp_path(config)
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)
    print(f"Experiment config.yaml saved to {config['exp_path']}")
    return config