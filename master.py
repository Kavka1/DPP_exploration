from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from env import Env_wrapper
from buffer import Buffer
from worker import Worker
from model import Policy_MLP, QFunction
from utils import soft_update, hard_update


class Master(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        
        self.env_config = config['env_config']
        self.model_config = config['model_config']
        self.learning_rate = config['lr']
        self.gamma = config['gamma']
        self.rho = config['rho']
        self.num_agents = config['num_agents']
        self.buffer_size = config['buffer_size']
        self.device = torch.device(config['device'])
        
        self.init_buffer()
        self.init_value_functions()
        self.init_policies()
        self.init_workers()
        self.init_optimizer()
        
    def init_buffer(self) -> None:
        self.buffer = Buffer(self.buffer_size)

    def init_value_functions(self) -> None:
        self.q1 = QFunction(self.model_config, self.device).to(self.device)
        self.q2 = QFunction(self.model_config, self.device).to(self.device)
        self.q1_tar = QFunction(self.model_config, self.device).to(self.device)
        self.q2_tar = QFunction(self.model_config, self.device).to(self.device)
        hard_update(source_model=self.q1, target_model=self.q1_tar)
        hard_update(source_model=self.q2, target_model=self.q2_tar)

    def init_policies(self) -> None:
        self.policies = []
        self.policies_tar = []
        for i in range(len(self.num_agents)):
            policy = Policy_MLP(self.model_config, self.device).to(self.device)
            policy_tar = Policy_MLP(self.model_config, self.device).to(self.device)
            hard_update(source_model=policy, target_model=policy_tar)
            self.policies.append(policy)
            self.policies_tar.append(policy_tar)

    def init_workers(self) -> None:
        self.workers = [
            Worker(Env_wrapper(self.env_config, self.policies[i])).remote()
            for i in range(len(self.num_agents))
        ]

    def init_optimizer(self) -> None:
        policy_params = []
        for i in range(len(self.num_agents)):
            policy_params += [self.policies[i].parameters()]
        self.optimizer_policy = optim.Adam(policy_params, self.learning_rate)
        self.optimizer_q = optim.Adam([self.q1.parameters()]+[self.q2.parameters()], self.learning_rate)

    