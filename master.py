from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pipe

from env import Env_wrapper
from buffer import Buffer
from worker import Worker
from model import Policy_MLP, QFunction
from utils import check_path, soft_update, hard_update


class Master(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        
        self.env_config = config['env_config']
        self.model_config = config['model_config']

        self.learning_rate = config['lr']
        self.gamma = config['gamma']
        self.rho = config['rho']
        self.train_policy_delay = config['train_policy_delay']
        
        self.num_agents = config['num_agents']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.device = torch.device(config['device'])

        self.exp_path = config['exp_path']
        self.train_count = 0
        self.best_scores = [0] * self.num_agents
        
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
        for i in range(self.num_agents):
            policy = Policy_MLP(self.model_config, self.device).to(self.device)
            policy_tar = Policy_MLP(self.model_config, self.device).to(self.device)
            hard_update(source_model=policy, target_model=policy_tar)
            self.policies.append(policy)
            self.policies_tar.append(policy_tar)

    def init_workers(self) -> None:
        self.parents_conn = []
        self.child_conn = []
        self.workers = []
        for i in range(self.num_agents):
            parent_conn, child_conn = Pipe()
            worker = Worker(i, Env_wrapper(self.env_config), child_conn)
            worker.start()
            self.parents_conn.append(parent_conn)
            self.child_conn.append(child_conn)
            self.workers.append(worker)

    def init_optimizer(self) -> None:
        policy_params = []
        for i in range(self.num_agents):
            policy_params += [self.policies[i].parameters()]
        self.optimizer_policy = optim.Adam([{'params': policy.parameters()} for policy in self.policies], self.learning_rate)
        self.optimizer_q = optim.Adam([{'params': self.q1.parameters()}, {'params': self.q2.parameters()}], self.learning_rate)

    def train(self) -> None:
        log_q_value, log_loss_q = 0, 0
        
        obs_batch, a_batch, r_batch, done_batch, next_obs_batch = self.buffer.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
        a_batch = torch.from_numpy(a_batch).float().to(self.device)
        r_batch = torch.from_numpy(r_batch).float().to(self.device).unsqueeze(dim=-1)
        done_batch = torch.from_numpy(done_batch).int().to(self.device).unsqueeze(dim=-1)
        next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self.device)

        with torch.no_grad():
            next_action_tar_batch = self.policies_tar[0].batch_forward(next_obs_batch)
            q1_next_tar = self.q1_tar(next_obs_batch, next_action_tar_batch)
            q2_next_tar = self.q2_tar(next_obs_batch, next_action_tar_batch)
            q_tar = torch.min(q1_next_tar, q2_next_tar)
        
        q_update_target = r_batch + (1 - done_batch) * self.gamma * q_tar
        q1_update_eval = self.q1(obs_batch, a_batch)
        q2_update_eval = self.q2(obs_batch, a_batch)
        loss_q = F.mse_loss(q1_update_eval, q_update_target) + F.mse_loss(q2_update_eval, q_update_target)
        
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        log_q_value += q1_update_eval.mean().item() + q2_update_eval.mean().item()
        log_loss_q += loss_q.item()
        
        if self.train_count % self.train_policy_delay == 0:          
            loss_policy = 0          
            for i in range(self.num_agents):
                new_action = self.policies[i].batch_forward(obs_batch)
                loss_policy += - self.q1(obs_batch, new_action).mean()

            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()
    
            for i in range(self.num_agents):
                soft_update(self.policies[i], self.policies_tar[i], self.rho)

        soft_update(self.q1, self.q1_tar, self.rho)
        soft_update(self.q2, self.q2_tar, self.rho)
        
        self.train_count += 1

        return log_q_value/self.num_agents, log_loss_q/self.num_agents

    def save_policy(self, remark: str, policy_id: int) -> None:
        model_path = self.exp_path + 'models/'
        check_path(model_path)
        torch.save(self.policies[policy_id].state_dict(), model_path+f'model_{policy_id}_{remark}')
        print(f"-------Models {policy_id} saved to {model_path}-------")