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
from utils import MeanStdFilter, check_path, soft_update, hard_update


class Master(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        
        self.env_config = config['env_config']
        self.model_config = config['model_config']

        self.learning_rate = config['lr']
        self.gamma = config['gamma']
        self.rho = config['rho']
        self.train_policy_delay = config['train_policy_delay']
        self.noise_std = config['noise_std']
        self.noise_clip = config['noise_clip']
        self.diversity_importance = config['diversity_importance']

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
        self.init_obs_filter()
        self.init_logger_value()
        
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

    def init_obs_filter(self) -> None:
        self.obs_filter = MeanStdFilter(shape=self.model_config['o_dim'])

    def init_logger_value(self) -> None:
        self.log_q_value = 0
        self.log_q_loss = 0
        self.log_policy_loss = 0
        self.log_diversity_loss = 0

    def adapt_div_importance(self, loss_policy, loss_diverse) -> None:
        if loss_policy > 0.1 * loss_diverse:
            self.diversity_importance *= 1.5
        else:
            self.diversity_importance *= 0.75
        
        self.diversity_importance = max(min(self.diversity_importance, 0.2), 0.05)

    def train(self) -> None:
        self.log_q_value, self.log_q_loss = 0, 0
        
        obs_batch, a_batch, r_batch, done_batch, next_obs_batch = self.buffer.sample(self.batch_size)

        self.obs_filter.push_batch(obs_batch)
        obs_batch, next_obs_batch = self.obs_filter(obs_batch), self.obs_filter(next_obs_batch)

        obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
        a_batch = torch.from_numpy(a_batch).float().to(self.device)
        r_batch = torch.from_numpy(r_batch).float().to(self.device).unsqueeze(dim=-1)
        done_batch = torch.from_numpy(done_batch).int().to(self.device).unsqueeze(dim=-1)
        next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self.device)

        with torch.no_grad():
            next_action_tar_batch = self.policies_tar[0].batch_forward(next_obs_batch)
            noise = torch.clamp(torch.randn_like(next_action_tar_batch).to(self.device) * self.noise_std, - self.noise_clip, self.noise_clip)
            next_action_tar_batch = next_action_tar_batch + noise
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

        self.log_q_value += q1_update_eval.mean().item() + q2_update_eval.mean().item()
        self.log_q_loss += loss_q.item()
        
        if self.train_count % self.train_policy_delay == 0:          
            loss_policy = 0
            pop_action_batch = []
            for i in range(self.num_agents):
                new_action = self.policies[i].batch_forward(obs_batch)
                loss_policy += - self.q1(obs_batch, new_action).mean()
                pop_action_batch.append(new_action)

            diversity_loss = self.compute_diversity_loss(pop_action_batch)

            self.log_policy_loss = loss_policy.item()
            self.log_diversity_loss = diversity_loss.item()

            loss_policy += self.diversity_importance * diversity_loss
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()
    
            for i in range(self.num_agents):
                soft_update(self.policies[i], self.policies_tar[i], self.rho)

        soft_update(self.q1, self.q1_tar, self.rho)
        soft_update(self.q2, self.q2_tar, self.rho)

        self.train_count += 1

        return self.log_q_value/2, self.log_q_loss/2, self.log_policy_loss, self.log_diversity_loss

    def compute_diversity_loss(self, pop_action_batch: List[torch.tensor]) -> torch.tensor:
        action_embedding = [action.flatten() for action in pop_action_batch]
        embedding = torch.stack(action_embedding, dim=0)
        left = embedding.unsqueeze(0).expand(embedding.size(0), -1, -1)
        right = embedding.unsqueeze(1).expand(-1, embedding.size(0), -1)
        matrix = torch.exp(-torch.square(left - right)).sum(-1) / (2)
        return - torch.logdet(matrix)

    def save_policy(self, remark: str, policy_id: int) -> None:
        model_path = self.exp_path + 'models/'
        check_path(model_path)
        torch.save(self.policies[policy_id].state_dict(), model_path+f'model_{policy_id}_{remark}')
        print(f"-------Models {policy_id} saved to {model_path}-------")

    def save_filter(self, remark: str) -> None:
        filter_path = self.exp_path + 'filters/'
        check_path(filter_path)
        filter_params = np.array([self.obs_filter.mean, self.obs_filter.square_sum, self.obs_filter.count])
        np.save(filter_path + remark, filter_params)
        print(f"-------Filter params saved to {filter_path}-------")