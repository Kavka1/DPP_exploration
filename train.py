from typing import Dict, List, Tuple, Type
import numpy as np
from torch.serialization import load
import yaml
import torch
from master import Master
from env import Env_wrapper
from utils import refine_config
from torch.utils.tensorboard import SummaryWriter


def load_config():
    config_path = '/home/xukang/GitRepo/DPP_exploration/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
    return config


def train():
    config = load_config()
    config = refine_config(config)

    master = Master(config)
    logger = SummaryWriter(config['exp_path'])

    log_q_value, log_q_loss = 0, 0

    obs_seq = [worker.initial_obs for worker in master.workers]
    for step in range(config['max_timestep']):
        a_seq, r_seq, done_seq, next_obs_seq, info_seq = [], [], [], [], []
        
        for i in range(config['num_agents']):
            action = master.policies[i].act(obs_seq[i])
            master.parents_conn[i].send(action)
            a_seq.append(action)

        for i in range(config['num_agents']):
            next_obs, r, done, info = master.parents_conn[i].recv()
            r_seq.append(r)
            done_seq.append(done)
            next_obs_seq.append(next_obs)
            info_seq.append(info)

        master.buffer.push_batch(list(zip(obs_seq, a_seq, r_seq, done_seq, next_obs_seq)))
        obs_seq = next_obs_seq
        
        if step > config['start_training_step']:
            log_q_value, log_q_loss = master.train()

        for i in range(config['num_agents']):
            if done_seq[i] == True:
                episode_r = info_seq[i]['episode_reward']
                episode_count = info_seq[i]['episode_count']
                episode_step = info_seq[i]['episode_step']
                
                if episode_count % config['save_interval'] == 0:
                    master.save_policy(remark=f'{episode_count}', policy_id = i)
                if episode_r > master.best_scores[i]:
                    master.save_policy(remark='best', policy_id=i)
                    master.best_scores[i] = episode_r
                
                logger.add_scalar(f'Scores/episode_reward_policy_{i}', episode_r, episode_count)

        logger.add_scalar('Train/q_value', log_q_value, step)
        logger.add_scalar('Train/q_loss', log_q_loss, step)


if __name__ == '__main__':
    train()