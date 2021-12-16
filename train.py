from typing import Dict, List, Tuple, Type
import numpy as np
import yaml
import pandas as pd
from master import Master
from utils import refine_config, seed_torch_np
from torch.utils.tensorboard import SummaryWriter


def load_config():
    config_path = '/home/xukang/GitRepo/DPP_exploration/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
    return config


def train():
    config = load_config()
    config = refine_config(config)

    seed_torch_np(config['seed'])

    master = Master(config)
    logger = SummaryWriter(config['exp_path'])

    log_q_value, log_q_loss, log_policy_loss, log_diversity_loss = 0, 0, 0, 0
    log_score = [{'episode': [] ,'episode_reward': []} for _ in range(config['num_agents'])]

    obs_seq = [worker.initial_obs for worker in master.workers]
    for step in range(config['max_timestep']):
        a_seq, r_seq, done_seq, next_obs_seq, info_seq = [], [], [], [], []
        
        for i in range(config['num_agents']):
            obs = master.obs_filter(obs_seq[i])
            action = master.policies[i].act(obs)
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
            log_q_value, log_q_loss, log_policy_loss, log_diversity_loss = master.train()

        for i in range(config['num_agents']):
            if done_seq[i] == True:
                episode_r = info_seq[i]['episode_reward']
                episode_count = info_seq[i]['episode_count']
                episode_step = info_seq[i]['episode_step']
                
                if episode_count % config['save_interval'] == 0:
                    master.save_policy(remark=f'{episode_count}', policy_id = i)
                    master.save_filter(remark=f'{episode_count}')
                if episode_r > master.best_scores[i]:
                    master.save_policy(remark='best', policy_id=i)
                    master.save_filter(remark='best')
                    master.best_scores[i] = episode_r
                
                logger.add_scalar(f'Scores/episode_reward_policy_{i}', episode_r, episode_count)

                log_score[i]['episode'].append(episode_count)
                log_score[i]['episode_reward'].append(episode_r)
                df = pd.DataFrame(log_score[i])
                df.to_csv(config['exp_path'] + f'stats_{i}.csv', index=False)

        logger.add_scalar('Train/q_value', log_q_value, step)
        logger.add_scalar('Train/q_loss', log_q_loss, step)
        logger.add_scalar('Train/policy_loss', log_policy_loss, step)
        logger.add_scalar('Train/diversity_loss', log_diversity_loss, step)

    for worker in master.workers:
        worker.terminate()


if __name__ == '__main__':
    train()