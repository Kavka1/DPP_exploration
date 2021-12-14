from typing import Dict, List, Tuple, Type
import numpy as np
import torch
from multiprocessing.connection import Connection
from torch.multiprocessing import Pipe
from torch.multiprocessing import Process

from env import Env_wrapper
from model import Policy_MLP


class Worker(Process):
    def __init__(self, index: int, env: Env_wrapper, child_conn: Connection) -> None:
        super().__init__()
        self.id = index
        self.env = env
        self.child_conn = child_conn

        self.episode_reward = 0
        self.episode_length = 0
        self.episode_count = 0
        self.initial_obs = self.env.reset()

    def run(self) -> None:
        super(Worker, self).run()
        while True:
            action = self.child_conn.recv()
            obs, r, done, info = self.env.step(action)
            self.episode_reward += r
            self.episode_length += 1

            if done:
                info = {'episode_step': self.episode_length, 'episode_reward': self.episode_reward, 'episode_count': self.episode_count}
                print(f"Worker {self.id} complete episode {self.episode_count} rewards: {self.episode_reward} length: {self.episode_length}")
                self.episode_length = 0
                self.episode_reward = 0
                self.episode_count += 1
                obs = self.env.reset()

            self.child_conn.send([obs, r, done, info])

"""
    def do_rollouts(self, num_rollouts: int) -> Tuple[int, np.array, np.array, List[float]]:
        o_seq, a_seq, r_seq = [], [], []
        total_step = 0
        for i_rollout in range(num_rollouts):
            step, o, a, r = self.rollouts()
            o_seq += o
            a_seq += a
            r_seq += r
            total_step += step
        return total_step, np.array(o_seq), np.array(a_seq), r

    def rollouts(self) -> Tuple[int, List[np.array], List[np.array], List[float]]:
        o, a, r = [], [], []
        step = 0
        done = False
        obs = self.env.reset()
        while not done:
            action = self.policy(obs)
            obs_, reward, done, info = self.env.step(action)
            o.append(obs)
            a.append(action)
            r.append(reward)
            obs = obs_
            step += 1
        return step, o, a, r
"""