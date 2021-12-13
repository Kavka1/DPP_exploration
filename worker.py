from _typeshed import Self
from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import ray

from env import Env_wrapper
from model import Policy_MLP


@ray.remote
class Worker(object):
    def __init__(self, env: Env_wrapper, policy: Policy_MLP, num_rollout: int) -> None:
        super().__init__()
        self.env = env
        self.policy = policy

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