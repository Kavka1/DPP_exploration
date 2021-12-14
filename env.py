from typing import List, Dict, Tuple, Type
import numpy as np
import gym


class Env_wrapper(object):
    def __init__(self, env_config: Dict) -> None:
        super().__init__()
        
        self.env_name = env_config['env_name']
        self._max_episode_length = env_config['max_episode_length']
        self.env = gym.make(self.env_name)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict]:
        return self.env.step(action)

    def reset(self) -> np.array:
        return self.env.reset()

    def render(self) -> None:
        return self.env.render()
    
    @property
    def observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return self.env.action_space