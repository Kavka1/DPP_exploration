from typing import List, Dict, Tuple, Type
import numpy as np
import gym
from gym.envs.mujoco import MujocoEnv
from gym.utils import EzPickle
import mujoco_py


class Env_wrapper(object):
    def __init__(self, env_config: Dict) -> None:
        super().__init__()
        
        self.env_name = env_config['env_name']
        self._max_episode_step = env_config['max_episode_step']
        self.seed = env_config['env_seed']
        self._elapsed_step = 0

        if self.env_name == 'Swimmer':
            self.env = SwimmerEnv('/home/xukang/GitRepo/DPP_exploration/assets/swimmer.xml', 4)
        elif self.env_name == 'SwimmerMR':
            self.env = SwimmerMultiRewardEnv('/home/xukang/GitRepo/DPP_exploration/assets/swimmer.xml', 4)
        elif self.env_name in ['Swimmer-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'Humanoid-v2', 'Ant-v2']:
            self.env = gym.make(self.env_name)
        else:
            raise ValueError(f'Env name illegal, no cressponding named {self.env_name}')
        
        self.env.seed(self.seed)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict]:
        obs, r, done, info = self.env.step(action)
        self._elapsed_step += 1
        if self._elapsed_step >= self._max_episode_step:
            done = True
        return obs, r, done, info

    def reset(self) -> np.array:
        self._elapsed_step = 0
        return self.env.reset()

    def render(self) -> None:
        return self.env.render()
    
    @property
    def observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return self.env.action_space


class SwimmerEnv(MujocoEnv, EzPickle):
    def __init__(self, model_path, frame_skip = 4):
        MujocoEnv.__init__(self, model_path, frame_skip)
        EzPickle.__init__(self)
    
    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()


class SwimmerMultiRewardEnv(MujocoEnv, EzPickle):
    def __init__(self, model_path, frame_skip):
        MujocoEnv.__init__(self, model_path, frame_skip)
        EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001

        xposbefore = self.sim.data.qpos[0]
        yposbefore = self.sim.data.qpos[1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        yposafter = self.sim.data.qpos[1]

        reward_x_fwd = (xposafter - xposbefore) / self.dt
        reward_y_up = (yposafter - yposbefore) / self.dt

        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()

        reward = reward_x_fwd + reward_y_up + reward_ctrl
        
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_up=reward_y_up, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()