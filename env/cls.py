import itertools
import numpy as np
import cv2
import gym

from utility.utils import batch_dicts
from env.typing import EnvOutput
from env import make_env


class Env(gym.Wrapper):
    def __init__(self, config, env_fn=make_env):
        self.env = env_fn(config)
        if 'seed' in config and hasattr(self.env, 'seed'):
            self.env.seed(config['seed'])
        self.name = config['name']
        self.max_episode_steps = self.env.max_episode_steps
        self.n_envs = 1
        self.env_type = 'Env'
        super().__init__(self.env)

    def reset(self, idxes=None):
        return self.env.reset()

    def random_action(self, *args, **kwargs):
        action = self.env.random_action() if hasattr(self.env, 'random_action') \
            else self.env.action_space.sample()
        return action
        
    def step(self, action, **kwargs):
        output = self.env.step(action, **kwargs)
        return output

    """ the following code is needed for ray """
    def score(self, *args):
        return self.env.score()

    def epslen(self, *args):
        return self.env.epslen()

    def mask(self, *args):
        return self.env.mask()

    def prev_obs(self):
        return self.env.prev_obs()

    def info(self):
        return self.env.info()

    def output(self):
        return self.env.output()

    def game_over(self):
        return self.env.game_over()
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            img = self.env.get_screen()
        else:
            img = self.env.render(mode='rgb_array')

        if size is not None and size != img.shape[:2]:
            # cv2 receive size of form (width, height)
            img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
            
        return img


class EnvVecBase(gym.Wrapper):
    def __init__(self):
        self.env_type = 'EnvVec'
        super().__init__(self.env)

    def _convert_batch(self, data, func=np.stack):
        if data != []:
            if isinstance(data[0], (np.ndarray, int, float, np.floating, np.integer)):
                data = func(data)
            elif isinstance(data[0], dict):
                data = batch_dicts(data, func)
            else:
                data = list(data)
        return data

    def _get_idxes(self, idxes):
        if idxes is None:
            idxes = list(range(self.n_envs))
        elif isinstance(idxes, int):
            idxes = [idxes]
        return idxes


class EnvVec(EnvVecBase):
    def __init__(self, config, env_fn=make_env):
        self.n_envs = n_envs = config.pop('n_envs', 1)
        self.name = config['name']
        self.envs = [env_fn(config) for i in range(n_envs)]
        self.env = self.envs[0]
        if 'seed' in config:
            [env.seed(config['seed'] + i) 
                for i, env in enumerate(self.envs)
                if hasattr(env, 'seed')]
        self.max_episode_steps = self.env.max_episode_steps
        super().__init__()

    def random_action(self, *args, **kwargs):
        return np.stack([env.random_action() if hasattr(env, 'random_action') \
            else env.action_space.sample() for env in self.envs])

    def reset(self, idxes=None, **kwargs):
        idxes = self._get_idxes(idxes)
        out = zip(*[self.envs[i].reset() for i in idxes])

        return EnvOutput(*[self._convert_batch(o) for o in out])

    def step(self, actions, **kwargs):
        return self._envvec_op('step', action=actions, **kwargs)

    def score(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].score() for i in idxes]

    def epslen(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].epslen() for i in idxes]

    def mask(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return np.stack([self.envs[i].mask() for i in idxes])

    def game_over(self):
        return np.stack([env.game_over() for env in self.envs])

    def prev_obs(self, idxes=None):
        idxes = self._get_idxes(idxes)
        obs = [self.envs[i].prev_obs() for i in idxes]
        if isinstance(obs[0], dict):
            obs = batch_dicts(obs)
        return obs

    def info(self, idxes=None, convert_batch=False):
        idxes = self._get_idxes(idxes)
        info = [self.envs[i].info() for i in idxes]
        if convert_batch:
            info = batch_dicts(info)
        return info

    def output(self, idxes=None):
        idxes = self._get_idxes(idxes)
        out = zip(*[self.envs[i].output() for i in idxes])

        return EnvOutput(*[self._convert_batch(o) for o in out])

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            imgs = np.stack([env.get_screen() for env in self.envs])
        else:
            imgs = np.stack([env.render(mode='rgb_array') for env in self.envs])

        if size is not None:
            # cv2 receive size of form (width, height)
            imgs = np.stack([cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
                            for i in imgs])
        
        return imgs

    def _envvec_op(self, name, **kwargs):
        method = lambda e: getattr(e, name)
        if kwargs:
            kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_envs)] 
                for k, v in kwargs.items()}
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
                for k, v in kwargs.items()])]
            out = zip(*[method(env)(**kw) for env, kw in zip(self.envs, kwargs)])
        else:
            out = zip(*[method(env)() for env in self.envs])

        return EnvOutput(*[self._convert_batch(o) for o in out])

    def close(self):
        if hasattr(self.env, 'close'):
            [env.close() for env in self.envs]


if __name__ == '__main__':
    config = dict(
        name='smac_6h_vs_8z',
        n_workers=8,
        n_envs=1,
        use_state_agent=True,
        use_mustalive=True,
        add_center_xy=True,
        timeout_done=True,
        add_agent_id=False,
        obs_agent_id=False,
    )
    env = Env(config)
    for k in range(100):
        o, r, d, re = env.step(env.random_action())
        print(k, d, re, o['episodic_mask'])
        print(r, env.score(), env.epslen())
