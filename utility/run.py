import collections
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RunMode:
    NSTEPS='nsteps'
    TRAJ='traj'


class Runner:
    def __init__(self, env, agent, step=0, nsteps=None, 
                run_mode=RunMode.NSTEPS, record_envs=None, info_func=None):
        self.env = env
        if env.max_episode_steps == int(1e9):
            logger.info(f'Maximum episode steps is not specified'
                f'and is by default set to {self.env.max_episode_steps}')
            # assert nsteps is not None
        self.agent = agent
        self.step = step
        if run_mode == RunMode.TRAJ and env.env_type == 'EnvVec':
            logger.warning('Runner.step is not the actual environment steps '
                f'as run_mode == {RunMode.TRAJ} and env_type == EnvVec')
        self.env_output = self.env.output()
        self.episodes = np.zeros(env.n_envs)
        assert getattr(self.env, 'auto_reset', None), getattr(self.env, 'auto_reset', None)
        self.run = {
            f'{RunMode.NSTEPS}-Env': self._run_env,
            f'{RunMode.NSTEPS}-EnvVec': self._run_envvec,
            f'{RunMode.TRAJ}-Env': self._run_traj_env,
            f'{RunMode.TRAJ}-EnvVec': self._run_traj_envvec,
        }[f'{run_mode}-{self.env.env_type}']

        self._frame_skip = getattr(env, 'frame_skip', 1)
        self._frames_per_step = self.env.n_envs * self._frame_skip
        self._default_nsteps = nsteps or env.max_episode_steps // self._frame_skip

        record_envs = record_envs or self.env.n_envs
        self._record_envs = list(range(record_envs))

        self._info_func = info_func

    def _run_env(self, *, action_selector=None, step_fn=None, nsteps=None):
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs

        for t in range(nsteps):
            action = action_selector(self.env_output, evaluation=False)
            obs, reset = self.step_env(obs, action, step_fn)

            # logging when env is reset 
            if reset:
                info = self.env.info()
                if 'score' in info:
                    self.store_info(info)
                    self.episodes += 1

        return self.step

    def _run_envvec(self, *, action_selector=None, step_fn=None, nsteps=None):
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        
        for t in range(nsteps):
            action = action_selector(self.env_output, evaluation=False)
            obs, reset = self.step_env(obs, action, step_fn)
            
            # logging when any env is reset 
            done_env_ids = [i for i, r in enumerate(reset)
                if (np.all(r) if isinstance(r, np.ndarray) else r) 
                and i in self._record_envs]
            if done_env_ids:
                info = self.env.info(done_env_ids)
                # further filter done caused by life loss
                info = [i for i in info if i.get('game_over')]
                if info:
                    self.store_info(info)
                self.episodes[done_env_ids] += 1

        return self.step

    def _run_traj_env(self, action_selector=None, step_fn=None):
        action_selector = action_selector or self.agent
        obs = self.env_output.obs
        
        for t in range(self._default_nsteps):
            action = action_selector(self.env_output, evaluation=False)
            obs, reset = self.step_env(obs, action, step_fn)

            if reset:
                break
        
        info = self.env.info()
        self.store_info(info)
        self.episodes += 1

        return self.step

    def _run_traj_envvec(self, action_selector=None, step_fn=None):
        action_selector = action_selector or self.agent
        obs = self.env_output.obs
        
        for t in range(self._default_nsteps):
            action = action_selector(self.env_output, evaluation=False)
            obs, reset = self.step_env(obs, action, step_fn)

            # logging when any env is reset 
            if np.all(reset):
                break

        info = [i for idx, i in enumerate(self.env.info()) if idx in self._record_envs]
        self.store_info(info)
        self.episodes += 1

        return self.step

    def step_env(self, obs, action, step_fn):
        prev_reset = self.env_output.reset
        if isinstance(action, tuple):
            if len(action) == 2:
                action, terms = action
                self.env_output = self.env.step(action)
                self.step += self._frames_per_step
            elif len(action) == 3:
                action, frame_skip, terms = action
                frame_skip += 1     # plus 1 as values returned start from zero
                self.env_output = self.env.step(action, frame_skip=frame_skip)
                self.step += np.sum(frame_skip)
            else:
                raise ValueError(f'Invalid action "{action}"')
        else:
            self.env_output = self.env.step(action)
            self.step += self._frames_per_step
            terms = {}

        next_obs, reward, discount, reset = self.env_output

        if step_fn:
            kwargs = dict(obs=obs, action=action, reward=reward,
                discount=discount, next_obs=next_obs)
            assert 'reward' not in terms, 'reward in terms is from the preivous timestep and should not be used to override here'
            # allow terms to overwrite the values in kwargs
            kwargs.update(terms)
            step_fn(self.env, self.step, prev_reset, **kwargs)

        return next_obs, reset
    
    def store_info(self, info):
        if isinstance(info, list):
            score = [i['score'] for i in info]
            epslen = [i['epslen'] for i in info]
        else:
            score = info['score']
            epslen = info['epslen']
        self.agent.store(score=score, epslen=epslen)
        if self._info_func is not None:
            self._info_func(self.agent, info)

def evaluate(env, 
             agent, 
             n=1, 
             record_video=False, 
             size=None, 
             video_len=1000, 
             step_fn=None, 
             record_stats=False,
             n_windows=4):
    scores = []
    epslens = []
    max_steps = env.max_episode_steps // getattr(env, 'frame_skip', 1)
    frames = [collections.deque(maxlen=video_len) 
        for _ in range(min(n_windows, env.n_envs))]
    if hasattr(agent, 'reset_states'):
        agent.reset_states()
    env_output = env.reset()
    n_run_eps = env.n_envs  # count the number of episodes that has begun to run
    n = max(n, env.n_envs)
    n_done_eps = 0
    frame_skip = None
    obs = env_output.obs
    prev_done = np.zeros(env.n_envs)
    while n_done_eps < n:
        for k in range(max_steps):
            if record_video:
                img = env.get_screen(size=size)
                if env.env_type == 'Env':
                    frames[0].append(img)
                else:
                    for i in range(len(frames)):
                        frames[i].append(img[i])
                    
            action = agent(
                env_output, 
                evaluation=True, 
                return_eval_stats=record_stats)
            terms = {}
            if isinstance(action, tuple):
                if len(action) == 2:
                    action, terms = action
                elif len(action) == 3:
                    action, frame_skip, terms = action
                else:
                    raise ValueError(f'Unkown model return: {action}')
            if frame_skip is not None:
                frame_skip += 1     # plus 1 as values returned start from zero
                env_output = env.step(action, frame_skip=frame_skip)
            else:
                env_output = env.step(action)
            next_obs, reward, discount, reset = env_output

            if step_fn:
                step_fn(obs=obs, action=action, reward=reward, 
                    discount=discount, next_obs=next_obs, 
                    reset=reset, **terms)
            obs = next_obs
            if env.env_type == 'Env':
                if env.game_over():
                    scores.append(env.score())
                    epslens.append(env.epslen())
                    n_done_eps += 1
                    if n_run_eps < n:
                        n_run_eps += 1
                        env_output = env.reset()
                        if hasattr(agent, 'reset_states'):
                            agent.reset_states()
                    break
            else:
                done = env.game_over()
                done_env_ids = [i for i, (d, pd) in 
                    enumerate(zip(done, prev_done)) if d and not pd]
                n_done_eps += len(done_env_ids)
                if done_env_ids:
                    score = env.score(done_env_ids)
                    epslen = env.epslen(done_env_ids)
                    scores += score
                    epslens += epslen
                    if n_run_eps < n:
                        reset_env_ids = done_env_ids[:n-n_run_eps]
                        n_run_eps += len(reset_env_ids)
                        eo = env.reset(reset_env_ids)
                        for t, s in zip(env_output, eo):
                            if isinstance(t, dict):
                                for k in t.keys():
                                    for i, ri in enumerate(reset_env_ids):
                                        t[k][ri] = s[k][i]
                            else:
                                for i, ri in enumerate(reset_env_ids):
                                    t[ri] = s[i]
                    elif n_done_eps == n:
                        break
                prev_done = done

    if record_video:
        max_len = np.max([len(f) for f in frames])
        # padding to make all sequences of the same length
        for i, f in enumerate(frames):
            while len(f) < max_len:
                f.append(f[-1])
            frames[i] = np.array(f)
        frames = np.array(frames)
        return scores, epslens, frames
    else:
        return scores, epslens, None
