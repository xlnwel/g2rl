import collections
from copy import deepcopy
import functools
import numpy as np
import tensorflow as tf

from utility.timer import Every, Timer
from algo.ppo.train import main


def random_run(env, step):
    reset = [False for _ in range(env.n_envs)]
    trajs = [collections.defaultdict(list) for _ in range(env.n_envs)]
    obs, _, _, _ = env.output()

    while not np.all(reset):
        next_obs, reward, discount, reset = env.step(env.random_action())
        assert np.all(reset[0] == reset), reset
        discount[np.any(discount, 1)] = 1
        kwargs = dict(
            obs=obs['obs'],
            global_state=obs['global_state'],
            reward=reward,
            discount=discount,
        )
        if env.use_life_mask:
            kwargs['life_mask'] = obs['life_mask']
        trans = [{k: v[i] for k, v in kwargs.items()} for i in range(env.n_envs)]
        info_list = env.info()
        for i, (info, traj) in enumerate(zip(info_list, trans)):
            if info['valid_step']:
                for k, v in traj.items():
                    trajs[i][k].append(v)
        
        obs = next_obs
        
    step += np.sum(env.epslen())
    
    data = {k: np.concatenate([traj[k] for traj in trajs if traj])
        for k in kwargs.keys()}
    keys = ('obs', 'global_state', 'life_mask') if env.use_life_mask \
        else ('obs', 'global_state')
    data = {k: np.concatenate(v) if k in keys else v
        for k, v in data.items()}
    
    return step, data

def run(agent, env, buffer, step):
    prev_reset = [False for _ in range(env.n_envs)]
    agent.reset_states()
    buffer.clear_buffer()

    env_output = env.output()
    last_env_output = deepcopy(env_output)

    while not np.all(prev_reset):
        action, terms = agent(env_output, evaluation=False)
        env_output = env.step(action)
        _, reward, discount, reset = env_output
        discount[np.any(discount, 1)] = 1
        kwargs = dict(
            action=action.reshape(env.n_envs, env.n_agents),
            reward=reward,
            discount=discount,
            **tf.nest.map_structure(
                lambda x: x.reshape(env.n_envs, env.n_agents, *x.shape[1:]), terms)
        )
        kwargs = [{k: v[i] for k, v in kwargs.items()} for i in range(env.n_envs)]
        info_list = env.info()
        for i, (info, data) in enumerate(zip(info_list, kwargs)):
            if info['bad_episode']:
                buffer.remove(i)
            elif info['valid_step']:
                buffer.add(i, **data)
                last_env_output[i] = env_output[i]
        prev_reset = reset

    score = env.score()
    epslen = env.epslen()
    win_rate = [info['won'] for info in info_list]
    step += np.sum(epslen)
    agent.store(score=score, epslen=epslen, win_rate=win_rate)

    return step, last_env_output

def train(agent, env, eval_env, buffer):
    def initialize_rms(step):
        if step == 0 and agent.actor.is_obs_normalized:
            print('Start to initialize running stats...')
            for i in range(10):
                step, data = random_run(env, step)
                life_mask = data.get('life_mask')
                agent.actor.update_obs_rms(data['obs'], mask=life_mask)
                agent.actor.update_obs_rms(data['global_state'], 
                    'global_state', mask=life_mask)
                agent.actor.update_reward_rms(data['reward'], data['discount'])
            agent.set_env_step(step)
            agent.save(print_terminal_info=True)
        return step
    step = initialize_rms(agent.get_env_step())

    # print("Initial running stats:", *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    rt = Timer('run')
    tt = Timer('train')
    lt = Timer('log')

    def collect_data(step, agent, env, buffer):
        with rt:
            step, last_env_output = run(agent, env, buffer, step)
        
        for i in range(env.n_envs):
            if not buffer.is_valid_traj(i):
                continue
            reward = buffer.get(i, 'reward')
            discount = buffer.get(i, 'discount')
            agent.actor.update_reward_rms(reward, discount)
            buffer.update_buffer(i, 'reward', agent.actor.normalize_reward(reward))
        agent.record_inputs_to_vf(last_env_output)
        value = agent.compute_value()
        buffer.finish(value)
        return step

    def record_stats():
        with lt:
            agent.store(**{
                'time/fps': (step - start_env_step) / rt.last(),
                'time/tps': (agent.get_train_step()-start_train_step)/tt.last(),
                'misc/train_step': agent.get_train_step(),
                'time/run': rt.total(), 
                'time/train': tt.total(),
                'time/log': lt.total(),
                'time/run_mean': rt.average(), 
                'time/train_mean': tt.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    print('Training starts...')
    while step < agent.MAX_STEPS:
        buffer.reset()
        start_env_step = agent.get_env_step()
        while not buffer.ready():
            step = collect_data(step, agent, env, buffer)
        start_train_step = agent.get_train_step()
        with tt:
            agent.train_record()
        agent.store(
            fps=(step - start_env_step) / rt.last(),
            tps=(agent.get_train_step()-start_train_step) / tt.last()
        )
        agent.set_env_step(step)
        if to_record(agent.get_train_step()) and agent.contains_stats('score'):
            record_stats()

main = functools.partial(main, train=train)
