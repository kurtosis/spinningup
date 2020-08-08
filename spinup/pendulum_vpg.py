from textwrap import dedent
import time

import gym

from spinup import vpg_pytorch, ppo_pytorch
from spinup.algos.pytorch.vpg.core import *
from spinup.agents.gym_pendulum import *


from spinup.user_config import DEFAULT_DATA_DIR, FORCE_DATESTAMP, \
                               DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH
from spinup.utils.logx import colorize

DIV_LINE_WIDTH = 80


def create_output_msg(logger_kwargs):
    plot_cmd = 'python -m spinup.run plot ' + logger_kwargs['output_dir']
    plot_cmd = colorize(plot_cmd, 'green')
    test_cmd = 'python -m spinup.run test_policy ' + logger_kwargs['output_dir']
    test_cmd = colorize(test_cmd, 'green')
    output_msg = '\n' * 5 + '=' * DIV_LINE_WIDTH + '\n' + dedent("""\
    End of experiment.
    
    
    Plot results from this run with:
    
    %s
    
    
    Watch the trained agent with:
    
    %s
    
    
    """ % (plot_cmd, test_cmd)) + '=' * DIV_LINE_WIDTH + '\n' * 5
    return output_msg


# env_str = 'Pendulum-v0'
# env_str = 'Hopper-v3'
env_str = 'Swimmer-v3'
env_dir = env_str.lower().replace('-','_')
output_dir = f'{DEFAULT_DATA_DIR}/{env_dir}/{int(time.time())}'
logger_kwargs = {'output_dir': output_dir}
ac_kwargs={
    'hidden_layers_mu' : [64,64],
    'hidden_layers_sigma' : [64,64],
    'hidden_layers_v' : [64,64],
}

output_msg = create_output_msg(logger_kwargs)


# actor_critic = PendulumAgent
actor_critic = MLPActorCritic

if actor_critic == PendulumAgent:
    ac_kwargs = {
        'hidden_layers_mu': [64, 64],
        # 'hidden_layers_sigma': [64, 64],
        'hidden_layers_sigma': [32],
        'hidden_layers_v': [64, 64],
    }
else:
    ac_kwargs = {}


seed = 310

# vpg_pytorch(lambda: gym.make(env_str), actor_critic=actor_critic, ac_kwargs=ac_kwargs,  seed=0,
#             steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
#             vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=200,
#             logger_kwargs=logger_kwargs, save_freq=10)

my_vpg(lambda: gym.make(env_str), actor_critic, ac_kwargs=ac_kwargs,  seed=0,
            steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
            vf_lr=1e-3, train_v_iters=80, lam=0.97, max_episode_len=200,
            logger_kwargs=logger_kwargs, save_freq=10, render=False)

# ppo_pytorch(lambda: gym.make(env_str), actor_critic=actor_critic, ac_kwargs=dict(), seed=seed,
# ppo_pytorch(lambda: gym.make(env_str), actor_critic=actor_critic, ac_kwargs=ac_kwargs, seed=seed,
#             steps_per_epoch=4000, epochs=120, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
#             vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
#             target_kl=0.01, logger_kwargs=logger_kwargs, save_freq=10)

print(output_msg)
