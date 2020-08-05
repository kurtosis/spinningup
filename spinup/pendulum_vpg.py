import gym
from spinup import vpg_pytorch
from agents.gym_pendulum import *


def env_fn(seed=3):
        env = gym.make('Pendulum-v0')
        env.seed(seed)
        return env

vpg_pytorch(env_fn, actor_critic=PendulumAgent, ac_kwargs=dict(),  seed=0,
            steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
            vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
            logger_kwargs=dict(), save_freq=10)