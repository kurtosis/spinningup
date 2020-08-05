import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import scipy.signal
import sys
# from scipy import special
sys.path.insert(0, '/Users/kurtsmith/research/pytorch_projects/reinforcement_learning/environments')
sys.path.insert(0, '/Users/kurtsmith/research/spinningup')
# from ultimatum_env import *
import gym
import importlib
import numpy as np
import pandas as pd
# import plotnine as pn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import ultimatum_env
importlib.reload(ultimatum_env)
import spinup.algos.pytorch.vpg.core as core
import spinup.algos.pytorch.vpg.vpg as vpg

from torch.distributions import Beta, Uniform, Normal


class PendulumActor(nn.Module):
    """
    Produces a Normal distribution for one var from a MLP for mu and sigma.
    Input dimension: 3 (observation)
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def net(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        #         log_sigma = 2.0*torch.tanh(self.sigma_head(x))
        sigma = F.softplus(self.sigma_head(x))
        #         std = torch.exp(self.log_std)
        return (mu, sigma)

    def _distribution(self, obs):
        #         o = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.net(obs)
        return Normal(mu, sigma)

    def _logprob_from_distr(self, pi, act):
        # Need sum for a Normal distribution
        return pi.log_prob(act).sum(axis=1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logprob = None
        if act is not None:
            logprob = self._logprob_from_distr(pi, act)
        return pi, logprob


class PendulumCritic(nn.Module):
    """
    Produces a value estimate from a MLP.
    Input dimension: 3 (observation)
    Output dimension: 1
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        v = self.v_head(x)
        return torch.squeeze(v, -1) # Critical to ensure v has right shape.


class PendulumAgent():
    """
    Contains an actor (to produce policy and act)
    and a critic (to estimate value function)
    """

    def __init__(self):
        self.actor = PendulumActor()
        self.critic = PendulumCritic()

    def step(self, obs):
        with torch.no_grad():
            pi = self.actor._distribution(obs)
            act = pi.sample()
            logprob = pi.log_prob(act)
            logprob = logprob.sum(axis=1)
            # logprob = logprob.sum()
            act = act.clamp(-2, 2)
            val = self.critic(obs)
        return act.numpy(), val.numpy(), logprob.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# my implementation
def discount_cumsum(x, discount):
    discounts = [discount ** ll for ll in range(len(x))]
    disc_seqs = [discounts] + [discounts[:-i] for i in range(1,len(x))]
    return np.array([np.dot(x[i:], disc_seqs[i]) for i in range(len(x))])


def discount_cumsum_orig(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def merge_shape(shape1, shape2=None):
    if shape2 is None:
        return (shape1,)
    elif np.isscalar(shape2):
        return (shape1, shape2)
    else:
        return (shape1, *shape2)


class TrajectoryBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lamb=0.95):
        self.obs = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(merge_shape(size, act_dim), dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)
        self.v = np.zeros(size, dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lamb = lamb
        self.ptr = 0
        self.path_start = 0
        self.max_size = size

    def store(self, obs, act, reward, val, logprob):
        """Add current step variables to buffer."""
        assert self.ptr < self.max_size
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.reward[self.ptr] = reward
        self.v[self.ptr] = val
        self.logprob[self.ptr] = logprob
        self.ptr += 1

    def finish_path(self, last_v=0):
        """
        We've logged most variables at each step in episode.
        There are two vars that can only be computed at end
        of episode (b/c they depend on future rewards):
        - Advantage (for GAE)
        - Return (using reward-to-go)
        Compute both of those here and save to buffer.
        Update start index for next episode.
        """
        # note location of current episode in buffer
        path_slice = slice(self.path_start, self.ptr)
        # get rewards and values of current episode, append the last step value
        rewards = np.append(self.reward[path_slice], last_v)
        values = np.append(self.v[path_slice], last_v)
        # compute advantage fn A(s_t,a_t) for each step in episode using GAE
        # write this to the buffer in the location of this episode
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv[path_slice] = discount_cumsum(deltas, self.gamma * self.lamb)
        # compute rewards to go
        self.ret[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]
        # Update start index for next episode
        self.path_start = self.ptr

    def get(self):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Note that advantage is normalized first.
        Reset pointers for next epoch.
        """
        # can only get data when buffer is full
        assert self.ptr == self.max_size
        # reset pointers for next epoch
        self.ptr = 0
        self.path_start = 0
        # Normalize adv for GAE
        adv_mean = self.adv.mean()
        adv_std = self.adv.std()
        self.adv = (self.adv - adv_mean) / adv_std
        # return needed variables as a dictionary
        data = {
            'obs': self.obs,
            'act': self.act,
            'adv': self.adv,
            'ret': self.ret,
            'logprob': self.logprob,
        }
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        return data


def compute_loss_pi(data, agent):
    # get data
    obs, act, adv, logprob_old = data['obs'], data['act'], data['adv'], data['logprob']

    # Get policy (given obs) and logprob of actions taken
    pi, logprob = agent.actor(obs, act)
    # The loss function equation for VPG (see docs)
    pi_loss = -(logprob * adv).mean()

    # TODO add info
    pi_info = None

    return pi_loss, pi_info


def compute_loss_v(data, agent):
    obs, ret = data['obs'], data['ret']
    v = agent.critic(obs)
    return ((v - ret) ** 2).mean()


def update(agent, buf, pi_optimizer, v_optimizer, train_v_iters=80):
    # TODO
    # - add logging at end
    # DONE: write loss computation

    # Get training data from buffer
    data = buf.get()

    # Compute policy/value losses
    pi_loss, _ = compute_loss_pi(data, agent)
    v_loss = compute_loss_v(data, agent).item()

    # Update policy by single step using optimizer
    pi_optimizer.zero_grad()
    pi_loss, _ = compute_loss_pi(data, agent)
    pi_loss.backward()
    pi_optimizer.step()

    # Update value function multiple steps using optimizer
    for i in range(train_v_iters):
        v_optimizer.zero_grad()
        v_loss = compute_loss_v(data, agent)
        v_loss.backward()
        v_optimizer.step()

    # add logging?


def print_info(x, name):
    print(f'{name} {x}')
    print(f'{name} type {type(x)}')
    print(f'{name} shape {x.shape}')


# TODO
# DONE finish buffer methods
# DONE create update step to learn agent models
# DONE figure out how to deal with epochs/episodes/steps
def train(seed=0, n_epochs=10, n_steps_per_epoch=200,
          log_interval=10, render=True,
          max_episode_length=100,
          pi_lr=3e-4, v_lr=1e-3):
    # Initialize environment, agent, auxilary objects
    env = gym.make('Pendulum-v0')
    env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    agent = PendulumAgent()
    training_records = []
    running_avg_return = -1000
    buf = TrajectoryBuffer(obs_dim, act_dim, n_steps_per_epoch)
    pi_optimizer = Adam(agent.actor.parameters(), lr=pi_lr)
    v_optimizer = Adam(agent.critic.parameters(), lr=v_lr)

    for i_epoch in range(n_epochs):
        # Start a new epoch
        obs = env.reset()
        episode_return = 0
        episode_length = 0
        episode_count = 0
        print(f'epoch {i_epoch}')
        for t in range(n_steps_per_epoch):
            print(f'step {t}')
            # Step agent given latest observation
            a, v, logprob = agent.step(torch.as_tensor([obs], dtype=torch.float32))

            # Step environment given latest agent action
            obs_next, reward, done, _ = env.step([a])
            # Store current step in buffer
            buf.store(obs, a, reward, v, logprob)
            # Visualize current state if desired
            if render:
                env.render()

            # update episode return and env state
            episode_length += 1
            episode_return += reward
            obs = obs_next

            episode_capped = (episode_length == max_episode_length)
            epoch_ended = (t == n_steps_per_epoch - 1)
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                episode_count += 1
                obs = env.reset()
                episode_return = 0
                episode_length = 0
                if not (done or episode_capped):
                    print(f'Trajectory terminated by end of epoch at step {episode_length}')
                # get last value function
                if episode_capped or epoch_ended:
                    _, v, _ = agent.step(torch.as_tensor([obs], dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)

        update(agent, buf, pi_optimizer, v_optimizer)
        # update running avg of episode score and training trajectory buffer
        running_avg_return = running_avg_return * 0.9 + episode_return * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        # Print running avg episode score at end of episode
        if i_ep % log_interval == 0:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))
        if running_reward > -200:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            #             agent.save_param()
            #             with open('log/ppo_training_records.pkl', 'wb') as f:
            #                 pickle.dump(training_records, f)
            break

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('VPG')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("./vpg.png")
    plt.show()


if __name__ == '__main__':
    train(seed=0, n_epochs=100, n_steps_per_epoch=10,
          log_interval=1, render=False,
          max_episode_length=500,
          pi_lr=3e-4, v_lr=1e-3)