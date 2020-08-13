import random
# import scipy as sp
from scipy import special
import torch
import torch.nn as nn
from torch.distributions import Beta, Uniform
import json
import gym
from gym import spaces
import pandas as pd
import plotnine as pn
import numpy as np
from spinup.algos.pytorch.vpg.core import *


class ResponderBot():
    def __init__(self, x0, k):
        self.x0 = x0
        self.k = k
    def accept_prob(self, offer):
        logit = special.logit(offer)
        return special.expit(self.k*(logit - self.x0))
    def expected_value(self, offer):
        prob = self.accept_prob(offer)
        return prob*(1-offer)
    def respond(self, offer):
        prob = self.accept_prob(offer)
        return np.random.binomial(1,prob)
    def plot_prob(self, steps=100):
        df = pd.DataFrame({'offer': np.arange(0, 1.0001, 1.0/steps)})
        df['prob'] = [self.accept_prob(o) for o in df['offer']]
        print(pn.ggplot(df, pn.aes(x='offer', y='prob')) + pn.geom_line())
    def plot_ev(self, steps=100):
        df = pd.DataFrame({'offer': np.arange(0, 1.0001, 1.0/steps)})
        df['ev'] = [self.expected_value(o) for o in df['offer']]
        print(pn.ggplot(df, pn.aes(x='offer', y='ev')) + pn.geom_line())
        # return max expected value
        return df.loc[df['ev'].idxmax(),:]


class UltimatumResponderBotEnv(gym.Env):
    """An environment consisting of a single Ultimatum responder bot based on OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, x0=None, k=None, n_turns=10):
        super(UltimatumResponderBotEnv, self).__init__()
        if x0 is None:
            self.x0 = np.random.uniform()
        else:
            self.x0 = x0
        if k is None:
            self.k = np.random.uniform(0, 10)
        else:
            self.k = k
        self.bot = ResponderBot(self.x0, self.k)
        self.reward_range = (0, 1)
        self.current_step = 0
        self.offer = 0
        self.response = 0
        self.n_turns = n_turns
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([1]), dtype=np.float16)

        # self.observation_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([1]), dtype=np.float16)

    def _get_response(self, action):
        return self.bot.respond(action)

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        self.offer = action
        self.response = self._get_response(action)
        reward = self.response * (1 - action)
        done = self.current_step >= self.n_turns
        # done = False
        # obs = self.response
        obs = np.array([[0.]])
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.x0 = np.random.uniform()
        self.k = np.random.uniform(0, 10)
        self.bot = ResponderBot(self.x0, self.k)
        self.current_step = 0
        self.offer = 0
        self.response = 0
        return np.array([[0.]])

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Last offer: {self.offer}')
        print(f'Last response: {self.response}')


class ProposerPolicy(nn.Module):
    def __init__(self):
        super(ProposerPolicy, self).__init__()
        self.beta = nn.Linear(1, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.beta(x)
        x = torch.abs(x)
#         x = F.relu(x) + 1e-8
        return x


class ProposerContinuousActor(Actor):

    def __init__(self, distr='beta_abs'):
        super().__init__()
        self.distr = distr
        layers = []
        if self.distr=='beta_abs':
            # use absolute values of output as Beta distrib coeffs
            layers.append(nn.Linear(1, 2))
            layers.append(nn.LeakyReLU(-1))
        elif self.distr=='beta_exp':
            # use exponents of output as Beta distrib coeffs
            layers.append(nn.Linear(1, 2))
        elif self.distr=='beta_softplus':
            # use absolute values of output as Beta distrib coeffs
            layers.append(nn.Linear(1, 2))
            layers.append(nn.Softplus())
        # elif self.distr=='logistic_norm':
        #     # let the offer distrib be the logistic function of a normal distribution
        #     layers.append(nn.Linear(1, 2))
        self.net = nn.Sequential(*layers)

    def _distribution(self, obs):
        if self.distr=='beta_abs':
            beta_coeffs = self.net(obs)
            return Beta(beta_coeffs[:, 0], beta_coeffs[:, 1])
        elif self.distr=='beta_exp':
            beta_coeffs = torch.exp(self.net(obs))
            return Beta(beta_coeffs[:, 0], beta_coeffs[:, 1])
        elif self.distr=='beta_softplus':
            beta_coeffs = self.net(obs)
            # print(f'obs {obs}')
            # print(f'beta {beta_coeffs}')
            return Beta(beta_coeffs[:, 0], beta_coeffs[:, 1])
            # return Beta(beta_coeffs[0], beta_coeffs[1])
        # elif self.distr=='logistic_norm':

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class ProposerCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class ProposerActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh, distr='beta_softplus'):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        # self.pi = ProposerContinuousActor(obs_dim, action_space.shape[0], hidden_sizes, activation, distr=distr)
        self.pi = ProposerContinuousActor(distr=distr)

        # build value function
        self.v = ProposerCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample().unsqueeze(-1)
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# policy = ProposerPolicy()
# optimizer = optim.Adam(policy.parameters(), lr=1e-3)
# eps = np.finfo(np.float32).eps.item()
#
# def select_action(state):
#     state = torch.from_numpy(state).float().unsqueeze(0)
#     beta_coeffs = policy(state).split(1)
#     m = Beta(beta_coeffs[0], beta_coeffs[1])
#     action = m.sample()
#     policy.saved_log_probs.append(m.log_prob(action))
#     return action.item()
