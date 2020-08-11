import numpy as np

import torch
from torch.distributions import Beta, Uniform, Normal
import torch.nn as nn


def mlp(layer_sizes, hidden_activation, final_activation):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(hidden_activation())
        else:
            layers.append(final_activation())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """
    Produces a Normal distribution for one var from a MLP for mu and sigma.
    Input dimension: 3 (observation)
    """

    def __init__(self, layer_sizes_mu, layer_sizes_sigma, activation):
        super().__init__()
        self.mu_net = mlp(layer_sizes_mu, activation, nn.Identity)
        # self.sigma_net = mlp(layer_sizes_sigma, activation, nn.Softplus)
        self.log_sigma_net = mlp(layer_sizes_sigma, activation, nn.Identity)
        # adjust this to use action size
        # self.log_sigma = torch.nn.Parameter(torch.Tensor([-0.5]))

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        # sigma = self.sigma_net(obs)
        sigma = torch.exp(self.log_sigma_net(obs))
        # sigma = torch.exp(self.log_sigma)
        return Normal(mu, sigma)

    def _logprob_from_distr(self, pi, act):
        # Need sum for a Normal distribution
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logprob = None
        if act is not None:
            logprob = self._logprob_from_distr(pi, act)
        return pi, logprob


class ValueCritic(nn.Module):
    """
    Produces a value estimate from a MLP.
    Input dimension: 3 (observation)
    Output dimension: 1
    """

    def __init__(self, layer_sizes_v, activation):
        super().__init__()
        self.v_net = mlp(layer_sizes_v, activation, nn.Identity)

    def forward(self, x):
        v = self.v_net(x)
        return torch.squeeze(v, -1)  # Critical to ensure v has right shape.


class ContinuousEstimator(nn.Module):
    """
    Generic MLP object to output continuous value(s).
    Can be used for:
        - V function (input: s, output: exp return)
        - Q function (input: (s, a), output: exp return)
        - deterministic policy (input: s, output: a)
    Layer sizes passed as argument.
    Input dimension: layer_sizes[0]
    Output dimension: layer_sizes[-1] (should be 1 for V,Q)
    """

    def __init__(self, layer_sizes, activation, final_activation=nn.Identity, **kwargs):
        super().__init__()
        self.net = mlp(layer_sizes, activation, final_activation)

    def forward(self, x):
        output = self.net(x)
        return torch.squeeze(output, -1)  # Critical to ensure v has right shape.


class BoundedContinuousActor(nn.Module):
    """
    MLP net for actor in bounded continuous action space.
    Returns deterministic action.
    Layer sizes passed as argument.
    Input dimension: layer_sizes[0]
    Output dimension: layer_sizes[-1] (should be 1 for V,Q)
    """

    def __init__(self, layer_sizes, activation, low, high, **kwargs):
        super().__init__()
        self.low = torch.as_tensor(low)
        self.width = torch.as_tensor(high - low)
        self.net = mlp(layer_sizes, activation, nn.Tanh)

    def forward(self, x):
        output = (self.net(x) + 1) * self.width / 2 + self.low
        return output


class GaussianActorCritic(nn.Module):
    """
    Contains an actor (to produce policy and act)
    and a critic (to estimate value function)
    """

    def __init__(self, observation_space, action_space,
                 hidden_layers_mu=[100], hidden_layers_sigma=[100], hidden_layers_v=[100],
                 activation=nn.Tanh,
                 **kwargs):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_mu = [obs_dim] + hidden_layers_mu + [act_dim]
        layer_sizes_sigma = [obs_dim] + hidden_layers_sigma + [act_dim]
        layer_sizes_v = [obs_dim] + hidden_layers_v + [1]
        self.pi = GaussianActor(layer_sizes_mu=layer_sizes_mu, layer_sizes_sigma=layer_sizes_sigma,
                                activation=activation)
        self.v = ValueCritic(layer_sizes_v=layer_sizes_v, activation=activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            act = pi.sample()
            logprob = pi.log_prob(act)
            logprob = logprob.sum(axis=-1)
            act = act.clamp(-2, 2)
            val = self.v(obs)
        return act.numpy(), val.numpy(), logprob.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class DDPGAgent(nn.Module):
    """
    Agent to be used in DDPG.
    Contains:
    - estimated Q*(s,a,)
    - policy

    """

    def __init__(self, observation_space, action_space,
                 hidden_layers_mu=[256, 256], hidden_layers_q=[256, 256],
                 activation=nn.ReLU,
                 final_activation=nn.Tanh,
                 noise_std=0.1, gamma=0.9,
                 **kwargs):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_low = action_space.low
        self.act_high = action_space.high

        layer_sizes_mu = [obs_dim] + hidden_layers_mu + [act_dim]
        layer_sizes_q = [obs_dim + act_dim] + hidden_layers_q + [1]
        self.noise_std = noise_std
        self.gamma = gamma
        self.policy = BoundedContinuousActor(layer_sizes=layer_sizes_mu, activation=activation,
                                             final_activation=final_activation, low=self.act_low, high=self.act_high,
                                             **kwargs)
        self.q = ContinuousEstimator(layer_sizes=layer_sizes_q, activation=activation, **kwargs)

    def act(self, obs, noise=False):
        """Return noisy action as numpy array, **without computing grads**"""
        # TO DO: fix how noise and clipping are handled for multiple dimensions.
        with torch.no_grad():
            act = self.policy(obs)
            if noise:
                act += self.noise_std * np.random.randn(self.act_dim)
            act = np.clip(act.numpy(), self.act_low[0], self.act_high[0])
        return act


class TD3Agent(nn.Module):
    """
    Agent to be used in TD3.
    Contains:
    - estimated Q*(s,a,)
    - policy

    """

    def __init__(self, observation_space, action_space,
                 hidden_layers_mu=[256, 256], hidden_layers_q=[256, 256],
                 activation=nn.ReLU,
                 final_activation=nn.Tanh,
                 noise_std=0.1, gamma=0.9,
                 **kwargs):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_low = action_space.low
        self.act_high = action_space.high

        layer_sizes_mu = [obs_dim] + hidden_layers_mu + [act_dim]
        layer_sizes_q = [obs_dim + act_dim] + hidden_layers_q + [1]
        self.noise_std = noise_std
        self.gamma = gamma
        self.policy = BoundedContinuousActor(layer_sizes=layer_sizes_mu, activation=activation,
                                             final_activation=final_activation, low=self.act_low, high=self.act_high,
                                             **kwargs)
        self.q1 = ContinuousEstimator(layer_sizes=layer_sizes_q, activation=activation, **kwargs)
        self.q2 = ContinuousEstimator(layer_sizes=layer_sizes_q, activation=activation, **kwargs)

    def act(self, obs, noise=False):
        """Return noisy action as numpy array, **without computing grads**"""
        # TO DO: fix how noise and clipping are handled for multiple dimensions.
        with torch.no_grad():
            act = self.policy(obs)
            if noise:
                act += self.noise_std * np.random.randn(self.act_dim)
            act = np.clip(act.numpy(), self.act_low[0], self.act_high[0])
        return act



def discount_cumsum(x, discount):
    """Compute cumsum with discounting used in GAE (generalized adv estimn).
    (My implementation)"""
    discounts = [discount ** ll for ll in range(len(x))]
    disc_seqs = [discounts] + [discounts[:-i] for i in range(1, len(x))]
    return np.array([np.dot(x[i:], disc_seqs[i]) for i in range(len(x))])


def merge_shape(shape1, shape2=None):
    if shape2 is None:
        return (shape1,)
    elif np.isscalar(shape2):
        return (shape1, shape2)
    else:
        return (shape1, *shape2)


# For VPG/PPO
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


class DDPGBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
        self.obs_next = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(merge_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.max_size = size
        self.filled_size = 0
        self.full = False

    def store(self, obs, act, reward, obs_next, done):
        """Add current step variables to buffer."""
        # Cycle through buffer, overwriting oldest entry.
        # Note that buffer is never flushed, unlike on-policy methods.
        if self.ptr >= self.max_size:
            self.ptr = self.ptr % self.max_size
            self.full = True
        if not self.full:
            self.filled_size += 1
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.reward[self.ptr] = reward
        self.obs_next[self.ptr] = obs_next
        self.done[self.ptr] = done
        self.ptr += 1

    def get(self, sample_size=None):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Reset pointers for next epoch.
        """
        # can only get data when buffer is full
        # if not self.full:
        #     raise Exception('Buffer cannot be sampled until it is full.')
        # return needed variables as a dictionary
        if sample_size is None:
            data = {
                'obs': torch.as_tensor(self.obs, dtype=torch.float32),
                'act': torch.as_tensor(self.act, dtype=torch.float32),
                'reward': torch.as_tensor(self.reward, dtype=torch.float32),
                'obs_next': torch.as_tensor(self.obs_next, dtype=torch.float32),
                'done': torch.as_tensor(self.done, dtype=torch.float32),
            }
        else:
            sample_indexes = np.random.randint(0, self.filled_size, sample_size)
            data = {
                'obs': torch.as_tensor(self.obs[sample_indexes], dtype=torch.float32),
                'act': torch.as_tensor(self.act[sample_indexes], dtype=torch.float32),
                'reward': torch.as_tensor(self.reward[sample_indexes], dtype=torch.float32),
                'obs_next': torch.as_tensor(self.obs_next[sample_indexes], dtype=torch.float32),
                'done': torch.as_tensor(self.done[sample_indexes], dtype=torch.float32),
            }
        return data
