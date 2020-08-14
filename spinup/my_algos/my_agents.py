import numpy as np

import torch
from torch.distributions import Normal
import torch.nn.functional as F
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


class NormalDistActor(nn.Module):
    """
    A stochastic policy for use in on-policy methods
    forward method: returns Normal dist and logprob of an action if passed
    model: N(mu, sigma) with MLP for mu, log(sigma) is a tunable parameter
    Input: observation/state
    """

    def __init__(self, layer_sizes_mu, act_dim, act_low, act_high, activation):
        super().__init__()
        self.mu_net = mlp(layer_sizes_mu, activation, nn.Identity)
        log_sigma = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_sigma = torch.nn.Parameter(torch.as_tensor(log_sigma))

    def distribution(self, obs):
        mu = self.mu_net(obs)
        sigma = torch.exp(self.log_sigma)
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
        # x should contain [obs, act] for off-policy methods
        output = self.net(x)
        return torch.squeeze(output, -1)  # Critical to ensure v has right shape.


class BoundedDeterministicActor(nn.Module):
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


class BoundedStochasticActor(nn.Module):
    """
    Produces a squashed Normal distribution for one var from a MLP for mu and sigma.
    """

    def __init__(
        self,
        layer_sizes,
        act_dim,
        act_low,
        act_high,
        activation=nn.ReLU,
        log_sigma_min=-20,
        log_sigma_max=2,
    ):
        super().__init__()
        self.act_low = torch.as_tensor(act_low)
        self.act_width = torch.as_tensor(act_high - act_low)
        self.shared_net = mlp(layer_sizes, activation, activation)
        self.mu_layer = nn.Linear(layer_sizes[-1], act_dim, activation)
        self.log_sigma_layer = nn.Linear(layer_sizes[-1], act_dim, activation)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, obs, deterministic=False, get_logprob=False):
        shared = self.shared_net(obs)
        mu = self.mu_layer(shared)
        log_sigma = self.log_sigma_layer(shared)
        log_sigma = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = torch.exp(log_sigma)
        pi = Normal(mu, sigma)
        if deterministic:
            # For evaluating performance at end of epoch, not for data collection
            act = mu
        else:
            act = pi.rsample()
        logprob = None
        if get_logprob:
            logprob = pi.log_prob(act).sum(axis=-1)
            # Convert pdf due to tanh transform
            logprob -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=1)
        act = torch.tanh(act)
        act = (act + 1) * self.act_width / 2 + self.act_low
        return act, logprob


class GaussianActorCritic(nn.Module):
    """
    Contains an actor (to produce policy and act)
    and a critic (to estimate value function)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_mu=(64, 64),
        hidden_layers_v=(64, 64),
        activation=nn.Tanh,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_mu = [obs_dim] + list(hidden_layers_mu) + [act_dim]
        layer_sizes_v = [obs_dim] + list(hidden_layers_v) + [1]
        self.pi = NormalDistActor(
            layer_sizes_mu=layer_sizes_mu,
            act_dim=act_dim,
            activation=activation,
        )
        self.v = ContinuousEstimator(layer_sizes_v=layer_sizes_v, activation=activation)
        self.act_low = action_space.low
        self.act_high = action_space.high

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi.distribution(obs)
            act = pi.sample()
            logprob = pi.log_prob(act)
            logprob = logprob.sum(axis=-1)
            act = act.clamp(self.act_low, self.act_high)
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

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_mu=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        final_activation=nn.Tanh,
        noise_std=0.1,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_low = action_space.low
        self.act_high = action_space.high

        layer_sizes_mu = [obs_dim] + list(hidden_layers_mu) + [self.act_dim]
        layer_sizes_q = [obs_dim + self.act_dim] + list(hidden_layers_q) + [1]
        self.noise_std = noise_std
        self.pi = BoundedDeterministicActor(
            layer_sizes=layer_sizes_mu,
            activation=activation,
            final_activation=final_activation,
            low=self.act_low,
            high=self.act_high,
            **kwargs
        )
        self.q = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )

    def act(self, obs, noise=False):
        """Return noisy action as numpy array, **without computing grads**"""
        # TO DO: fix how noise and clipping are handled for multiple dimensions.
        with torch.no_grad():
            act = self.pi(obs).numpy()
            if noise:
                act += self.noise_std * np.random.randn(self.act_dim)
            act = np.clip(act, self.act_low[0], self.act_high[0])
        return act


class TD3Agent(nn.Module):
    """
    Agent to be used in TD3.
    Contains:
    - estimated Q*(s,a,)
    - policy

    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_mu=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        final_activation=nn.Tanh,
        noise_std=0.1,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_low = action_space.low
        self.act_high = action_space.high

        layer_sizes_mu = [obs_dim] + list(hidden_layers_mu) + [act_dim]
        layer_sizes_q = [obs_dim + act_dim] + list(hidden_layers_q) + [1]
        self.noise_std = noise_std
        self.pi = BoundedDeterministicActor(
            layer_sizes=layer_sizes_mu,
            activation=activation,
            final_activation=final_activation,
            low=self.act_low,
            high=self.act_high,
            **kwargs
        )
        self.q1 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )
        self.q2 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )

    def act(self, obs, noise=False):
        """Return noisy action as numpy array, **without computing grads**"""
        # TO DO: fix how noise and clipping are handled for multiple dimensions.
        with torch.no_grad():
            act = self.pi(obs).numpy()
            if noise:
                act += self.noise_std * np.random.randn(self.act_dim)
            act = np.clip(act, self.act_low[0], self.act_high[0])
        return act


class SACAgent(nn.Module):
    """
    Agent to be used in SAC.
    Contains:
    - stochastic policy (bounded by tanh)
    - estimated Q*(s,a,)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_pi=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_pi = [obs_dim] + list(hidden_layers_pi)
        layer_sizes_q = [obs_dim + act_dim] + list(hidden_layers_q) + [1]
        self.pi = BoundedStochasticActor(
            layer_sizes_pi,
            act_dim,
            action_space.low,
            action_space.high,
            activation=activation,
            **kwargs
        )
        self.q1 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )
        self.q2 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )

    def act(self, obs, deterministic=False):
        """Return noisy action as numpy array, **without computing grads**"""
        with torch.no_grad():
            act, _ = self.pi(obs, deterministic=deterministic)
        return act.numpy()


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


# For on-policy VPG/PPO
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
            "obs": self.obs,
            "act": self.act,
            "adv": self.adv,
            "ret": self.ret,
            "logprob": self.logprob,
        }
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        return data


# For off-policy methods
class TransitionBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(merge_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.obs_next = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
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
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.reward[self.ptr] = reward
        self.obs_next[self.ptr] = obs_next
        self.done[self.ptr] = done
        self.ptr += 1
        if not self.full:
            self.filled_size += 1

    def get(self, sample_size=100):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Reset pointers for next epoch.
        """
        # can only get data when buffer is full
        # if not self.full:
        #     raise Exception('Buffer cannot be sampled until it is full.')
        # return needed variables as a dictionary
        sample_indexes = np.random.randint(0, self.filled_size, sample_size)
        data = {
            "obs": torch.as_tensor(self.obs[sample_indexes], dtype=torch.float32),
            "act": torch.as_tensor(self.act[sample_indexes], dtype=torch.float32),
            "reward": torch.as_tensor(self.reward[sample_indexes], dtype=torch.float32),
            "obs_next": torch.as_tensor(
                self.obs_next[sample_indexes], dtype=torch.float32
            ),
            "done": torch.as_tensor(self.done[sample_indexes], dtype=torch.float32),
        }
        return data
