import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys

sys.path.insert(
    0, "/Users/kurtsmith/research/pytorch_projects/reinforcement_learning/environments"
)
sys.path.insert(0, "/Users/kurtsmith/research/spinningup")

from copy import deepcopy
import numpy as np

import torch
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn

from spinup.agents.my_agents import *

def mlp(layer_sizes, hidden_activation, final_activation):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(hidden_activation())
        else:
            layers.append(final_activation())
    return nn.Sequential(*layers)


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
        # x should contain [obs, act]
        output = self.net(x)
        return torch.squeeze(output, -1)  # Critical to ensure v has right shape.


class TanhStochasticActor(nn.Module):
    """
    Produces a squashed Normal distribution for one var from a MLP for mu and sigma.
    """

    def __init__(
        self,
        layer_sizes,
        act_dim,
        low,
        high,
        activation=nn.ReLU,
        log_sigma_min=-20,
        log_sigma_max=2,
    ):
        super().__init__()
        self.low = torch.as_tensor(low)
        self.width = torch.as_tensor(high - low)
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
        act = (act + 1) * self.width / 2 + self.low
        return act, logprob


class DualUltimatumSACAgent(nn.Module):
    """
    Agent to be used in SAC for Dual Ultimatum.
    (Is this the same as a normal SAC agent?)
    Contains:
    - stochastic policy (bounded by tanh)
    - estimated Q*(s,a,)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_pi=[256, 256],
        hidden_layers_q=[256, 256],
        activation=nn.ReLU,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_pi = [obs_dim] + hidden_layers_pi
        layer_sizes_q = [obs_dim + act_dim] + hidden_layers_q + [1]
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
            act, _ = self.policy(obs, deterministic=deterministic)
        return act.numpy()



def target_update(net_main, net_target, polyak=0.9):
    """Update a lagged target network by Polya averaging."""
    new_state_dict = net_target.state_dict()
    for key in new_state_dict:
        new_state_dict[key] = (
            polyak * net_target.state_dict()[key]
            + (1 - polyak) * net_main.state_dict()[key]
        )
    net_target.load_state_dict(new_state_dict)
    return net_target


def my_sac(
    env_fn,
    agent_fn=UltimDualAgent,
    seed=0,
    epochs=100,
    steps_per_epoch=4000,
    replay_size=1000000,
    sample_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    test_episodes=10,
    log_interval=10,
    max_episode_len=1000,
    gamma=0.99,
    polyak=0.995,
    alpha=0.2,
    policy_lr=1e-3,
    qf_lr=1e-3,
    agent_kwargs=dict(),
    logger_kwargs=dict(),
    save_freq=10,
):
    """Run SAC training."""
    # Initialize environment, agent, auxiliary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = env_fn()
    # env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    act_low = env.action_space.low
    act_high = env.action_space.high
    agent = agent_fn(env.observation_space, env.action_space, **agent_kwargs)
    agent_target = deepcopy(agent)

    # Freeze target mu, Q so they are not updated by optimizers
    for p in agent_target.parameters():
        p.requires_grad = False

    var_counts = tuple(core.count_vars(module) for module in [agent.policy, agent.q1])
    logger.log(
        f"\nNumber of parameters \t policy: {var_counts[0]} q1/q2: {var_counts[1]}\n"
    )

    buf = TransitionBuffer(obs_dim, act_dim, replay_size)
    policy_optimizer = Adam(agent.policy.parameters(), lr=policy_lr)
    q1_optimizer = Adam(agent.q1.parameters(), lr=qf_lr)
    q2_optimizer = Adam(agent.q2.parameters(), lr=qf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(agent)

    def compute_q_target(data):
        r, o_next, d = data["reward"], data["obs_next"], data["done"]
        with torch.no_grad():
            a_next, logprob_next = agent.policy(o_next, get_logprob=True)
            q1_target = agent_target.q1(torch.cat((o_next, a_next), dim=-1))
            q2_target = agent_target.q2(torch.cat((o_next, a_next), dim=-1))
            q_target = torch.min(q1_target, q2_target)
            q_target = r + gamma * (1 - d) * (q_target - alpha * logprob_next)
        return q_target

    def compute_loss_q(q_model, data, q_target, qvals="QVals"):
        o, a = data["obs"], data["act"]
        q = q_model(torch.cat((o, a), dim=-1))
        loss = ((q - q_target) ** 2).mean()
        q_loss_info = {qvals: q.detach().numpy()}
        return loss, q_loss_info

    def compute_loss_policy(data):
        o = data["obs"]
        # Get actions that agent policy would take at each step
        a, logprob = agent.policy(o, get_logprob=True)
        q1 = agent.q1(torch.cat((o, a), dim=-1))
        q2 = agent.q2(torch.cat((o, a), dim=-1))
        q = torch.min(q1, q2)  # Useful info for logging
        loss = -(q - alpha * logprob).mean()
        policy_info = dict(LogPi=logprob.detach().numpy())
        return loss, policy_info

    def update():
        # Get training data from buffer
        data = buf.get(sample_size=sample_size)

        # Update Q functions
        q_target = compute_q_target(data)

        q1_optimizer.zero_grad()
        q1_loss, q1_loss_info = compute_loss_q(agent.q1, data, q_target, qvals="Q1Vals")
        q1_loss.backward()
        q1_optimizer.step()

        q2_optimizer.zero_grad()
        q2_loss, q2_loss_info = compute_loss_q(agent.q2, data, q_target, qvals="Q2Vals")
        q2_loss.backward()
        q2_optimizer.step()

        logger.store(
            LossQ1=q1_loss.item(),
            LossQ2=q2_loss.item(),
            **q1_loss_info,
            **q2_loss_info,
        )

        # Freeze Q params during policy update to save time
        for p in agent.q1.parameters():
            p.requires_grad = False
        for p in agent.q2.parameters():
            p.requires_grad = False
        # Update policy
        policy_optimizer.zero_grad()
        policy_loss, policy_info = compute_loss_policy(data)
        policy_loss.backward()
        policy_optimizer.step()
        # Unfreeze Q params after policy update
        for p in agent.q1.parameters():
            p.requires_grad = True
        for p in agent.q2.parameters():
            p.requires_grad = True
        logger.store(LossPi=policy_loss.item(), **policy_info)

        with torch.no_grad():
            agent_target.q1 = target_update(agent.q1, agent_target.q1, polyak=polyak)
            agent_target.q2 = target_update(agent.q2, agent_target.q2, polyak=polyak)

    def deterministic_policy_test():
        for _ in range(test_episodes):
            o = test_env.reset()
            ep_ret = 0
            ep_len = 0
            d = False
            while not d and not ep_len == max_episode_len:
                a = agent.act(
                    torch.as_tensor(o, dtype=torch.float32), deterministic=True
                )
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()

    # Begin training phase.
    for epoch in range(epochs):
        obs = env.reset()
        episode_return = 0
        episode_length = 0
        for t in range(steps_per_epoch):
            # Take random actions for first n steps to do broad exploration
            if t + epoch * steps_per_epoch < start_steps:
                act = env.action_space.sample()
            else:
                act = agent.act(torch.as_tensor(obs, dtype=torch.float32))
            # Step environment given latest agent action
            obs_next, reward, done, _ = env.step(act)

            episode_return += reward
            episode_length += 1

            # Store current step in buffer
            buf.store(obs, act, reward, obs_next, done)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                if done or episode_capped:
                    logger.store(EpRet=episode_return, EpLen=episode_length)
                obs = env.reset()
                episode_return = 0
                episode_length = 0

            if t >= update_after and (t + 1) % update_every == 0:
                # update_start = time.time()
                for _ in range(update_every):
                    update()
                # update_end = time.time()
                # print(f'update time {update_end - update_start}')

        deterministic_policy_test()
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("TestEpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("TestEpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("Q1Vals", with_min_and_max=True)
        logger.log_tabular("Q2Vals", with_min_and_max=True)
        logger.log_tabular("LogPi", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossQ1", average_only=True)
        logger.log_tabular("LossQ2", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
