import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys

sys.path.insert(
    0, "/Users/kurtsmith/research/pytorch_projects/reinforcement_learning/environments"
)
sys.path.insert(0, "/Users/kurtsmith/research/spinningup")

from copy import deepcopy
import numpy as np
import time

import torch
from torch.optim import Adam

# import spinup.algos.pytorch.vpg.core as core
from spinup.agents.my_agents import *
from spinup.utils.logx import EpochLogger


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


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


def my_vpg(
    env_fn,
    actor_critic,
    seed=0,
    epochs=10,
    steps_per_epoch=200,
    log_interval=10,
    render=True,
    max_episode_len=100,
    gamma=0.99,
    lam=0.97,
    train_v_iters=80,
    pi_lr=3e-4,
    vf_lr=1e-3,
    ac_kwargs=dict(),
    logger_kwargs=dict(),
    save_freq=10,
):
    """Run VPG training."""
    # Initialize environment, agent, auxilary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    # env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log(f"\nNumber of parameters \t pi: {var_counts[0]} v: {var_counts[1]}\n")

    # training_records = []
    # running_avg_return = -1000
    buf = TrajectoryBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    v_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def compute_loss_pi(data):
        # get data
        obs, act, adv, logprob_old = (
            data["obs"],
            data["act"],
            data["adv"],
            data["logprob"],
        )

        # Get policy (given obs) and logprob of actions taken
        pi, logprob = ac.pi(obs, act)
        # The loss function equation for VPG (see docs)
        pi_loss = -(logprob * adv).mean()

        # KL-div is approx difference in log probs
        approx_kl = (logprob_old - logprob).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = {"kl": approx_kl, "ent": ent}
        return pi_loss, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        v = ac.v(obs)
        return ((v - ret) ** 2).mean()

    def update():
        # TODO
        # - add logging at end
        # DONE: write loss computation

        # Get training data from buffer
        data = buf.get()

        # Compute policy/value losses
        pi_loss_old, pi_info_old = compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = compute_loss_v(data).item()

        # Update policy by single step using optimizer
        pi_optimizer.zero_grad()
        pi_loss, pi_info = compute_loss_pi(data)
        pi_loss.backward()
        pi_optimizer.step()

        # Update value function multiple steps using optimizer
        for i in range(train_v_iters):
            v_optimizer.zero_grad()
            v_loss = compute_loss_v(data)
            v_loss.backward()
            v_optimizer.step()

        # add logging?
        kl, ent = pi_info["kl"], pi_info["ent"]
        logger.store(
            LossPi=pi_loss_old,
            LossV=v_loss_old,
            KL=kl,
            Entropy=ent,
            DeltaLossPi=pi_loss.item() - pi_loss_old,
            DeltaLossV=v_loss.item() - v_loss_old,
        )

    start_time = time.time()

    for epoch in range(epochs):
        # Start a new epoch
        obs = env.reset()
        episode_return = 0
        episode_length = 0
        episode_count = 0
        for t in range(steps_per_epoch):
            # print(f'step {t}')
            # Step agent given latest observation
            a, v, logprob = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            # Step environment given latest agent action
            obs_next, reward, done, _ = env.step(a)
            # Visualize current state if desired
            if render:
                env.render()

            episode_return += reward
            episode_length += 1

            # Store current step in buffer
            buf.store(obs, a, reward, v, logprob)
            logger.store(VVals=v)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                episode_count += 1
                if not (done or episode_capped):
                    print(
                        f"Trajectory terminated by end of epoch at step {episode_length}"
                    )
                # get last value function
                if episode_capped or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if done or episode_capped:
                    logger.store(EpRet=episode_return, EpLen=episode_length)
                obs = env.reset()
                episode_return = 0
                episode_length = 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)
        update()

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()

        # update running avg of episode score and training trajectory buffer
        # running_avg_return = running_avg_return * 0.9 + episode_return * 0.1
        # training_records.append(TrainingRecord(i_ep, running_reward))

        # # Print running avg episode score at end of episode
        # if i_ep % log_interval == 0:
        #     print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))
        # if running_reward > -200:
        #     print("Solved! Moving average score is now {}!".format(running_reward))
        #     env.close()
        #     #             ac.save_param()
        #     #             with open('log/ppo_training_records.pkl', 'wb') as f:
        #     #                 pickle.dump(training_records, f)
        #     break

    # plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    # plt.title('VPG')
    # plt.xlabel('Episode')
    # plt.ylabel('Moving averaged episode reward')
    # plt.savefig("./vpg.png")
    # plt.show()
    # end_time = time.time()
    # print(f'total time: {end_time - start_time}')


def my_ppo(
    env_fn,
    actor_critic=GaussianActorCritic,
    seed=0,
    epochs=50,
    steps_per_epoch=4000,
    log_interval=10,
    render=False,
    max_episode_len=1000,
    gamma=0.99,
    lam=0.97,
    train_pi_iters=80,
    train_v_iters=80,
    pi_lr=3e-4,
    vf_lr=1e-3,
    clip_ratio=0.2,
    target_kl=0.015,
    ac_kwargs=dict(),
    logger_kwargs=dict(),
    save_freq=10,
):
    """Run PPO training."""
    # Initialize environment, agent, auxilary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    # env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log(f"\nNumber of parameters \t pi: {var_counts[0]} v: {var_counts[1]}\n")

    buf = TrajectoryBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    v_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def compute_loss_pi(data, clip_ratio=clip_ratio):
        # get data
        obs, act, adv, logprob_old = (
            data["obs"],
            data["act"],
            data["adv"],
            data["logprob"],
        )

        # Get policy (given obs) and logprob of actions taken
        pi, logprob = ac.pi(obs, act)

        pi_ratio = torch.exp(logprob - logprob_old)
        clips = 1.0 + clip_ratio * torch.sign(adv)
        pi_loss = -torch.min(pi_ratio * adv, clips * adv).mean()

        # KL-div is approx difference in log probs
        approx_kl = (logprob_old - logprob).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = {"kl": approx_kl, "ent": ent}
        return pi_loss, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        v = ac.v(obs)
        return ((v - ret) ** 2).mean()

    def update():
        # Get training data from buffer
        data = buf.get()

        # Compute policy/value losses
        pi_loss_old, pi_info_old = compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = compute_loss_v(data).item()

        # Update value function multiple steps using optimizer
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            pi_loss, pi_info = compute_loss_pi(data)
            if pi_info["kl"] > target_kl:
                logger.log(f"Early stopping pi update after {i} steps due to KL max.")
                break
            pi_loss.backward()
            pi_optimizer.step()

        # Update value function multiple steps using optimizer
        for i in range(train_v_iters):
            v_optimizer.zero_grad()
            v_loss = compute_loss_v(data)
            v_loss.backward()
            v_optimizer.step()

        # add logging?
        kl, ent = pi_info["kl"], pi_info["ent"]
        logger.store(
            LossPi=pi_loss_old,
            LossV=v_loss_old,
            KL=kl,
            Entropy=ent,
            DeltaLossPi=pi_loss.item() - pi_loss_old,
            DeltaLossV=v_loss.item() - v_loss_old,
        )

    start_time = time.time()

    for epoch in range(epochs):
        # Start a new epoch
        obs = env.reset()
        episode_return = 0
        episode_length = 0
        episode_count = 0
        for t in range(steps_per_epoch):
            # Step agent given latest observation
            a, v, logprob = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            # Step environment given latest agent action
            obs_next, reward, done, _ = env.step(a)
            # Visualize current state if desired
            if render:
                env.render()

            episode_return += reward
            episode_length += 1

            # Store current step in buffer
            buf.store(obs, a, reward, v, logprob)
            logger.store(VVals=v)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                episode_count += 1
                if not (done or episode_capped):
                    print(
                        f"Trajectory terminated by end of epoch at step {episode_length}"
                    )
                # get last value function
                if episode_capped or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if done or episode_capped:
                    logger.store(EpRet=episode_return, EpLen=episode_length)
                obs = env.reset()
                episode_return = 0
                episode_length = 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)
        update()

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()

        # update running avg of episode score and training trajectory buffer
        # running_avg_return = running_avg_return * 0.9 + episode_return * 0.1
        # training_records.append(TrainingRecord(i_ep, running_reward))

        # # Print running avg episode score at end of episode
        # if i_ep % log_interval == 0:
        #     print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))
        # if running_reward > -200:
        #     print("Solved! Moving average score is now {}!".format(running_reward))
        #     env.close()
        #     #             ac.save_param()
        #     #             with open('log/ppo_training_records.pkl', 'wb') as f:
        #     #                 pickle.dump(training_records, f)
        #     break

    # plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    # plt.title('VPG')
    # plt.xlabel('Episode')
    # plt.ylabel('Moving averaged episode reward')
    # plt.savefig("./vpg.png")
    # plt.show()
    # end_time = time.time()
    # print(f'total time: {end_time - start_time}')


def my_ddgp(
    env_fn,
    agent_fn=DDPGAgent,
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
    policy_lr=1e-3,
    qf_lr=1e-3,
    agent_kwargs=dict(),
    logger_kwargs=dict(),
    save_freq=10,
):
    """Run DDPG training."""
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
    agent = agent_fn(env.observation_space, env.action_space, **agent_kwargs)
    agent_target = deepcopy(agent)

    # Freeze target mu, Q so they are not updated by optimizers
    for p in agent_target.parameters():
        p.requires_grad = False

    var_counts = tuple(count_vars(module) for module in [agent.policy, agent.q])
    logger.log(
        f"\nNumber of parameters \t policy: {var_counts[0]} q: {var_counts[1]}\n"
    )

    buf = TransitionBuffer(obs_dim, act_dim, replay_size)
    policy_optimizer = Adam(agent.policy.parameters(), lr=policy_lr)
    q_optimizer = Adam(agent.q.parameters(), lr=qf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(agent)

    def compute_loss_policy(data):
        # get data
        o = data["obs"]
        # Get actions that agent policy would take at each step
        a = agent.policy(o)
        return -agent.q(torch.cat((o, a), dim=-1)).mean()

    def compute_q_target(data):
        r, o_next, d = data["reward"], data["obs_next"], data["done"]
        with torch.no_grad():
            a_next = agent_target.policy(o_next)
            q_target = agent_target.q(torch.cat((o_next, a_next), dim=-1))
            q_target = r + gamma * (1 - d) * q_target
        return q_target

    def compute_loss_q(data, q_target):
        o, a = data["obs"], data["act"]
        q = agent.q(torch.cat((o, a), dim=-1))
        q_loss_info = {"QVals": q.detach().numpy()}
        return ((q - q_target) ** 2).mean(), q_loss_info

    def update():
        # Get training data from buffer
        data = buf.get(sample_size=sample_size)

        # Update Q function
        q_optimizer.zero_grad()
        q_target = compute_q_target(data)
        q_loss, q_loss_info = compute_loss_q(data, q_target)
        q_loss.backward()
        q_optimizer.step()

        # Freeze Q params during policy update to save time
        for p in agent.q.parameters():
            p.requires_grad = False
        # Update policy
        policy_optimizer.zero_grad()
        policy_loss = compute_loss_policy(data)
        policy_loss.backward()
        policy_optimizer.step()
        # Unfreeze Q params after policy update
        for p in agent.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            agent_target.q = target_update(agent.q, agent_target.q, polyak=polyak)
            agent_target.policy = target_update(
                agent.policy, agent_target.policy, polyak=polyak
            )

        logger.store(LossPi=policy_loss.item(), LossQ=q_loss.item(), **q_loss_info)

    def deterministic_policy_test():
        for _ in range(test_episodes):
            o = test_env.reset()
            ep_ret = 0
            ep_len = 0
            d = False
            while not d and not ep_len == max_episode_len:
                with torch.no_grad():
                    a = agent.act(torch.as_tensor(o, dtype=torch.float32), noise=False)
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
                act = agent.act(torch.as_tensor(obs, dtype=torch.float32), noise=True)
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
                # episode_count += 1
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
        logger.log_tabular("QVals", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()


if __name__ == "__main__":
    my_vpg(
        seed=0,
        epochs=100,
        steps_per_epoch=10,
        log_interval=1,
        render=False,
        max_episode_len=500,
        pi_lr=3e-4,
        v_lr=1e-3,
    )
