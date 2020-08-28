from textwrap import dedent
import time

import gym

from spinup import vpg_pytorch, ppo_pytorch, ddpg_pytorch, td3_pytorch, sac_pytorch

from spinup.my_algos.my_training import *
from spinup.my_algos.ultimatum_agents import *
from spinup.environments.tournament_env import *

from spinup.user_config import (
    DEFAULT_DATA_DIR,
    FORCE_DATESTAMP,
    DEFAULT_SHORTHAND,
    WAIT_BEFORE_LAUNCH,
)
from spinup.utils.logx import colorize

DIV_LINE_WIDTH = 80


def create_output_msg(logger_kwargs):
    plot_cmd = "python -m spinup.run plot " + logger_kwargs["output_dir"]
    plot_cmd = colorize(plot_cmd, "green")
    test_cmd = "python -m spinup.run test_policy " + logger_kwargs["output_dir"]
    test_cmd = colorize(test_cmd, "green")
    output_msg = (
        "\n" * 5
        + "=" * DIV_LINE_WIDTH
        + "\n"
        + dedent(
            """\
End of experiment.


Plot results from this run with:

%s


Watch the trained agent with:

%s


"""
            % (plot_cmd, test_cmd)
        )
        + "=" * DIV_LINE_WIDTH
        + "\n" * 5
    )
    return output_msg


def logging_info(env_str, subdir=None):
    env_dir = env_str.lower().replace("-", "_")
    if subdir is None:
        output_dir = f"{DEFAULT_DATA_DIR}/{env_dir}/{int(time.time())}"
    else:
        output_dir = f"{DEFAULT_DATA_DIR}/{env_dir}/{subdir}/{int(time.time())}"
    logger_kwargs = {"output_dir": output_dir}
    output_msg = create_output_msg(logger_kwargs)
    return logger_kwargs, output_msg


seed = 4153
epochs = 10
steps_per_epoch = 4000
runs_per_method = 4
runs_per_method_ppo = 4
max_episode_len = 100

logger_kwargs, output_msg = logging_info("dual_ultimatum", subdir="dual_ultimatum")

# const_offer = 0.5
# const_threshold = 0.5
# dualultimatum_ddpg(
#     agent_fn=DDPGAgent,
#     player_2=ConstantBot(offer=const_offer, threshold=const_threshold),
#     env_fn=DualUltimatum,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     max_episode_len=max_episode_len,
#     test_episodes=200,
# )


static_agent_kwargs = dict(
    mean_offer=0.2, std_offer=0.00001, mean_threshold=0.55, std_threshold=0.00001
)

# dualultimatum_ddpg(
#     agent_1_fn=DDPGAgent,
#     agent_2_fn=StaticDistribBot,
#     agent_2_kwargs=static_agent_kwargs,
#     env_fn=DualUltimatum,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     max_episode_len=max_episode_len,
#     test_episodes=200,
# )
#
# dualultimatum_ddpg(
#     agent_2_fn=DDPGAgent,
#     agent_1_fn=StaticDistribBot,
#     agent_1_kwargs=static_agent_kwargs,
#     env_fn=DualUltimatum,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     max_episode_len=max_episode_len,
#     test_episodes=200,
# )

# dualultimatum_ddpg(
#     agent_1_fn=DDPGAgent,
#     agent_1_kwargs=static_agent_kwargs,
#     agent_2_fn=DDPGAgent,
#     agent_2_kwargs=static_agent_kwargs,
#     env_fn=DualUltimatum,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     max_episode_len=max_episode_len,
#     test_episodes=200,
# )


# tournament_ddpg(seed=8571,
#                 # bottom_cutoff=1,
#                 steps_per_epoch=4000,)


# Test all bots
# logger_kwargs, output_msg = logging_info("tournament", subdir="dual_ultimatum")
# env_kwargs = dict(top_cutoff=2, bottom_cutoff=None, top_reward=1.0, bottom_reward=1.0,)
# agent_kwargs_constantbot = dict(offer=0.6, threshold=0.2)
# agent_kwargs_distrib = dict(
#     mean_offer=0.3, std_offer=0.01, mean_threshold=0.9, std_threshold=0.01,
# )
# tournament_ddpg(
#     seed=42,
#     steps_per_epoch=4000,
#     num_agents=4,
#     env_fn=RoundRobinTournament,
#     # agent_fn=ConstantBot,
#     # agent_kwargs=agent_kwargs_constantbot,
#     # agent_fn=StaticDistribBot,
#     # agent_kwargs=agent_kwargs_distrib,
#     logger_kwargs=logger_kwargs,
#     env_kwargs=env_kwargs,
# )

# Test constant bot w/ guaranteed winner
# logger_kwargs, output_msg = logging_info("tournament", subdir="dual_ultimatum")
# env_kwargs = dict(top_cutoff=2, bottom_cutoff=None, top_reward=1.0, bottom_reward=1.0,)
# agent_kwargs_constantbot = dict(offer=0.6, threshold=0.2)
# agents = [ConstantBot, ConstantBot, ConstantBot, ConstantBot]
# agent_kwargs_winning = dict(offer=0.200001, threshold=0.0)
# agents_kwargs = [agent_kwargs_winning, agent_kwargs_constantbot, agent_kwargs_constantbot, agent_kwargs_constantbot]
# tournament_ddpg(
#     seed=42,
#     steps_per_epoch=4000,
#     num_agents=4,
#     env_fn=RoundRobinTournament,
#     agents=agents,
#     agents_kwargs=agents_kwargs,
#     logger_kwargs=logger_kwargs,
#     env_kwargs=env_kwargs,
# )

# Test DDPG vs constant bots
logger_kwargs, output_msg = logging_info("tournament", subdir="dual_ultimatum")
env_kwargs = dict(
    num_rounds=1,
    round_length=10,
    noise_size=0,
    top_cutoff=2,
    bottom_cutoff=None,
    top_reward=1.0,
    bottom_reward=1.0,
    score_reward=True,
    per_turn_reward=False,
    hide_obs=True,
    game_kwargs=dict(reward="l1"),
)
agent_kwargs_constantbot = dict(offer=0.5, threshold=0.5)
# agent_kwargs_ddpg = dict(hidden_layers_mu=(256, 256), hidden_layers_q=(256, 256))
agent_kwargs_ddpg = dict(
    hidden_layers_mu=(1,),
    hidden_layers_q=(64, 64, 64),
    # hidden_layers_q=(64, 64, 64, 64),
    noise_std=0.5,
    pi_lr=1e-3,
    q_lr=1e-3,
    # gamma=0,
)
agents = [DDPGAgent, ConstantBot, ConstantBot, ConstantBot]
agents_kwargs = [
    agent_kwargs_ddpg,
    agent_kwargs_constantbot,
    agent_kwargs_constantbot,
    agent_kwargs_constantbot,
]
tournament_ddpg(
    seed=736,
    steps_per_epoch=1000,
    epochs=50,
    save_freq=1,
    start_steps=0,
    sample_size=4096,
    update_after=0,
    env_fn=RoundRobinTournament,
    agents=agents,
    agents_kwargs=agents_kwargs,
    logger_kwargs=logger_kwargs,
    env_kwargs=env_kwargs,
    q_file="q",
)


# tournament_ddpg(
#     seed=42,
#     steps_per_epoch=4000,
#     num_agents=4,
#     logger_kwargs=logger_kwargs,
#     env_kwargs=env_kwargs,
# )
#
# logger_kwargs, output_msg = logging_info("tournament", subdir="dual_ultimatum")
# env_kwargs = dict(top_cutoff=1, bottom_cutoff=1, top_reward=1.0, bottom_reward=1.0,)
# tournament_ddpg(
#     seed=42,
#     steps_per_epoch=4000,
#     num_agents=4,
#     logger_kwargs=logger_kwargs,
#     env_kwargs=env_kwargs,
# )


# env_fn = DualUltimatumTournament
# num_agents = 4

# dualultimatum_bots(
#     ConstantBot(offer=0.5, threshold=0.5),
#     ConstantBot(offer=0.5, threshold=0.5),
#     seed=seed,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     logger_kwargs=logger_kwargs,
# )

# dualultimatum_bots(
#     ConstantBot(offer=0.49, threshold=0.5),
#     ConstantBot(offer=0.5, threshold=0.5),
#     seed=seed,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     logger_kwargs=logger_kwargs,
# )

# dualultimatum_bots(
#     ConstantBot(offer=0.9, threshold=0.5),
#     ConstantBot(offer=0.5, threshold=0.5),
#     seed=seed,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     logger_kwargs=logger_kwargs,
# )

# dualultimatum_bots(
#     StaticDistribBot(),
#     ConstantBot(offer=0.5, threshold=0.5),
#     seed=seed,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     logger_kwargs=logger_kwargs,
# )

# dualultimatum_bots(
#     ConstantBot(offer=0.5, threshold=0.5),
#     StaticDistribBot(),
#     seed=seed,
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     logger_kwargs=logger_kwargs,
# )
