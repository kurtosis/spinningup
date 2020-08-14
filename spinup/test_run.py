from textwrap import dedent
import time

import gym

from spinup import vpg_pytorch, ppo_pytorch, ddpg_pytorch, td3_pytorch, sac_pytorch

# from spinup.algos.pytorch.vpg.core import *
# from spinup.algos.pytorch.ppo.core import *
# from spinup.algos.pytorch.ddpg.core import *

# from spinup.agents.my_agents import *
from spinup.my_algos.my_training import *

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


# env_str = "Hopper-v3"
env_list = ["Hopper-v3", "HalfCheetah-v3", "Swimmer-v3", "Walker2d-v3", "Ant-v3", ]
# env_list = ["Walker2d-v3"]


def logging_info(env_str, subdir=None):
    env_dir = env_str.lower().replace("-", "_")
    if subdir is None:
        output_dir = f"{DEFAULT_DATA_DIR}/{env_dir}/{int(time.time())}"
    else:
        output_dir = f"{DEFAULT_DATA_DIR}/{env_dir}/{subdir}/{int(time.time())}"
    logger_kwargs = {"output_dir": output_dir}
    output_msg = create_output_msg(logger_kwargs)
    return logger_kwargs, output_msg


ac_kwargs = {}

seed = 5213
epochs = 100
steps_per_epoch = 4000
runs_per_method = 2
# runs_per_method_ppo = 4

# PPO only (fast)
# for env_str in env_list:
#     for i in range(runs_per_method_ppo):
#         seed += 1
#         print(f"ppo {env_str}")
#         logger_kwargs, output_msg = logging_info(env_str, subdir="ppo")
#         my_ppo(
#             lambda: gym.make(env_str),
#             seed=seed,
#             epochs=epochs,
#             steps_per_epoch=steps_per_epoch,
#             logger_kwargs=logger_kwargs,
#         )
#         print(f"ppo {env_str}")
#         # print(output_msg)
#
#         seed += 1
#         print(f"spinningup ppo {env_str}")
#         logger_kwargs, output_msg = logging_info(env_str, subdir="su_ppo")
#         ppo_pytorch(
#             lambda: gym.make(env_str),
#             seed=seed,
#             epochs=epochs,
#             steps_per_epoch=steps_per_epoch,
#             logger_kwargs=logger_kwargs,
#         )
#         print(f"spinningup ppo {env_str}")
#         # print(output_msg)

# non-PPO methods
for env_str in env_list:
    for i in range(runs_per_method):

        seed += 1
        print(f"sac {env_str}")
        logger_kwargs, output_msg = logging_info(env_str, subdir="sac")
        my_sac(
            lambda: gym.make(env_str),
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            logger_kwargs=logger_kwargs,
        )
        print(f"sac {env_str}")

        seed += 1
        print(f"spinningup sac {env_str}")
        logger_kwargs, output_msg = logging_info(env_str, subdir="su_sac")
        sac_pytorch(
            lambda: gym.make(env_str),
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            logger_kwargs=logger_kwargs,
        )
        print(f"spinningup sac {env_str}")

        seed += 1
        print(f"td3 {env_str}")
        logger_kwargs, output_msg = logging_info(env_str, subdir="td3")
        my_td3(
            lambda: gym.make(env_str),
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            logger_kwargs=logger_kwargs,
        )
        print(f"td3 {env_str}")

        seed += 1
        print(f"spinningup td3 {env_str}")
        logger_kwargs, output_msg = logging_info(env_str, subdir="su_td3")
        td3_pytorch(
            lambda: gym.make(env_str),
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            logger_kwargs=logger_kwargs,
        )
        print(f"spinningup td3 {env_str}")

        seed += 1
        print(f"ddpg {env_str}")
        logger_kwargs, output_msg = logging_info(env_str, subdir="ddpg")
        my_ddgp(
            lambda: gym.make(env_str),
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            logger_kwargs=logger_kwargs,
        )
        print(f"ddpg {env_str}")

        seed += 1
        print(f"spinningup ddpg {env_str}")
        logger_kwargs, output_msg = logging_info(env_str, subdir="su_ddpg")
        ddpg_pytorch(
            lambda: gym.make(env_str),
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            logger_kwargs=logger_kwargs,
        )
        print(f"spinningup ddpg {env_str}")




