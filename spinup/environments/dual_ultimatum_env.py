import gym
from gym import spaces
from spinup.algos.pytorch.vpg.core import *


class UltimatumDualEnv(gym.Env):
    """An environment consisting of a single Ultimatum responder bot based on OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(UltimatumDualEnv, self).__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # self.observation_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2, 2), dtype=np.float32
        )

    def step(self, actions):
        offer_1, threshold_1, offer_2, threshold_2 = actions
        if offer_1 >= threshold_2 and offer_2 >= threshold_1:
            reward_1 = (1 - offer_1) + offer_2
            reward_2 = offer_1 + (1 - offer_2)
            reward = [reward_1, reward_2]
        else:
            reward = [0, 0]
        obs = actions
        done = False
        return obs, reward, done, {}

    def reset(self):
        # Nothing do reset for this environment
        pass
