import gym
from gym import spaces
from spinup.algos.pytorch.vpg.core import *

EPS = 1e-8
class DualUltimatum(gym.Env):
    """An environment consisting of a 'dual ultimatum' game'"""

    # metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(DualUltimatum, self).__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # self.observation_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def step(self, actions):
        offer_1, threshold_1 = actions[0,:]
        offer_2, threshold_2 = actions[1,:]

        if offer_1 + EPS >= threshold_2 and offer_2 + EPS >= threshold_1:
            reward_1 = (1 - offer_1) + offer_2
            reward_2 = offer_1 + (1 - offer_2)
            reward = np.array([reward_1, reward_2])
        else:
            reward = np.array([0, 0])
        obs = np.concatenate((actions[0,:], actions[1,:]))
        done = False
        return obs, reward, done, {}

    def reset(self):
        # Nothing do reset for this environment
        return np.array([0.5, 0.5, 0.5, 0.5])
        pass
