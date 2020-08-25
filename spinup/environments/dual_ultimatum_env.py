from random import sample
from scipy.stats import rankdata

import gym
from gym import spaces
from spinup.algos.pytorch.vpg.core import *

EPS = 1e-8


class OneHot(gym.Space):
    """
    One-hot space. Used as the observation space.
    """

    def __init__(self, n):
        super(OneHot, self).__init__()
        self.n = n

    def sample(self):
        return np.random.multinomial(1, [1.0 / self.n] * self.n)

    def contains(self, x):
        return (
            isinstance(x, np.ndarray)
            and x.shape == (self.n,)
            and np.all(np.logical_or(x == 0, x == 1))
            and np.sum(x) == 1
        )

    @property
    def shape(self):
        return (self.n,)

    def __repr__(self):
        return "OneHot(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n


class DualUltimatum(gym.Env):
    """An environment consisting of a 'dual ultimatum' game'"""

    def __init__(self):
        super(DualUltimatum, self).__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # self.observation_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def step(self, actions):
        offer_0, threshold_0 = actions[0, :]
        offer_1, threshold_1 = actions[1, :]

        if offer_0 + EPS >= threshold_1 and offer_1 + EPS >= threshold_0:
            reward_0 = (1 - offer_0) + offer_1
            reward_1 = offer_0 + (1 - offer_1)
            reward = np.array([reward_0, reward_1])
        else:
            reward = np.array([0, 0])
        obs = np.concatenate((actions[0, :], actions[1, :]))
        done = False
        return obs, reward, done, {}

    def reset(self):
        # Nothing do reset for this environment
        return np.array([0.5, 0.5, 0.5, 0.5])

    def render(self, mode="human"):
        pass


class DualUltimatum2(gym.Env):
    """An environment consisting of a 'dual ultimatum' game'"""

    def __init__(self):
        super(DualUltimatum2, self).__init__()
        self.action_space = spaces.Tuple(
            [
                spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
                for _ in range(2)
            ]
        )
        self.observation_space = spaces.Tuple(
            [
                spaces.Tuple(
                    (
                        spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                        spaces.Discrete(2),
                    )
                )
                for _ in range(2)
            ]
        )

    def step(self, actions):
        offer_0, threshold_0 = actions[0, :]
        offer_1, threshold_1 = actions[1, :]

        if offer_0 + EPS >= threshold_1 and offer_1 + EPS >= threshold_0:
            reward_0 = (1 - offer_0) + offer_1
            reward_1 = offer_0 + (1 - offer_1)
            rewards = np.array([reward_0, reward_1])
        else:
            rewards = np.array([0, 0])

        obs = np.array(
            [
                np.concatenate((actions[0, :], actions[1, :])),
                np.concatenate((actions[1, :], actions[0, :])),
            ]
        )
        # Add flag indicating this is not the first step
        obs = np.concatenate((obs, np.zeros((2, 1))), axis=1)
        done = False
        return obs, rewards, done, {}

    def reset(self):
        # Create init state obs, with flag indicating this is the first step
        obs = np.zeros((2, 5))
        obs[:, -1] = 1
        return obs

    def render(self, mode="human"):
        pass


class MatrixGame(gym.Env):
    """An environment consisting of a matrix game with stochastic outomes"""

    NUM_ACTIONS = 3
    NUM_STATES = NUM_ACTIONS ** 2 + 1

    def __init__(
        self,
        payout_mean=np.array([[10, 5, -5], [0, 0, 5], [20, -5, 0]]),
        payout_std=np.array([[0, 0, 0], [0, 0, 0], [0, 20, 20]]),
    ):
        super(MatrixGame, self).__init__()
        self.num_actions = 3
        self.action_space = spaces.Tuple(
            [spaces.Discrete(self.num_actions) for _ in range(2)]
        )
        self.observation_space = spaces.Tuple(
            [OneHot(self.NUM_STATES) for _ in range(2)]
        )
        self.payout_mean_matrix = payout_mean
        self.payout_std_matrix = payout_std

    def step(self, actions):
        a0, a1 = actions

        reward = np.array(
            [
                self.payout_mean_matrix[a0, a1]
                + self.payout_std_matrix[a0, a1] * np.random.randn(1)[0],
                self.payout_mean_matrix[a1, a0]
                + self.payout_std_matrix[a1, a0] * np.random.randn(1)[0],
            ]
        )

        obs0 = np.zeros(self.NUM_STATES)
        obs1 = np.zeros(self.NUM_STATES)
        obs0[a0 * self.NUM_ACTIONS + a1] = 1
        obs1[a1 * self.NUM_ACTIONS + a0] = 1
        obs = np.array([obs0, obs1])
        done = False
        return obs, reward, done, {}

    def reset(self):
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        obs = np.array([init_state, init_state])
        return obs

    def render(self, mode="human"):
        pass


def assign_match_pairs(num_agents):
    """Create random pairings for an even number of agents"""
    assert num_agents % 2 == 0
    shuffled = sample(range(num_agents), k=num_agents)
    match_pairs = [shuffled[i : i + 2] for i in range(0, num_agents, 2)]
    return match_pairs


class DualUltimatumTournament(gym.Env):
    """An environment for of a tournament of some pairwise game."""

    def __init__(
        self,
        num_agents,
        num_rounds=10,
        round_length=10,
        noise_size=1,
        top_cutoff=2,
        bottom_cutoff=1,
        top_reward=1.0,
        bottom_reward=1.0,
        game_fn=DualUltimatum,
    ):
        super(DualUltimatumTournament, self).__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # hard coded for now
        self.obs_dim = 4 + num_agents + 1 + 1
        # Observations are:
        # 4 (match obs)
        # 1 rounds left
        # 1 opponent score
        # num_agents (scores)
        # ? should we add more for rankings/thresholds?
        # opponent id, OHE??? useful when we add bots, or other differences?

        # Hard code number of dimensions in obs space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.num_agents = num_agents
        self.num_matches = int(num_agents / 2)
        self.num_rounds = num_rounds
        self.round_length = round_length
        self.match_env_list = [game_fn()] * int(num_agents / 2)
        self.current_round = self.num_rounds
        self.current_turn = self.round_length
        self.noise_size = noise_size
        self.top_cutoff = top_cutoff
        self.bottom_cutoff = bottom_cutoff
        self.top_reward = top_reward
        self.bottom_reward = bottom_reward
        self.scores = np.random.randn(self.num_agents) * self.noise_size
        self.match_pairs = assign_match_pairs(num_agents)
        self.agent_opponent = np.zeros(self.num_agents, dtype=np.int32)
        self.agent_match = np.zeros(self.num_agents, dtype=np.int32)
        self.agent_position = np.zeros(self.num_agents, dtype=np.int32)
        # self.agent_opponent_dict = dict()
        # self.agent_match_dict = dict()
        for i, m in enumerate(self.match_pairs):
            # Note each agent's opponent
            self.agent_opponent[m[0]] = m[1]
            self.agent_opponent[m[1]] = m[0]
            # Note each agent's match
            self.agent_match[m[0]] = i
            self.agent_match[m[1]] = i
            # Note each agent's position (0/1) in match
            self.agent_position[m[0]] = 0
            self.agent_position[m[1]] = 1

        # Generate initial obs for each match
        self.env_obs_list = [env.reset() for env in self.match_env_list]
        self.ag_obs_list = [
            self.env_obs_list[self.agent_match[ag]] for ag in range(self.num_agents)
        ]
        self.ag_obs_next_list = self.ag_obs_list

        self.all_obs = np.zeros((self.num_agents, self.obs_dim))

    # def _take_turn(self, match_actions):
    #
    # def _next_round(self):
    #     self.current_round -= 1
    #     self.current_turn = self.round_length

    def _final_reward(self):
        ranks = rankdata(-self.scores)
        reward = (ranks <= self.top_cutoff) * self.top_reward
        if self.bottom_cutoff is not None:
            ranks = rankdata(+self.scores)
            reward += (ranks <= self.bottom_cutoff) * self.bottom_reward
        return reward

    def step(self, actions):
        done = False
        reward = np.zeros(self.num_agents)
        # Rearrange actions to as input for each match environment
        match_actions = [
            np.array([actions[i] for i in match]) for match in self.match_pairs
        ]

        # Pass actions to each match env, get next obs/reward
        match_outputs = [
            match_env.step(acts)
            for acts, match_env in zip(match_actions, self.match_env_list)
        ]
        # Update scores/obs based on env steps
        for pair, output in zip(self.match_pairs, match_outputs):
            o, r, d, _ = output
            self.scores[pair] += r
            # obs - match, last moves
            self.all_obs[pair[0], :4] = o
            self.all_obs[pair[1], :4] = np.roll(o, 2)

        # obs - current round / rounds left
        self.all_obs[:, 4] = self.current_round
        # obs - opponent score
        self.all_obs[:, 5] = [self.scores[i] for i in self.agent_opponent]
        # obs - all scores
        self.all_obs[:, 6 : (6 + self.num_agents)] = self.scores

        self.current_turn -= 1
        if self.current_turn == 0:
            self.current_round -= 1
            self.current_turn = self.round_length
            self.all_obs[:, :4] = 0.5 * np.ones((self.num_agents, 4))
            if self.current_round == 0:
                reward = self._final_reward()
                done = True
            else:
                for env in self.match_env_list:
                    env.reset()

        return self.all_obs, reward, done, {}

    def reset(self):
        self.current_round = self.num_rounds
        self.current_turn = self.round_length
        self.scores = np.random.randn(self.num_agents) * self.noise_size

        match_outputs = [match_env.reset() for match_env in self.match_env_list]

        for i in range(len(match_outputs)):
            p0 = self.match_pairs[i][0]
            p1 = self.match_pairs[i][1]
            self.all_obs[p0, :4] = match_outputs[i]
            self.all_obs[p1, :4] = np.roll(match_outputs[i], 2)
        # obs - current round / rounds left
        self.all_obs[:, 4] = self.current_round
        # obs - opponent score
        self.all_obs[:, 5] = [self.scores[i] for i in self.agent_opponent]
        # obs - all scores
        self.all_obs[:, 6 : (6 + self.num_agents)] = self.scores
        return self.all_obs

    def render(self, mode="human"):
        pass


class RoundRobinTournament(gym.Env):
    """An environment for of a tournament of some pairwise game."""

    def __init__(
        self,
        num_agents,
        num_rounds=10,
        round_length=10,
        noise_size=1,
        top_cutoff=2,
        bottom_cutoff=1,
        top_reward=1.0,
        bottom_reward=1.0,
        game_fn=DualUltimatum2,
        relative_reward=False,
        per_turn_reward=False,
    ):
        super(RoundRobinTournament, self).__init__()

        self.num_agents = num_agents
        self.num_matches = int(num_agents / 2)
        self.num_rounds = num_rounds
        self.round_length = round_length
        self.noise_size = noise_size
        self.top_cutoff = top_cutoff
        self.bottom_cutoff = bottom_cutoff
        self.top_reward = top_reward
        self.bottom_reward = bottom_reward
        self.relative_reward = relative_reward
        self.per_turn_reward = per_turn_reward

        self.match_env_list = [game_fn()] * int(num_agents / 2)
        self.action_space = spaces.Tuple(
            [env.action_space for env in self.match_env_list]
        )

        # hard coded for now (to dualultimatum)
        # observation space for single match
        self.match_obs_dim = 5
        # observation space for tournament
        self.obs_dim = self.match_obs_dim + num_agents + 1 + 1
        # Observations are:
        # 4 (match obs), contin
        # 1 first turn flag, binary
        # 1 rounds left, int
        # 1 opponent score, contin
        # num_agents (scores), contin
        # ? should we add more for rankings/thresholds?
        # opponent id, OHE??? useful when we add bots, or other differences?

        # Create tuple of observations for all agents. Extend match-level obs with tournament-level obs features
        self.observation_space = spaces.Tuple(
            [
                spaces.Tuple(
                    [space for space in self.match_env_list[0].observation_space[0]]
                    + [
                        spaces.Discrete(self.num_rounds),
                        spaces.Box(
                            low=0.0,
                            high=np.inf,
                            shape=(self.num_agents + 1,),
                            dtype=np.float32,
                        ),
                    ]
                )
                for _ in range(self.num_agents)
            ]
        )

    def _final_reward(self, relative_score=True):
        if relative_score:
            return self.scores
        else:
            ranks = rankdata(-self.scores)
            reward = (ranks <= self.top_cutoff) * self.top_reward
            if self.bottom_cutoff is not None:
                ranks = rankdata(+self.scores)
                reward += (ranks <= self.bottom_cutoff) * self.bottom_reward
            return reward

    def step(self, actions):
        # Rearrange actions as input for each match environment
        match_actions = [
            np.array([actions[i] for i in match]) for match in self.match_pairs
        ]

        # Pass actions to each match env, get next obs/reward
        match_outputs = [
            match_env.step(acts)
            for acts, match_env in zip(match_actions, self.match_env_list)
        ]

        all_obs = np.zeros((self.num_agents, self.obs_dim))
        # Update scores/obs based on env steps
        for pair, output in zip(self.match_pairs, match_outputs):
            o, r, d, _ = output
            self.scores[pair] += r
            # obs - match, last moves
            all_obs[pair[0], : self.match_obs_dim] = o[0]
            all_obs[pair[1], : self.match_obs_dim] = o[1]

        self.scores -= np.mean(self.scores)
        # obs - current round / rounds left
        all_obs[:, self.match_obs_dim] = self.current_round / self.num_rounds
        # obs - opponent score
        all_obs[:, self.match_obs_dim + 1] = [
            self.scores[i] for i in self.agent_opponent
        ]
        # obs - all agents' scores
        all_obs[:, self.match_obs_dim + 2 :] = self.scores

        reward = np.zeros(self.num_agents)
        done = False
        self.current_turn -= 1
        if self.current_turn == 0:
            self.current_round -= 1
            self.current_turn = self.round_length
            # to do: check that this actually resets state for match environments
            all_obs[:, : self.match_obs_dim] = np.concatenate(
                [match.reset() for match in self.match_env_list]
            )
            if self.current_round == 0:
                reward = self._final_reward()
                done = True

        # Per turn reward based on change to relative score
        if self.per_turn_reward:
            reward = np.zeros(self.num_agents)
            for pair, output in zip(self.match_pairs, match_outputs):
                _, r, _, _ = output
                reward[pair] = r - np.sum(r)/self.num_agents

        return all_obs, reward, done, {}

    def reset(self):
        self.current_round = self.num_rounds
        self.current_turn = self.round_length
        self.scores = np.random.randn(self.num_agents) * self.noise_size
        self.match_pairs = assign_match_pairs(self.num_agents)
        self.agent_opponent = np.zeros(self.num_agents, dtype=np.int32)
        self.agent_match = np.zeros(self.num_agents, dtype=np.int32)
        for i, m in enumerate(self.match_pairs):
            # Note each agent's opponent
            self.agent_opponent[m[0]] = m[1]
            self.agent_opponent[m[1]] = m[0]
            # Note each agent's match
            self.agent_match[m[0]] = i
            self.agent_match[m[1]] = i

        # Generate initial obs for each match
        all_obs = np.zeros((self.num_agents, self.obs_dim))

        match_obs = [match_env.reset() for match_env in self.match_env_list]

        for i in range(self.num_matches):
            p0 = self.match_pairs[i][0]
            p1 = self.match_pairs[i][1]
            all_obs[p0, : self.match_obs_dim] = match_obs[i][0]
            all_obs[p1, : self.match_obs_dim] = match_obs[i][1]
        # obs - current round / rounds left
        all_obs[:, self.match_obs_dim] = self.current_round / self.num_rounds
        # obs - opponent score
        all_obs[:, self.match_obs_dim + 1] = [
            self.scores[i] for i in self.agent_opponent
        ]
        # obs - all agents' scores
        all_obs[:, self.match_obs_dim + 2 :] = self.scores
        return all_obs

    def render(self, mode="human"):
        pass
