import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class BitFlip(gym.GoalEnv):
    """
    A multi-goal environment based on BitFlip environment from [1]

     [1] Hindsight Experience Replay - https://arxiv.org/abs/1707.01495
    TODO add support for mean = zero
    """

    def __init__(self, bit_length=16, max_steps=None):

        super(BitFlip, self).__init__()

        assert bit_length >= 1, 'bit_length must be >= 1, found {}'.format(bit_length)

        self.bit_length = bit_length

        if max_steps is None:
            self.max_steps = bit_length
        else:
            self.max_steps = max_steps

        self.last_action = -1  # -1 for reset
        self.steps = 0
        self.seed()
        self.action_space = spaces.Discrete(bit_length + 1)  # index = n means to not flip any bit
        # achieved goal and observation are identical in bit_flip environment, however it is made this way to be
        # compatible with Openai GoalEnv
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(low=0, high=1, shape=(bit_length,), dtype=np.int32),
            achieved_goal=spaces.Box(low=0, high=1, shape=(bit_length,), dtype=np.int32),
            desired_goal=spaces.Box(low=0, high=1, shape=(bit_length,), dtype=np.int32),
        ))

        self.reset()

    def step(self, action):
        obs = self._get_obs()

        if not action == self.bit_length:  # ignore moves that do flip
            self._bitflip(action)

        self.last_action = action
        self.steps += 1

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        done = self._terminate()
        info = {}  # TODO add useful debug information in info

        return obs, reward, done, info

    def render(self, mode='human'):
        print("Observation_space: {}, last_action: {}, num_steps: {}".format(self._get_obs(),
                                                                             self.last_action,
                                                                             self.steps))

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0 if np.array_equal(achieved_goal, desired_goal) else -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _terminate(self):
        return np.array_equal(self.state, self.goal) or self.steps >= self.max_steps

    def reset(self):
        self.steps = 0
        self.last_action = -1  # -1 for reset

        initial_state = self.observation_space.sample()

        self.state = initial_state['observation']
        self.goal = initial_state['desired_goal']

        return self._get_obs()

    def _get_obs(self):
        return {
            'observation': self.state.copy(),
            'achieved_goal': self.state.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _bitflip(self, index):
        self.state[index] = not self.state[index]
