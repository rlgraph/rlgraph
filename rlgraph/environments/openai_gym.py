# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import gym
import numpy as np
from six.moves import xrange as range_

from rlgraph.environments import Environment
from rlgraph.spaces import *
from rlgraph.utils.rlgraph_errors import RLGraphError


class OpenAIGymEnv(Environment):
    """
    OpenAI Gym adapter for RLgraph: https://gym.openai.com/.
    """

    def __init__(
            self, gym_env, frameskip=None, max_num_noops=0, noop_action=0, episodic_life=False, fire_reset=False,
            monitor=None, monitor_safe=False, monitor_video=0, visualize=False,
            force_float32=True, **kwargs
    ):
        """
        Args:
            gym_env (Union[str,gym.Env]): OpenAI Gym environment ID or actual gym.Env. See https://gym.openai.com/envs
            frameskip (Optional[Tuple[int,int],int]): Number of game frames that should be skipped with each action
                (repeats given action for this number of game frames and accumulates reward).
                Default: (2,5) -> Uniformly pull from set [2,3,4].
            max_num_noops (Optional[int]): How many no-ops to maximally perform when resetting
                the environment before returning the reset state.
            noop_action (any): The action representing no-op. 0 for Atari.
            episodic_life (bool): If true, losing a life will lead to episode end from the perspective
                of the agent. Internally, th environment will keep stepping the game and manage the true
                termination (end of game).
            fire_reset (Optional[bool]): If true, fire off environment after reset.
            monitor: Output directory. Setting this to None disables monitoring.
            monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.
            visualize: If set True, the program will visualize the trainings of gym's environment. Note that such
                visualization is probably going to slow down the training.
            force_float32 (bool): Whether to convert all state signals (iff the state space is of dtype float64) into
                float32. Note: This does not affect any int-type state spaces.
                Default: True.
        """
        if isinstance(gym_env, str):
            self.gym_env = gym.make(gym_env)  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv
        else:
            self.gym_env = gym_env

        # Multi-goal environments states comes in a dict{observation: dtype, desired_goal: dtype, achieved_goal:dtype}
        if hasattr(gym, "GoalEnv") and isinstance(self.gym_env.env, gym.GoalEnv):
            self.gym_env = gym.wrappers.FlattenDictWrapper(self.gym_env, dict_keys=['observation', 'desired_goal'])
            self.achieved_goal = self.translate_space(self.gym_env.env.observation_space.spaces['achieved_goal'],
                                                      force_float32=force_float32)
        # Manually set the frameskip property.
        self.frameskip = None
        if frameskip is not None:
            # Skip externally.
            if "NoFrameskip" in gym_env:
                self.state_buffer = np.zeros((2,) + self.gym_env.observation_space.shape, dtype=np.uint8)
                self.frameskip = frameskip
            else:
                # Set gym property.
                self.gym_env.env.frameskip = frameskip

        # In Atari environments, 0 is no-op.
        self.noop_action = noop_action
        self.max_num_noops = max_num_noops

        # Manage life as episodes.
        self.episodic_life = episodic_life
        self.true_terminal = True
        self.lives = 0
        self.fire_after_reset = fire_reset
        self.force_float32 = False  # Set to False for now, later overwrite with a correct value.

        if self.fire_after_reset:
            assert self.gym_env.unwrapped.get_action_meanings()[1] == 'FIRE'
            assert len(self.gym_env.unwrapped.get_action_meanings()) >= 3

        self.visualize = visualize
        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            self.gym_env = gym.wrappers.Monitor(self.gym_env, monitor, force=not monitor_safe,
                                                video_callable=video_callable)

        self.action_space = self.translate_space(self.gym_env.action_space)

        # Don't trust gym's own information on dtype. Find out what the observation space really is.
        # Gym_env.observation_space's low/high used to be float64 ndarrays, but the actual output was uint8.
        self.state_space = self.translate_space(self.gym_env.observation_space, dtype=self.reset().dtype,
                                                force_float32=force_float32)

        super(OpenAIGymEnv, self).__init__(self.state_space, self.action_space, **kwargs)

        # If state_space is not a FloatBox -> Set force_float32 to False.
        if not isinstance(self.state_space, FloatBox):
            force_float32 = False

        self.force_float32 = force_float32

    def seed(self, seed=None):
        if seed is None:
            seed = time.time()
        self.gym_env.seed(seed)
        return seed

    def reset(self):
        if self.fire_after_reset:
            self.episodic_reset()
            state, _, terminal, _ = self.step(1)
            if terminal:
                self.episodic_reset()
            state, _, terminal, _ = self.step(2)
            if terminal:
                self.episodic_reset()
            return state if self.force_float32 is False else np.array(state, dtype=np.float32)
        else:
            return self.episodic_reset()

    def episodic_reset(self):
        if self.episodic_life:
            # If the last terminal was actually the end of the episode.
            if self.true_terminal:
                state = self.noop_reset()
            else:
                # If not, step.
                state, _, _, _ = self._step_and_skip(self.noop_action)
            # Update live property.
            self.lives = self.gym_env.unwrapped.ale.lives()
            return state if self.force_float32 is False else np.array(state, dtype=np.float32)
        else:
            return self.noop_reset()

    def noop_reset(self):
        """
        Steps through reset and warm-start.
        """
        if isinstance(self.gym_env, gym.wrappers.Monitor):
            self.gym_env.stats_recorder.done = True
        state = self.gym_env.reset()
        if self.max_num_noops > 0:
            num_noops = np.random.randint(low=1, high=self.max_num_noops + 1)
            # Do a number of noops to vary starting positions.
            for _ in range_(num_noops):
                state, reward, terminal, info = self.gym_env.step(self.noop_action)
                if terminal:
                    state = self.gym_env.reset()
        return state if self.force_float32 is False else np.array(state, dtype=np.float32)

    def reset_flow(self):
        return self.reset()

    def terminate(self):
        self.gym_env.close()
        self.gym_env = None

    def _step_and_skip(self, actions):
        # TODO - allow for goal reward substitution for multi-goal envs
        if self.frameskip is None:
            # Frames kipping is unset or set as env property.
            return self.gym_env.step(actions)
        else:
            # Do frameskip loop in our wrapper class.
            step_reward = 0.0
            terminal = None
            info = None
            for i in range_(self.frameskip):
                state, reward, terminal, info = self.gym_env.step(actions)
                if i == self.frameskip - 2:
                    self.state_buffer[0] = state
                if i == self.frameskip - 1:
                    self.state_buffer[1] = state
                step_reward += reward
                if terminal:
                    break

            max_frame = self.state_buffer.max(axis=0)

            return max_frame, step_reward, terminal, info

    def step(self, actions):
        if self.visualize:
            self.gym_env.render()
        state, reward, terminal, info = self._step_and_skip(actions)

        # Manage lives if necessary.
        if self.episodic_life:
            self.true_terminal = terminal
            lives = self.gym_env.unwrapped.ale.lives()
            # lives < self.lives -> lost a life so show terminal = true to learner.
            if self.lives > lives > 0:
                terminal = True
            self.lives = lives

        if self.force_float32 is True:
            state = np.array(state, dtype=np.float32)

        return state, np.asarray(reward, dtype=np.float32), terminal, info

    def step_flow(self, actions):
        state, reward, terminal, _ = self.step(actions)
        if terminal:
            state = self.reset_flow()
        return state, reward, terminal

    def render(self):
        self.gym_env.render("human")

    @staticmethod
    def translate_space(space, dtype=None, force_float32=False):
        """
        Translates openAI spaces into RLGraph Space classes.

        Args:
            space (gym.spaces.Space): The openAI Space to be translated.

        Returns:
            Space: The translated rlgraph Space.
        """
        if isinstance(space, gym.spaces.Discrete):
            return IntBox(space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return BoolBox(shape=(space.n,))
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return IntBox(low=np.zeros((space.nvec.ndim,), dtype("uint8", "np")), high=space.nvec)
        elif isinstance(space, gym.spaces.Box):
            # Decide by dtype:
            box_dtype = str(dtype or space.low.dtype)
            if "int" in box_dtype:
                return IntBox(low=space.low, high=space.high, dtype=box_dtype)
            elif "float" in box_dtype:
                return FloatBox(
                    low=space.low, high=space.high, dtype="float32" if force_float32 is True else box_dtype
                )
            elif "bool" in box_dtype:
                return BoolBox(shape=space.shape)
        elif isinstance(space, gym.spaces.Tuple):
            return Tuple(*[OpenAIGymEnv.translate_space(s) for s in space.spaces])
        elif isinstance(space, gym.spaces.Dict):
            return Dict({key: OpenAIGymEnv.translate_space(value, dtype, force_float32)
                         for key, value in space.spaces.items()})

        raise RLGraphError("Unknown openAI gym Space class ({}) for state_space!".format(space))

    def __str__(self):
        return "OpenAIGym({})".format(self.gym_env)
