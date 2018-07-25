# Copyright 2018 The RLgraph authors. All Rights Reserved.
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

import gym
import numpy as np
import time

from rlgraph import RLGraphError
from rlgraph.utils.util import dtype
from rlgraph.environments import Environment
from rlgraph.spaces import *


class OpenAIGymEnv(Environment):
    """
    OpenAI Gym Integration: https://gym.openai.com/.
    """
    def __init__(self, gym_env, frameskip=None, monitor=None, monitor_safe=False, monitor_video=0, visualize=False):
        """
        Args:
            gym_env (Union[str,gym.Env]): OpenAI Gym environment ID or actual gym.Env. See https://gym.openai.com/envs
            frameskip (Optional[Tuple[int,int],int]): How many frames should be skipped with (repeated action and
                accumulated reward). Default: (2,5) -> Uniformly pull from set [2,3,4].
            monitor: Output directory. Setting this to None disables monitoring.
            monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.
            visualize: If set True, the program will visualize the trainings of gym's environment. Note that such
                visualization is probably going to slow down the training.
        """
        if isinstance(gym_env, str):
            self.gym_env = gym.make(gym_env)  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv
        else:
            self.gym_env = gym_env

        # Manually set the frameskip property.
        if frameskip is not None:
            self.gym_env.env.frameskip = frameskip

        observation_space = self.translate_space(self.gym_env.observation_space)
        action_space = self.translate_space(self.gym_env.action_space)
        super(OpenAIGymEnv, self).__init__(observation_space, action_space)

        self.visualize = visualize
        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            self.gym_env = gym.wrappers.Monitor(self.gym_env, monitor, force=not monitor_safe,
                                                video_callable=video_callable)

    def seed(self, seed=None):
        if seed is None:
            seed = time.time()
        self.gym_env.seed(seed)
        return seed

    def reset(self):
        if isinstance(self.gym_env, gym.wrappers.Monitor):
            self.gym_env.stats_recorder.done = True
        return self.gym_env.reset()

    def terminate(self):
        self.gym_env.close()
        self.gym_env = None

    def step(self, actions):
        if self.visualize:
            self.gym_env.render()
        state, reward, terminal, info = self.gym_env.step(actions)
        return state, reward, terminal, info

    def render(self):
        self.gym_env.render("human")

    @staticmethod
    def translate_space(space):
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
            return FloatBox(low=space.low, high=space.high)
        elif isinstance(space, gym.spaces.Tuple):
            return Tuple(*[OpenAIGymEnv.translate_space(s) for s in space.spaces])
        elif isinstance(space, gym.spaces.Dict):
            return Dict({key: OpenAIGymEnv.translate_space(value) for key, value in space.spaces.items()})
        else:
            raise RLGraphError("Unknown openAI gym Space class for state_space!")

    def __str__(self):
        return "OpenAIGym({})".format(self.gym_env)

