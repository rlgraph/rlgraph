# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl import Specifiable


class Env(Specifiable):
    """
    An Env class used to run experiment-based RL.
    """
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def seed(self, seed=None):
        """
        Sets the random seed of the environment to the given value (current time if None).

        Args:
            seed (int): The seed to use (default: current epoch seconds).

        Returns: The seed actually used.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.

        Returns: A tuple of (state, reward, is-terminal, info).
        """
        raise NotImplementedError

    def step(self, **kwargs):
        """
        Run one time step of the environment's dynamics. When the end of an episode is reached, reset() should be
        called to reset the environment's internal state.

        Args:
            kwargs (any): The action(s) to be executed by the environment. Actions have to be members of this
                Environment's action_space (a call to self.action_space.contains(action) must return True)

        Returns: A tuple of (state after(!) executing the given actions(s), reward, is-terminal, info).
        """
        raise NotImplementedError

    def render(self):
        """
        Should render the Environment in its current state. May be implemented or not.
        """
        pass

    def terminate(self):
        """
        Clean up operation. May be implemented or not.
        """
        pass

    def __str__(self):
        raise NotImplementedError


