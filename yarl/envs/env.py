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
from yarl.spaces import Space


class Environment(Specifiable):
    """
    An Env class used to run experiment-based RL.
    """
    def __init__(self, state_space, action_space):
        """
        Args:
            state_space (Union[dict,Space]): The spec-dict for generating the state Space or the state Space object
                itself.
            action_space (Union[dict,Space]): The spec-dict for generating the action Space or the action Space object
                itself.
        """
        self.state_space = Space.from_spec(state_space)
        self.action_space = Space.from_spec(action_space)

    def seed(self, seed=None):
        """
        Sets the random seed of the environment to the given value.

        Args:
            seed (int): The seed to use (default: current epoch seconds).

        Returns:
            int: The seed actually used.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.

        Returns:
            tuple: The Env's state after the reset.
        """
        raise NotImplementedError

    def step(self, **kwargs):
        """
        Run one time step of the environment's dynamics. When the end of an episode is reached, reset() should be
        called to reset the environment's internal state.

        Args:
            kwargs (any): The action(s) to be executed by the environment. Actions have to be members of this
                Environment's action_space (a call to self.action_space.contains(action) must return True)

        Returns:
            tuple: A tuple of (state after(!) executing the given actions(s), reward, is-terminal, info).
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


