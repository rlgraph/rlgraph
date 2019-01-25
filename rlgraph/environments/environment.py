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

from rlgraph.utils.specifiable import Specifiable
from rlgraph.spaces import Space


class Environment(Specifiable):
    """
    An Env class used to run experiment-based RL.
    """
    def __init__(self, state_space, action_space, seed=None):
        """
        Args:
            state_space (Union[dict,Space]): The spec-dict for generating the state Space or the state Space object
                itself.
            action_space (Union[dict,Space]): The spec-dict for generating the action Space or the action Space object
                itself.
            #reward_clipping (Optionalp[Tuple[float,float],float]: An optional reward clipping setting used
            #    to restrict all rewards produced by the Environment to be in a certain range.
            #    None for no clipping. Single float for clipping between -`reward_clipping` and +`reward_clipping`.
        """
        super(Environment, self).__init__()

        self.state_space = Space.from_spec(state_space)
        self.action_space = Space.from_spec(action_space)
        # self.reward_clipping = reward_clipping

        # Add some seeding to the created Env.
        if seed is not None:
            self.seed(seed)

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
            The Env's state after the reset.
        """
        raise NotImplementedError

    def reset_flow(self):
        """
        A special implementation of `reset` in which the state after the reset is returned as a tuple of flat
        state-component iff a Dict state is given.

        Returns:
            The Env's state (flat components if Dict) after the reset.
        """
        pass  # optional

    def step(self, actions, **kwargs):
        """
        Run one time step of the environment's dynamics. When the end of an episode is reached, reset() should be
        called to reset the environment's internal state.

        Args:
            actions (any): The action(s) to be executed by the environment. Actions have to be members of this
                Environment's action_space (a call to self.action_space.contains(action) must return True)

        Returns:
            tuple:
                - The state s' after(!) executing the given actions(s).
                - The reward received after taking a in s.
                - Whether s' is a terminal state.
                - Some Environment specific info.
        """
        raise NotImplementedError

    def step_flow(self, **kwargs):
        """
        A special implementation of `step` in which `reset` is called automatically if a terminal is encountered, such
        that only a sequence of `step_flow` is needed in any loop. Always returns the next state or - if terminal -
        the first state after the reset (and then the last reward before the reset and True for terminal).
        Also, if a Dict state is given, will flatten it into its single components.

        Args:
            kwargs (any): The action(s) to be executed by the environment. Actions have to be members of this
                Environment's action_space (a call to self.action_space.contains(action) must return True)

        Returns:
            tuple:
                - The state s' after(!) executing the given actions(s) or after a reset if the action lead to a terminal
                    state.
                - The reward received after taking a in s.
                - Whether s' is a terminal state.
        """
        pass  # optional

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


