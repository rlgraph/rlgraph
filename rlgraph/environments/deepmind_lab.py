# Copyright 2018 The RLgraph authors, All Rights Reserved.
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

import deepmind_lab
import numpy as np
import time

from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.environments.environment import Environment
from rlgraph.spaces import *
from rlgraph.utils.util import force_list, default_dict


class DeepmindLabEnv(Environment):
    """
    Deepmind Lab Environment adapter for RLgraph:
    https://arxiv.org/abs/1612.03801
    https://github.com/deepmind/lab

    Also note this paper, which uses the deepmind Lab as environment:
    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, level_id, observations="RGB_INTERLEAVED", actions=None, frameskip=4, config=None,
                 renderer="software", seed=None, level_cache=None):
        """
        Args:
            level_id (str): Specifier of the level to play, e.g. 'seekavoid_arena_01'.
            observations (Union[str,List[str]]): String specifier(s) for the observation(s) to be used with the
                given level. Will be converted into either a (single) BoxSpace or a Tuple (of BoxSpaces).
                See deepmind's documentation for all available observations.
            actions (Optional[List[dict]]): The RLgraph action spec (currently, only IntBox (shape=()) RLgraph action
                spaces are supported) that will be translated from and to the deepmind Lab actions.
                List slots correspond to the single int-actions, list items are dicts with:
                key=deepmind Lab partial action name e.g. LOOK_LEFT_RIGHT_PIXELS_PER_FRAME.
                value=the value for that deepmind Lab partial action e.g. -100.
            frameskip (Optional[Tuple[int,int],int]): How many frames should be skipped with (repeated action and
                accumulated reward). E.g. (2,5) -> Uniformly pull from set [2,3,4].
                Default: 4.
            config (Optional[dict]): The `config` parameter to be passed into the Lab's constructor.
                Supports 'width', 'height', 'fps', and other useful parameters.
                Values must be given as string values. e.g. dict(width='96')
            renderer (str): The `renderer` parameter to be passed into the Lab's constructor.
            seed (Optional[int]): An optional seed to use to initialize a numpy random state object, which is then used
                to seed all occurring resets in a deterministic fashion.
            level_cache (Optional[object]): An optional custom level caching object to help increase performance
                when playing many repeating levels. Will be passed as is into the Lab's constructor.
        """
        # Create the wrapped deepmind lab level object.
        self.level_id = level_id
        observations = force_list(observations)
        config = default_dict(config, dict(width='96', height='72', fps='60'))  # Default config.
        self.level = deepmind_lab.Lab(
            self.level_id, observations, config=config, renderer=renderer, level_cache=level_cache
        )

        # Dict mapping a discrete action (int) - we don't support continuous actions yet - into a
        # deepmind Lab action vector.
        self.action_list, action_space = self.define_actions(actions)
        observation_space = self.define_observations(observations)
        super(DeepmindLabEnv, self).__init__(observation_space, action_space)

        self.frameskip = frameskip
        self.random_state = np.random.RandomState(seed=seed or int(time.time()))

    def terminate(self):
        """
        Shuts down the underlying Quake III Arena instance.
        Invalidates `self.level` such that no other method calls are possible afterwards.
        """
        self.level.close()
        self.level = None

    def reset(self):
        print("\n----------------------------\nResetting DM Lab Env.\n----------------------------\n")
        self.level.reset(seed=self.random_state.randint(0, 2 ** 31 - 1))
        state = self.level.observations()
        return state

    def reset_for_env_stepper(self):
        state = self.reset()
        if isinstance(self.state_space, Dict):
            return [state[key] for key in self.state_space]
        else:
            return state[next(iter(state))]

    def step(self, actions):
        # Do the actual step.
        reward = self.level.step(action=self.action_list[actions], num_steps=self.frameskip)
        terminal = not self.level.is_running()
        # Quirk in DM Lab: A terminal state cannot be observed anymore
        # (which is fine as its state-value (and action choices) is always 0 anyway).
        state = self.level.observations() if terminal is False else self.state_space.zeros()

        # Return state, reward, terminal, and None (info).
        return state, np.array(reward, dtype=np.float32), terminal, None

    def step_for_env_stepper(self, actions):
        # Do the actual step.
        # TODO: Remove this impala hack again (just to see whether looking up action in graph is faster).
        reward = self.level.step(action=self.action_list[actions], num_steps=self.frameskip)
        #reward = self.level.step(action=actions, num_steps=self.frameskip)
        terminal = np.array(not self.level.is_running())
        # Flow Env logic.
        if terminal:
            state = self.reset()
        else:
            state = self.level.observations()

        # Return state, reward, terminal, and None (info).
        if isinstance(self.state_space, Dict):
            return [state[key] for key in self.state_space] + [np.array(reward, dtype=np.float32), terminal]
        else:
            return [state[next(iter(state))], np.array(reward, dtype=np.float32), terminal]

    @staticmethod
    def define_actions(actions_spec=None):
        """
        Translates and maps Rlgraph IntBox(shape=()) actions - provided by user - to the correct deepmind Lab
        representation for the calls to `step`.

        Args:
            actions_spec (List[dict]): The discrete action definitions to be supported by this Environment.

        Returns:
            tuple:
            - A lookup list of deepmind actions, where the slot is the RLgraph IntBox value
            and the items are numpy arrays (with dtype=np.intc) that are understood by deepmind Lab.
            - The RLgraph action Space (IntBox(shape=(), n)), where n is the number of discrete actions.
        """
        # Default actions: The ones used in the IMPALA paper (see [1]).
        if actions_spec is None:
            actions_spec = [
                dict(MOVE_BACK_FORWARD=1),  # forward
                dict(MOVE_BACK_FORWARD=-1),  # backward
                dict(STRAFE_LEFT_RIGHT=1),  # strafe right
                dict(STRAFE_LEFT_RIGHT=-1),  # strafe left
                dict(LOOK_LEFT_RIGHT_PIXELS_PER_FRAME=-20),  # look left
                dict(LOOK_LEFT_RIGHT_PIXELS_PER_FRAME=20),  # look right
                dict(MOVE_BACK_FORWARD=1, LOOK_LEFT_RIGHT_PIXELS_PER_FRAME=-20),  # forward + look left
                dict(MOVE_BACK_FORWARD=1, LOOK_LEFT_RIGHT_PIXELS_PER_FRAME=20),  # forward + look right
                dict(FIRE=1),  # fire
            ]

        # Build the lookup dict mapping ints to deepmind-readable actions (numpy intc arrays).
        lookup_list = list()
        for action in actions_spec:
            assert isinstance(action, dict), "ERROR: Single action spec '{}' must be a dict!".format(action)
            lookup_list.append(np.array([0] * 7, dtype=np.intc))
            for name, value in action.items():
                # TODO: Sanity check values for deepmind lab bounds.
                slot = 0 if name == "LOOK_LEFT_RIGHT_PIXELS_PER_FRAME" else 1 \
                    if name == "LOOK_DOWN_UP_PIXELS_PER_FRAME" else 2 \
                    if name == "STRAFE_LEFT_RIGHT" else 3 if name == "MOVE_BACK_FORWARD" \
                    else 4 if name == "FIRE" else 5 if name == "JUMP" else 6  # 6=CROUCH
                lookup_list[-1][slot] = value

        # Return the lookup_list and the RLgraph action Space.
        return lookup_list, IntBox(len(actions_spec))

    def define_observations(self, observation_spec):
        """
        Creates a RLgraph Space for the given deepmind Lab's observation specifier.

        Args:
            observation_spec (List[str]): A list with the wanted names from the deepmind Lab available observations.
                Each available observation is a dict with the following keys: name, shape and dtype.

        Returns:
            Space: The RLgraph equivalent observation Space.
        """
        dict_space = dict()
        space = None
        available_observations = self.level.observation_spec()
        for observation_name in observation_spec:
            # Find the observation_item in the observation_spec of the Env.
            observation_item = [o for o in available_observations if o["name"] == observation_name][0]
            if "float" in str(observation_item["dtype"]):
                space = FloatBox(shape=observation_item["shape"], dtype=observation_item["dtype"])
            elif "int" in str(observation_item["dtype"]):
                space = IntBox(shape=observation_item["shape"], dtype=observation_item["dtype"])
            elif "str" in str(observation_item["dtype"]):
                space = TextBox(shape=observation_item["shape"])
            else:
                raise RLGraphError("Unknown Deepmind Lab Space class for state_space!")

            dict_space[observation_name] = space

        if len(dict_space) == 1:
            return space
        else:
            return Dict(dict_space)

    def __str__(self):
        return "DeepMindLab({})".format(self.level_id)
