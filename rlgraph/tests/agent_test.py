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

from rlgraph.utils import root_logger
from rlgraph.tests.test_util import recursive_assert_almost_equal
from rlgraph.execution.worker import Worker


class AgentTest(object):
    """
    A simple (and limited) Agent-wrapper to test an Agent in an easy, straightforward way.
    """
    def __init__(self, worker, seed=10, logging_level=None, enable_profiler=False):
        """
        Args:
            worker (Worker): The Worker (holding the Env and Agent) to use for stepping.
            #seed (Optional[int]): The seed to use for random-seeding the Model object.
            #    If None, do not seed the Graph (things may behave non-deterministically).
            logging_level (Optional[int]): When provided, sets RLGraph's root_logger's logging level to this value.
            enable_profiler (Optional(bool)): When enabled, activates backend profiling.
        """
        self.worker = worker
        self.agent = self.worker.agent
        self.env = self.worker.vector_env.get_env()
        self.seed = seed
        if logging_level is not None:
            root_logger.setLevel(logging_level)

        # Simply use the Agent's GraphExecutor.
        self.graph_executor = self.agent.graph_executor

    def step(self, num_timesteps=1, use_exploration=False, frameskip=None, reset=False):
        """
        Performs n steps in the environment, picking up from where the Agent/Environment was before (no reset).

        Args:
            num_timesteps (int): How many time steps to perform using the Worker.
            use_exploration (Optional[bool]): Indicates whether to utilize exploration (epsilon or noise based)
                when picking actions. Default: False (b/c we are testing).
            frameskip (Union[int,None]): Number of repeated (same) actions in one "step".
            reset (bool): Whether to reset the previous run(s) and start from scratch.
                Default: False. Picks up from a previous `step` (even if in the middle of an episode).

        Returns:
            dict: The stats dict returned by the worker after num_timesteps have been taken.
        """
        # Perform n steps and return stats.
        return self.worker.execute_timesteps(num_timesteps=num_timesteps, use_exploration=use_exploration,
                                             frameskip=frameskip, reset=reset)

    def check_env(self, prop, expected_value, decimals=7):
        """
        Checks a property of our environment for (almost) equality.

        Args:
            prop (str): The name of the Environment's property to check.
            expected_value (any): The expected value of the given property.
            decimals (Optional[int]): The number of digits after the floating point up to which to compare actual
                and expected values.
        """
        is_value = getattr(self.env, prop, None)
        recursive_assert_almost_equal(is_value, expected_value, decimals=decimals)

    def check_agent(self, prop, expected_value, decimals=7, key_or_index=None):
        """
        Checks a property of our Agent for (almost) equality.

        Args:
            prop (str): The name of the Agent's property to check.
            expected_value (any): The expected value of the given property.
            decimals (Optional[int]): The number of digits after the floating point up to which to compare actual
                and expected values.
            key_or_index (Optional[int, str]): Optional key or index into the propery in case of nested data structure.
        """
        is_value = getattr(self.agent, prop, None)
        if key_or_index is not None:
            is_value = is_value[key_or_index]
        recursive_assert_almost_equal(is_value, expected_value, decimals=decimals)

    def check_var(self, variable, expected_value, decimals=7):
        """
        Checks a value of our an Agent's variable for (almost) equality against an expected one.

        Args:
            variable (str): The global scope (within Agent's root-component) of the variable to check.
            expected_value (any): The expected value of the given variable.
            decimals (Optional[int]): The number of digits after the floating point up to which to compare actual
                and expected values.
        """
        variables_dict = self.agent.root_component.variables
        assert variable in variables_dict, "ERROR: Variable '{}' not found in Agent '{}'!".\
            format(variable, self.agent.name)
        var = variables_dict[variable]
        value = self.graph_executor.read_variable_values(var)
        recursive_assert_almost_equal(value, expected_value, decimals=decimals)
