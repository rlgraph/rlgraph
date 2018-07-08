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

from yarl.components.component import Component
from yarl.utils import root_logger
from yarl.agents.agent import Agent
from yarl.envs.environment import Environment
from yarl.tests.test_util import recursive_assert_almost_equal
from yarl.execution.worker import Worker


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
            logging_level (Optional[int]): When provided, sets YARL's root_logger's logging level to this value.
            enable_profiler (Optional(bool)): When enabled, activates backend profiling.
        """
        self.worker = worker
        self.agent = self.worker.agent
        self.env = self.worker.environment
        self.seed = seed
        if logging_level is not None:
            root_logger.setLevel(logging_level)

        # Use the Agent's GraphBuilder.
        self.graph_executor = self.agent.graph_executor

    def test(self, steps=1, checks=None, deterministic=True, decimals=7):
        """
        Performs n steps in the environment, then checks some variables or other values for (almost) equality.

        Args:
            steps (int): How many time steps to perform using the Worker.
            checks (Optional[List[Tuple]]): An optional list of checks to perform. Each item in the list is a tuple:
                Either:
                (some-value, desired-value): Checks whether som-value is almost equal (`decimals`) to desired-value.
                (variables-dict, desired-values): Checks whether an entire variables dict is almost (`decimals`) equal
                    to the given dict of values.
                (variables-dict, key, desired-value): Checks whether a variable (defined by key) in the given
                    variables dict is almost (`decimals`) equal to the desired-value.
            deterministic
            decimals (Optional[int]): The number of digits after the floating point up to which to compare actual
                outputs and expected values.
            #fn_test (Optional[callable]): Test function to call with (self, outs) as parameters.
        """
        # Perform n steps.
        self.worker.execute_timesteps(num_timesteps=steps, deterministic=deterministic)

        # Perform some checks.
        for i, check in enumerate(checks):
            assert isinstance(check, (tuple, list)) and len(check) == 3 and isinstance(check[1], str)

            # Variable check.
            if isinstance(check[0], Component):
                component = check[0]
                var_key = component.global_scope+"/"+check[1]
                variables_dict = component.variables
                assert var_key in variables_dict, "ERROR: Variable '{}' not found in Component '{}'!".\
                    format(var_key, component.global_scope)
                var = variables_dict[var_key]
                value = self.graph_executor.read_variable_values(var)
                try:
                    recursive_assert_almost_equal(value, check[2], decimals=decimals)
                except AssertionError:
                    self.agent.logger.error("Mismatch in check #{} (Variable {}).".format(i+1, check[1]))
                    raise
            # Simple value check.
            else:
                obj = check[0]
                property = check[1]
                is_value = getattr(obj, property, None)
                desired_value = check[2]
                try:
                    recursive_assert_almost_equal(is_value, desired_value, decimals=decimals)
                except AssertionError:
                    self.agent.logger.error("Mismatch in check #{}:".format(i+1))
                    raise
