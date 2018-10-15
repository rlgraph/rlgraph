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

import numpy as np
import random

from rlgraph import get_backend
from rlgraph.utils import root_logger
from rlgraph.utils.input_parsing import parse_execution_spec
from rlgraph.graphs import GraphBuilder
from rlgraph.graphs.graph_executor import GraphExecutor
from rlgraph.tests.test_util import recursive_assert_almost_equal


class ComponentTest(object):
    """
    A simple (and limited) Graph-wrapper to test a single component in an easy, straightforward way.
    """
    def __init__(
        self,
        component,
        input_spaces=None,
        action_space=None,
        seed=10,
        logging_level=None,
        execution_spec=None,
        # TODO: Move all the below into execution_spec just like for Agent class.
        enable_profiler=False,
        disable_monitoring=False,
        device_strategy="default",
        device_map=None,
        backend=None,
        auto_build=True
    ):
        """
        Args:
            component (Component): The Component to be tested (may contain sub-components).
            input_spaces (Optional[dict]): Dict with component's API input-parameter' names as keys and Space objects
                or Space specs as values. Describes the input Spaces for the component.
                None, if the Component to be tested has no API methods with input parameters.
            action_space (Optional[Space]): The action space to pass into the GraphBuilder.
            seed (Optional[int]): The seed to use for random-seeding the Model object.
                If None, do not seed the Graph (things may behave non-deterministically).
            logging_level (Optional[int]): When provided, sets RLGraph's root_logger's logging level to this value.
            execution_spec (Optional[dict]): Specification dict for execution settings.
            enable_profiler (bool): When enabled, activates backend profiling. Default: False.
            disable_monitoring (bool): When True, will not use a monitored session. Default: False.
            device_strategy (str): Optional device-strategy to be passed into GraphExecutor.
            device_map (Optional[Dict[str,str]]): Optional device-map to be passed into GraphExecutor.
            backend (Optional[str]): Override global backend settings for a test by passing in a specific
                backend, convenience method.
            auto_build (Optional[bool]): If false, build has to be triggered manually to eval build stats.
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        if logging_level is not None:
            root_logger.setLevel(logging_level)

        # Create a GraphBuilder.
        self.graph_builder = GraphBuilder(action_space=action_space)
        self.component = component
        self.input_spaces = input_spaces

        # Build the model.
        execution_spec = parse_execution_spec(execution_spec or dict(
            seed=self.seed,
            enable_profiler=enable_profiler,
            profiler_frequency=1,
            device_strategy=device_strategy,
            disable_monitoring=disable_monitoring,
            device_map=device_map
        ))
        use_backend = backend if backend is not None else get_backend()
        self.graph_executor = GraphExecutor.from_spec(
            use_backend,
            graph_builder=self.graph_builder,
            execution_spec=execution_spec
        )
        if auto_build:
            self.build()
        else:
            print("Auto-build false, did not build. Waiting for manual build..")

    def build(self):
        return self.graph_executor.build([self.component], self.input_spaces)

    def test(self, *api_method_calls, **kwargs):
        """
        Does one test pass through the component to test.

        Args:
            api_method_calls (Union[str,list,tuple]): See rlgraph.graphs.graph_executor for details.
            A specifier for an API-method call.
                - str: Call the API-method that has the given name w/o any input args.
                - tuple len=2: 0=the API-method name to call; 1=the input args to use for the call.
                - tuple len=3: same as len=2, AND 2=list of returned op slots to pull (e.g. [0]: only pull
                    the first op).

        Keyword Args:
            expected_outputs (Optional[any]): The expected return value(s) generated by the API-method.
                If None, no checks will be done on the output.
            decimals (Optional[int]): The number of digits after the floating point up to which to compare actual
                outputs and expected values.
            fn_test (Optional[callable]): Test function to call with (self, outs) as parameters.

        Returns:
            any: The actual returned values when calling the API-method with the given parameters.
        """
        expected_outputs = kwargs.pop("expected_outputs", None)
        decimals = kwargs.pop("decimals", 7)
        fn_test = kwargs.pop("fn_test", None)
        assert not kwargs

        # Get the outs ..
        outs = self.graph_executor.execute(*api_method_calls)

        #  Optionally do test asserts here.
        if expected_outputs is not None:
            self.assert_equal(outs, expected_outputs, decimals=decimals)

        if callable(fn_test):
            fn_test(self, outs)

        return outs

    def variable_test(self, variables, expected_values):
        """
        Asserts that all given `variables` have the `expected_values`.
        Variables can be given in an arbitrary structure including nested ones.

        Args:
            variables (any): Any structure that contains variables.
            expected_values (any): Matching structure with the expected values for the given variables.
        """
        values = self.read_variable_values(variables)
        self.assert_equal(values, expected_values)

    def read_variable_values(self, *variables):
        """
        Executes a session to retrieve the values of the provided variables.

        Args:
            variables (Union[variable,List[variable]]): Variable objects whose values to retrieve from the graph.

        Returns:
            any: Values of the variables provided.
        """
        ret = self.graph_executor.read_variable_values(variables)
        if len(variables) == 1:
            return ret[0]
        return ret

    @staticmethod
    def assert_equal(outs, expected_outputs, decimals=7):
        """
        Convenience wrapper: See implementation of `recursive_assert_almost_equal` for details.
        """
        recursive_assert_almost_equal(outs, expected_outputs, decimals=decimals)

    def terminate(self):
        """
        Terminates this ComponentTest object (so it can no longer be used) allowing for cleanup
        operations to be executed.
        """
        self.graph_executor.terminate()
