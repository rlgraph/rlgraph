# Copyright 2018/2019 The Rlgraph Authors, All Rights Reserved.
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

from rlgraph.components.component import Component
from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.preprocessing.container_splitter import ContainerSplitter
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.tests.dummy_components import DummyWithVar, SimpleDummyWithVar, DummyCallingOneAPIFromWithinOther
from rlgraph.utils.decorators import rlgraph_api, graph_fn


class DummyWithSubComponents(Component):
    """
    A dummy component with a couple of sub-components that have their own API methods.

    API:
        run(input_): Result of input_ + sub_comp.run(input_) + `self.constant_value`
    """
    def __init__(self, scope="dummy-with-sub-components", constant_value=1.0, **kwargs):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(DummyWithSubComponents, self).__init__(scope=scope, **kwargs)
        self.constant_value = constant_value

        # Create a sub-Component and add it.
        self.sub_comp = DummyWithVar()
        self.add_components(self.sub_comp)

    @rlgraph_api
    def run1(self, input_):
        # Explicit definition of an API-method using one of our graph_fn and one of
        # our child API-methods.
        result = self.sub_comp.run_plus(input_)
        result2 = self._graph_fn_call(result)
        return result, result2

    @rlgraph_api
    def run2(self, input_):
        result1 = self.sub_comp.run_minus(input_)
        result3 = self._graph_fn_call(result1)
        return result3

    @graph_fn(returns=1)
    def _graph_fn_call(self, input_):
        return input_ + self.constant_value


class DummyNNWithDictInput(NeuralNetwork):
    """
    Dummy NN with dict input taking a dict with keys "a" and "b" passes them both through two different (parallel,
    not connected in any way) dense layers and then concatenating the outputs to yield the final output.
    """

    def __init__(self, num_units_a=3, num_units_b=2, scope="dummy-nn-with-dict-input", **kwargs):
        super(DummyNNWithDictInput, self).__init__(scope=scope, **kwargs)

        self.num_units_a = num_units_a
        self.num_units_b = num_units_b

        # Splits the input into two streams.
        self.splitter = ContainerSplitter("a", "b")
        self.stack_a = DenseLayer(units=self.num_units_a, scope="dense-a")
        self.stack_b = DenseLayer(units=self.num_units_b, scope="dense-b")
        self.concat_layer = ConcatLayer()

        # Add all sub-components to this one.
        self.add_components(self.splitter, self.stack_a, self.stack_b, self.concat_layer)

    @rlgraph_api
    def call(self, input_dict):
        # Split the input dict into two streams.
        input_a, input_b = self.splitter.call(input_dict)

        # Get the two stack outputs.
        output_a = self.stack_a.call(input_a)
        output_b = self.stack_b.call(input_b)

        # Concat everything together, that's the output.
        concatenated_data = self.concat_layer.call(output_a, output_b)

        return concatenated_data


class DummyCallingSubComponentsAPIFromWithinGraphFn(Component):
    """
    A dummy component with one sub-component that has variables and an API-method.
    This dummy calls the sub-componet's API-method from within its graph_fn.

    API:
        run(input_): Result of input_ + sub_comp.run(input_) + `self.constant_value`
    """

    def __init__(self, scope="dummy-calling-sub-components-api-from-within-graph-fn", constant_value=1.0, **kwargs):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(DummyCallingSubComponentsAPIFromWithinGraphFn, self).__init__(scope=scope, **kwargs)
        self.constant_value = constant_value

        # Create a sub-Component and add it.
        self.sub_comp = SimpleDummyWithVar()
        self.add_components(self.sub_comp)

    @rlgraph_api
    def run(self, input_):
        # Returns 2*input_ + 10.0.
        sub_comp_result = self.sub_comp.run(input_)  # input_ + 3.0
        self_result = self._graph_fn_call(sub_comp_result)  # 2*(input_ + 3.0) + 4.0 = 2*input_ + 10.0
        return self_result

    @graph_fn(returns=1)
    def _graph_fn_call(self, input_):
        # Returns: input_ + [(input_ + 1.0) + 3.0] = 2*input_ + 4.0
        intermediate_result = input_ + self.constant_value
        after_api_call = self.sub_comp.run(intermediate_result)
        return input_ + after_api_call


class DummyProducingInputIncompleteBuild(Component):
    """
    A dummy component which produces an input-incomplete build.
    """
    def __init__(self, scope="dummy-calling-sub-components-api-from-within-graph-fn", **kwargs):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(DummyProducingInputIncompleteBuild, self).__init__(scope=scope, **kwargs)

        # Create the "problematic" sub-Component and add it.
        self.sub_comp = DummyCallingOneAPIFromWithinOther()
        self.add_components(self.sub_comp)

    @rlgraph_api
    def run(self, input_):
        return self.sub_comp.run(input_)


