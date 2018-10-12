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

from rlgraph.utils.ops import FlattenedDataOp
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.components.common.container_splitter import ContainerSplitter
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.nn.concat_layer import ConcatLayer


class Dummy1To1(Component):
    def __init__(self, scope="dummy-1-to-1", constant_value=1.0, **kwargs):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(Dummy1To1, self).__init__(scope=scope, **kwargs)
        self.constant_value = constant_value

    @rlgraph_api(name="run", returns=1)
    def _graph_fn_1to1(self, input_):
        """
        Returns:
            `output` (SingleDataOp): Result of input_ + `self.constant_value`.
        """
        return input_ + self.constant_value


class Dummy2To1(Component):
    """
    API:
        run(input1, input2): Result of input1 + input2.
    """
    def __init__(self, scope="dummy-2-to-1"):
        super(Dummy2To1, self).__init__(scope=scope)

    @rlgraph_api(name="run", returns=1)
    def _graph_fn_2to1(self, input1, input2):
        return input1 + input2


class Dummy1To2(Component):
    """
    API:
        run(input1): (input + `self.constant_value`, input * `self.constant_value`).
    """
    def __init__(self, scope="dummy-1-to-2", constant_value=1.0):
        super(Dummy1To2, self).__init__(scope=scope)
        self.constant_value = constant_value

    @rlgraph_api(name="run", returns=2)
    def _graph_fn_1to2(self, input_):
        return input_ + self.constant_value, input_ * self.constant_value


class Dummy2To2(Component):
    """
    API:
        run(input1, input2): (input1 + `self.constant_value`, input2 * `self.constant_value`).
    """
    def __init__(self, scope="dummy-2-to-2", constant_value=1.0):
        super(Dummy2To2, self).__init__(scope=scope)
        self.constant_value = constant_value

    @rlgraph_api(name="run", returns=2)
    def _graph_fn_2to2(self, input1, input2):
        return input1 + self.constant_value, input2 * self.constant_value


class Dummy2To2WithDefaultValue(Component):
    """
    API:
        run(input1, input2=None): (input1 + `self.constant_value`, (input2 or 5.0) * `self.constant_value`).
    """
    def __init__(self, scope="dummy-2-to-2-w-default-value", constant_value=1.0):
        super(Dummy2To2WithDefaultValue, self).__init__(scope=scope)
        self.constant_value = constant_value

    @rlgraph_api(name="run", returns=2)
    def _graph_fn_2to2(self, input1, input2=None):
        return input1 + self.constant_value, (input2 or 5.0) * self.constant_value


class Dummy0To1(Component):
    """
    A dummy component with one graph_fn without api_methods and one output.

    API:
        run() -> fixed value stored in a variable
    """
    def __init__(self, scope="dummy-0-to-1", var_value=1.0):
        super(Dummy0To1, self).__init__(scope=scope)
        self.var_value = var_value
        self.var = None

    def create_variables(self, input_spaces, action_space=None):
        self.var = self.get_variable(initializer=self.var_value)

    @rlgraph_api(name="run", returns=1)
    def _graph_fn_0to1(self):
        return self.var


class Dummy2GraphFns1To1(Component):
    """
    API:
        run(input_): Result of input_ + `self.constant_value` - 2x`self.constant_value`
    """
    def __init__(self, scope="dummy-2graph_fns-1to1", constant_value=1.0):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(Dummy2GraphFns1To1, self).__init__(scope=scope)
        self.constant_value = constant_value

    @rlgraph_api
    def run(self, input_):
        # Explicit definition of an API-method using both our graph_fn.
        result_1to1 = self._graph_fn_1to1(input_)
        result_1to1_neg = self._graph_fn_1to1_neg(result_1to1)
        return result_1to1_neg

    @graph_fn
    def _graph_fn_1to1(self, input_):
        return input_ + self.constant_value

    @graph_fn
    def _graph_fn_1to1_neg(self, input_):
        return input_ - self.constant_value * 2


class DummyWithVar(Component):
    """
    A dummy component with a couple of sub-components that have their own API methods.

    API:
        run_plus(input_): input_ + `self.constant_variable`
        run_minus(input_): input_ - `self.constant_value`
    """
    def __init__(self, scope="dummy-with-var", constant_value=2.0, **kwargs):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(DummyWithVar, self).__init__(scope=scope, **kwargs)
        self.constant_value = constant_value
        self.constant_variable = None

    def create_variables(self, input_spaces, action_space=None):
        self.constant_variable = self.get_variable(name="constant-variable", initializer=2.0)

    @rlgraph_api
    def run_plus(self, input_):
        # Explicit definition of an API-method using one of our graph_fn.
        result = self._graph_fn_1(input_)
        return result

    @graph_fn(returns=1)
    def _graph_fn_1(self, input_):
        return input_ + self.constant_value

    @rlgraph_api(name="run_minus", returns=1)
    def _graph_fn_2(self, input_):
        return input_ - self.constant_variable


class SimpleDummyWithVar(Component):
    """
    A simpler dummy component with only one variable and one  graph_fn.

    API:
        run(input_): input_ + `self.variable`(3.0)
    """
    def __init__(self, scope="simple-dummy-with-var", **kwargs):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(SimpleDummyWithVar, self).__init__(scope=scope, **kwargs)
        self.constant_variable = None

    def create_variables(self, input_spaces, action_space=None):
        self.constant_variable = self.get_variable(name="constant-variable", initializer=3.0)

    @rlgraph_api
    def run(self, input_):
        # Explicit definition of an API-method using one of our graph_fn.
        result = self._graph_fn_1(input_)
        return result

    @graph_fn(returns=1)
    def _graph_fn_1(self, input_):
        return input_ + self.constant_variable


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
        result2 = self._graph_fn_apply(result)
        return result, result2

    @rlgraph_api
    def run2(self, input_):
        result1 = self.sub_comp.run_minus(input_)
        result3 = self._graph_fn_apply(result1)
        return result3

    @graph_fn(returns=1)
    def _graph_fn_apply(self, input_):
        return input_ + self.constant_value


class DummyCallingSubCompAPIFromWithinGraphFn(Component):
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
        super(DummyCallingSubCompAPIFromWithinGraphFn, self).__init__(scope=scope, **kwargs)
        self.constant_value = constant_value

        # Create a sub-Component and add it.
        self.sub_comp = SimpleDummyWithVar()
        self.add_components(self.sub_comp)

    @rlgraph_api
    def run(self, input_):
        # Returns 2*input_ + 10.0.
        sub_comp_result = self.sub_comp.run(input_)  # input_ + 3.0
        self_result = self._graph_fn_apply(sub_comp_result)  # 2*(input_ + 3.0) + 4.0 = 2*input_ + 10.0
        return self_result

    @graph_fn
    def _graph_fn_apply(self, input_):
        # Returns: input_ + [(input_ + 1.0) + 3.0] = 2*input_ + 4.0
        intermediate_result = input_ + self.constant_value
        after_api_call = self.sub_comp.run(intermediate_result)
        return input_ + after_api_call


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
        self_result = self._graph_fn_apply(sub_comp_result)  # 2*(input_ + 3.0) + 4.0 = 2*input_ + 10.0
        return self_result

    @graph_fn(returns=1)
    def _graph_fn_apply(self, input_):
        # Returns: input_ + [(input_ + 1.0) + 3.0] = 2*input_ + 4.0
        intermediate_result = input_ + self.constant_value
        after_api_call = self.sub_comp.run(intermediate_result)
        return input_ + after_api_call


class FlattenSplitDummy(Component):
    """
    A dummy component with a 2-to-2 graph_fn mapping with flatten/split settings set to True.
    """
    def __init__(self, scope="dummy-2-to-2-all-options", constant_value=1.0, **kwargs):
        super(FlattenSplitDummy, self).__init__(scope=scope, **kwargs)

        self.constant_value = np.array(constant_value)

    @rlgraph_api(name="run", returns=2, flatten_ops=True, split_ops=True)
    def _graph_fn_2_to_2(self, input1, input2):
        """
        Returns:
            Tuple:
                - in1 + 1.0
                - in1 + in2
        """
        return input1 + self.constant_value, input1 + input2


class NoFlattenNoSplitDummy(Component):
    """
    A dummy component with a 2-to-2 graph_fn mapping with flatten/split settings all set to False.
    """
    def __init__(self, scope="dummy-2-to-2-no-options", **kwargs):
        super(NoFlattenNoSplitDummy, self).__init__(scope=scope, **kwargs)

    @rlgraph_api(name="run", returns=2)
    def _graph_fn_2_to_2(self, input1, input2):
        """
        Returns:
            Tuple:
                - in2
                - in1
        """
        return input2, input1


class OnlyFlattenDummy(Component):
    """
    A dummy component with a 2-to-3 graph_fn mapping with only flatten_ops=True.
    """
    def __init__(self, scope="dummy-2-to-2-all-options", constant_value=1.0, **kwargs):
        super(OnlyFlattenDummy, self).__init__(scope=scope, **kwargs)

        self.constant_value = np.array(constant_value)

    @rlgraph_api(name="run", returns=3, flatten_ops=True)
    def _graph_fn_2_to_3(self, input1, input2):
        """
        NOTE: Both input1 and input2 are flattened dicts.

        Returns:
            Tuple:
                - in1 + in2
                - in1 - in2
                - in2

        """
        ret = FlattenedDataOp()
        ret2 = FlattenedDataOp()
        for key, value in input1.items():
            ret[key] = value + input2[""]
            ret2[key] = value - input2[""]
        return ret, ret2, input2


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
    def apply(self, input_dict):
        # Split the input dict into two streams.
        input_a, input_b = self.splitter.split(input_dict)

        # Get the two stack outputs.
        output_a = self.stack_a.apply(input_a)
        output_b = self.stack_b.apply(input_b)

        # Concat everything together, that's the output.
        concatenated_data = self.concat_layer.apply(output_a, output_b)

        return dict(output=concatenated_data)
