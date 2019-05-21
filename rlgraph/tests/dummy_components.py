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

import numpy as np

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.components.optimizers.local_optimizers import GradientDescentOptimizer
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import FlattenedDataOp

if get_backend() == "tf":
    import tensorflow as tf


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
    def __init__(self, scope="dummy-2-to-2-w-default-value", constant_value=1.0):
        super(Dummy2To2WithDefaultValue, self).__init__(scope=scope)
        self.constant_value = constant_value

    @rlgraph_api(name="run", returns=2)
    def _graph_fn_2to2(self, input1, input2=None):
        return input1 + self.constant_value, (input2 or 5.0) * self.constant_value


class Dummy3To1WithDefaultValues(Component):
    """
    Tests forwarding a `None` arg in the middle of the API-call signature correctly to a graph_fn.
    """
    def __init__(self, scope="dummy-3-to-1-w-default-value"):
        super(Dummy3To1WithDefaultValues, self).__init__(scope=scope)

    @rlgraph_api
    def run(self, input1, input2=1.0, input3=None):
        return self._graph_fn_run(input1, input2, input3)

    @rlgraph_api
    def run2(self, input1, input3=None, input4=1.0):
        return self._graph_fn_run(input1, input4, input3)

    @graph_fn
    def _graph_fn_run(self, input1, input2, input3=None):
        if input3 is None:
            return input1 + input2
        return input3


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
        self.var = self.get_variable(name="var", initializer=self.var_value)

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


class Dummy2NestedGraphFnCalls(Component):
    """
    API:
        run(input_): Result of (input_ - `self.constant_value` * 2) + `self.constant_value`
    """
    def __init__(self, scope="dummy-2nested-graph_fn-calls", constant_value=1.0):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(Dummy2NestedGraphFnCalls, self).__init__(scope=scope)
        self.constant_value = constant_value

    @rlgraph_api
    def run(self, input_):
        result = self._graph_fn_outer(input_)
        return result

    @graph_fn
    def _graph_fn_outer(self, input_):
        intermediate_result = self._graph_fn_inner(input_)
        return intermediate_result + self.constant_value

    @graph_fn
    def _graph_fn_inner(self, input_):
        return input_ - self.constant_value * 2


class DummyThatDefinesCustomAPIMethod(Component):
    def __init__(self, scope="dummy-that-defines-custom-api-method", **kwargs):
        super(DummyThatDefinesCustomAPIMethod, self).__init__(scope=scope, **kwargs)

        self.define_api_methods()

    def define_api_methods(self):
        @rlgraph_api(component=self)
        def some_custom_api_method(self, input_):
            return self._graph_fn_1to1(input_)

    @graph_fn
    def _graph_fn_1to1(self, input_):
        return input_


class DummyThatDefinesCustomGraphFn(Component):
    def __init__(self, scope="dummy-that-defines-custom-graph-fn", **kwargs):
        super(DummyThatDefinesCustomGraphFn, self).__init__(scope=scope, **kwargs)

        self.define_graph_fns()

    @rlgraph_api
    def run(self, input_):
        return self._graph_fn_some_custom_graph_fn(input_)

    def define_graph_fns(self):
        @graph_fn(component=self)
        def _graph_fn_some_custom_graph_fn(self, input_):
            return input_


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
    def __init__(self, variable_value=3.0, scope="simple-dummy-with-var", **kwargs):
        """
        Args:
            variable_value (float): The initial value of our variable.
        """
        super(SimpleDummyWithVar, self).__init__(scope=scope, **kwargs)
        self.variable = None
        self.variable_value = variable_value

    def create_variables(self, input_spaces, action_space=None):
        self.variable = self.get_variable(name="variable", initializer=self.variable_value)

    @rlgraph_api
    def run(self, input_):
        # Explicit definition of an API-method using one of our graph_fn.
        result = self._graph_fn_1(input_)
        return result

    @graph_fn(returns=1)
    def _graph_fn_1(self, input_):
        return input_ + self.variable


class DummyWithOptimizer(SimpleDummyWithVar):
    def __init__(self, variable_value=3.0, learning_rate=0.1, scope="dummy-with-optimizer", **kwargs):
        super(DummyWithOptimizer, self).__init__(variable_value=variable_value, scope=scope, **kwargs)

        assert isinstance(learning_rate, float), "ERROR: Only float (constant) values allowed in `learning_rate`!"

        self.optimizer = GradientDescentOptimizer(learning_rate=learning_rate)
        self.add_components(self.optimizer)

    @rlgraph_api
    def calc_grads(self, time_percentage):
        loss = self._graph_fn_simple_square_loss()
        return self.optimizer.calculate_gradients(self.variables(), loss, time_percentage)

    @rlgraph_api
    def step(self, time_percentage):
        loss = self._graph_fn_simple_square_loss()
        return self.optimizer.step(self.variables(), loss, loss, time_percentage)

    @graph_fn
    def _graph_fn_simple_square_loss(self):
        loss = None
        if get_backend() == "tf":
            loss = tf.square(x=tf.log(self.variable))
        return loss


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


class DummyCallingOneAPIFromWithinOther(Component):
    """
    A Component with 2 API-methods (A and B) that calls B from within A.
    Hence, if a parent component does not call B directly, this component will never be input-complete.
    """
    def __init__(self, scope="dummy-calling-one-api-from-within-other", **kwargs):
        super(DummyCallingOneAPIFromWithinOther, self).__init__(scope=scope, **kwargs)
        self.my_var = None

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["inner_input"]
        self.my_var = self.get_variable(
            name="memory", trainable=False, from_space=in_space, flatten=False, initializer=0
        )

    @rlgraph_api
    def run(self, outer_input):
        # Call another graph_fn here to force space coming out of it (space of tmp_out) to be unknown.
        tmp_out = self._graph_fn_2(outer_input)
        # At this point during the build, we have no idea what space tmp_out has and cannot complete this Component
        # unless the parent itself calls `run_inner`.
        return self.run_inner(tmp_out)

    @rlgraph_api(name="run_inner")
    def _graph_fn_1(self, inner_input):
        return inner_input * self.my_var

    @graph_fn
    def _graph_fn_2(self, inner_input_2):
        return inner_input_2 + 1.0
