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

import copy
import numpy as np

from yarl.utils.ops import FlattenedDataOp, DataOpRecord
from yarl.utils.util import force_list
from yarl.components import Component


class Dummy1to1(Component):
    """
    A dummy component with one graph_fn mapping one input to one output.

    API:
    ins:
        input
    outs:
        output
    """
    def __init__(self, scope="dummy-1-to-1", constant_value=1.0):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(Dummy1to1, self).__init__(scope=scope)
        self.constant_value = constant_value
        self.define_inputs("input")
        self.define_outputs("output")
        self.add_graph_fn("input", "output", self._graph_fn_1to1)

    def _graph_fn_1to1(self, input_):
        return input_ + self.constant_value


class Dummy1to1NEW(Component):
    """
    A dummy component with one API method (run) mapping one input to one output.

    API:
        run(input_): Result of input_ + `self.constant_value` - 2x`self.constant_value`
        ... <- add more API method here
    """
    def __init__(self, scope="dummy-1-to-1", constant_value=1.0):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(Dummy1to1NEW, self).__init__(scope=scope)
        self.constant_value = constant_value

    def run(self, input_):
        result_1to1 = self.call(from_=input_, graph_fn=self._graph_fn_1to1)
        result_1to1_neg = self.call(from_=result_1to1, graph_fn=self._graph_fn_1to1_neg)
        return result_1to1_neg

    def _graph_fn_1to1(self, input_):
        return input_ + self.constant_value

    def _graph_fn_1to1_neg(self, input_):
        return input_ - self.constant_value * 2


class Dummy1to2(Component):
    """
    A dummy component with one graph_fn mapping one input to two outputs.

    API:
    ins:
        input
    outs:
        output1
        output2
    """
    def __init__(self, scope="dummy-1-to-2", **kwargs):
        super(Dummy1to2, self).__init__(scope=scope, **kwargs)
        self.define_inputs("input")
        self.define_outputs("output1", "output2")
        self.add_graph_fn("input", ["output1", "output2"], self._graph_fn_1to2)

    def _graph_fn_1to2(self, input_):
        return input_, input_ + 1.0


class Dummy2to1(Component):
    """
    A dummy component with one graph_fn mapping two api_methods to one output.

    API:
    ins:
        input1
        input2
    outs:
        output
    """
    def __init__(self, scope="dummy-2-to-1"):
        super(Dummy2to1, self).__init__(scope=scope)
        self.define_inputs("input1", "input2")
        self.define_outputs("output")
        self.add_graph_fn(["input1", "input2"], "output", self._graph_fn_2to1)

    def _graph_fn_2to1(self, input1, input2):
        return input1 + input2


class Dummy2to1Where1ConnectedWithConstant(Component):
    """
    A dummy component with one graph_fn mapping two api_methods to one output and
    one of the api_methods is already connected with a constant value.

    API:
    ins:
        input1
        input2 (not exposed b/c constant-value-blocked)
    outs:
        output
    """
    def __init__(self, scope="dummy-2-to-1"):
        super(Dummy2to1Where1ConnectedWithConstant, self).__init__(scope=scope)
        self.define_inputs("input1")
        self.define_outputs("output")
        self.add_graph_fn(["input1", 3.0], "output", self._graph_fn_1_and_1blocked_to1)

    def _graph_fn_1_and_1blocked_to1(self, input1, input2):
        return input1 + input2


class Dummy0to1(Component):
    """
    A dummy component with one graph_fn without api_methods and one output.

    API:
    outs:
        output
    """
    def __init__(self, scope="dummy-0-to-1"):
        super(Dummy0to1, self).__init__(scope=scope)
        self.define_outputs("output")
        self.add_graph_fn(None, "output", self._graph_fn_0to1)
        self.var = None

    def create_variables(self, input_spaces, action_space):
        self.var = self.get_variable(initializer=8.0)

    def _graph_fn_0to1(self):
        return self.var


