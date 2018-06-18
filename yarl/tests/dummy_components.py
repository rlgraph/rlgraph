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

import numpy as np

from yarl.utils.ops import FlattenedDataOp
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
        super(Dummy1to1, self).__init__(scope=scope, flatten_ops=False)
        self.constant_value = constant_value
        self.define_inputs("input")
        self.define_outputs("output")
        self.add_graph_fn("input", "output", self._graph_fn_1to1)

    def _graph_fn_1to1(self, input_):
        return input_ + self.constant_value


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
        super(Dummy1to2, self).__init__(scope=scope, flatten_ops=kwargs.pop("flatten_ops", False), **kwargs)
        self.define_inputs("input")
        self.define_outputs("output1", "output2")
        self.add_graph_fn("input", ["output1", "output2"], self._graph_fn_1to2)

    def _graph_fn_1to2(self, input_):
        return input_, input_ + 1.0


class Dummy2to1(Component):
    """
    A dummy component with one graph_fn mapping two inputs to one output.

    API:
    ins:
        input1
        input2
    outs:
        output
    """
    def __init__(self, scope="dummy-2-to-1"):
        super(Dummy2to1, self).__init__(scope=scope, flatten_ops=False)
        self.define_inputs("input1", "input2")
        self.define_outputs("output")
        self.add_graph_fn(["input1", "input2"], "output", self._graph_fn_2to1)

    def _graph_fn_2to1(self, input1, input2):
        return input1 + input2


class Dummy2to1Where1ConnectedWithConstant(Component):
    """
    A dummy component with one graph_fn mapping two inputs to one output and
    one of the inputs is already connected with a constant value.

    API:
    ins:
        input1
        input2 (not exposed b/c constant-value-blocked)
    outs:
        output
    """
    def __init__(self, scope="dummy-2-to-1"):
        super(Dummy2to1Where1ConnectedWithConstant, self).__init__(scope=scope, flatten_ops=False)
        self.define_inputs("input1")
        self.define_outputs("output")
        self.add_graph_fn(["input1", 3.0], "output", self._graph_fn_1_and_1blocked_to1)

    def _graph_fn_1_and_1blocked_to1(self, input1, input2):
        return input1 + input2


class Dummy0to1(Component):
    """
    A dummy component with one graph_fn without inputs and one output.

    API:
    outs:
        output
    """
    def __init__(self, scope="dummy-0-to-1"):
        super(Dummy0to1, self).__init__(scope=scope, flatten_ops=False)
        self.define_outputs("output")
        self.add_graph_fn(None, "output", self._graph_fn_0to1)
        self.var = None

    def create_variables(self, input_spaces, action_space):
        self.var = self.get_variable(initializer=8.0)

    def _graph_fn_0to1(self):
        return self.var


class NoFlattenNoSplitDummy(Component):
    """
    A dummy component with a 2-to-2 graph_fn mapping with flatten/split settings all set to False.

    API:
    ins:
        in1
        in2
    outs:
        out1: in2
        out2: in1
    """
    def __init__(self, scope="dummy-2-to-2-no-options", **kwargs):
        super(NoFlattenNoSplitDummy, self).__init__(scope=scope, **kwargs)

        self.define_inputs("in1", "in2")
        self.define_outputs("out1", "out2")
        self.add_graph_fn(["in1", "in2"], ["out1", "out2"],
                          self._graph_fn_2_to_2,
                          flatten_ops=False, split_ops=False)

    def _graph_fn_2_to_2(self, input1, input2):
        return input2, input1


class FlattenSplitDummy(Component):
    """
    A dummy component with a 2-to-2 graph_fn mapping with flatten/split settings set to True.

    API:
    ins:
        in1_fsu (flat, split, unflat)
        in2_fsu
    outs:
        out1_fsu: in1_fsu + 1.0
        out2_fsu: in1_fsu + in2_fsu
    """
    def __init__(self, scope="dummy-2-to-2-all-options", constant_value=1.0, **kwargs):
        super(FlattenSplitDummy, self).__init__(scope=scope, **kwargs)

        self.constant_value = np.array(constant_value)

        self.define_inputs("in1_fsu", "in2_fsu")
        self.define_outputs("out1_fsu", "out2_fsu")
        self.add_graph_fn(["in1_fsu", "in2_fsu"], ["out1_fsu", "out2_fsu"],
                          self._graph_fn_2_to_2,
                          flatten_ops=True, split_ops=True)

    def _graph_fn_2_to_2(self, input1, input2):
        return input1 + self.constant_value, input1 + input2


class OnlyFlattenDummy(Component):
    """
    A dummy component with a 2-to-3 graph_fn mapping with only flatten_ops=True.

    API:
    ins:
        in1_f (flat)
        in2_f
    outs:
        out1: dict(in1_f key: in1_f value + in2_f)
        out2: dict(in1_f key: in1_f value - in2_f)
        out3: in2_f
    """
    def __init__(self, scope="dummy-2-to-2-all-options", constant_value=1.0, **kwargs):
        super(OnlyFlattenDummy, self).__init__(scope=scope, **kwargs)

        self.constant_value = np.array(constant_value)

        self.define_inputs("in1", "in2")
        self.define_outputs("out1", "out2", "out3")
        self.add_graph_fn(["in1", "in2"], ["out1", "out2", "out3"],
                          self._graph_fn_2_to_3,
                          flatten_ops=True, split_ops=False)

    def _graph_fn_2_to_3(self, input1, input2):
        """
        NOTE: Both input1 and input2 are flattened dicts.
        """
        ret = FlattenedDataOp()
        ret2 = FlattenedDataOp()
        for k, v in input1.items():
            ret[k] = v + input2[""]
            ret2[k] = v - input2[""]
        return ret, ret2, input2

