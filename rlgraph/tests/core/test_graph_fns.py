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

import logging
import unittest

from rlgraph.tests import ComponentTest
from rlgraph.utils import root_logger
import rlgraph.spaces as spaces
from rlgraph.tests.dummy_components import *


class TestGraphFns(unittest.TestCase):
    """
    Tests for different ways to send DataOps through GraphFunctions.
    Tests flattening, splitting, etc.. operations.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_2_containers_flattening_splitting(self):
        """
        Adds a single component with 2-to-2 graph_fn to the core and passes two containers through it
        with flatten/split options enabled.
        """
        input1_space = spaces.Dict(a=float, b=spaces.FloatBox(shape=(1, 2)))
        input2_space = spaces.Dict(a=float, b=float)

        component = FlattenSplitDummy()
        test = ComponentTest(
            component=component,
            input_spaces=dict(input1=input1_space, input2=input2_space)
        )

        # Options: fsu=flat/split/un-flat.
        in1_fsu = dict(a=np.array(0.234), b=np.array([[0.0, 3.0]]))
        in2_fsu = dict(a=np.array(5.0), b=np.array(5.5))
        # Result of sending 'a' keys through graph_fn: (in1[a]+1.0=1.234, in1[a]+in2[a]=5.234)
        # Result of sending 'b' keys through graph_fn: (in1[b]+1.0=[[1, 4]], in1[b]+in2[b]=[[5.5, 8.5]])
        out1_fsu = dict(a=1.234, b=np.array([[1.0, 4.0]]))
        out2_fsu = dict(a=np.array(5.234, dtype=np.float32), b=np.array([[5.5, 8.5]]))
        test.test(("run", [in1_fsu, in2_fsu]), expected_outputs=[out1_fsu, out2_fsu])

    def test_1_containers_1_float_flattening_splitting(self):
        """
        Adds a single component with 2-to-2 graph_fn to the core and passes one container and one float through it
        with flatten/split options all disabled.
        """
        input1_space = spaces.Dict(a=float, b=spaces.FloatBox(shape=(1, 2)))
        input2_space = spaces.FloatBox(shape=(1,1))

        component = FlattenSplitDummy()
        test = ComponentTest(component=component, input_spaces=dict(input1=input1_space, input2=input2_space))

        # Options: fsu=flat/split/un-flat.
        in1_fsu = dict(a=np.array(0.234), b=np.array([[0.0, 3.0]]))
        in2_fsu = np.array([[2.0]])
        # Result of sending 'a' keys through graph_fn: (in1[a]+1.0=1.234, in1[a]+in2=2.234)
        # Result of sending 'b' keys through graph_fn: (in1[b]+1.0=[[1, 4]], in1[b]+in2=[[2.0, 5.0]])
        out1_fsu = dict(a=1.234, b=np.array([[1.0, 4.0]]))
        out2_fsu = dict(a=np.array([[2.234]], dtype=np.float32), b=np.array([[2.0, 5.0]]))
        test.test(("run", [in1_fsu, in2_fsu]), expected_outputs=[out1_fsu, out2_fsu])

    def test_2_containers_no_options(self):
        """
        Adds a single component with 2-to-2 graph_fn to the core and passes one container and one float through it
        with no flatten/split options enabled.
        """
        input1_space = spaces.Dict(a=int, b=bool)
        input2_space = spaces.Dict(c=bool, d=int)

        component = NoFlattenNoSplitDummy()
        test = ComponentTest(component=component, input_spaces=dict(input1=input1_space, input2=input2_space))

        # Options: fsu=flat/split.
        in1 = dict(a=5, b=True)
        in2 = dict(c=False, d=3)
        # Expect reversal (see graph_fn)
        out1 = in2
        out2 = in1
        test.test(("run", [in1, in2]), expected_outputs=[out1, out2])

    def test_1_container_1_float_only_flatten(self):
        """
        Adds a single component with 2-to-3 graph_fn to the core and passes one container and one float through it
        with only the flatten option enabled.
        """
        input1_space = spaces.Dict(a=float, b=float, c=spaces.Tuple(float))
        input2_space = spaces.FloatBox(shape=(1,))

        component = OnlyFlattenDummy(constant_value=5.0)
        test = ComponentTest(component=component, input_spaces=dict(input1=input1_space, input2=input2_space))

        # Options: only flatten_ops=True.
        in1 = dict(a=5.4, b=3.4, c=tuple([3.2]))
        in2 = np.array([1.2])
        # out1: dict(in1_f key: in1_f value + in2_f[""])
        # out2: in2_f
        # out3: self.constant_value
        out1 = dict(a=in1["a"] + in2, b=in1["b"] + in2, c=tuple([in1["c"][0] + in2]))
        out2 = dict(a=in1["a"] - in2, b=in1["b"] - in2, c=tuple([in1["c"][0] - in2]))
        out3 = in2
        test.test(("run", [in1, in2]), expected_outputs=[out1, out2, out3], decimals=5)

    def test_calling_graph_fn_from_inside_another_graph_fn(self):
        """
        One graph_fn gets called from within another. Must return actual ops from inner one so that the outer one
        can handle it.
        """
        input_space = spaces.FloatBox(shape=(2,))
        component = Dummy2NestedGraphFnCalls()
        test = ComponentTest(component=component, input_spaces=dict(
            input_=input_space
        ))

        input_ = input_space.sample()
        expected = input_ - 1.0
        test.test(("run", input_), expected_outputs=expected, decimals=5)

    def test_component_that_defines_custom_graph_fns(self):
        a = DummyThatDefinesCustomGraphFn()

        test = ComponentTest(component=a, input_spaces=dict(input_=float))

        test.test(("run", 3.4567), expected_outputs=3.4567, decimals=3)

    def test_calling_graph_fn_with_default_args_in_middle(self):
        a = Dummy3To1WithDefaultValues()
        test = ComponentTest(component=a, input_spaces=dict(input1=float))
        # Will put default float into input2.
        test.test(("run", 1.0), expected_outputs=2.0, decimals=3)

        b = Dummy3To1WithDefaultValues()
        test = ComponentTest(component=b, input_spaces=dict(input1=int, input3=int))
        test.test(("run", [5, 6]), expected_outputs=6, decimals=0)

        c = Dummy3To1WithDefaultValues()
        test = ComponentTest(component=c, input_spaces=dict(input1=float, input4=float))  # TODO: if we leave out input4, should create a default-placeholder with default value = 1.0 (see api-method)
        test.test(("run2", [1.0, 2.0]), expected_outputs=3.0, decimals=3)
