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


class Dummy1To1(Component):
    """
    API:
        run(input_): Result of input_ + `self.constant_value`
    """
    def __init__(self, scope="dummy-1-to-1", constant_value=1.0):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(Dummy1To1, self).__init__(scope=scope)
        self.constant_value = constant_value

        # Automatically generate an API-method around our graph_fn.
        self.define_api_method("run", self._graph_fn_1to1)

    def _graph_fn_1to1(self, input_):
        return input_ + self.constant_value


class Dummy2To1(Component):
    """
    API:
        run(input1, input2): Result of input1 + input2.
    """
    def __init__(self, scope="dummy-2-to-1"):
        super(Dummy2To1, self).__init__(scope=scope)

        # Automatically generate an API-method around our graph_fn.
        self.define_api_method("run", self._graph_fn_2to1)

    def _graph_fn_2to1(self, input1, input2):
        return input1 + input2


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

    def run(self, input_):
        # Explicit definition of an API-method using both our graph_fn.
        result_1to1 = self.call(self._graph_fn_1to1, input_)
        result_1to1_neg = self.call(self._graph_fn_1to1_neg, result_1to1)
        return result_1to1_neg

    def _graph_fn_1to1(self, input_):
        return input_ + self.constant_value

    def _graph_fn_1to1_neg(self, input_):
        return input_ - self.constant_value * 2


class DummyInputComplete(Component):
    """
    A dummy component with a couple of sub-components that have their own API methods.

    API:
        run_plus(input_): input_ + `self.constant_variable`
        run_minus(input_): input_ - `self.constant_value`
    """
    def __init__(self, scope="dummy-for-testing-input-completeness", constant_value=2.0):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(DummyInputComplete, self).__init__(scope=scope)
        self.constant_value = constant_value
        self.constant_variable = None

        # Automatically generate an API-method around one of our graph_fn.
        self.define_api_method("run_minus", self._graph_fn_2)

    def create_variables(self, input_spaces, action_space):
        self.constant_variable = self.create_variable(name="constant-variable", initializer=2.0)

    def run_plus(self, input_):
        # Explicit definition of an API-method using one of our graph_fn.
        result = self.call(self._graph_fn_1, input_)
        return result

    def _graph_fn_1(self, input_):
        return input_ + self.constant_value

    def _graph_fn_2(self, input_):
        return input_ - self.constant_variable


class DummyWithSubComponents(Component):
    """
    A dummy component with a couple of sub-components that have their own API methods.

    API:
        run(input_): Result of input_ + sub_comp.run(input_) + `self.constant_value`
    """
    def __init__(self, scope="dummy-with-sub-components", constant_value=1.0):
        """
        Args:
            constant_value (float): A constant to add to input in our graph_fn.
        """
        super(DummyWithSubComponents, self).__init__(scope=scope)
        self.constant_value = constant_value

        # Create a sub-Component and add it.
        self.sub_comp = DummyInputComplete()
        self.add_components(self.sub_comp)

    def run1(self, input_):
        # Explicit definition of an API-method using one of our graph_fn and one of
        # our child API-methods.
        result = self.call(self.sub_comp.run_plus, input_)
        result2 = self.call(self._graph_fn_apply, result)
        return result, result2

    def run2(self, input_):
        # Explicit definition of an API-method using one of our graph_fn and both of
        # our child API-methods.
        result1 = self.call(self.sub_comp.run_plus, input_)
        result2 = self.call(self.sub_comp.run_minus, result1)
        result3 = self.call(self._graph_fn_apply, result2)
        return result3

    def _graph_fn_apply(self, input_):
        return input_ + self.constant_value

