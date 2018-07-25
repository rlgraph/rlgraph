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

from rlgraph.components.layers.preprocessing import PreprocessLayer


class Multiply(PreprocessLayer):
    """
    Scales an input by a constant scaling-factor.
    """
    def __init__(self, factor, scope="multiply", **kwargs):
        """
        Args:
            scaling_factor (float): The factor to scale with.
        """
        super(Multiply, self).__init__(scope=scope, **kwargs)
        self.factor = factor

    def _graph_fn_apply(self, tensor):
        """
        Multiplies the input with our factor.

        Args:
            tensor (tensor): The input to be scaled.

        Returns:
            op: The op to scale the input.
        """
        return tensor * self.factor


class Divide(PreprocessLayer):
    """
    Divides an input by a constant value.
    """
    def __init__(self, divisor, scope="divide", **kwargs):
        """
        Args:
            scaling_factor (float): The factor to scale with.
        """
        super(Divide, self).__init__(scope=scope, **kwargs)
        self.divisor = divisor

    def _graph_fn_apply(self, tensor):
        """
        Divides the input by with our divisor.

        Args:
            tensor (tensor): The input to be divided.

        Returns:
            DataOp: The op to divide the input.
        """
        return tensor / self.divisor
