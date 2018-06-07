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

from yarl.components.layers.preprocessing import PreprocessLayer


class Scale(PreprocessLayer):
    """
    Simply scales an input by a constant scaling-factor.
    """

    def __init__(self, scaling_factor, scope="scale", **kwargs):
        """
        Args:
            scaling_factor (float): The factor to scale with.
        """
        super(Scale, self).__init__(scope=scope, **kwargs)
        self.scaling_factor = scaling_factor

    def _graph_fn_apply(self, tensor):
        """
        Simply multiplies the input with our scaling factor.

        Args:
            tensor (tensor): The input to be scaled.

        Returns:
            op: The op to scale the input.
        """
        return tensor * self.scaling_factor
