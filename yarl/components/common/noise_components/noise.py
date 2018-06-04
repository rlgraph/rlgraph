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

import tensorflow as tf

from yarl import backend
from yarl.components import Component


class NoiseComponent(Component):
    """
    A base class Component that takes an action input and outputs some noise value.

    API:
    ins:
        action (float): The action value input.
    outs:
        noise (float): The noise value to be added to the action.
    """
    def __init__(self, action_space, scope="noise", **kwargs):
        """

        Args:
            action_space: The action space.
        """
        super(NoiseComponent, self).__init__(scope=scope, **kwargs)

        self.action_space = action_space

        # Our interface.
        self.define_outputs("noise")
        self.add_graph_fn(None, "noise", self._graph_fn_value)

    def noise(self):
        """
        The function that returns the DataOp to actually compute the noise.

        Returns:
            DataOp: The noise value.
        """
        return tf.constant(0.0)

    def _graph_fn_value(self):
        """
        Args:
            action (DataOp): The float-type action value.

        Returns:
            DataOp: The noise value.
        """
        return self.noise()
