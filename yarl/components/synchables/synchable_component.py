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

from yarl import YARLError
from yarl.utils.util import get_shape
from yarl.components.layers import StackComponent


class SynchableComponent(StackComponent):
    """
    The SynchableComponent adds a simple synchronization API to any Component that exposes this sub-component's
    interface. This is useful for constructions like a target network in DQN or
    for distributed setups where e.g. local policies need to be sync'd from a global model from time to time.
    """
    def __init__(self, *args, **kwargs):
        super(SynchableComponent, self).__init__(*args, **kwargs)

    def _computation_sync(self, sync_from):
        """
        Generates the op that syncs this approximators' trainable variable values from another
        FunctionApproximator object.

        Args:
            sync_from (OrderedDict): An OrderedDict of variables (coming from the "sync-from"-Socket) that need to be
                assigned to their counterparts in this Component. The keys and order in the OrderedDict refer to
                the names of the trainable variables and must match the names and order of the trainable variables
                in this component.

        Returns:
            op: The single op that executes the syncing.
        """
        # Loop through all incoming vars and our own and collect assign ops.
        syncs = list()
        for (key_from, var_from), (key_to, var_to) in zip(sync_from, self.get_trainable_variables()):
            # Sanity checking
            if key_from != key_to:
                raise YARLError("ERROR: Variable names for synching must match in order and name! "
                                "Mismatch at from={} and to={}.".format(key_from, key_to))
            elif get_shape(var_from) != get_shape(var_to):
                raise YARLError("ERROR: Variable shapes for synching must match! "
                                "Shape mismatch between from={} ({}) and to={} ({}).".
                                format(key_from, get_shape(var_from), key_to, get_shape(var_to)))
            syncs.append(self.assign_variable(var_to, var_from))

        # Bundle everything into one "sync"-op.
        with tf.control_dependencies(syncs):
            return tf.no_op()

