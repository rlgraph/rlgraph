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

from collections import OrderedDict

from yarl import YARLError, backend
from yarl.utils.ops import DataOpTuple
from yarl.utils.util import get_shape
from yarl.components import Component


class Synchronizable(Component):
    """
    The Synchronizable adds a simple synchronization API as a mix-in class to arbitrary Components.
    This is useful for constructions like a target network in DQN or
    for distributed setups where e.g. local policies need to be sync'd from a global model from time to time.
    """
    def __init__(self, collections=None, writable=True, *args, **kwargs):
        """
        Args:
            collections (set): A set of specifiers (currently only tf), that determine which Variables to synchronize.
            writable (bool): Whether this Component is synchronizable/writable by another Synchronizable.
                If False, this Component can only push out its own values (and thus overwrite other Synchronizables).
                Default: True.
        """
        super(Synchronizable, self).__init__(*args, **kwargs)

        self.collections = collections
        self.writable = writable

        # Add a simple syncing API.
        # Outgoing data (to overwrite another Synchronizable Component's data).
        self.define_outputs("synch_out")
        # Add the sending out data operation.
        self.add_graph_fn(None, "synch_out", self._graph_fn_synch_out)

        # The synch op, actually doing the overwriting from synch_in to our variables.
        # This is only possible if `self.writable` is True (default).
        if self.writable is True:
            # Socket for incoming data (the data that this Component will get overwritten with).
            self.define_inputs("synch_in")
            self.define_outputs("synch")
            # Add the synching operation.
            self.add_graph_fn("synch_in", "synch", self._graph_fn_synch)

    def _graph_fn_synch(self, sync_in):
        """
        Generates the op that syncs this approximators' trainable variable values from another
        FunctionApproximator object.

        Args:
            sync_in (OrderedDict): An OrderedDict of variables (coming from the "sync-from"-Socket) that need to be
                assigned to their counterparts in this Component. The keys and order in the OrderedDict refer to
                the names of the trainable variables and must match the names and order of the trainable variables
                in this component.

        Returns:
            op: The single op that executes the syncing.
        """
        # Loop through all incoming vars and our own and collect assign ops.
        syncs = list()
        for (key_from, var_from), (key_to, var_to) in zip(sync_in, self.get_variables(collections=self.collections)):
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
        if backend == "tf":
            import tensorflow as tf
            with tf.control_dependencies(syncs):
                return tf.no_op()

    def _graph_fn_synch_out(self):
        """
        Outputs all of this Component's variables that match our collection(s) specifier in a tuple-op.

        Returns:
            DataOpTuple: All our Variables that match the given collection (see c'tor).
        """
        variables_dict = self.get_variables(collections=self.collections)
        ordered_dict = OrderedDict(sorted(variables_dict.items()))
        # Need to use DataOpTuple to enforce everything being returned in one out-Socket.
        ret = DataOpTuple(list(ordered_dict.values()))
        return ret
