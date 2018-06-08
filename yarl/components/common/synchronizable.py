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

from yarl import YARLError, backend
from yarl.utils.ops import DataOpDict
from yarl.utils.util import get_shape
from yarl.components import Component

if backend == "tf":
    import tensorflow as tf


class Synchronizable(Component):
    """
    The Synchronizable adds a simple synchronization API as a mix-in class to arbitrary Components.
    This is useful for constructions like a target network in DQN or
    for distributed setups where e.g. local policies need to be sync'd from a global model from time to time.
    """
    def __init__(self, *args, **kwargs):
        """
        Keyword Args:
            collections (set): A set of specifiers (currently only tf), that determine which Variables to synchronize.
            writable (bool): Whether this Component is synchronizable/writable by another Synchronizable.
                If False, this Component can only push out its own values (and thus overwrite other Synchronizables).
                Default: True.
        """
        self.collections = kwargs.pop("collections", None)
        self.writable = kwargs.pop("writable", True)

        super(Synchronizable, self).__init__(*args, **kwargs)

        # Add a simple syncing API.
        # Outgoing data (to overwrite another Synchronizable Component's data).
        self.define_outputs("sync_out")
        # Add the sending-out-data operation.
        self.add_graph_fn(None, "own_vars", self._graph_fn_sync_out, flatten_ops=False)
        self.connect("own_vars", "sync_out")

        # The sync op, actually doing the overwriting from sync_in to our variables.
        # This is only possible if `self.writable` is True (default).
        if self.writable is True:
            # Socket for incoming data (the data that this Component will get overwritten with).
            self.define_inputs("sync_in")
            self.define_outputs("sync")
            # Add the syncing operation.
            self.add_graph_fn(["sync_in", "own_vars"], "sync", self._graph_fn_sync, flatten_ops=False)

    def _graph_fn_sync(self, sync_in, own_vars):
        """
        Generates the op that syncs this Synchronizable's variable values from another Synchronizable Component.

        Args:
            sync_in (DataOpDict): The dict of variable values (coming from the "sync_out"-Socket of the other
                Synchronizable) that need to be assigned to this Component's variables.
                The keys in the dict refer to the names of our own variables and must match their names.
            own_vars (DataOpDict): The dict of our own variables that need to be overwritten by `sync_in`.

        Returns:
            DataOp: The op that executes the syncing.
        """
        # Loop through all incoming vars and our own and collect assign ops.
        syncs = list()
        # Sanity checking
        syncs_from, syncs_to = (sync_in.items(), own_vars.items())
        if len(syncs_from) != len(syncs_to):
            raise YARLError("ERROR: Number of Variables to sync must match! "
                            "We have {} syncs_from and {} syncs_to.".format(len(syncs_from), len(syncs_to)))
        for (key_from, var_from), (key_to, var_to) in zip(syncs_from, syncs_to):
            # Sanity checking. TODO: Check the names' ends? Without the global scope?
            #if key_from != key_to:
            #    raise YARLError("ERROR: Variable names for syncing must match in order and name! "
            #                    "Mismatch at from={} and to={}.".format(key_from, key_to))
            if get_shape(var_from) != get_shape(var_to):
                raise YARLError("ERROR: Variable shapes for syncing must match! "
                                "Shape mismatch between from={} ({}) and to={} ({}).".
                                format(key_from, get_shape(var_from), key_to, get_shape(var_to)))
            syncs.append(self.assign_variable(var_to, var_from))

        # Bundle everything into one "sync"-op.
        if backend == "tf":
            with tf.control_dependencies(syncs):
                return tf.no_op()

    def _graph_fn_sync_out(self):
        """
        Outputs all of this Component's variables that match our collection(s) specifier in a DataOpDict.

        Returns:
            DataOpDict: Dict with keys=variable names and values=variable (SingleDataOp).
                Only Variables that match `self.collections` (see c'tor) are part of the dict.
        """
        # Must use custom_scope_separator here b/c YARL doesn't allow Dict with '/'-chars in the keys.
        # '/' could collide with a FlattenedDataOp's keys and mess up the un-flatten process.
        variables_dict = self.get_variables(collections=self.collections, custom_scope_separator="-")
        assert len(variables_dict) > 0, \
            "ERROR: Synchronizable Component '{}' does not have any variables!".format(self.name)
        return DataOpDict(variables_dict)
