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

if backend == 'tf':
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

        super(Synchronizable, self).__init__(*args, **kwargs)  # ignore other args/kwargs

        # Add a simple syncing API.
        # Outgoing data (to overwrite another Synchronizable Component's data).
        self.define_outputs("synch_out")
        # Add the sending-out-data operation.
        self.add_graph_fn(None, "own_vars", self._graph_fn_synch_out, flatten_ops=False)
        self.connect("own_vars", "synch_out")

        # The synch op, actually doing the overwriting from synch_in to our variables.
        # This is only possible if `self.writable` is True (default).
        if self.writable is True:
            # Socket for incoming data (the data that this Component will get overwritten with).
            self.define_inputs("synch_in")
            self.define_outputs("synch")
            # Add the synching operation.
            self.add_graph_fn(["synch_in", "own_vars"], "synch", self._graph_fn_synch, flatten_ops=False)

    def _graph_fn_synch(self, sync_in, own_vars):
        """
        Generates the op that syncs this approximators' trainable variable values from another
        FunctionApproximator object.

        Args:
            sync_in (DataOpDict): An dict of variables (coming from the "sync_out"-Socket) that need to be
                assigned to their counterparts in this Component. The keys in the dicts refer to
                the names of the variables and must match the names and variables of this component.

        Returns:
            op: The single op that executes the syncing.
        """
        # Loop through all incoming vars and our own and collect assign ops.
        syncs = list()
        # Sanity checking
        syncs_from, syncs_to = (sync_in.items(), own_vars.items())
        if len(syncs_from) != len(syncs_to):
            raise YARLError("ERROR: Number of Variables to synch must match! "
                            "We have {} syncs_from and {} syncs_to.".format(len(syncs_from), len(syncs_to)))
        for (key_from, var_from), (key_to, var_to) in zip(syncs_from, syncs_to):
            # Sanity checking. TODO: Check the names' ends? Without the global scope?
            #if key_from != key_to:
            #    raise YARLError("ERROR: Variable names for synching must match in order and name! "
            #                    "Mismatch at from={} and to={}.".format(key_from, key_to))
            if get_shape(var_from) != get_shape(var_to):
                raise YARLError("ERROR: Variable shapes for synching must match! "
                                "Shape mismatch between from={} ({}) and to={} ({}).".
                                format(key_from, get_shape(var_from), key_to, get_shape(var_to)))
            syncs.append(self.assign_variable(var_to, var_from))

        # Bundle everything into one "sync"-op.
        if backend == "tf":
            with tf.control_dependencies(syncs):
                return tf.no_op()

    def _graph_fn_synch_out(self):
        """
        Outputs all of this Component's variables that match our collection(s) specifier in a tuple-op.

        Returns:
            DataOpDict: Dict with keys=variable names and values=variable (SingleDataOp).
                Only Variables that match the given collection (see c'tor) are part of the dict.
        """
        # Must use custom_scope_separator here b/c YARL doesn't allow Dict with '/'-chars in the keys.
        # '/' could collide with a FlattenedDataOp's keys and mess up the un-flatten process.
        variables_dict = self.get_variables(collections=self.collections, custom_scope_separator="-")
        assert len(variables_dict) > 0, \
            "ERROR: Synchronizable Component '{}' does not have any variables!".format(self.name)
        return DataOpDict(variables_dict)
