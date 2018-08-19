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

from rlgraph import get_backend
from rlgraph.utils.rlgraph_error import RLGraphError
from rlgraph.utils.ops import DataOpDict
from rlgraph.utils.util import get_shape
from rlgraph.components import Component

if get_backend() == "tf":
    import tensorflow as tf


class Synchronizable(Component):
    """
    The Synchronizable Component adds a simple synchronization API to arbitrary Components to which this
    Synchronizable is added (and connected via `connections=CONNECT_ALL`).
    This is useful for constructions like a target network in DQN or for distributed setups where e.g.
    local policies need to be sync'd from a global model from time to time.
    """
    def __init__(self, *args, **kwargs):
        """
        Keyword Args:
            collections (set): A set of specifiers (currently only tf), that determine which Variables
                of the parent Component to synchronize.
        """
        self.collections = kwargs.pop("collections", None)

        super(Synchronizable, self).__init__(*args, scope=kwargs.pop("scope", "synchronizable"), **kwargs)

        # Add the syncing operation.
        self.define_api_method(name="sync", func=self._graph_fn_sync)

    def check_input_completeness(self):
        # Overwrites this method as any Synchronizable should only be input-complete once the parent
        # component is variable-complete (not counting this component!). Also, the number of variables in
        # the parent component must match the number of variables that the sync-in component sends in its API-call
        # to our `sync` API-method.
        if self.parent_component.input_complete is True:
            parents_vars = self.parent_component.get_variables(collections=self.collections, custom_scope_separator="-")
            # We don't have any value yet from the sync-in Component OR
            # some of the parent's variables (or its other children) have not been created yet.
            if self.api_method_inputs["values_"] is None or len(self.api_method_inputs["values_"]) != len(parents_vars):
                return False
            else:
                # Check our own input-completeness (have to wait for the incoming values which to sync to).
                return super(Synchronizable, self).check_input_completeness()
        # If parent component not input complete, we cannot be either.
        return False

    def _graph_fn_sync(self, values_):
        """
        Generates the op that syncs this Synchronizable's parent's variable values from another Synchronizable
        Component.

        Args:
            values_ (DataOpDict): The dict of variable values (coming from the "_variables"-Socket of any other
                Component) that need to be assigned to this Component's parent's variables.
                The keys in the dict refer to the names of our parent's variables and must match their names.

        Returns:
            DataOp: The op that executes the syncing.
        """
        # Loop through all incoming vars and our own and collect assign ops.
        syncs = list()
        parents_vars = self.parent_component.get_variables(collections=self.collections, custom_scope_separator="-")

        # Sanity checking
        syncs_from, syncs_to = (sorted(values_.items()), sorted(parents_vars.items()))
        if len(syncs_from) != len(syncs_to):
            raise RLGraphError("ERROR: Number of Variables to sync must match! "
                               "We have {} syncs_from and {} syncs_to.".format(len(syncs_from), len(syncs_to)))
        for (key_from, var_from), (key_to, var_to) in zip(syncs_from, syncs_to):
            # Sanity checking. TODO: Check the names' ends? Without the global scope?
            #if key_from != key_to:
            #    raise RLGraphError("ERROR: Variable names for syncing must match in order and name! "
            #                    "Mismatch at from={} and to={}.".format(key_from, key_to))
            if get_shape(var_from) != get_shape(var_to):
                raise RLGraphError("ERROR: Variable shapes for syncing must match! "
                                   "Shape mismatch between from={} ({}) and to={} ({}).".
                                   format(key_from, get_shape(var_from), key_to, get_shape(var_to)))
            syncs.append(self.assign_variable(var_to, var_from))

        # Bundle everything into one "sync"-op.
        if get_backend() == "tf":
            with tf.control_dependencies(syncs):
                return tf.no_op()
