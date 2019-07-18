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

from __future__ import absolute_import, division, print_function

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import DataOpDict
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.utils.util import get_shape

if get_backend() == "tf":
    import tensorflow as tf


class Synchronizable(Component):
    """
    The Synchronizable Component adds a simple synchronization API to arbitrary Components to which this
    Synchronizable is added (and exposed via `expose_apis='sync'`).
    This is useful for constructions like a target network in DQN or for distributed setups where e.g.
    local policies need to be sync'd from a global model from time to time.
    """
    def __init__(self, sync_tau=1.0, sync_every_n_times=1, collections=None, scope="synchronizable", **kwargs):
        """
        Args:
            sync_tau (float): Factor for soft synching:
                [new values] = tau * [sync source] + (1.0 - tau) * [old values]
                Default: 1.0 (complete synching).

            sync_every_n_times (int): If >1, only sync every n times the `sync` API is called.

            collections (set): A set of specifiers (currently only tf), that determine which Variables
                of the parent Component to synchronize.
        """
        self.sync_tau = sync_tau
        self.sync_every_n_times = sync_every_n_times
        self.collections = collections

        # Variable.
        self.steps_since_last_sync = None  # type: tf.Variable

        super(Synchronizable, self).__init__(scope=scope, **kwargs)

    def create_variables(self, input_spaces, action_space=None):
        self.steps_since_last_sync = self.get_variable(
            "steps-since-last-sync", dtype="int", initializer=0, trainable=False  #, private=True
        )

    def check_variable_completeness(self):
        # Overwrites this method as any Synchronizable should only be variable-complete once the parent
        # component is variable-complete (not counting this component!).

        # Shortcut.
        if self.variable_complete:
            return True

        # If this component is not used at all (no calls to API-method: `sync` are made), return True.
        if len(self.api_methods["sync"].in_op_columns) == 0:
            return super(Synchronizable, self).check_variable_completeness()

        # Recheck input-completeness of parent.
        if self.parent_component.input_complete is False:
            self.graph_builder.build_component_when_input_complete(self.parent_component)

        # Pretend we are input- and variable-complete (which we may not be) and then check parent's variable
        # completeness under this condition.
        if self.parent_component.variable_complete is False:
            self.variable_complete = True
            self.graph_builder.build_component_when_input_complete(self.parent_component)  #.check_variable_completeness()  #build_when_complete=False)
            self.variable_complete = False

        if self.parent_component.variable_complete is True:
            return super(Synchronizable, self).check_variable_completeness()

        return False

    @rlgraph_api(returns=1, requires_variable_completeness=True)
    def _graph_fn_sync(self, values_, tau=None):
        """
        Generates the op that syncs this Synchronizable's parent's variable values from another Synchronizable
        Component.

        Args:
            values_ (DataOpDict): The dict of variable values (coming from the "variables"-API of any other
                Component) that need to be assigned to this Component's parent's variables.
                The keys in the dict refer to the names of our parent's variables and must match their names.

            tau (Optional[float]): An optional tau parameter which - when given - will override `self.sync_tau`.

        Returns:
            DataOp: The op that executes the syncing (or no_op if sync_every_n_times-condition not fulfilled).
        """
        if tau is None:
            tau = self.sync_tau

        syncs = []

        if get_backend() == "tf":
            # Check whether we should sync at all.
            inc_op = tf.assign_add(self.steps_since_last_sync, 1)

            def reset_op():
                op = tf.assign(self.steps_since_last_sync, 0)
                with tf.control_dependencies([op]):
                    return tf.no_op()

            with tf.control_dependencies([inc_op]):
                sync_counter_op = tf.cond(
                    self.steps_since_last_sync >= self.sync_every_n_times,
                    true_fn=reset_op,
                    false_fn=tf.no_op
                )

            def sync_op():
                parents_vars = self.parent_component.get_variables(collections=self.collections, custom_scope_separator="-")
                # Remove all our own variables from the parent-list. These should not be synched.
                own_vars = self.get_variables(collections=self.collections, custom_scope_separator="-")
                for key in own_vars:
                    del parents_vars[key]
                syncs_from, syncs_to = (sorted(values_.items()), sorted(parents_vars.items()))
                if len(syncs_from) != len(syncs_to):
                    raise RLGraphError(
                        "ERROR: Number of Variables to sync must match! We have {} syncs_from and {} syncs_to.".
                        format(len(syncs_from), len(syncs_to))
                    )

                for (key_from, var_from), (key_to, var_to) in zip(syncs_from, syncs_to):
                    # Sanity checking. TODO: Check the names' ends? Without the global scope?
                    #if key_from != key_to:
                    #    raise RLGraphError("ERROR: Variable names for syncing must match in order and name! "
                    #                    "Mismatch at from={} and to={}.".format(key_from, key_to))
                    if get_shape(var_from) != get_shape(var_to):
                        raise RLGraphError(
                            "ERROR: Variable shapes for syncing must match! Shape mismatch between from={} ({}) and "
                            "to={} ({}).".format(key_from, get_shape(var_from), key_to, get_shape(var_to))
                        )
                    if tau == 1.0:
                        syncs.append(self.assign_variable(var_to, var_from))
                    else:
                        syncs.append(self.assign_variable(var_to, tau * var_from + (1.0 - tau) * var_to))

                with tf.control_dependencies(syncs):
                    return tf.no_op()

            # Bundle everything into one "sync"-op.
            with tf.control_dependencies([sync_counter_op]):
                cond_sync_op = tf.cond(
                    tf.equal(self.steps_since_last_sync, 0),
                    true_fn=sync_op,
                    false_fn=tf.no_op,
                    name="sync-to-{}".format(self.parent_component.name)
                )
                return cond_sync_op

        elif get_backend() == "pytorch":
            # Check whether we should sync at all.
            self.steps_since_last_sync += 1

            # Synchronize and reset our counter.
            if self.steps_since_last_sync >= self.sync_every_n_times:
                self.steps_since_last_sync = 0

                # Get refs(!)
                parents_vars = self.parent_component.get_variables(
                    collections=self.collections, custom_scope_separator="-", get_ref=True
                )
                parents_var_values = self.parent_component.get_variables(
                    collections=self.collections, custom_scope_separator="-", get_ref=False
                )
                syncs_from, sync_to_ref, sync_to_value = (
                    sorted(values_.items()), sorted(parents_vars.items()), sorted(parents_var_values.items())
                )

                if self.sync_tau == 1.0:
                    for (key_from, var_from), (key_to, ref_to) in zip(syncs_from, sync_to_ref):
                        ref_to.set_value(var_from)
                else:
                    for (key_from, var_from), (key_to, ref_to), (_, var_to) in zip(syncs_from, sync_to_ref, sync_to_value):
                        ref_to.set_value(self.sync_tau * var_from + (1.0 - self.sync_tau) * var_to)

            return None
