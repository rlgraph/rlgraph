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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf


class AlgorithmComponent(Component):
    """
    The root component of some Algorithm/Agent.
    """
    def __init__(self, agent, scope="algorithm-root", **kwargs):
        super(AlgorithmComponent, self).__init__(scope=scope, **kwargs)

        self.agent = agent
        self.nesting_level = 0

        if self.agent.value_function is not None:
            # This avoids variable-incompleteness for the value-function component in a multi-GPU setup, where the root
            # value-function never performs any forward pass (only used as variable storage).
            @rlgraph_api(component=self)
            def get_state_values(root, preprocessed_states):
                vf = root.get_sub_component_by_name(agent.value_function.scope)
                return vf.value_output(preprocessed_states)

    @rlgraph_api
    def update_from_memory(self, **kwargs):
        raise NotImplementedError

    @rlgraph_api
    def update_from_external_batch(self, batch, **kwargs):
        raise NotImplementedError

    # Add API methods for syncing.
    @rlgraph_api
    def get_weights(self):
        policy = self.get_sub_component_by_name(self.agent.policy.scope)
        policy_weights = policy.variables()
        value_function_weights = None
        if self.agent.value_function is not None:
            value_func = self.get_sub_component_by_name(self.agent.value_function.scope)
            value_function_weights = value_func.variables()
        return dict(policy_weights=policy_weights, value_function_weights=value_function_weights)

    @rlgraph_api(must_be_complete=False)
    def set_weights(self, policy_weights, value_function_weights=None):
        policy = self.get_sub_component_by_name(self.agent.policy.scope)
        policy_sync_op = policy.sync(policy_weights)
        if value_function_weights is not None:
            assert self.agent.value_function is not None
            vf = self.get_sub_component_by_name(self.agent.value_function.scope)
            vf_sync_op = vf.sync(value_function_weights)
            return self._graph_fn_group(policy_sync_op, vf_sync_op)
        else:
            return policy_sync_op

    # TODO: Replace this with future on-the-fly-API-components.
    @graph_fn
    def _graph_fn_group(self, *ops):
        if get_backend() == "tf":
            return tf.group(*ops)
        return ops[0]

    # To pre-process external data if needed.
    @rlgraph_api
    def preprocess_states(self, states):
        preprocessor_stack = self.get_sub_component_by_name(self.agent.preprocessor.scope)
        return preprocessor_stack.preprocess(states)

    @graph_fn
    def _graph_fn_training_step(self, other_step_op=None):
        """
        Increases the global training timestep by 1. Should be called by all training API-methods to
        timestamp each training/update step.

        Args:
            other_step_op (Optional[DataOp]): Another DataOp (e.g. a step_op) which should be
                executed before the increase takes place.

        Returns:
            DataOp: no_op.
        """
        if get_backend() == "tf":
            add_op = tf.assign_add(self.agent.graph_executor.global_training_timestep, 1)
            op_list = [add_op] + [other_step_op] if other_step_op is not None else []
            with tf.control_dependencies(op_list):
                if other_step_op is None or hasattr(other_step_op, "type") and other_step_op.type == "NoOp":
                    return tf.no_op()
                else:
                    return tf.identity(other_step_op)
        elif get_backend == "pytorch":
            self.agent.graph_executor.global_training_timestep += 1
            return None
