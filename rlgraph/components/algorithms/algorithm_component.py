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
from rlgraph.components import Exploration, PreprocessorStack, Synchronizable, Policy, Optimizer, \
    ContainerMerger, ContainerSplitter
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.input_parsing import parse_value_function_spec

if get_backend() == "tf":
    import tensorflow as tf


class AlgorithmComponent(Component):
    """
    The root component of some Algorithm/Agent.
    """
    def __init__(self, agent, preprocessing_spec=None, policy_spec=None, network_spec = None, value_function_spec=None,
                 exploration_spec=None, optimizer_spec=None, value_function_optimizer_spec=None,
                 scope="algorithm-component", **kwargs):
        super(AlgorithmComponent, self).__init__(scope=scope, **kwargs)

        self.agent = agent
        self.nesting_level = 0

        # Construct the Preprocessor.
        self.preprocessor = PreprocessorStack.from_spec(preprocessing_spec)
        self.preprocessed_state_space = self.preprocessor.get_preprocessed_space(self.agent.state_space)
        self.preprocessing_required = preprocessing_spec is not None and len(preprocessing_spec) > 0
        if self.preprocessing_required:
            self.logger.info("Preprocessing required.")
            self.logger.info("Parsed preprocessed-state space definition: {}".format(self.preprocessed_state_space))
        else:
            self.logger.info("No preprocessing required.")

        # Construct the Policy and its NeuralNetwork.
        # Adjust/auto-generate a policy_spec so it always contains a network spec and action_space.
        policy_spec = policy_spec or {}
        if "network_spec" not in policy_spec:
            policy_spec["network_spec"] = network_spec
        if "action_space" not in policy_spec:
            policy_spec["action_space"] = self.agent.action_space
        self.policy = Policy.from_spec(policy_spec)
        # Done by default.
        self.policy.add_components(Synchronizable(), expose_apis="sync")

        # Create non-shared baseline network.
        self.value_function = parse_value_function_spec(value_function_spec)
        # TODO move this to specific agents.
        if self.value_function is not None:
            self.vars_merger = ContainerMerger("policy", "vf", scope="variable-dict-merger")
            self.vars_splitter = ContainerSplitter("policy", "vf", scope="variable-container-splitter")
        else:
            self.vars_merger = ContainerMerger("policy", scope="variable-dict-merger")
            self.vars_splitter = ContainerSplitter("policy", scope="variable-container-splitter")

        self.exploration = Exploration.from_spec(exploration_spec)

        # An object implementing the loss function interface is only strictly needed
        # if automatic device strategies like multi-gpu are enabled. This is because
        # the device strategy needs to know the name of the loss function to infer the appropriate
        # operations.
        self.loss_function = None

        # Create the Agent's optimizer based on optimizer_spec and execution strategy.
        self.optimizer = None
        if optimizer_spec is not None:
            self.optimizer = Optimizer.from_spec(optimizer_spec)

        self.value_function_optimizer = None
        if self.value_function is not None:
            if value_function_optimizer_spec is None:
                vf_optimizer_spec = optimizer_spec
            else:
                vf_optimizer_spec = value_function_optimizer_spec
            vf_optimizer_spec["scope"] = "value-function-optimizer"
            self.value_function_optimizer = Optimizer.from_spec(vf_optimizer_spec)

        self.add_components(self.preprocessor, self.policy, self.value_function, self.vars_merger, self.vars_splitter,
                            self.exploration, self.optimizer, self.value_function_optimizer)

        if self.value_function is not None:
            # This avoids variable-incompleteness for the value-function component in a multi-GPU setup, where the root
            # value-function never performs any forward pass (only used as variable storage).
            @rlgraph_api(component=self)
            def get_state_values(self_, preprocessed_states):
                #vf = self_.get_sub_component_by_name(self_.value_function.scope)
                return self_.value_function.value_output(preprocessed_states)

    @rlgraph_api
    def update_from_memory(self, **kwargs):
        raise NotImplementedError

    @rlgraph_api
    def update_from_external_batch(self, batch, **kwargs):
        raise NotImplementedError

    # Add API methods for syncing.
    @rlgraph_api
    def get_weights(self):
        #policy = self.policy(self.agent.policy.scope)
        policy_weights = self.policy.variables()
        value_function_weights = None
        if self.value_function is not None:
            #value_func = self.get_sub_component_by_name(self.agent.value_function.scope)
            value_function_weights = self.value_function.variables()
        return dict(policy_weights=policy_weights, value_function_weights=value_function_weights)

    @rlgraph_api(must_be_complete=False)
    def set_weights(self, policy_weights, value_function_weights=None):
        #policy = self.get_sub_component_by_name(self.agent.policy.scope)
        policy_sync_op = self.policy.sync(policy_weights)
        if value_function_weights is not None:
            assert self.value_function is not None
            #vf = self.get_sub_component_by_name(self.agent.value_function.scope)
            vf_sync_op = self.value_function.sync(value_function_weights)
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
        #preprocessor = self.get_sub_component_by_name(self.agent.preprocessor.scope)
        return self.preprocessor.preprocess(states)

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
