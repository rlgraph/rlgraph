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
from rlgraph.components import Exploration, PreprocessorStack, Synchronizable, Policy, Optimizer, \
    ValueFunction
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.rlgraph_errors import RLGraphObsoletedError

if get_backend() == "tf":
    import tensorflow as tf


class AlgorithmComponent(Component):
    """
    The root component of some Algorithm/Agent.
    """
    def __init__(self, agent, *, discount=0.98, memory_batch_size=None, preprocessing_spec=None, policy_spec=None,
                 network_spec=None, value_function_spec=None,
                 exploration_spec=None, optimizer_spec=None, value_function_optimizer_spec=None,
                 scope="algorithm-component", **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).

            memory_batch_size (Optional[int]): The batch size to use when pulling data from memory.

            preprocessing_spec (Optional[list,PreprocessorStack]): The spec list for the different necessary states
                preprocessing steps or a PreprocessorStack object itself.

            policy_spec (Optional[dict]): An optional dict for further kwargs passing into the Policy c'tor.

            network_spec (Optional[list,NeuralNetwork]): Spec list for a NeuralNetwork Component or the NeuralNetwork
                object itself.

            value_function_spec (list, dict, ValueFunction): Neural network specification for baseline or instance
                of ValueFunction.

            exploration_spec (Optional[dict]): The spec-dict to create the Exploration Component.

            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.

            value_function_optimizer_spec (dict): Optimizer config for value function optimizer. If None, the optimizer
                spec for the policy is used (same learning rate and optimizer type).
        """

        super(AlgorithmComponent, self).__init__(scope=scope, **kwargs)

        # Our owning Agent object (may be None, e.g. for testing purposes).
        self.agent = agent

        # Root component, set nesting level to 0.
        self.nesting_level = 0

        # Some generic properties that all Agents have.
        self.discount = discount
        self.memory_batch_size = memory_batch_size

        # Construct the Preprocessor.
        self.preprocessor = PreprocessorStack.from_spec(preprocessing_spec)
        self.preprocessing_required = preprocessing_spec is not None and len(preprocessing_spec) > 0
        if self.preprocessing_required:
            self.logger.info("Preprocessing required.")
        else:
            self.logger.info("No preprocessing required.")

        # Construct the Policy and its NeuralNetwork (if policy-spec or network given).
        if policy_spec is None and network_spec is None:
            self.policy = None
        elif isinstance(policy_spec, dict):
            # Adjust/auto-generate a policy_spec so it always contains a network spec and action_space.
            policy_spec = policy_spec or {}
            if "network_spec" not in policy_spec:
                policy_spec["network_spec"] = network_spec
            if "action_space" not in policy_spec:
                policy_spec["action_space"] = self.agent.action_space
            self.policy = Policy.from_spec(policy_spec)
            self.policy.add_components(Synchronizable(), expose_apis="sync")
        else:
            # Make sure policy is Synchronizable (will raise ERROR otherwise).
            self.policy = policy_spec
            assert self.policy.get_sub_component_by_name("synchronizable")

        # Create non-shared baseline network (and add synchronization feature).
        self.value_function = ValueFunction.from_spec(value_function_spec)
        self.value_function.add_components(Synchronizable(), expose_apis="sync")
        # TODO move this to specific agents.
        #if self.value_function is not None:
        #    self.vars_merger = ContainerMerger("policy", "vf", scope="variable-dict-merger")
        #    self.vars_splitter = ContainerSplitter("policy", "vf", scope="variable-container-splitter")
        #else:
        #    self.vars_merger = ContainerMerger("policy", scope="variable-dict-merger")
        #    self.vars_splitter = ContainerSplitter("policy", scope="variable-container-splitter")

        # Optional exploration object. None if no exploration needed (usually None for PG algos, as they use
        # stochastic policies, not epsilon greedy Q).
        self.exploration = Exploration.from_spec(exploration_spec)

        # An object implementing the loss function interface is only strictly needed
        # if automatic device strategies like multi-gpu are enabled. This is because
        # the device strategy needs to know the name of the loss function to infer the appropriate
        # operations.
        self.loss_function = None

        # List of all of this Component's optimizer sub-components.
        self.all_optimizers = []

        # Create the Agent's optimizer based on optimizer_spec and execution strategy.
        self.optimizer = None
        if optimizer_spec is not None:
            self.optimizer = Optimizer.from_spec(optimizer_spec)
            self.all_optimizers.append(self.optimizer)

        self.value_function_optimizer = None
        if self.value_function is not None:
            if value_function_optimizer_spec is None:
                vf_optimizer_spec = optimizer_spec
            else:
                vf_optimizer_spec = value_function_optimizer_spec

            # Change name to avoid scope-collision.
            if isinstance(vf_optimizer_spec, dict):
                vf_optimizer_spec["scope"] = "value-function-optimizer"
            else:
                vf_optimizer_spec.scope = "value-function-optimizer"
                vf_optimizer_spec.propagate_scope()

            self.value_function_optimizer = Optimizer.from_spec(vf_optimizer_spec)
            self.all_optimizers.append(self.value_function_optimizer)

        self.add_components(self.preprocessor, self.policy, self.value_function, #self.vars_merger, self.vars_splitter,
                            self.exploration, self.optimizer, self.value_function_optimizer)

        # Add reset-preprocessor API, if necessary.
        if self.preprocessing_required:
            @rlgraph_api(component=self)
            def reset_preprocessor(self):
                reset_op = self.preprocessor.reset()
                return reset_op

        # Get state value default API.
        if self.value_function is not None:
            # This avoids variable-incompleteness for the value-function component in a multi-GPU setup, where the root
            # value-function never performs any forward pass (only used as variable storage).
            @rlgraph_api(component=self)
            def get_state_values(self_, preprocessed_states):
                #vf = self_.get_sub_component_by_name(self_.value_function.scope)
                return self_.value_function.call(preprocessed_states)

        # Default getter and setter for policy/vf weights.
        if self.policy is not None:
            @rlgraph_api(component=self)
            def get_weights(self):
                # policy = self.policy(self.agent.policy.scope)
                policy_weights = self.policy.variables()
                value_function_weights = None
                if self.value_function is not None:
                    # value_func = self.get_sub_component_by_name(self.agent.value_function.scope)
                    value_function_weights = self.value_function.variables()
                return dict(policy_weights=policy_weights, value_function_weights=value_function_weights)

            @rlgraph_api(component=self, must_be_complete=False)
            def set_weights(self, policy_weights, value_function_weights=None):
                # policy = self.get_sub_component_by_name(self.agent.policy.scope)
                policy_sync_op = self.policy.sync(policy_weights)
                if value_function_weights is not None:
                    assert self.value_function is not None
                    # vf = self.get_sub_component_by_name(self.agent.value_function.scope)
                    vf_sync_op = self.value_function.sync(value_function_weights)
                    return self._graph_fn_group(policy_sync_op, vf_sync_op)
                else:
                    return policy_sync_op

    def add_components(self, *sub_components, expose_apis=None):
        super(AlgorithmComponent, self).add_components(*sub_components, expose_apis=expose_apis)

        # Keep track of all our Optimizers.
        for sub_component in self.get_all_sub_components(exclude_self=True):
            if isinstance(sub_component, Optimizer):
                self.all_optimizers.append(sub_component)

    @rlgraph_api
    def get_preprocessed_states(self, states):
        """
        Args:
            states (DataOpRec): The states to preprocess via the Component's PreprocessorStack (if any).

        Returns:
            DataOpRec: The preprocessed states (or the original states if there is no PreprocessorStack).
        """
        if self.preprocessing_required is True:
            return self.preprocessor.preprocess(states)
        return states

    @rlgraph_api
    def get_actions(self, states, deterministic=False, time_percentage=None):
        """
        Args:
            states (DataOpRec): The states from which to derive actions (via mapping through the policy's NNs).
            deterministic (DataOpRec(bool)): Whether to draw the action deterministically (deterministic sampling
                from the Policy's distribution(s)).

        Returns:
            DataOpRec(dict):
                `actions`: The drawn action.
                `preprocessed_states`: The preprocessed states.
                `[other data]`: Depending on the AlgorithmComponent.
        """
        preprocessed_states = self.get_preprocessed_states(states)
        return self.get_actions_from_preprocessed_states(preprocessed_states, deterministic, time_percentage)

    @rlgraph_api
    def get_actions_from_preprocessed_states(self, preprocessed_states, deterministic=False, time_percentage=None):
        """
        Args:
            preprocessed_states (DataOpRec): The already preprocessed states from which to derive actions
                (via mapping through the policy's NNs).

            deterministic (DataOpRec(bool)): Whether to draw the action deterministically (deterministic sampling
                from the Policy's distribution(s)).

            time_percentage (SingleDataOpRec): The percentage (between 0.0 and 1.0) of the time steps already done
                with respect to some total (maximum) number of time steps.

        Returns:
            DataOpRec(dict):
                `actions`: The drawn action.
                `preprocessed_states`: The preprocessed states.
                `[other data]`: Depending on the AlgorithmComponent.
        """
        out = self.policy.get_action(preprocessed_states, deterministic=deterministic)
        if self.exploration is not None:
            out["action"] = self.exploration.get_action(out["action"], time_percentage, not deterministic)
        return dict(actions=out["action"], preprocessed_states=preprocessed_states)

    @rlgraph_api
    def update_from_memory(self, **kwargs):
        """
        Updates this Component's Policy/ValueFunction or other optimizable sub-components via pulling samples from
        the memory.
        """
        raise NotImplementedError

    @rlgraph_api
    def update_from_external_batch(self, **kwargs):
        """
        Updates this Component's Policy/ValueFunction or other optimizable sub-components via some (external) data.
        """
        raise NotImplementedError

    # TODO: Replace this with future on-the-fly-API-components.
    @graph_fn
    def _graph_fn_group(root, *ops):
        if get_backend() == "tf":
            return tf.group(*ops)
        # For pytorch (define-by-run), ops don't matter.
        return ops[0]

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

    def preprocess_states(self, nn_inputs):
        raise RLGraphObsoletedError("API-method", "preprocess_states", "get_preprocessed_states")

    def get_preprocessed_state_and_action(self, states, time_percentage=None, use_exploration=True):
        raise RLGraphObsoletedError("API-method", "get_preprocessed_state_and_action", "get_actions(deterministic=not use_exploration!)")

    def action_from_preprocessed_state(self, preprocessed_states, time_percentage=None, use_exploration=True):
        raise RLGraphObsoletedError("API-method", "action_from_preprocessed_state", "get_actions_from_preprocessed_states(deterministic=not use_exploration!)")
