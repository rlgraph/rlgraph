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

import numpy as np

from rlgraph import get_backend
from rlgraph.agents import Agent
from rlgraph.components import Synchronizable, Memory, PrioritizedReplay
from rlgraph.components.algorithms.algorithm_component import AlgorithmComponent
from rlgraph.components.loss_functions.sac_loss_function import SACLossFunction
from rlgraph.components.policies.policy import Policy
from rlgraph.execution.rules.sync_rules import SyncRules
from rlgraph.spaces import FloatBox, BoolBox, IntBox, ContainerSpace
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import flatten_op

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class SACAgent(Agent):
    def __init__(
        self,
        state_space,
        action_space,
        *,
        discount=0.98,
        python_buffer_size=0,
        custom_python_buffers=None,
        memory_batch_size=None,
        preprocessing_spec=None,
        network_spec=None,
        internal_states_space=None,
        policy_spec=None,
        value_function_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        value_function_optimizer_spec=None,
        observe_spec=None,
        update_spec=None,
        sync_rules=None,
        summary_spec=None,
        saver_spec=None,
        auto_build=True,
        name="sac-agent",
        double_q=True,
        initial_alpha=1.0,
        gumbel_softmax_temperature=1.0,
        target_entropy=None,
        memory_spec=None,
        q_function_sync_rules=None
    ):
        """
        This is an implementation of the Soft-Actor Critic algorithm.

        Paper: http://arxiv.org/abs/1801.01290

        Args:
            state_space (Union[dict,Space]): Spec dict for the state Space or a direct Space object.
            action_space (Union[dict,Space]): Spec dict for the action Space or a direct Space object.
            preprocessing_spec (Optional[list,PreprocessorStack]): The spec list for the different necessary states
                preprocessing steps or a PreprocessorStack object itself.
            discount (float): The discount factor (gamma).
            network_spec (Optional[list,NeuralNetwork]): Spec list for a NeuralNetwork Component or the NeuralNetwork
                object itself.
            internal_states_space (Optional[Union[dict,Space]]): Spec dict for the internal-states Space or a direct
                Space object for the Space(s) of the internal (RNN) states.
            policy_spec (Optional[dict]): An optional dict for further kwargs passing into the Policy c'tor.
            value_function_spec (list, dict, ValueFunction): Neural network specification for baseline or instance
                of ValueFunction.
            execution_spec (Optional[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.
            value_function_optimizer_spec (dict): Optimizer config for value function optimizer. If None, the optimizer
                spec for the policy is used (same learning rate and optimizer type).
            observe_spec (Optional[dict]): Obsoleted: Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Obsoleted: Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.
            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.
            name (str): Some name for this Agent object.
            double_q (bool): Whether to train two q networks independently.
            initial_alpha (float): "The temperature parameter α determines the
                relative importance of the entropy term against the reward".
            gumbel_softmax_temperature (float): Temperature parameter for the Gumbel-Softmax distribution used
                for discrete actions.
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
            update_spec (dict): Here we can have sync_interval or sync_tau (for the value network update).
        """
        super(SACAgent, self).__init__(
            state_space=state_space,
            action_space=action_space,
            python_buffer_size=python_buffer_size,
            custom_python_buffers=custom_python_buffers,
            internal_states_space=internal_states_space,
            execution_spec=execution_spec,
            observe_spec=observe_spec,  # Obsolete.
            update_spec=update_spec,  # Obsolete.
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            name=name
        )

        self.double_q = double_q
        # Keep track of when to sync the target network (every n updates).
        if isinstance(sync_rules, dict) and "sync_tau" not in sync_rules:
            sync_rules["sync_tau"] = 0.005  # The value mentioned in the paper
        self.sync_rules = SyncRules.from_spec(sync_rules)
        self.steps_since_target_net_sync = 0

        self.root_component = SACAlgorithmComponent(
            agent=self,
            policy=policy_spec,
            network_spec=network_spec,
            q_function_spec=value_function_spec,  # q-functions
            preprocessing_spec=preprocessing_spec,
            memory_spec=memory_spec,
            discount=discount,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
            #num_iterations=num_iterations,
            memory_batch_size=memory_batch_size,
            gumbel_softmax_temperature=gumbel_softmax_temperature,
            optimizer_spec=optimizer_spec,
            value_function_optimizer_spec=value_function_optimizer_spec,
            #alpha_optimizer=self.alpha_optimizer,
            q_function_sync_rules=q_function_sync_rules,
            num_q_functions=2 if self.double_q is True else 1
        )

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.root_component.preprocessor.get_preprocessed_space(self.state_space).\
            with_batch_rank()
        float_action_space = self.action_space.with_batch_rank().map(
            mapping=lambda flat_key, space: space.as_one_hot_float_space() if isinstance(space, IntBox) else space
        )
        self.input_spaces.update(dict(
            env_actions=self.action_space.with_batch_rank(),
            actions=float_action_space,
            preprocessed_states=preprocessed_state_space,
            rewards=FloatBox(add_batch_rank=True),
            terminals=BoolBox(add_batch_rank=True),
            next_states=preprocessed_state_space,
            states=self.state_space.with_batch_rank(add_batch_rank=True),
            importance_weights=FloatBox(add_batch_rank=True),
            deterministic=bool,
            weights="variables:{}".format(self.root_component.policy.scope)
        ))

        if auto_build is True:
            self.build(build_options=dict(optimizers=self.root_component.all_optimizers))

    def set_weights(self, policy_weights, value_function_weights=None):
        return self.graph_executor.execute((self.root_component.set_policy_weights, policy_weights))

    def get_weights(self):
        return dict(policy_weights=self.graph_executor.execute(self.root_component.get_policy_weights))

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None,
                   time_percentage=None):
        # Call super.
        ret = super(SACAgent, self).get_action(
            states, internals, use_exploration, apply_preprocessing, extra_returns, time_percentage
        )
        actions = ret[0] if "preprocessed_states" in extra_returns else ret

        # Convert Gumble (relaxed one-hot) sample back into int type for all discrete composite actions.
        if isinstance(self.action_space, ContainerSpace):
            actions = actions.map(
                mapping=lambda key, action: np.argmax(action, axis=-1).astype(action.dtype)
                if isinstance(self.flat_action_space[key], IntBox) else action
            )
        elif isinstance(self.action_space, IntBox):
            actions = np.argmax(actions, axis=-1).astype(self.action_space.dtype)

        if "preprocessed_states" in extra_returns:
            return actions, ret[1]
        else:
            return actions

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals, **kwargs):
        next_states = kwargs.pop("next_states")
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, next_states, terminals]))

    def update(self, batch=None, time_percentage=None, **kwargs):
        self.num_updates += 1

        if batch is None:
            #size = self.graph_executor.execute(self.root_component.get_memory_size)
            # TODO: is this necessary?
            #if size < self.batch_size:
            #    return 0.0, 0.0, 0.0
            ret = self.graph_executor.execute(("update_from_memory", [time_percentage]))
        else:
            # No sequence indices means terminals are used in place.
            batch_input = [
                batch["states"], batch["actions"], batch["rewards"], batch["terminals"], batch["next_states"],
                time_percentage
            ]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input))

        return \
            ret["actor_loss"] + ret["critic_loss"] + ret["alpha_loss"],\
            ret["actor_loss_per_item"] + ret["critic_loss_per_item"] + ret["alpha_loss_per_item"]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.root_component.preprocessing_required and len(self.root_component.preprocessor.variables) > 0:
            self.graph_executor.execute("reset_preprocessor")
        self.graph_executor.execute("reset_targets")

    def __repr__(self):
        return "SACAgent(double-q={}, initial-alpha={}, target-entropy={})".format(
            self.double_q, self.root_component.initial_alpha, self.root_component.target_entropy
        )


class SACAlgorithmComponent(AlgorithmComponent):
    def __init__(self, agent, memory_spec, q_function_spec, initial_alpha=1.0, gumbel_softmax_temperature=1.0,
                 target_entropy=None, q_function_sync_rules=None, num_q_functions=2,
                 scope="sac-agent-component", **kwargs):

        # If VF spec is a network spec, wrap with SAC vf type. The VF must concatenate actions and states,
        # which can require splitting the network in the case of e.g. conv-inputs.
        if isinstance(q_function_spec, list):
            q_function_spec = dict(type="sac_value_function", network_spec=q_function_spec)
            self.logger.info("Using default SAC value function.")
        elif isinstance(q_function_spec, ValueFunction):
            self.logger.info("Using value function object {}".format(ValueFunction))

        # Default Policy (non-deterministic):
        # Continuous action space: Use squashed normal.
        # Discrete: Gumbel-softmax.
        policy_spec = Policy.set_policy_deterministic(kwargs.pop("policy_spec", dict(
            deterministic=False, distributions_spec=dict(
                bounded_distribution_type="squashed", discrete_distribution_type="gumbel_softmax",
                gumbel_softmax_temperature=gumbel_softmax_temperature
            )
        )), False)
        super(SACAlgorithmComponent, self).__init__(
            agent, policy_spec=policy_spec, value_function_spec=q_function_spec, scope=scope, **kwargs
        )
        # Make base value_function (template for all our q-functions) synchronizable.
        if "synchronizable" not in self.value_function.sub_components:
            self.value_function.add_components(Synchronizable(), expose_apis="sync")

        self.q_function_sync_rules = SyncRules.from_spec(q_function_sync_rules)

        self.memory = Memory.from_spec(memory_spec)
        # Copy value function (q-function) n times to reach num_q_functions.
        self.q_functions = [self.value_function] + [
            self.value_function.copy(scope="{}-{}".format(self.value_function.scope, i + 2), trainable=True)
            for i in range(num_q_functions - 1)
        ]

        # Set number of return values for get_q_values graph_fn.
        self.graph_fn_num_outputs["_graph_fn_get_q_values"] = num_q_functions

        # Produce target q-functions from respective base q-functions (which now also contain
        # the Synchronizable component).
        self.target_q_functions = [q.copy(scope="target-" + q.scope, trainable=False) for q in self.q_functions]

        self.target_entropy = target_entropy
        self.alpha_optimizer = self.optimizer.copy(scope="alpha-" + self.optimizer.scope) if self.target_entropy is not None else None
        self.initial_alpha = initial_alpha
        self.log_alpha = None

        self.loss_function = SACLossFunction(
            target_entropy=target_entropy, discount=self.discount, num_q_functions=num_q_functions
        )

        self.steps_since_last_sync = None
        self.env_action_space = None

        self.add_components(self.memory, self.loss_function, self.alpha_optimizer)
        self.add_components(*self.q_functions[1:])  # Skip the 1st value-function (already added via super-call)
        self.add_components(*self.target_q_functions)

    def check_input_spaces(self, input_spaces, action_space=None):
        for s in ["states", "actions", "env_actions", "preprocessed_states", "rewards", "terminals"]:
            sanity_check_space(input_spaces[s], must_have_batch_rank=True)

        self.env_action_space = input_spaces["env_actions"].flatten()

    def create_variables(self, input_spaces, action_space=None):
        self.steps_since_last_sync = self.get_variable("steps_since_last_sync", dtype="int", initializer=0)
        self.log_alpha = self.get_variable("log_alpha", dtype="float", initializer=np.log(self.initial_alpha))

    @rlgraph_api
    def get_policy_weights(self):
        return self.policy.variables()

    #@rlgraph_api
    #def get_q_weights(self):
    #    #merged_weights = self._q_vars_merger.merge(*[q.variables() for q in self.q_functions])
    #    #q_names = ["q_{}".format(i) for i in range(len(self.q_functions))]
    #    #self._q_vars_merger = ContainerMerger(*q_names, scope="q_vars_merger")
    #    merged_weights = {"q_{}".format(i): q.variables() for i, q in enumerate(self.q_functions)}
    #    return merged_weights

    @rlgraph_api(must_be_complete=False)
    def set_policy_weights(self, weights):
        return self.policy.sync(weights)

    """ TODO: need to define the input space
    @rlgraph_api(must_be_complete=False)
    def set_q_weights(self, q_weights):
        split_weights = self._q_vars_splitter.call(q_weights)
        assert len(split_weights) == len(self.q_functions)
        update_ops = [q.sync(q_weights) for q_weights, q in zip(split_weights, self.q_functions)]
        update_ops.extend([q.sync(q_weights) for q_weights, q in zip(split_weights, self.target_q_functions)])
        return tuple(update_ops)
    """

    @rlgraph_api
    def insert_records(self, preprocessed_states, env_actions, rewards, next_states, terminals):
        records = dict(
            states=preprocessed_states, env_actions=env_actions, rewards=rewards, next_states=next_states,
            terminals=terminals
        )
        return self.memory.insert_records(records)

    @rlgraph_api
    def update_from_memory(self, time_percentage=None):
        records, sample_indices, importance_weights = self.memory.get_records(self.memory_batch_size)
        result = self.update_from_external_batch(
            records["states"], records["actions"], records["rewards"], records["terminals"],
            records["next_states"], importance_weights, time_percentage
        )

        if isinstance(self.memory, PrioritizedReplay):
            update_pr_step_op = self.memory.update_records(sample_indices, result["critic_loss_per_item"])
            result["step_op"] = self._graph_fn_group(result["step_op"], update_pr_step_op)

        return result

    @rlgraph_api
    def update_from_external_batch(self, preprocessed_states, env_actions, rewards, terminals, next_states,
                                   importance_weights, time_percentage=None
    ):
        actions = self._graph_fn_one_hot(env_actions)
        actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item, alpha_loss, alpha_loss_per_item = \
            self.get_losses(preprocessed_states, actions, rewards, terminals, next_states, importance_weights)

        policy_vars = self.policy.variables()
        merged_q_vars = {"q_{}".format(i): q.variables() for i, q in enumerate(self.q_functions)}
        critic_step_op = self.value_function_optimizer.step(
            merged_q_vars, critic_loss, critic_loss_per_item, time_percentage
        )

        actor_step_op = self.optimizer.step(
            policy_vars, actor_loss, actor_loss_per_item, time_percentage
        )

        if self.target_entropy is not None:
            alpha_step_op = self._graph_fn_update_alpha(alpha_loss, alpha_loss_per_item, time_percentage)
        else:
            alpha_step_op = self._graph_fn_no_op()
        # TODO: optimizer for alpha
        sync_op = self.sync_targets()

        # Increase the global training step counter.
        alpha_step_op = self._graph_fn_training_step(alpha_step_op)

        return dict(
            actor_step_op=actor_step_op,
            critic_step_op=critic_step_op,
            sync_op=sync_op,
            alpha_step_op=alpha_step_op,
            actor_loss=actor_loss,
            actor_loss_per_item=actor_loss_per_item,
            critic_loss=critic_loss,
            critic_loss_per_item=critic_loss_per_item,
            alpha_loss=alpha_loss,
            alpha_loss_per_item=alpha_loss_per_item
        )

    @graph_fn(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_one_hot(self, key, env_actions):
        if isinstance(self.env_action_space[key], IntBox):
            env_actions = tf.one_hot(env_actions, depth=self.env_action_space[key].num_categories, axis=-1)
        return env_actions

    @graph_fn(requires_variable_completeness=True)
    def _graph_fn_update_alpha(self, alpha_loss, alpha_loss_per_item, time_percentage):
        alpha_step_op = self.alpha_optimizer.step(
            #DataOpTuple([self.log_alpha]), alpha_loss, alpha_loss_per_item, time_percentage)
            (self.log_alpha,), alpha_loss, alpha_loss_per_item, time_percentage
        )
        return alpha_step_op

    @rlgraph_api  # `returns` are determined in ctor
    def _graph_fn_get_q_values(self, preprocessed_states, actions, target=False):
        backend = get_backend()

        flat_actions = flatten_op(actions)
        actions = []
        for flat_key, action_component in self.policy.action_space.flatten().items():
            actions.append(flat_actions[flat_key])

        if backend == "tf":
            actions = tf.concat(actions, axis=-1)
        elif backend == "pytorch":
            actions = torch.cat(actions, dim=-1)

        q_funcs = self.q_functions if target is False else self.target_q_functions

        # We do not concat states yet because we might pass states through a conv stack before merging it
        # with actions.
        return tuple(q.call((preprocessed_states, actions)) for q in q_funcs)

    @rlgraph_api
    def get_losses(self, preprocessed_states, actions, rewards, terminals, next_states, importance_weights):
        # TODO: internal states
        samples_next = self.policy.get_action_and_log_likelihood(next_states, deterministic=False)
        next_sampled_actions = samples_next["action"]
        log_probs_next_sampled = samples_next["log_likelihood"]

        q_values_next_sampled = self.get_q_values(
            next_states, next_sampled_actions, target=True
        )
        q_values = self.get_q_values(preprocessed_states, actions)
        samples = self.policy.get_action_and_log_likelihood(preprocessed_states, deterministic=False)
        sampled_actions = samples["action"]
        log_probs_sampled = samples["log_likelihood"]
        q_values_sampled = self.get_q_values(preprocessed_states, sampled_actions)

        alpha = self._graph_fn_compute_alpha()

        return self.loss_function.loss(
            alpha,
            log_probs_next_sampled,
            q_values_next_sampled,
            q_values,
            log_probs_sampled,
            q_values_sampled,
            rewards,
            terminals
        )

    @rlgraph_api(requires_variable_completeness=True)
    def reset_targets(self):
        ops = (target_q.sync(q.variables()) for q, target_q in zip(self.q_functions, self.target_q_functions))
        return tuple(ops)

    @rlgraph_api(requires_variable_completeness=True)
    def sync_targets(self):
        should_sync = self._graph_fn_get_should_sync()
        return self._graph_fn_sync(should_sync)

    @rlgraph_api
    def get_memory_size(self):
        return self.memory.get_size()

    @graph_fn
    def _graph_fn_compute_alpha(self):
        backend = get_backend()
        if backend == "tf":
            return tf.exp(self.log_alpha)
        elif backend == "pytorch":
            return torch.exp(self.log_alpha)

    # TODO: Move this into generic AgentRootComponent.
    @graph_fn
    def _graph_fn_training_step(self, other_step_op=None):
        if self.agent is not None:
            add_op = tf.assign_add(self.agent.graph_executor.global_training_timestep, 1)
            op_list = [add_op] + [other_step_op] if other_step_op is not None else []
            with tf.control_dependencies(op_list):
                return tf.no_op() if other_step_op is None else other_step_op
        else:
            return tf.no_op() if other_step_op is None else other_step_op

    @graph_fn(returns=1, requires_variable_completeness=True)
    def _graph_fn_get_should_sync(self):
        if get_backend() == "tf":
            inc_op = tf.assign_add(self.steps_since_last_sync, 1)
            should_sync = inc_op >= self.q_function_sync_rules.sync_interval

            def reset_op():
                op = tf.assign(self.steps_since_last_sync, 0)
                with tf.control_dependencies([op]):
                    return tf.no_op()

            sync_op = tf.cond(
                pred=inc_op >= self.q_function_sync_rules.sync_interval,
                true_fn=reset_op,
                false_fn=tf.no_op
            )
            with tf.control_dependencies([sync_op]):
                return tf.identity(should_sync)
        else:
            raise NotImplementedError("TODO")

    # TODO: Move `should_sync` logic into Agent (python side?).
    @graph_fn(returns=1, requires_variable_completeness=True)
    def _graph_fn_sync(self, should_sync):
        assign_ops = []
        tau = self.q_function_sync_rules.sync_tau
        if tau != 1.0:
            all_source_vars = [source.get_variables(collections=None, custom_scope_separator="-") for source in self.q_functions]
            all_dest_vars = [destination.get_variables(collections=None, custom_scope_separator="-") for destination in self.target_q_functions]
            for source_vars, dest_vars in zip(all_source_vars, all_dest_vars):
                for (source_key, source_var), (dest_key, dest_var) in zip(sorted(source_vars.items()), sorted(dest_vars.items())):
                    assign_ops.append(tf.assign(dest_var, tau * source_var + (1.0 - tau) * dest_var))
        else:
            all_source_vars = [source.variables() for source in self.q_functions]
            for source_vars, destination in zip(all_source_vars, self.target_q_functions):
                assign_ops.append(destination.sync(source_vars))
        assert len(assign_ops) > 0
        grouped_op = tf.group(assign_ops)

        def assign_op():
            # Make sure we are returning no_op as opposed to reference
            with tf.control_dependencies([grouped_op]):
                return tf.no_op()

        cond_assign_op = tf.cond(should_sync, true_fn=assign_op, false_fn=tf.no_op)
        with tf.control_dependencies([cond_assign_op]):
            return tf.no_op()

    @graph_fn
    def _graph_fn_no_op(self):
        return tf.no_op()
