# Copyright 2018/2019 The Rlgraph Authors, All Rights Reserved.
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

import numpy as np
from rlgraph import get_backend
from rlgraph.components import Synchronizable, Memory, PrioritizedReplay
from rlgraph.components.algorithms.algorithm_component import AlgorithmComponent
from rlgraph.components.loss_functions.sac_loss_function import SACLossFunction
from rlgraph.components.neural_networks.value_function import ValueFunction
from rlgraph.spaces import IntBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import flatten_op, DataOpTuple

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class SyncSpecification(object):
    """Describes a synchronization schedule, used to update the target value weights. The target values are gradually
    updates using exponential moving average as suggested by the paper."""
    def __init__(self, sync_interval=None, sync_tau=None):
        """
        Arguments:
            sync_interval: How often to update the target.
            sync_tau: The smoothing constant to use in the averaging. Setting to 1 replaces the values each iteration.
        """
        self.sync_interval = sync_interval
        self.sync_tau = sync_tau


class SACAlgorithmComponent(AlgorithmComponent):
    def __init__(self, agent, memory_spec, q_function, initial_alpha=1.0, gumbel_softmax_temperature=1.0, target_entropy=None,
                 q_sync_spec=None, num_q_functions=2,
                 scope="sac-agent-component", **kwargs):

        # If VF spec is a network spec, wrap with SAC vf type. The VF must concatenate actions and states,
        # which can require splitting the network in the case of e.g. conv-inputs.
        value_function_spec = kwargs.pop("value_function_spec", None)
        if isinstance(value_function_spec, list):
            value_function_spec = dict(type="sac_value_function", network_spec=value_function_spec)
            self.logger.info("Using default SAC value function.")
        elif isinstance(value_function_spec, ValueFunction):
            self.logger.info("Using value function object {}".format(ValueFunction))

        policy_spec = kwargs.pop("policy_spec", None)
        # Force set deterministic to False.
        if policy_spec is not None:
            policy_spec["deterministic"] = False
        else:
            # Continuous action space: Use squashed normal.
            # Discrete: Gumbel-softmax.
            policy_spec = dict(deterministic=False, distributions_spec=dict(
                bounded_distribution_type="squashed", discrete_distribution_type="gumbel_softmax",
                gumbel_softmax_temperature=gumbel_softmax_temperature
            ))

        super(SACAlgorithmComponent, self).__init__(
            agent, policy_spec=policy_spec, value_function_spec=value_function_spec, scope=scope, **kwargs
        )

        if q_sync_spec is None:
            q_sync_spec = SyncSpecification(
                sync_interval=self.agent.update_spec["sync_interval"] // self.agent.update_spec["update_interval"],
                sync_tau=self.agent.update_spec["sync_tau"] if "sync_tau" in self.agent.update_spec else 5e-3
            )

        self.memory = Memory.from_spec(memory_spec)
        self.q_functions = [q_function] + \
                           [q_function.copy(scope="{}-{}".format(q_function.scope, i + 1), trainable=True)
                            for i in range(num_q_functions - 1)]

        # Set number of return values for get_q_values graph_fn.
        self.graph_fn_num_outputs["_graph_fn_get_q_values"] = num_q_functions

        for q in self.q_functions:
            # TODO: is there a better way to do this?
            if "synchronizable" not in q.sub_components:
                q.add_components(Synchronizable(), expose_apis="sync")
        self.target_q_functions = [q.copy(scope="target-" + q.scope, trainable=True) for q in self.q_functions]
        for target_q in self.target_q_functions:
            # TODO: is there a better way to do this?
            if "synchronizable" not in target_q.sub_components:
                target_q.add_components(Synchronizable(), expose_apis="sync")

        self.target_entropy = target_entropy
        self.alpha_optimizer = self.optimizer.copy(scope="alpha-" + self.optimizer.scope) if self.target_entropy is not None else None
        self.initial_alpha = initial_alpha
        self.log_alpha = None

        self.loss_function = SACLossFunction(
            target_entropy=target_entropy, discount=self.discount, num_q_functions=num_q_functions
        )

        self.steps_since_last_sync = None
        self.q_sync_spec = q_sync_spec
        self.env_action_space = None

        #q_names = ["q_{}".format(i) for i in range(len(self.q_functions))]
        #self._q_vars_merger = ContainerMerger(*q_names, scope="q_vars_merger")

        self.add_components(self.memory, self.loss_function, self.alpha_optimizer)  # self._merger, self._q_vars_merger)
        self.add_components(*self.q_functions)
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
        #memory_items = ["states", "actions", "rewards", "next_states", "terminals"]
        #self._merger = ContainerMerger(*memory_items)
        #records = self._merger.merge(preprocessed_states, env_actions, rewards, next_states, terminals)
        # Auto-merge via dict.
        records = dict(states=preprocessed_states, env_actions=env_actions, rewards=rewards, next_states=next_states,
                       terminals=terminals)
        return self.memory.insert_records(records)

    @rlgraph_api
    def update_from_memory(self, batch_size=64):
        records, sample_indices, importance_weights = self.memory.get_records(batch_size)
        result = self.update_from_external_batch(
            records["states"], records["actions"], records["rewards"], records["terminals"],
            records["next_states"], importance_weights
        )

        if isinstance(self.memory, PrioritizedReplay):
            update_pr_step_op = self.memory.update_records(sample_indices, result["critic_loss_per_item"])
            result["update_pr_step_op"] = update_pr_step_op

        return result

    @rlgraph_api
    def update_from_external_batch(self, preprocessed_states, env_actions, rewards, terminals, next_states, importance_weights):
        actions = self._graph_fn_one_hot(env_actions)
        actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item, alpha_loss, alpha_loss_per_item = \
            self.get_losses(preprocessed_states, actions, rewards, terminals, next_states, importance_weights)

        policy_vars = self.policy.variables()
        #q_vars = [q_func.variables() for q_func in self.q_functions]
        merged_q_vars = {"q_{}".format(i): q.variables() for i, q in enumerate(self.q_functions)}  #self._q_vars_merger.merge(*q_vars)
        critic_step_op, critic_loss, critic_loss_per_item = \
            self.value_function_optimizer.step(merged_q_vars, critic_loss, critic_loss_per_item)

        actor_step_op, actor_loss, actor_loss_per_item = \
            self.optimizer.step(policy_vars, actor_loss, actor_loss_per_item)

        if self.target_entropy is not None:
            alpha_step_op = self._graph_fn_update_alpha(alpha_loss, alpha_loss_per_item)
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
    def _graph_fn_update_alpha(self, alpha_loss, alpha_loss_per_item):
        alpha_step_op, _, _ = self.alpha_optimizer.step(
            DataOpTuple([self.log_alpha]), alpha_loss, alpha_loss_per_item)
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
        return tuple(q.state_action_value(preprocessed_states, actions) for q in q_funcs)

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
            should_sync = inc_op >= self.q_sync_spec.sync_interval

            def reset_op():
                op = tf.assign(self.steps_since_last_sync, 0)
                with tf.control_dependencies([op]):
                    return tf.no_op()

            sync_op = tf.cond(
                pred=inc_op >= self.q_sync_spec.sync_interval,
                true_fn=reset_op,
                false_fn=tf.no_op
            )
            with tf.control_dependencies([sync_op]):
                return tf.identity(should_sync)
        else:
            raise NotImplementedError("TODO")

    @graph_fn(returns=1, requires_variable_completeness=True)
    def _graph_fn_sync(self, should_sync):
        assign_ops = []
        tau = self.q_sync_spec.sync_tau
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

