from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from rlgraph import get_backend
from rlgraph.agents import Agent
from rlgraph.utils import RLGraphError
from rlgraph.spaces import FloatBox, BoolBox, IntBox
from rlgraph.components import Component, Synchronizable
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.components import Memory, DictMerger, ContainerSplitter, PrioritizedReplay, LossFunction
from rlgraph.utils.util import strip_list
from rlgraph.utils.ops import flatten_op, DataOpTuple


if get_backend() == "tf":
    import tensorflow as tf
    from rlgraph.utils import tf_util
if get_backend() == "pytorch":
    import torch


class SACLossFunction(LossFunction):
    def __init__(self, alpha=1.0, discount=0.99, scope="sac-loss-function", **kwargs):
        super(SACLossFunction, self).__init__(discount=discount, scope=scope, **kwargs)
        self.alpha = alpha

    @rlgraph_api
    def loss(self, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
             q_values_sampled, rewards, terminals):
        actor_loss_per_item, critic_loss_per_item = self.loss_per_item(
            log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
            q_values_sampled, rewards, terminals
        )
        actor_loss = self.loss_average(actor_loss_per_item)
        critic_loss = self.loss_average(critic_loss_per_item)
        return actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item

    @graph_fn
    def _graph_fn__critic_loss(self, log_probs_next_sampled, q_values_next_sampled, q_values, rewards, terminals):
        q_min_next = tf.reduce_min(tf.concat(q_values_next_sampled, axis=1), axis=1, keepdims=True)
        soft_state_value = q_min_next - self.alpha * log_probs_next_sampled
        q_target = rewards + self.discount * (1.0 - tf.cast(terminals, tf.float32)) * soft_state_value
        q_target = tf.stop_gradient(q_target)
        total_loss = 0.0
        for i, q_value in enumerate(q_values):
            loss = 0.5 * (q_value - q_target) ** 2
            loss = tf.identity(loss, "critic_loss_per_item_{}".format(i + 1))
            total_loss += loss
        total_loss = tf.identity(total_loss, "critic_loss_per_item")
        return total_loss

    @graph_fn
    def _graph_fn__actor_loss(self, log_probs_sampled, q_values_sampled):
        q_min = tf.reduce_min(tf.concat(q_values_sampled, axis=1), axis=1, keepdims=True)
        loss = self.alpha * log_probs_sampled - q_min
        loss = tf.identity(loss, "actor_loss_per_item")
        return loss

    @rlgraph_api
    def loss_per_item(self, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
                      q_values_sampled, rewards, terminals):
        policy_loss, value_loss = self._graph_fn_loss_per_item(
            log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled, q_values_sampled, rewards,
            terminals
        )
        return policy_loss, value_loss

    @graph_fn
    def _graph_fn_loss_per_item(self, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
                                q_values_sampled, rewards, terminals):
        assert log_probs_next_sampled.shape.as_list() == [None, 1]
        assert all(q.shape.as_list() == [None, 1] for q in q_values_next_sampled)
        assert all(q.shape.as_list() == [None, 1] for q in q_values)
        assert log_probs_sampled.shape.as_list() == [None, 1]
        assert all(q.shape.as_list() == [None, 1] for q in q_values_sampled)
        assert rewards.shape.as_list() == [None]
        assert terminals.shape.as_list() == [None]
        rewards = tf.expand_dims(rewards, axis=-1)
        terminals = tf.expand_dims(terminals, axis=-1)

        critic_loss_per_item = self._graph_fn__critic_loss(
            log_probs_next_sampled=log_probs_next_sampled,
            q_values_next_sampled=q_values_next_sampled,
            q_values=q_values,
            rewards=rewards,
            terminals=terminals
        )
        critic_loss_per_item = tf.squeeze(critic_loss_per_item, axis=1)

        actor_loss_per_item = self._graph_fn__actor_loss(
            log_probs_sampled=log_probs_sampled,
            q_values_sampled=q_values_sampled
        )
        actor_loss_per_item = tf.squeeze(actor_loss_per_item, axis=1)

        return actor_loss_per_item, critic_loss_per_item


class SyncSpecification(object):
    def __init__(self, sync_interval=None, sync_tau=None):
        self.sync_interval = sync_interval
        self.sync_tau = sync_tau


class SACAgentComponent(Component):
    def __init__(self, policy, q_functions, preprocessor, memory, discount,
                 alpha, optimizer, vf_optimizer, q_sync_spec):
        super(SACAgentComponent, self).__init__()
        self._policy = policy
        self._preprocessor = preprocessor
        self._memory = memory
        self._q_functions = q_functions
        self._target_q_functions = [q.copy(scope="target-" + q.scope, trainable=True) for q in q_functions]
        for target_q in self._target_q_functions:
            target_q.add_components(Synchronizable(), expose_apis="sync")
        self._optimizer = optimizer
        self.vf_optimizer = vf_optimizer
        self.loss_function = SACLossFunction(alpha=alpha, discount=discount)

        memory_items = ["states", "actions", "rewards", "next_states", "terminals"]
        self._merger = DictMerger(*memory_items)
        self._splitter = ContainerSplitter(*memory_items)

        q_names = ["q_{}".format(i) for i in range(len(self._q_functions))]
        self._q_vars_merger = DictMerger(*q_names, scope="q_vars_merger")
        self._q_vars_splitter = ContainerSplitter(*q_names, scope="q_vars_splitter")

        self.add_components(policy, preprocessor, memory, self._merger, self._splitter, self.loss_function,
                            optimizer, vf_optimizer, self._q_vars_merger, self._q_vars_splitter)
        self.add_components(*self._q_functions)
        self.add_components(*self._target_q_functions)

        self.steps_since_last_sync = None
        self.sync_interval = q_sync_spec.sync_interval

    @rlgraph_api
    def get_policy_weights(self):
        # TODO: why is _variables() "protected"?
        return self._policy._variables()

    @rlgraph_api
    def get_q_weights(self):
        merged_weights = self._q_vars_merger.merge(*[q._variables() for q in self._q_functions])
        return merged_weights

    @rlgraph_api(must_be_complete=False)
    def set_policy_weights(self, weights):
        return self._policy.sync(weights)

    @rlgraph_api(must_be_complete=False)
    def set_q_weights(self, weights):
        #return self._value_function.sync(vf_weights)
        return None

    @rlgraph_api
    def preprocess_states(self, states):
        return self._preprocessor.preprocess(states)

    @rlgraph_api
    def insert_records(self, preprocessed_states, actions, rewards, next_states, terminals):
        records = self._merger.merge(preprocessed_states, actions, rewards, next_states, terminals)
        return self._memory.insert_records(records)

    @rlgraph_api
    def update_from_memory(self, batch_size):
        records, sample_indices, importance_weights = self._memory.get_records(batch_size)
        preprocessed_s, actions, rewards, preprocessed_s_prime, terminals = self._splitter.split(records)

        actor_step_op, critic_step_op, actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item\
            = self.update_from_external_batch(
                preprocessed_s, actions, rewards, terminals, preprocessed_s_prime, importance_weights
            )

        ret = [actor_step_op, critic_step_op, actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item]
        if isinstance(self._memory, PrioritizedReplay):
            update_pr_step_op = self._memory.update_records(sample_indices, critic_loss_per_item)
            ret.append(update_pr_step_op)

        return tuple(ret)

    @rlgraph_api
    def update_from_external_batch(self, preprocessed_states, actions, rewards, terminals,
                                             preprocessed_s_prime, importance_weights):
        actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item = \
            self.get_losses(preprocessed_states, actions, rewards, terminals, preprocessed_s_prime, importance_weights)

        policy_vars = self._policy._variables()
        q_vars = [q_func._variables() for q_func in self._q_functions]
        merged_q_vars = self._q_vars_merger.merge(*q_vars)
        critic_step_op, critic_loss, critic_loss_per_item = \
            self.vf_optimizer.step(merged_q_vars, critic_loss, critic_loss_per_item)
        # TODO: every q-function needs to be optimized separately - we need multiple optimizers

        actor_step_op, actor_loss, actor_loss_per_item = \
            self._optimizer.step(policy_vars, actor_loss, actor_loss_per_item)

        #sync_op = self.sync_targets()

        return actor_step_op, critic_step_op, actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item

    def _compute_q_values(self, q_functions, states, actions):
        flat_actions = flatten_op(actions)
        state_actions = [states]
        for flat_key, action_component in self._policy.action_space.flatten().items():
            if isinstance(action_component, IntBox):
                state_actions.append(self._graph_fn__one_hot(flat_actions[flat_key]))
            else:
                state_actions.append(flat_actions[flat_key])
        state_actions = self._graph_fn__concat(*state_actions)
        return tuple(q.value_output(state_actions) for q in q_functions)

    @rlgraph_api
    def get_losses(self, preprocessed_states, actions, rewards, terminals, preprocessed_next_states, importance_weights):
        # TODO: internal states

        next_sampled_actions = self._policy.get_action(preprocessed_next_states, deterministic=False)["action"]
        log_probs_next_sampled = self._policy.get_action_log_probs(preprocessed_states, next_sampled_actions)[
            "action_log_probs"]
        q_values_next_sampled = self._compute_q_values(
            self._target_q_functions, preprocessed_next_states, next_sampled_actions
        )
        q_values = self._compute_q_values(self._q_functions, preprocessed_states, actions)

        sampled_actions = self._policy.get_action(preprocessed_states, deterministic=False)["action"]
        debug_data = self._policy.get_logits_parameters_log_probs(preprocessed_states)
        if sampled_actions.op is not None:
            with tf.control_dependencies([tf.print("debug:", sampled_actions.op, debug_data["parameters"].op)]):
                sampled_actions.op = tf.identity(sampled_actions.op)

        log_probs_sampled = self._policy.get_action_log_probs(preprocessed_states, sampled_actions)["action_log_probs"]
        q_values_sampled = self._compute_q_values(self._q_functions, preprocessed_states, sampled_actions)

        total_policy_loss, policy_loss_per_item, total_values_loss, values_loss_per_item = self.loss_function.loss(
            log_probs_next_sampled,
            q_values_next_sampled,
            q_values,
            log_probs_sampled,
            q_values_sampled,
            rewards,
            terminals
        )
        return total_policy_loss, policy_loss_per_item, total_values_loss, values_loss_per_item

    @rlgraph_api
    def get_preprocessed_state_and_action(self, states, deterministic=False):
        preprocessed_states = self._preprocessor.preprocess(states)
        return self.action_from_preprocessed_state(preprocessed_states, deterministic)

    @rlgraph_api
    def action_from_preprocessed_state(self, preprocessed_states, deterministic=False):
        out = self._policy.get_action(preprocessed_states, deterministic=deterministic)
        return preprocessed_states, out["action"]

    @rlgraph_api
    def sync_target_qnet(self):
        pass

    @graph_fn(returns=1)
    def _graph_fn__concat(self, *tensors):
        backend = get_backend()
        if backend == "tf":
            return tf.concat([tf_util.ensure_batched(t) for t in tensors], axis=1)
        elif backend == "pytorch":
            raise NotImplementedError("TODO: pytorch support")

    @graph_fn
    def _graph_fn__one_hot(self, tensor):
        backend = get_backend()
        if backend == "tf":
            return tf.one_hot(tensor, depth=5)
        elif backend == "pytorch":
            raise NotImplementedError("TODO: pytorch support")

    """
    @rlgraph_api(must_be_complete=False)
    def sync_targets(self):
        should_sync = self._graph_fn_get__should_sync()
        all_source_vars = [source._variables() for source in self._q_functions]
        assign_ops = []
        for source_vars, destination in zip(all_source_vars, self._target_q_functions):
            assign_ops.append(destination.sync(source_vars))
        grouped_op = self._graph_fn__group(*assign_ops)
        return self._graph_fn__sync(should_sync, grouped_op)

    @graph_fn(returns=1)
    def _graph_fn_get__should_sync(self):
        if get_backend() == "tf":
            inc_op = tf.assign_add(self.steps_since_last_sync, 1)
            should_sync = inc_op >= self.sync_interval

            def reset_op():
                op = tf.assign(self.steps_since_last_sync, 0)
                with tf.control_dependencies([op]):
                    return tf.no_op()

            sync_op = tf.cond(
                pred=inc_op >= self.sync_interval,
                true_fn=reset_op,
                false_fn=tf.no_op
            )
            with tf.control_dependencies([sync_op]):
                return tf.identity(should_sync)
        else:
            raise NotImplementedError("TODO")

    @graph_fn(returns=1)
    def _graph_fn__sync(self, should_sync, grouped_op):
        def assign_op():
            # Make sure we are returning no_op as opposed to reference
            with tf.control_dependencies([grouped_op, tf.print("syncing")]):
                return tf.no_op()

        cond_assign_op = tf.cond(should_sync, true_fn=assign_op, false_fn=tf.no_op)
        with tf.control_dependencies([cond_assign_op]):
            return tf.no_op()

    @graph_fn(returns=1)
    def _graph_fn__group(self, *ops):
        return tf.group(*ops)
    """

    def create_variables(self, input_spaces, action_space=None):
        self.steps_since_last_sync = self.get_variable("steps_since_last_sync", dtype="int", initializer=0)


class SACAgent(Agent):
    def __init__(self, double_q=True, alpha=1.0, memory_spec=None, **kwargs):
        """
        This is an implementation of the Soft-Actor Critic algorithm.

        Paper: http://arxiv.org/abs/1801.01290

        Args:
            double_q (bool): Whether to train two q networks independently.
            alpha (float): "The temperature parameter α determines the relative importance of the entropy term against the reward".
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
            update_spec (dict): Here we can have sync_interval or sync_tau (for the value network update).
        """
        super(SACAgent, self).__init__(
            policy_spec=dict(deterministic=False),
            name=kwargs.pop("name", "sac-agent"),
            **kwargs
        )

        self.double_q = double_q
        self.alpha = alpha

        # Assert that the synch interval is a multiple of the update_interval.
        if "sync_interval" in self.update_spec:
            if self.update_spec["sync_interval"] / self.update_spec["update_interval"] != \
                    self.update_spec["sync_interval"] // self.update_spec["update_interval"]:
                raise RLGraphError(
                    "ERROR: sync_interval ({}) must be multiple of update_interval "
                    "({})!".format(self.update_spec["sync_interval"], self.update_spec["update_interval"])
                )
        elif "sync_tau" in self.update_spec:
            if self.update_spec["sync_tau"] <= 0 or self.update_spec["sync_tau"] > 1.0:
                raise RLGraphError(
                    "sync_tau ({}) must be in interval (0.0, 1.0]!".format(self.update_spec["sync_tau"])
                )
        else:
            self.update_spec["sync_tau"] = 0.005  # The value mentioned in the paper

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)

        self.iterations = self.update_spec["num_iterations"]
        self.sample_size = self.update_spec["sample_size"]
        self.batch_size = self.update_spec["batch_size"]

        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            time_step=int,
            preprocessed_states=preprocessed_state_space,
            rewards=reward_space,
            terminals=terminal_space,
            next_states=self.state_space.with_batch_rank(),
            preprocessed_next_states=preprocessed_state_space
        ))

        self.memory = Memory.from_spec(memory_spec)
        self.root_component = SACAgentComponent(
            policy=self.policy,
            value_function=self.value_function,
            target_value_function=self.target_value_function,
            q_functions=[],
            preprocessor=self.preprocessor,
            memory=self.memory,
            discount=self.discount,
            alpha=self.alpha,
            optimizer=self.optimizer
        )

    def define_graph_api(self, *args, **kwargs):
        pass

    def set_weights(self, weights):
        # TODO: Overrides parent but should this be policy of value function?
        return self.graph_executor.execute((self.root_component.set_policy_weights, weights))

    def get_weights(self):
        return self.graph_executor.execute(self.root_component.get_policy_weights)

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None):
        # TODO: common pattern - move to Agent
        """
        Args:
            extra_returns (Optional[Set[str],str]): Optional string or set of strings for additional return
                values (besides the actions). Possible values are:
                - 'preprocessed_states': The preprocessed states after passing the given states through the
                preprocessor stack.
                - 'internal_states': The internal states returned by the RNNs in the NN pipeline.
                - 'used_exploration': Whether epsilon- or noise-based exploration was used or not.

        Returns:
            tuple or single value depending on `extra_returns`:
                - action
                - the preprocessed states
        """
        extra_returns = {extra_returns} if isinstance(extra_returns, str) else (extra_returns or set())
        # States come in without preprocessing -> use state space.
        if apply_preprocessing:
            call_method = self.root_component.get_preprocessed_state_and_action
            batched_states = self.state_space.force_batch(states)
        else:
            call_method = self.root_component.action_from_preprocessed_state
            batched_states = states
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1

        # Increase timesteps by the batch size (number of states in batch).
        batch_size = len(batched_states)
        self.timesteps += batch_size

        # Control, which return value to "pull" (depending on `additional_returns`).
        return_ops = [1, 0] if "preprocessed_states" in extra_returns else [1]
        ret = self.graph_executor.execute((
            call_method,
            [batched_states, not use_exploration],  # deterministic = not use_exploration
            # 0=preprocessed_states, 1=action
            return_ops
        ))
        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        self.graph_executor.execute((self.root_component.insert_records, [preprocessed_states, actions, rewards, next_states, terminals]))

    def update(self, batch=None):
        # [0] = policy_loss; [1] = policy_loss_per_item, [2] = values_loss, [3] = values_loss_per_item
        return_ops = [0, 1, 2, 3]
        if batch is None:
            ret = self.graph_executor.execute((self.root_component.update_from_memory, None, return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]
        else:
            # No sequence indices means terminals are used in place.
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"], batch["next_states"]]
            ret = self.graph_executor.execute((self.root_component.update_from_external_batch, batch_input, return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_external_batch"]

        # [0] loss, [1] loss per item
        return ret[0], ret[1]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.preprocessing_required and len(self.preprocessor.variables) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def __repr__(self):
        return "SACAgent(double_q={}, alpha={})".format(self.double_q, self.alpha)
