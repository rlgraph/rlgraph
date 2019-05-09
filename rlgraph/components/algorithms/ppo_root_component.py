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
from rlgraph.components.algorithms.algorithm_component import AlgorithmComponent
from rlgraph.components.common.container_merger import ContainerMerger
from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.components.loss_functions.ppo_loss_function import PPOLossFunction
from rlgraph.components.memories import Memory, RingBuffer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.define_by_run_ops import define_by_run_flatten
from rlgraph.utils.ops import flatten_op, unflatten_op, DataOpDict, ContainerDataOp, FlattenedDataOp

if get_backend() == "tf":
    import tensorflow as tf
if get_backend() == "pytorch":
    import torch


class PPOAgentComponent(AlgorithmComponent):
    def __init__(self, agent,
                 memory_spec, gae_lambda, clip_rewards, clip_ratio, value_function_clipping, weight_entropy,
                 scope="ppo-agent-component", **kwargs):
        super(PPOAgentComponent, self).__init__(agent, scope=scope, **kwargs)

        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer), "ERROR: PPO memory must be ring-buffer for episode-handling!"

        # Make sure the python buffer is not larger than our memory capacity.
        assert self.agent.observe_spec["buffer_size"] <= self.memory.capacity, \
            "ERROR: Buffer's size ({}) in `observe_spec` must be smaller or equal to the memory's capacity ({})!". \
            format(self.agent.observe_spec["buffer_size"], self.memory.capacity)

        # The merger to merge inputs into one record Dict going into the memory.
        self.merger = ContainerMerger("states", "actions", "rewards", "terminals")

        self.gae_function = GeneralizedAdvantageEstimation(
            gae_lambda=gae_lambda, discount=self.agent.discount, clip_rewards=clip_rewards
        )
        self.loss_function = PPOLossFunction(
            clip_ratio=clip_ratio, value_function_clipping=value_function_clipping, weight_entropy=weight_entropy
        )

        # Add the Agent's components.
        self.add_components(
            self.agent.preprocessor, self.agent.policy, self.agent.value_function, self.agent.exploration,
            self.agent.optimizer, self.agent.value_function_optimizer,
            self.agent.vars_merger, self.agent.vars_splitter
        )

        # Add our self-created ones.
        self.add_components(self.memory, self.merger, self.gae_function, self.loss_function)

        # TODO: For now, use `self.agent` everywhere.
        # TODO: But try to move as many sub-components' references (e.g. self.agent.policy) directly here into this one.
        # TODO: self.agent should only be used for non-default settings (e.g. `self.agent.update_spec`).

    @rlgraph_api
    def reset_preprocessor(self):
        reset_op = self.agent.preprocessor.reset()
        return reset_op

    # Act from preprocessed states.
    @rlgraph_api
    def action_from_preprocessed_state(self, preprocessed_states, deterministic=False):
        out = self.agent.policy.get_action(preprocessed_states, deterministic=deterministic)
        return out["action"], preprocessed_states

    # State (from environment) to action with preprocessing.
    @rlgraph_api
    def get_preprocessed_state_and_action(self, states, deterministic=False):
        preprocessed_states = self.agent.preprocessor.preprocess(states)
        return self.action_from_preprocessed_state(preprocessed_states, deterministic)

    # Insert into memory.
    @rlgraph_api
    def insert_records(self, preprocessed_states, actions, rewards, terminals):
        records = self.agent.merger.merge(preprocessed_states, actions, rewards, terminals)
        return self.agent.memory.insert_records(records)

    @rlgraph_api
    def post_process(self, preprocessed_states, rewards, terminals, sequence_indices):
        baseline_values = self.agent.value_function.value_output(preprocessed_states)
        pg_advantages = self.agent.gae_function.calc_gae_values(baseline_values, rewards, terminals, sequence_indices)
        return pg_advantages

    # Learn from memory.
    @rlgraph_api
    def update_from_memory(self, apply_postprocessing):
        if self.agent.sample_episodes:
            records = self.agent.memory.get_episodes(self.agent.update_spec["batch_size"])
        else:
            records = self.agent.memory.get_records(self.agent.update_spec["batch_size"])

        # Route to post process and update method.
        # Use terminals as sequence indices.
        sequence_indices = records["terminals"]
        return self.update_from_external_batch(
            records["states"], records["actions"], records["rewards"], records["terminals"],
            sequence_indices, apply_postprocessing
        )

    # N.b. this is here because the iterative_optimization would need policy/losses as sub-components, but
    # multiple parents are not allowed currently.
    @rlgraph_api
    def _graph_fn_update_from_external_batch(
            self, preprocessed_states, actions, rewards, terminals, sequence_indices, apply_postprocessing
    ):
        """
        Calls iterative optimization by repeatedly sub-sampling.
        """
        multi_gpu_sync_optimizer = self.sub_components.get("multi-gpu-synchronizer")

        # Return values.
        loss, loss_per_item, vf_loss, vf_loss_per_item = None, None, None, None

        policy = self.get_sub_component_by_name(self.agent.policy.scope)
        value_function = self.get_sub_component_by_name(self.agent.value_function.scope)
        optimizer = self.get_sub_component_by_name(self.agent.optimizer.scope)
        loss_function = self.get_sub_component_by_name(self.agent.loss_function.scope)
        value_function_optimizer = self.get_sub_component_by_name(self.agent.value_function_optimizer.scope)
        vars_merger = self.get_sub_component_by_name(self.agent.vars_merger.scope)
        gae_function = self.get_sub_component_by_name(self.agent.gae_function.scope)
        prev_log_probs = policy.get_log_likelihood(preprocessed_states, actions)["log_likelihood"]

        if get_backend() == "tf":
            # Log probs before update.
            prev_log_probs = tf.stop_gradient(prev_log_probs)
            batch_size = tf.shape(preprocessed_states)[0]
            prior_baseline_values = tf.stop_gradient(value_function.value_output(preprocessed_states))

            # Advantages are based on prior baseline values.
            advantages = tf.cond(
                pred=apply_postprocessing,
                true_fn=lambda: gae_function.calc_gae_values(
                    prior_baseline_values, rewards, terminals, sequence_indices),
                false_fn=lambda: rewards
            )

            if self.agent.standardize_advantages:
                mean, std = tf.nn.moments(x=advantages, axes=[0])
                advantages = (advantages - mean) / std

            def opt_body(index_, loss_, loss_per_item_, vf_loss_, vf_loss_per_item_):
                start = tf.random_uniform(shape=(), minval=0, maxval=batch_size - 1, dtype=tf.int32)
                indices = tf.range(start=start, limit=start + self.agent.sample_size) % batch_size
                sample_states = tf.gather(params=preprocessed_states, indices=indices)
                if isinstance(actions, ContainerDataOp):
                    sample_actions = FlattenedDataOp()
                    for name, action in flatten_op(actions).items():
                        sample_actions[name] = tf.gather(params=action, indices=indices)
                    sample_actions = unflatten_op(sample_actions)
                else:
                    sample_actions = tf.gather(params=actions, indices=indices)

                sample_prior_log_probs = tf.gather(params=prev_log_probs, indices=indices)
                sample_rewards = tf.gather(params=rewards, indices=indices)
                sample_terminals = tf.gather(params=terminals, indices=indices)
                sample_sequence_indices = tf.gather(params=sequence_indices, indices=indices)
                sample_advantages = tf.gather(params=advantages, indices=indices)
                sample_advantages.set_shape((self.agent.sample_size, ))

                sample_baseline_values = value_function.value_output(sample_states)
                sample_prior_baseline_values = tf.gather(params=prior_baseline_values, indices=indices)

                # If we are a multi-GPU root:
                # Simply feeds everything into the multi-GPU sync optimizer's method and return.
                if multi_gpu_sync_optimizer is not None:
                    main_policy_vars = self.agent.policy.variables()
                    main_vf_vars = self.agent.value_function.variables()
                    all_vars = self.agent.vars_merger.merge(main_policy_vars, main_vf_vars)
                    # grads_and_vars, loss, loss_per_item, vf_loss, vf_loss_per_item = \
                    out = multi_gpu_sync_optimizer.calculate_update_from_external_batch(
                        all_vars,
                        sample_states, sample_actions, sample_rewards, sample_terminals, sample_sequence_indices,
                        apply_postprocessing=apply_postprocessing
                    )
                    avg_grads_and_vars_policy, avg_grads_and_vars_vf = self.agent.vars_splitter.call(
                        out["avg_grads_and_vars_by_component"]
                    )
                    policy_step_op = self.agent.optimizer.apply_gradients(avg_grads_and_vars_policy)
                    vf_step_op = self.agent.value_function_optimizer.apply_gradients(avg_grads_and_vars_vf)
                    step_op = self._graph_fn_group(policy_step_op, vf_step_op)
                    step_and_sync_op = multi_gpu_sync_optimizer.sync_variables_to_towers(
                        step_op, all_vars
                    )
                    loss_vf, loss_per_item_vf = out["additional_return_0"], out["additional_return_1"]

                    # Have to set all shapes here due to strict loop-var shape requirements.
                    out["loss"].set_shape(())
                    loss_vf.set_shape(())
                    loss_per_item_vf.set_shape((self.agent.sample_size,))
                    out["loss_per_item"].set_shape((self.agent.sample_size,))

                    with tf.control_dependencies([step_and_sync_op]):
                        if index_ == 0:
                            # Increase the global training step counter.
                            out["loss"] = self._graph_fn_training_step(out["loss"])
                        return index_ + 1, out["loss"], out["loss_per_item"], loss_vf, loss_per_item_vf

                policy_probs = policy.get_log_likelihood(sample_states, sample_actions)["log_likelihood"]
                baseline_values = value_function.value_output(tf.stop_gradient(sample_states))
                sample_rewards = tf.cond(
                    pred=apply_postprocessing,
                    true_fn=lambda: gae_function.calc_gae_values(
                        baseline_values, sample_rewards, sample_terminals, sample_sequence_indices),
                    false_fn=lambda: sample_rewards
                )
                sample_rewards.set_shape((self.agent.sample_size, ))
                entropy = policy.get_entropy(sample_states)["entropy"]

                loss, loss_per_item, vf_loss, vf_loss_per_item = \
                    loss_function.loss(
                        policy_probs, sample_prior_log_probs,
                        sample_baseline_values, sample_prior_baseline_values, sample_advantages, entropy
                    )

                if hasattr(self, "is_multi_gpu_tower") and self.is_multi_gpu_tower is True:
                    policy_grads_and_vars = optimizer.calculate_gradients(policy.variables(), loss)
                    vf_grads_and_vars = value_function_optimizer.calculate_gradients(
                        value_function.variables(), vf_loss
                    )
                    grads_and_vars_by_component = vars_merger.merge(policy_grads_and_vars, vf_grads_and_vars)
                    return grads_and_vars_by_component, loss, loss_per_item, vf_loss, vf_loss_per_item
                else:
                    step_op, loss, loss_per_item = optimizer.step(
                        policy.variables(), loss, loss_per_item
                    )
                    loss.set_shape(())
                    loss_per_item.set_shape((self.agent.sample_size,))

                    vf_step_op, vf_loss, vf_loss_per_item = value_function_optimizer.step(
                        value_function.variables(), vf_loss, vf_loss_per_item
                    )
                    vf_loss.set_shape(())
                    vf_loss_per_item.set_shape((self.agent.sample_size,))

                    with tf.control_dependencies([step_op, vf_step_op]):
                        return index_ + 1, loss, loss_per_item, vf_loss, vf_loss_per_item

            def cond(index_, loss_, loss_per_item_, v_loss_, v_loss_per_item_):
                return index_ < self.agent.iterations

            init_loop_vars = [
                0,
                tf.zeros(shape=(), dtype=tf.float32),
                tf.zeros(shape=(self.agent.sample_size,)),
                tf.zeros(shape=(), dtype=tf.float32),
                tf.zeros(shape=(self.agent.sample_size,))
            ]

            if hasattr(self, "is_multi_gpu_tower") and self.is_multi_gpu_tower is True:
                return opt_body(*init_loop_vars)
            else:
                index, loss, loss_per_item, vf_loss, vf_loss_per_item = tf.while_loop(
                    cond=cond,
                    body=opt_body,
                    loop_vars=init_loop_vars,
                    parallel_iterations=1
                )
                # Increase the global training step counter.
                loss = self._graph_fn_training_step(loss)
                return loss, loss_per_item, vf_loss, vf_loss_per_item

        elif get_backend() == "pytorch":
            if isinstance(prev_log_probs, dict):
                for name in actions.keys():
                    prev_log_probs[name] = prev_log_probs[name].detach()
            else:
                prev_log_probs = prev_log_probs.detach()
            batch_size = preprocessed_states.shape[0]
            sample_size = min(batch_size, self.agent.sample_size)
            prior_baseline_values = value_function.value_output(preprocessed_states).detach()
            if apply_postprocessing:
                advantages = gae_function.calc_gae_values(
                    prior_baseline_values, rewards, terminals, sequence_indices)
            else:
                advantages = rewards
            if self.agent.standardize_advantages:
                advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

            for _ in range(self.agent.iterations):
                start = int(torch.rand(1) * (batch_size - 1))
                indices = torch.arange(start=start, end=start + sample_size, dtype=torch.long) % batch_size
                sample_states = torch.index_select(preprocessed_states, 0, indices)

                if isinstance(actions, dict):
                    sample_actions = DataOpDict()
                    sample_prior_log_probs = DataOpDict()
                    for name, action in define_by_run_flatten(actions, scope_separator_at_start=False).items():
                        sample_actions[name] = torch.index_select(action, 0, indices)
                        sample_prior_log_probs[name] = torch.index_select(prev_log_probs[name], 0, indices)
                else:
                    sample_actions = torch.index_select(actions, 0, indices)
                    sample_prior_log_probs = torch.index_select(prev_log_probs, 0, indices)

                sample_advantages = torch.index_select(advantages, 0, indices)
                sample_prior_baseline_values = torch.index_select(prior_baseline_values, 0, indices)

                policy_probs = policy.get_log_likelihood(sample_states, sample_actions)["log_likelihood"]
                sample_baseline_values = value_function.value_output(sample_states)

                entropy = policy.get_entropy(sample_states)["entropy"]
                loss, loss_per_item, vf_loss, vf_loss_per_item = loss_function.loss(
                    policy_probs, sample_prior_log_probs,
                    sample_baseline_values,  sample_prior_baseline_values, sample_advantages, entropy
                )

                # Do not need step op.
                _, loss, loss_per_item = optimizer.step(policy.variables(), loss, loss_per_item)
                _, vf_loss, vf_loss_per_item = \
                    value_function_optimizer.step(value_function.variables(), vf_loss, vf_loss_per_item)
            return loss, loss_per_item, vf_loss, vf_loss_per_item

