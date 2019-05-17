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


class PPOAlgorithmComponent(AlgorithmComponent):
    def __init__(self, agent, memory_spec=None, gae_lambda=1.0, clip_rewards=0.0, clip_ratio=0.2,
                 value_function_clipping=None, weight_entropy=None, sample_episodes=True, standardize_advantages=False,
                 batch_size=128, sample_size=32, num_iterations=10,
                 scope="ppo-agent-component", **kwargs):
        """
        Args:
            sample_episodes (bool): If True, the update method interprets the batch_size as the number of
                episodes to fetch from the memory. If False, batch_size will refer to the number of time-steps. This
                is especially relevant for environments where episode lengths may vastly differ throughout training. For
                example, in CartPole, a losing episode is typically 10 steps, and a winning episode 200 steps.

            standardize_advantages (bool): If true, standardize advantage values in update.
        """
        policy_spec = kwargs.pop("policy_spec", None)
        if policy_spec is not None:
            policy_spec["deterministic"] = False
        else:
            policy_spec = dict(deterministic=False)

        super(PPOAlgorithmComponent, self).__init__(agent, policy_spec=policy_spec, scope=scope, **kwargs)

        self.sample_episodes = sample_episodes
        self.standardize_advantages = standardize_advantages
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_iterations = num_iterations

        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer), "ERROR: PPO memory must be ring-buffer for episode-handling!"

        # Make sure the python buffer is not larger than our memory capacity.
        assert self.agent.observe_spec["buffer_size"] <= self.memory.capacity, \
            "ERROR: Buffer's size ({}) in `observe_spec` must be smaller or equal to the memory's capacity ({})!". \
            format(self.agent.observe_spec["buffer_size"], self.memory.capacity)

        self.gae_function = GeneralizedAdvantageEstimation(
            gae_lambda=gae_lambda, discount=self.discount, clip_rewards=clip_rewards
        )
        self.loss_function = PPOLossFunction(
            clip_ratio=clip_ratio, value_function_clipping=value_function_clipping, weight_entropy=weight_entropy
        )

        # Add our self-created ones.
        self.add_components(self.memory, self.gae_function, self.loss_function)

    # Insert into memory.
    @rlgraph_api
    def insert_records(self, preprocessed_states, actions, rewards, terminals, sequence_indices):
        records = dict(
            states=preprocessed_states, actions=actions, rewards=rewards,
            terminals=terminals, sequence_indices=sequence_indices
        )
        return self.memory.insert_records(records)

    @rlgraph_api
    def post_process(self, preprocessed_states, rewards, terminals, sequence_indices):
        state_values = self.get_state_values(preprocessed_states)
        advantages = self.gae_function.calc_gae_values(state_values, rewards, terminals, sequence_indices)
        return advantages

    # Learn from memory.
    @rlgraph_api
    def update_from_memory(self, apply_postprocessing=True):
        if self.sample_episodes:
            records = self.memory.get_episodes(self.batch_size)
        else:
            records = self.memory.get_records(self.batch_size)

        # Route to post process and update method.
        return self.update_from_external_batch(
            records["states"], records["actions"], records["rewards"], records["terminals"],
            records["sequence_indices"], apply_postprocessing
        )

    # Retrieve some records from memory.
    @rlgraph_api
    def get_records(self, num_records=1):
        return self.memory.get_records(num_records)

    # N.b. this is here because the iterative_optimization would need policy/losses as sub-components, but
    # multiple parents are not allowed currently.
    @rlgraph_api
    def _graph_fn_update_from_external_batch(
            self, preprocessed_states, actions, rewards, terminals, sequence_indices, apply_postprocessing=True
    ):
        """
        Calls iterative optimization by repeatedly sub-sampling.
        """
        multi_gpu_sync_optimizer = self.sub_components.get("multi-gpu-synchronizer")

        # Return values.
        loss, loss_per_item, vf_loss, vf_loss_per_item = None, None, None, None

        prev_log_probs = self.policy.get_log_likelihood(preprocessed_states, actions)["log_likelihood"]
        prev_state_values = self.value_function.value_output(preprocessed_states)

        if get_backend() == "tf":
            batch_size = tf.shape(preprocessed_states)[0]

            # Log probs before update (stop-gradient as these are used in target term).
            prev_log_probs = tf.stop_gradient(prev_log_probs)
            # State values before update (stop-gradient as these are used in target term).
            prev_state_values = tf.stop_gradient(prev_state_values)

            # Advantages are based on previous state values.
            advantages = tf.cond(
                pred=apply_postprocessing,
                true_fn=lambda: self.gae_function.calc_gae_values(
                    prev_state_values, rewards, terminals, sequence_indices
                ),
                false_fn=lambda: rewards
            )
            if self.standardize_advantages:
                mean, std = tf.nn.moments(x=advantages, axes=[0])
                advantages = (advantages - mean) / std

            def opt_body(index_, loss_, loss_per_item_, vf_loss_, vf_loss_per_item_):
                start = tf.random_uniform(shape=(), minval=0, maxval=batch_size - 1, dtype=tf.int32)
                indices = tf.range(start=start, limit=start + self.sample_size) % batch_size
                sample_states = tf.gather(params=preprocessed_states, indices=indices)
                if isinstance(actions, ContainerDataOp):
                    sample_actions = FlattenedDataOp()
                    for name, action in flatten_op(actions).items():
                        sample_actions[name] = tf.gather(params=action, indices=indices)
                    sample_actions = unflatten_op(sample_actions)
                else:
                    sample_actions = tf.gather(params=actions, indices=indices)

                sample_prev_log_probs = tf.gather(params=prev_log_probs, indices=indices)
                sample_rewards = tf.gather(params=rewards, indices=indices)
                sample_terminals = tf.gather(params=terminals, indices=indices)
                sample_sequence_indices = tf.gather(params=sequence_indices, indices=indices)
                sample_advantages = tf.gather(params=advantages, indices=indices)
                sample_advantages.set_shape((self.sample_size,))

                sample_state_values = self.value_function.value_output(sample_states)
                sample_prev_state_values = tf.gather(params=prev_state_values, indices=indices)

                # If we are a multi-GPU root:
                # Simply feeds everything into the multi-GPU sync optimizer's method and return.
                if multi_gpu_sync_optimizer is not None:
                    main_policy_vars = self.policy.variables()
                    main_vf_vars = self.value_function.variables()
                    all_vars = self.vars_merger.merge(main_policy_vars, main_vf_vars)
                    # grads_and_vars, loss, loss_per_item, vf_loss, vf_loss_per_item = \
                    out = multi_gpu_sync_optimizer.calculate_update_from_external_batch(
                        all_vars,
                        sample_states, sample_actions, sample_rewards, sample_terminals, sample_sequence_indices,
                        apply_postprocessing=apply_postprocessing
                    )
                    avg_grads_and_vars_policy, avg_grads_and_vars_vf = self.vars_splitter.call(
                        out["avg_grads_and_vars_by_component"]
                    )
                    policy_step_op = self.optimizer.apply_gradients(avg_grads_and_vars_policy)
                    vf_step_op = self.value_function_optimizer.apply_gradients(avg_grads_and_vars_vf)
                    step_op = self._graph_fn_group(policy_step_op, vf_step_op)
                    step_and_sync_op = multi_gpu_sync_optimizer.sync_variables_to_towers(
                        step_op, all_vars
                    )
                    loss_vf, loss_per_item_vf = out["additional_return_0"], out["additional_return_1"]

                    # Have to set all shapes here due to strict loop-var shape requirements.
                    out["loss"].set_shape(())
                    loss_vf.set_shape(())
                    loss_per_item_vf.set_shape((self.sample_size,))
                    out["loss_per_item"].set_shape((self.sample_size,))

                    with tf.control_dependencies([step_and_sync_op]):
                        if index_ == 0:
                            # Increase the global training step counter.
                            out["loss"] = self._graph_fn_training_step(out["loss"])
                        return index_ + 1, out["loss"], out["loss_per_item"], loss_vf, loss_per_item_vf

                sample_log_probs = self.policy.get_log_likelihood(sample_states, sample_actions)["log_likelihood"]

                entropy = self.policy.get_entropy(sample_states)["entropy"]

                loss, loss_per_item, vf_loss, vf_loss_per_item = \
                    self.loss_function.loss(
                        sample_log_probs, sample_prev_log_probs,
                        sample_state_values, sample_prev_state_values, sample_advantages, entropy
                    )

                if hasattr(self, "is_multi_gpu_tower") and self.is_multi_gpu_tower is True:
                    policy_grads_and_vars = self.optimizer.calculate_gradients(self.policy.variables(), loss)
                    vf_grads_and_vars = self.value_function_optimizer.calculate_gradients(
                        self.value_function.variables(), vf_loss
                    )
                    grads_and_vars_by_component = self.vars_merger.merge(policy_grads_and_vars, vf_grads_and_vars)
                    return grads_and_vars_by_component, loss, loss_per_item, vf_loss, vf_loss_per_item
                else:
                    step_op, loss, loss_per_item = self.optimizer.step(
                        self.policy.variables(), loss, loss_per_item
                    )
                    loss.set_shape(())
                    loss_per_item.set_shape((self.sample_size,))

                    vf_step_op, vf_loss, vf_loss_per_item = self.value_function_optimizer.step(
                        self.value_function.variables(), vf_loss, vf_loss_per_item
                    )
                    vf_loss.set_shape(())
                    vf_loss_per_item.set_shape((self.sample_size,))

                    with tf.control_dependencies([step_op, vf_step_op]):
                        return index_ + 1, loss, loss_per_item, vf_loss, vf_loss_per_item

            def cond(index_, loss_, loss_per_item_, v_loss_, v_loss_per_item_):
                return index_ < self.num_iterations

            init_loop_vars = [
                0,
                tf.zeros(shape=(), dtype=tf.float32),
                tf.zeros(shape=(self.sample_size,)),
                tf.zeros(shape=(), dtype=tf.float32),
                tf.zeros(shape=(self.sample_size,))
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
            batch_size = preprocessed_states.shape[0]
            sample_size = min(batch_size, self.sample_size)

            if isinstance(prev_log_probs, dict):
                for name in actions.keys():
                    prev_log_probs[name] = prev_log_probs[name].detach()
            else:
                prev_log_probs = prev_log_probs.detach()
            prev_state_values = self.value_function.value_output(preprocessed_states).detach()
            if apply_postprocessing:
                advantages = self.gae_function.calc_gae_values(prev_state_values, rewards, terminals, sequence_indices)
            else:
                advantages = rewards
            if self.standardize_advantages:
                advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

            for _ in range(self.num_iterations):
                start = int(torch.rand(1) * (batch_size - 1))
                indices = torch.arange(start=start, end=start + sample_size, dtype=torch.long) % batch_size
                sample_states = torch.index_select(preprocessed_states, 0, indices)

                if isinstance(actions, dict):
                    sample_actions = DataOpDict()
                    sample_prev_log_probs = DataOpDict()
                    for name, action in define_by_run_flatten(actions, scope_separator_at_start=False).items():
                        sample_actions[name] = torch.index_select(action, 0, indices)
                        sample_prev_log_probs[name] = torch.index_select(prev_log_probs[name], 0, indices)
                else:
                    sample_actions = torch.index_select(actions, 0, indices)
                    sample_prev_log_probs = torch.index_select(prev_log_probs, 0, indices)

                sample_advantages = torch.index_select(advantages, 0, indices)
                sample_prev_state_values = torch.index_select(prev_state_values, 0, indices)

                sample_log_probs = self.policy.get_log_likelihood(sample_states, sample_actions)["log_likelihood"]
                sample_state_values = self.value_function.value_output(sample_states)

                entropy = self.policy.get_entropy(sample_states)["entropy"]
                loss, loss_per_item, vf_loss, vf_loss_per_item = self.loss_function.loss(
                    sample_log_probs, sample_prev_log_probs,
                    sample_state_values,  sample_prev_state_values, sample_advantages, entropy
                )

                # Do not need step op.
                _, loss, loss_per_item = self.optimizer.step(self.policy.variables(), loss, loss_per_item)
                _, vf_loss, vf_loss_per_item = \
                    self.value_function_optimizer.step(self.value_function.variables(), vf_loss, vf_loss_per_item)
            return loss, loss_per_item, vf_loss, vf_loss_per_item

