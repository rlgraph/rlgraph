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
from rlgraph.components.algorithms.algorithm_component import AlgorithmComponent
from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.components.loss_functions.ppo_loss_function import PPOLossFunction
from rlgraph.components.memories import Memory, RingBuffer
from rlgraph.spaces import BoolBox, FloatBox
from rlgraph.utils import util
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.define_by_run_ops import define_by_run_flatten
from rlgraph.utils.ops import flatten_op, DataOp, DataOpDict

if get_backend() == "tf":
    import tensorflow as tf
    setattr(tf.Tensor, "map", DataOp.map)
if get_backend() == "pytorch":
    import torch


class PPOAgent(Agent):
    """
    Proximal policy optimization is a variant of policy optimization in which
    the likelihood ratio between updated and prior policy is constrained by clipping, and
    where updates are performed via repeated sub-sampling of the input batch.

    Paper: https://arxiv.org/abs/1707.06347
    """

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
        max_timesteps=None,
        update_spec=None,
        summary_spec=None,
        saver_spec=None,
        auto_build=True,
        clip_ratio=0.2,
        gae_lambda=1.0,
        clip_rewards=0.0,
        value_function_clipping=None,
        standardize_advantages=False,
        sample_episodes=True,
        weight_entropy=None,
        sample_size=None,
        num_iterations=None,
        memory_spec=None,
        name="ppo-agent"
    ):
        """
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
            max_timesteps (Optional[int]): Obsoleted: An optional max timesteps hint for Workers.
            update_spec (Optional[dict]): Obsoleted: Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.

            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.

            name (str): Some name for this Agent object.
            clip_ratio (float): Clipping parameter for importance sampling (IS) likelihood ratio.
            gae_lambda (float): Lambda for generalized advantage estimation.

            clip_rewards (float): Reward clipping value. If not 0, rewards will be clipped within a +/- `clip_rewards`
                range.

            value_function_clipping (Optional[float]): If not None, uses clipped value function objective. If None,
                uses simple value function objective.

            standardize_advantages (bool): If true, standardize advantage values in update.

            sample_episodes (bool): If True, the update method interprets the batch_size as the number of
                episodes to fetch from the memory. If False, batch_size will refer to the number of time-steps. This
                is especially relevant for environments where episode lengths may vastly differ throughout training. For
                example, in CartPole, a losing episode is typically 10 steps, and a winning episode 200 steps.

            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).

            memory_spec (Optional[dict,Memory]): The spec for the Memory to use. Should typically be
                a ring-buffer.
        """
        super(PPOAgent, self).__init__(
            state_space=state_space,
            action_space=action_space,
            python_buffer_size=python_buffer_size,
            custom_python_buffers=custom_python_buffers,
            internal_states_space=internal_states_space,
            execution_spec=execution_spec,
            observe_spec=observe_spec,
            max_timesteps=max_timesteps,
            update_spec=update_spec,  # Obsoleted.
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            name=name
        )
        # Change our root-component to PPO.
        self.root_component = PPOAlgorithmComponent(
            agent=self, discount=discount, memory_batch_size=memory_batch_size, memory_spec=memory_spec,
            gae_lambda=gae_lambda, clip_rewards=clip_rewards,
            clip_ratio=clip_ratio, value_function_clipping=value_function_clipping, weight_entropy=weight_entropy,
            preprocessing_spec=preprocessing_spec, policy_spec=policy_spec, network_spec=network_spec,
            value_function_spec=value_function_spec,
            exploration_spec=None, optimizer_spec=optimizer_spec,
            value_function_optimizer_spec=value_function_optimizer_spec,
            sample_episodes=sample_episodes, standardize_advantages=standardize_advantages,
            sample_size=sample_size,
            num_iterations=num_iterations
        )

        # Extend input Space definitions to this Agent's specific API-methods.
        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            policy_weights="variables:policy",
            value_function_weights="variables:value-function",
            deterministic=bool,
            preprocessed_states=self.root_component.preprocessed_state_space.with_batch_rank(),
            rewards=FloatBox(add_batch_rank=True),
            terminals=BoolBox(add_batch_rank=True),
            sequence_indices=BoolBox(add_batch_rank=True),
            apply_postprocessing=bool,
            num_records=int,
            time_percentage=float
        ))

        if auto_build is True:
            self.build(dict(vf_optimizer=self.root_component.value_function_optimizer))

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals, **kwargs):
        self.graph_executor.execute(
            ("insert_records", [preprocessed_states, actions, rewards, terminals])
        )

    def update(self, batch=None, time_percentage=None, sequence_indices=None, apply_postprocessing=True):
        """
        Args:
            sequence_indices (Optional[np.ndarray, list]): Sequence indices are used in multi-env batches where
                partial episode fragments may be concatenated within the trajectory. For a single env, these are equal
                to terminals. If None are given, terminals will be used as sequence indices. A sequence index is True
                where an episode fragment ends and False otherwise. The reason separate indices are necessary is so that
                e.g. in GAE discounting, correct boot-strapping is applied depending on whether a true terminal state
                was reached, or a partial episode fragment of an environment ended.

                Example: If env_1 has terminals [0 0 0] for an episode fragment and env_2 terminals = [0 0 1],
                    we may pass them in as one combined array [0 0 0 0 0 1] with sequence indices showing where each
                    episode ends: [0 0 1 0 0 1].
            apply_postprocessing (Optional[(bool]): If True, apply post-processing such as generalised
                advantage estimation to collected batch in-graph. If False, update assumed post-processing has already
                been applied. The purpose of internal versus external post-processing is to be able to off-load
                post-processing in large scale distributed scenarios.
        """
        # TODO: Move update_spec to Worker. Agent should not hold these execution details.
        if time_percentage is None:
            time_percentage = self.timesteps / (self.max_timesteps or 1e6)

        self.num_updates += 1

        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", [True, time_percentage]))
        else:
            # No sequence indices means terminals are used in place.
            if sequence_indices is None:
                sequence_indices = batch["terminals"]

            pps_dtype = self.root_component.preprocessed_state_space.dtype
            batch["states"] = np.asarray(batch["states"], dtype=util.convert_dtype(dtype=pps_dtype, to='np'))

            ret = self.graph_executor.execute(
                ("update_from_external_batch", [
                    batch["states"], batch["actions"], batch["rewards"], batch["terminals"], sequence_indices,
                    apply_postprocessing, time_percentage
                ])
            )

        # Do some assertions that all iterations were run.
        assert ret["index"] == ret["step_op"] == self.root_component.num_iterations

        return ret["loss"] + ret["vf_loss"], ret["loss_per_item"] + ret["vf_loss_per_item"]

    def get_records(self, num_records=1):
        return self.graph_executor.execute(("get_records", num_records))

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.root_component.preprocessing_required and len(self.root_component.preprocessor.variable_registry) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def post_process(self, batch):
        batch_input = [batch["states"], batch["rewards"], batch["terminals"], batch["sequence_indices"]]
        ret = self.graph_executor.execute(("post_process", batch_input))
        return ret

    def __repr__(self):
        return "PPOAgent()"


class PPOAlgorithmComponent(AlgorithmComponent):
    def __init__(self, agent, *,
                 memory_spec=None, gae_lambda=1.0, clip_rewards=0.0, clip_ratio=0.2,
                 value_function_clipping=None, weight_entropy=None, sample_episodes=True, standardize_advantages=False,
                 sample_size=32, num_iterations=10,
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
        self.sample_size = sample_size
        self.num_iterations = num_iterations

        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer), "ERROR: PPO memory must be ring-buffer for episode-handling!"

        # Make sure the python buffer is not larger than our memory capacity.
        assert self.agent.python_buffer_size <= self.memory.capacity, \
            "ERROR: Buffer's size ({}) must be smaller or equal to the memory's capacity ({})!". \
            format(self.agent.python_buffer_size, self.memory.capacity)

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
    def insert_records(self, preprocessed_states, actions, rewards, terminals):
        records = dict(
            states=preprocessed_states, actions=actions, rewards=rewards,
            terminals=terminals
        )
        return self.memory.insert_records(records)

    @rlgraph_api
    def post_process(self, preprocessed_states, rewards, terminals, sequence_indices):
        state_values = self.get_state_values(preprocessed_states)
        advantages = self.gae_function.calc_gae_values(state_values, rewards, terminals, sequence_indices)
        return advantages

    # Learn from memory.
    @rlgraph_api
    def update_from_memory(self, apply_postprocessing=True, time_percentage=None):
        if self.sample_episodes:
            records = self.memory.get_episodes(self.memory_batch_size)
        else:
            records = self.memory.get_records(self.memory_batch_size)

        # Route to post process and update method.
        sequence_indices = records["terminals"]  # TODO: memory must return sequence_indices automatically.
        return self.update_from_external_batch(
            records["states"], records["actions"], records["rewards"], records["terminals"],
            sequence_indices, apply_postprocessing=apply_postprocessing, time_percentage=time_percentage
        )

    # Retrieve some records from memory.
    @rlgraph_api
    def get_records(self, num_records=1):
        return self.memory.get_records(num_records)

    # N.b. this is here because the iterative_optimization would need policy/losses as sub-components, but
    # multiple parents are not allowed currently.
    @rlgraph_api
    def _graph_fn_update_from_external_batch(
            self, preprocessed_states, actions, rewards, terminals, sequence_indices, apply_postprocessing=True,
            time_percentage=None
    ):
        """
        Performs iterative optimization by repeatedly sub-sampling from a batch.
        """
        multi_gpu_sync_optimizer = self.sub_components.get("multi-gpu-synchronizer")

        # Return values.
        loss, loss_per_item, vf_loss, vf_loss_per_item = None, None, None, None

        prev_log_probs = self.policy.get_log_likelihood(preprocessed_states, actions)["log_likelihood"]
        prev_state_values = self.value_function.value_output(preprocessed_states)

        if get_backend() == "tf":
            batch_size = tf.shape(list(flatten_op(preprocessed_states).values())[0])[0]

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
                start = tf.random_uniform(shape=(), minval=0, maxval=batch_size, dtype=tf.int32)
                indices = tf.range(start=start, limit=start + self.sample_size) % batch_size
                # Use `map` here in case we have container states/actions.
                sample_states = preprocessed_states.map(lambda k, v: tf.gather(v, indices))
                sample_actions = actions.map(lambda k, v: tf.gather(v, indices))

                #sample_states = tf.gather(params=preprocessed_states, indices=indices)
                #if isinstance(actions, ContainerDataOp):
                #    sample_actions = FlattenedDataOp()
                #    for name, action in flatten_op(actions).items():
                #        sample_actions[name] = tf.gather(params=action, indices=indices)
                #    sample_actions = unflatten_op(sample_actions)
                #else:
                #    sample_actions = tf.gather(params=actions, indices=indices)

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
                        apply_postprocessing=apply_postprocessing, time_percentage=time_percentage
                    )
                    avg_grads_and_vars_policy, avg_grads_and_vars_vf = self.vars_splitter.call(
                        out["avg_grads_and_vars_by_component"]
                    )
                    # Have to set all shapes here due to strict loop-var shape requirements.
                    out["loss"].set_shape(())
                    out["loss_vf"].set_shape(())
                    out["loss_per_item_vf"].set_shape((self.sample_size,))
                    out["loss_per_item"].set_shape((self.sample_size,))

                    policy_step_op = self.optimizer.apply_gradients(avg_grads_and_vars_policy, time_percentage)
                    vf_step_op = self.value_function_optimizer.apply_gradients(avg_grads_and_vars_vf, time_percentage)
                    step_op = self._graph_fn_group(policy_step_op, vf_step_op)
                    step_and_sync_op = multi_gpu_sync_optimizer.sync_variables_to_towers(step_op, all_vars)

                    with tf.control_dependencies([step_and_sync_op]):
                        if index_ == 0:
                            # Increase the global training step counter (only once per update).
                            out["loss"] = self._graph_fn_training_step(out["loss"])
                        # Return tuple not dict as we are in loop.
                        return index_ + 1, out["loss"], out["loss_per_item"], out["loss_vf"], out["loss_per_item_vf"]

                sample_log_probs = self.policy.get_log_likelihood(sample_states, sample_actions)["log_likelihood"]

                entropy = self.policy.get_entropy(sample_states)["entropy"]

                loss, loss_per_item, vf_loss, vf_loss_per_item = \
                    self.loss_function.loss(
                        sample_log_probs, sample_prev_log_probs,
                        sample_state_values, sample_prev_state_values, sample_advantages, entropy,
                        time_percentage=time_percentage
                    )

                if hasattr(self, "is_multi_gpu_tower") and self.is_multi_gpu_tower is True:
                    policy_grads_and_vars = self.optimizer.calculate_gradients(self.policy.variables(), loss)
                    vf_grads_and_vars = self.value_function_optimizer.calculate_gradients(
                        self.value_function.variables(), vf_loss
                    )
                    grads_and_vars_by_component = self.vars_merger.merge(policy_grads_and_vars, vf_grads_and_vars)
                    # Return tuple not dict as we are in loop.
                    return grads_and_vars_by_component, loss, loss_per_item, vf_loss, vf_loss_per_item
                else:
                    step_op = self.optimizer.step(self.policy.variables(), loss, loss_per_item, time_percentage)
                    loss.set_shape(())
                    loss_per_item.set_shape((self.sample_size,))

                    vf_step_op = self.value_function_optimizer.step(
                        self.value_function.variables(), vf_loss, vf_loss_per_item, time_percentage
                    )
                    vf_loss.set_shape(())
                    vf_loss_per_item.set_shape((self.sample_size,))

                    with tf.control_dependencies([step_op, vf_step_op]):
                        # Return tuple not dict as we are in loop.
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
                out = opt_body(*init_loop_vars)
                return dict(
                    grads_and_vars_by_component=out[0], loss=out[1], loss_per_item=out[2],
                    vf_loss=out[3], vf_loss_per_item=out[4]
                )
            else:
                index, loss, loss_per_item, vf_loss, vf_loss_per_item = tf.while_loop(
                    cond=cond,
                    body=opt_body,
                    loop_vars=init_loop_vars,
                    parallel_iterations=1
                )
                # Increase the global training step counter.
                loss = self._graph_fn_training_step(loss)
                # Return index as step_op (represents the collection of all step_ops).
                return dict(
                    step_op=index, loss=loss, loss_per_item=loss_per_item,
                    vf_loss=vf_loss, vf_loss_per_item=vf_loss_per_item,
                    index=index
                )

        elif get_backend() == "pytorch":
            batch_size = list(flatten_op(preprocessed_states).values())[0].shape[0]
            sample_size = min(batch_size, self.sample_size)

            # TODO: Add `map` support for pytorch tensors (like done for tf).
            if isinstance(prev_log_probs, dict):
                for name in actions.keys():
                    prev_log_probs[name] = prev_log_probs[name].detach()
            else:
                prev_log_probs = prev_log_probs.detach()
            # TODO: Add support for container spaces (via `map).
            prev_state_values = self.value_function.value_output(preprocessed_states).detach()

            if apply_postprocessing:
                advantages = self.gae_function.calc_gae_values(prev_state_values, rewards, terminals, sequence_indices)
            else:
                advantages = rewards

            if self.standardize_advantages:
                std = torch.std(advantages)
                # Std must not be 0.0 (would be the case for pytorch "test run" during build).
                if not np.isnan(std):
                    advantages = (advantages - torch.mean(advantages)) / std

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
                    sample_state_values,  sample_prev_state_values, sample_advantages, entropy,
                    time_percentage=time_percentage
                )

                # Do not need step op.
                self.optimizer.step(self.policy.variables(), loss, loss_per_item, time_percentage)
                self.value_function_optimizer.step(
                    self.value_function.variables(), vf_loss, vf_loss_per_item, time_percentage
                )
            return dict(
                step_op=None, loss=loss, loss_per_item=loss_per_item, vf_loss=vf_loss, vf_loss_per_item=vf_loss_per_item
            )

    @rlgraph_api
    def _graph_fn_get_gae(self, num_records=1):
        """
        Returns unstandardized(!) advantages for debugging purposes.
        """
        records = self.get_records(num_records)
        prev_state_values = self.value_function.value_output(records["states"])

        sequence_indices = records["terminals"]  # TODO: make ring-buffer return sequence indices automatically

        if get_backend() == "tf":
            # State values before update (stop-gradient as these are used in target term).
            prev_state_values = tf.stop_gradient(prev_state_values)

            # Advantages are based on previous state values.
            advantages = self.gae_function.calc_gae_values(
                prev_state_values, records["rewards"], records["terminals"], sequence_indices
            )
            # Advantages are based on previous state values.
            td_errors = self.gae_function.calc_td_errors(
                prev_state_values, records["rewards"], records["terminals"], sequence_indices
            )
        return prev_state_values, advantages, td_errors, records["rewards"], records["terminals"], sequence_indices
