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

import numpy as np

from rlgraph import get_backend
from rlgraph.agents import Agent
from rlgraph.components import Memory, RingBuffer, PPOLossFunction
from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.spaces import BoolBox, FloatBox
from rlgraph.utils import util
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.define_by_run_ops import define_by_run_flatten
from rlgraph.utils.ops import flatten_op, unflatten_op, DataOpDict, ContainerDataOp, FlattenedDataOp
from rlgraph.utils.util import strip_list

if get_backend() == "tf":
    import tensorflow as tf
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
        discount=0.98,
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
        summary_spec=None,
        saver_spec=None,
        auto_build=True,
        name="ppo-agent",
        clip_ratio=0.2,
        gae_lambda=1.0,
        clip_rewards=0.0,
        value_function_clipping=None,
        standardize_advantages=False,
        sample_episodes=True,
        weight_entropy=None,
        memory_spec=None
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

            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Spec-dict to specify `Agent.update()` settings.
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
        if policy_spec is not None:
            policy_spec["deterministic"] = False
        else:
            policy_spec = dict(deterministic=False)
        super(PPOAgent, self).__init__(
            state_space=state_space,
            action_space=action_space,
            discount=discount,
            preprocessing_spec=preprocessing_spec,
            network_spec=network_spec,
            internal_states_space=internal_states_space,
            policy_spec=policy_spec,
            value_function_spec=value_function_spec,
            execution_spec=execution_spec,
            optimizer_spec=optimizer_spec,
            value_function_optimizer_spec=value_function_optimizer_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            name=name,
            auto_build=auto_build
        )
        self.sample_episodes = sample_episodes

        # TODO: Have to manually set it here for multi-GPU synchronizer to know its number
        # TODO: of return values when calling _graph_fn_calculate_update_from_external_batch.
        # self.root_component.graph_fn_num_outputs["_graph_fn_update_from_external_batch"] = 4

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)

        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            policy_weights="variables:policy",
            value_function_weights="variables:value-function",
            deterministic=bool,
            preprocessed_states=preprocessed_state_space,
            rewards=reward_space,
            terminals=terminal_space,
            sequence_indices=BoolBox(add_batch_rank=True),
            apply_postprocessing=bool
        ))

        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer), "ERROR: PPO memory must be ring-buffer for episode-handling!"

        # Make sure the python buffer is not larger than our memory capacity.
        assert self.observe_spec["buffer_size"] <= self.memory.capacity, \
            "ERROR: Buffer's size ({}) in `observe_spec` must be smaller or equal to the memory's capacity ({})!". \
            format(self.observe_spec["buffer_size"], self.memory.capacity)

        # The splitter for splitting up the records coming from the memory.
        self.standardize_advantages = standardize_advantages
        self.gae_function = GeneralizedAdvantageEstimation(
            gae_lambda=gae_lambda, discount=self.discount, clip_rewards=clip_rewards
        )
        self.loss_function = PPOLossFunction(
            clip_ratio=clip_ratio, value_function_clipping=value_function_clipping, weight_entropy=weight_entropy
        )

        self.iterations = self.update_spec["num_iterations"]
        self.sample_size = self.update_spec["sample_size"]
        self.batch_size = self.update_spec["batch_size"]

        # Add all our sub-components to the core.
        self.root_component.add_components(
            self.preprocessor, self.memory,
            self.policy, self.exploration,
            self.loss_function, self.optimizer, self.value_function, self.value_function_optimizer, self.vars_merger,
            self.vars_splitter, self.gae_function
        )
        # Define the Agent's (root-Component's) API.
        self.define_graph_api()
        self.build_options = dict(vf_optimizer=self.value_function_optimizer)

        if self.auto_build:
            self._build_graph(
                [self.root_component], self.input_spaces, optimizer=self.optimizer,
                # Important: Use sample-size, not batch-size as the sub-samples (from a batch) are the ones that get
                # multi-gpu-split.
                batch_size=self.update_spec["sample_size"],
                build_options=self.build_options
            )
            self.graph_built = True

    def define_graph_api(self):
        super(PPOAgent, self).define_graph_api()

        agent = self

        # Reset operation (resets preprocessor).
        if self.preprocessing_required:
            @rlgraph_api(component=self.root_component)
            def reset_preprocessor(root):
                reset_op = agent.preprocessor.reset()
                return reset_op

        # Act from preprocessed states.
        @rlgraph_api(component=self.root_component)
        def action_from_preprocessed_state(root, preprocessed_states, deterministic=False):
            out = agent.policy.get_action(preprocessed_states, deterministic=deterministic)
            return out["action"], preprocessed_states

        # State (from environment) to action with preprocessing.
        @rlgraph_api(component=self.root_component)
        def get_preprocessed_state_and_action(root, states, deterministic=False):
            preprocessed_states = agent.preprocessor.preprocess(states)
            return root.action_from_preprocessed_state(preprocessed_states, deterministic)

        # Insert into memory.
        @rlgraph_api(component=self.root_component)
        def insert_records(root, preprocessed_states, actions, rewards, terminals):
            records = dict(states=preprocessed_states, actions=actions, rewards=rewards, terminals=terminals)
            return agent.memory.insert_records(records)

        @rlgraph_api(component=self.root_component)
        def post_process(root, preprocessed_states, rewards, terminals, sequence_indices):
            baseline_values = agent.value_function.value_output(preprocessed_states)
            pg_advantages = agent.gae_function.calc_gae_values(baseline_values, rewards, terminals, sequence_indices)
            return pg_advantages

        # Learn from memory.
        @rlgraph_api(component=self.root_component)
        def update_from_memory(root, apply_postprocessing=True, time_percentage=None):
            if agent.sample_episodes:
                records = agent.memory.get_episodes(self.update_spec["batch_size"])
            else:
                records = agent.memory.get_records(self.update_spec["batch_size"])

            # Route to post process and update method.
            sequence_indices = records["terminals"]  # TODO: return correct seq-indices automatically from mem
            return root.update_from_external_batch(
                records["states"], records["actions"], records["rewards"], records["terminals"],
                sequence_indices, apply_postprocessing, time_percentage
            )

        # N.b. this is here because the iterative_optimization would need policy/losses as sub-components, but
        # multiple parents are not allowed currently.
        @rlgraph_api(component=self.root_component)
        def _graph_fn_update_from_external_batch(
                root, preprocessed_states, actions, rewards, terminals, sequence_indices, apply_postprocessing=True,
                time_percentage=None
        ):
            """
            Calls iterative optimization by repeatedly sub-sampling.
            """
            multi_gpu_sync_optimizer = root.sub_components.get("multi-gpu-synchronizer")

            # Return values.
            loss, loss_per_item, vf_loss, vf_loss_per_item = None, None, None, None

            policy = root.get_sub_component_by_name(agent.policy.scope)
            value_function = root.get_sub_component_by_name(agent.value_function.scope)
            optimizer = root.get_sub_component_by_name(agent.optimizer.scope)
            loss_function = root.get_sub_component_by_name(agent.loss_function.scope)
            value_function_optimizer = root.get_sub_component_by_name(agent.value_function_optimizer.scope)
            vars_merger = root.get_sub_component_by_name(agent.vars_merger.scope)
            gae_function = root.get_sub_component_by_name(agent.gae_function.scope)

            prev_log_probs = policy.get_log_likelihood(preprocessed_states, actions)["log_likelihood"]
            prev_state_values = value_function.value_output(preprocessed_states)

            if get_backend() == "tf":
                batch_size = tf.shape(preprocessed_states)[0]

                # Log probs before update (stop-gradient as these are used in target term).
                prev_log_probs = tf.stop_gradient(prev_log_probs)
                # State values before update (stop-gradient as these are used in target term).
                prev_state_values = tf.stop_gradient(prev_state_values)

                # Advantages are based on previous state values.
                advantages = tf.cond(
                    pred=apply_postprocessing,
                    true_fn=lambda: gae_function.calc_gae_values(
                        prev_state_values, rewards, terminals, sequence_indices
                    ),
                    false_fn=lambda: rewards
                )
                if self.standardize_advantages:
                    mean, std = tf.nn.moments(x=advantages, axes=[0])
                    advantages = (advantages - mean) / std

                def opt_body(index_, loss_, loss_per_item_, vf_loss_, vf_loss_per_item_):
                    start = tf.random_uniform(shape=(), minval=0, maxval=batch_size, dtype=tf.int32)
                    indices = tf.range(start=start, limit=start + agent.sample_size) % batch_size
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
                    sample_advantages.set_shape((agent.sample_size,))

                    sample_state_values = value_function.value_output(sample_states)
                    sample_prev_state_values = tf.gather(params=prev_state_values, indices=indices)

                    # If we are a multi-GPU root:
                    # Simply feeds everything into the multi-GPU sync optimizer's method and return.
                    if multi_gpu_sync_optimizer is not None:
                        main_policy_vars = agent.policy.variables()
                        main_vf_vars = agent.value_function.variables()
                        all_vars = agent.vars_merger.merge(main_policy_vars, main_vf_vars)
                        # grads_and_vars, loss, loss_per_item, vf_loss, vf_loss_per_item = \
                        out = multi_gpu_sync_optimizer.calculate_update_from_external_batch(
                            all_vars,
                            sample_states, sample_actions, sample_rewards, sample_terminals, sample_sequence_indices,
                            apply_postprocessing=apply_postprocessing
                        )
                        avg_grads_and_vars_policy, avg_grads_and_vars_vf = agent.vars_splitter.call(
                            out["avg_grads_and_vars_by_component"]
                        )
                        policy_step_op = agent.optimizer.apply_gradients(avg_grads_and_vars_policy)
                        vf_step_op = agent.value_function_optimizer.apply_gradients(avg_grads_and_vars_vf)
                        step_op = root._graph_fn_group(policy_step_op, vf_step_op)
                        step_and_sync_op = multi_gpu_sync_optimizer.sync_variables_to_towers(
                            step_op, all_vars
                        )
                        loss_vf, loss_per_item_vf = out["additional_return_0"], out["additional_return_1"]

                        # Have to set all shapes here due to strict loop-var shape requirements.
                        out["loss"].set_shape(())
                        loss_vf.set_shape(())
                        loss_per_item_vf.set_shape((agent.sample_size,))
                        out["loss_per_item"].set_shape((agent.sample_size,))

                        with tf.control_dependencies([step_and_sync_op]):
                            if index_ == 0:
                                # Increase the global training step counter.
                                out["loss"] = root._graph_fn_training_step(out["loss"])
                            return index_ + 1, out["loss"], out["loss_per_item"], loss_vf, loss_per_item_vf

                    sample_log_probs = policy.get_log_likelihood(sample_states, sample_actions)["log_likelihood"]

                    entropy = policy.get_entropy(sample_states)["entropy"]

                    loss, loss_per_item, vf_loss, vf_loss_per_item = \
                        loss_function.loss(
                            sample_log_probs, sample_prev_log_probs,
                            sample_state_values, sample_prev_state_values, sample_advantages, entropy, time_percentage
                        )

                    if hasattr(root, "is_multi_gpu_tower") and root.is_multi_gpu_tower is True:
                        policy_grads_and_vars = optimizer.calculate_gradients(policy.variables(), loss, time_percentage)
                        vf_grads_and_vars = value_function_optimizer.calculate_gradients(
                            value_function.variables(), vf_loss, time_percentage
                        )
                        grads_and_vars_by_component = vars_merger.merge(policy_grads_and_vars, vf_grads_and_vars)
                        return grads_and_vars_by_component, loss, loss_per_item, vf_loss, vf_loss_per_item
                    else:
                        step_op = optimizer.step(policy.variables(), loss, loss_per_item, time_percentage)
                        loss.set_shape(())
                        loss_per_item.set_shape((agent.sample_size,))

                        vf_step_op = value_function_optimizer.step(
                            value_function.variables(), vf_loss, vf_loss_per_item, time_percentage
                        )
                        vf_loss.set_shape(())
                        vf_loss_per_item.set_shape((agent.sample_size,))

                        with tf.control_dependencies([step_op, vf_step_op]):
                            return index_ + 1, loss, loss_per_item, vf_loss, vf_loss_per_item

                def cond(index_, loss_, loss_per_item_, v_loss_, v_loss_per_item_):
                    return index_ < agent.iterations

                init_loop_vars = [
                    0,
                    tf.zeros(shape=(), dtype=tf.float32),
                    tf.zeros(shape=(agent.sample_size,)),
                    tf.zeros(shape=(), dtype=tf.float32),
                    tf.zeros(shape=(agent.sample_size,))
                ]

                if hasattr(root, "is_multi_gpu_tower") and root.is_multi_gpu_tower is True:
                    return opt_body(*init_loop_vars)
                else:
                    index, loss, loss_per_item, vf_loss, vf_loss_per_item = tf.while_loop(
                        cond=cond,
                        body=opt_body,
                        loop_vars=init_loop_vars,
                        parallel_iterations=1
                    )
                    # Increase the global training step counter.
                    loss = root._graph_fn_training_step(loss)
                    return loss, loss_per_item, vf_loss, vf_loss_per_item

            elif get_backend() == "pytorch":
                batch_size = preprocessed_states.shape[0]
                sample_size = min(batch_size, agent.sample_size)

                if isinstance(prev_log_probs, dict):
                    for name in actions.keys():
                        prev_log_probs[name] = prev_log_probs[name].detach()
                else:
                    prev_log_probs = prev_log_probs.detach()
                prev_state_values = value_function.value_output(preprocessed_states).detach()
                if apply_postprocessing:
                    advantages = gae_function.calc_gae_values(prev_state_values, rewards, terminals, sequence_indices)
                else:
                    advantages = rewards
                if self.standardize_advantages:
                    advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

                for _ in range(agent.iterations):
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

                    sample_log_probs = policy.get_log_likelihood(sample_states, sample_actions)["log_likelihood"]
                    sample_state_values = value_function.value_output(sample_states)

                    entropy = policy.get_entropy(sample_states)["entropy"]
                    loss, loss_per_item, vf_loss, vf_loss_per_item = loss_function.loss(
                        sample_log_probs, sample_prev_log_probs,
                        sample_state_values, sample_prev_state_values, sample_advantages, entropy, time_percentage
                    )

                    # Do not need step op.
                    optimizer.step(policy.variables(), loss, loss_per_item, time_percentage)
                    value_function_optimizer.step(value_function.variables(), vf_loss, vf_loss_per_item, time_percentage)
                return loss, loss_per_item, vf_loss, vf_loss_per_item

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None,
                   time_percentage=None):
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
            call_method = "get_preprocessed_state_and_action"
            batched_states = self.state_space.force_batch(states)
        else:
            call_method = "action_from_preprocessed_state"
            batched_states = states
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1

        # Increase timesteps by the batch size (number of states in batch).
        batch_size = len(batched_states)
        self.timesteps += batch_size

        # Control, which return value to "pull" (depending on `additional_returns`).
        return_ops = [0, 1] if "preprocessed_states" in extra_returns else [0]  # 1=preprocessed_states, 0=action
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

    # TODO make next states optional in observe API.
    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

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
            time_percentage = self.timesteps / self.update_spec.get("max_timesteps", 1e6)

        # [0] = the loss; [1] = loss-per-item, [2] = vf-loss, [3] = vf-loss- per item
        return_ops = [0, 1, 2, 3]
        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", [True, time_percentage], return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]
        else:
            # No sequence indices means terminals are used in place.
            if sequence_indices is None:
                sequence_indices = batch["terminals"]

            pps_dtype = self.preprocessed_state_space.dtype
            batch["states"] = np.asarray(batch["states"], dtype=util.convert_dtype(dtype=pps_dtype, to='np'))

            ret = self.graph_executor.execute(
                ("update_from_external_batch", [
                    batch["states"], batch["actions"], batch["rewards"], batch["terminals"], sequence_indices,
                    apply_postprocessing, time_percentage
                ], return_ops)
            )
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
        if self.preprocessing_required and len(self.preprocessor.variable_registry) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def post_process(self, batch):
        batch_input = [batch["states"], batch["rewards"], batch["terminals"], batch["sequence_indices"]]
        ret = self.graph_executor.execute(("post_process", batch_input))
        return ret

    def __repr__(self):
        return "PPOAgent()"
