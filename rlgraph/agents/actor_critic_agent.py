# Copyright 2018 The RLgraph authors. All Rights Reserved.
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

from rlgraph.agents import Agent
from rlgraph.components import Memory, DictMerger, ContainerSplitter, RingBuffer, ValueFunction, Optimizer
from rlgraph.components.loss_functions.actor_critic_loss_function import ActorCriticLossFunction
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import strip_list


class ActorCriticAgent(Agent):
    """
    Basic actor-critic policy gradient architecture with generalized advantage estimation,
    and entropy regularization. Suitable for execution with A2C, A3C.
    """
    def __init__(self,  value_function_spec, value_function_optimizer_spec=None,
                 gae_lambda=1.0, sample_episodes=False, weight_entropy=None, memory_spec=None, **kwargs):
        """
        Args:
            value_function_spec (list): Neural network specification for baseline.
            value_function_optimizer_spec (dict): Optimizer config for value function otpimizer. If None, the optimizer
                spec for the policy is used (same learning rate and optimizer type).
            gae_lambda (float): Lambda for generalized advantage estimation.
            sample_episodes (bool): If true, the update method interprets the batch_size as the number of
                episodes to fetch from the memory. If false, batch_size will refer to the number of time-steps. This
                is especially relevant for environments where episode lengths may vastly differ throughout training. For
                example, in CartPole, a losing episode is typically 10 steps, and a winning episode 200 steps.
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use. Should typically be
            a ring-buffer.
        """
        super(ActorCriticAgent, self).__init__(
            policy_spec=dict(deterministic=False),  # Set policy to stochastic.
            name=kwargs.pop("name", "actor-critic-agent"), **kwargs
        )
        self.sample_episodes = sample_episodes

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)

        # Create non-shared baseline network.
        self.value_function = ValueFunction(network_spec=value_function_spec)

        # Cannot use the default scope for another optimizer again.
        if value_function_optimizer_spec is None:
            vf_optimizer_spec = self.optimizer_spec
        else:
            vf_optimizer_spec = value_function_optimizer_spec

        vf_optimizer_spec["scope"] = "value-function-optimizer"
        self.value_function_optimizer = Optimizer.from_spec(vf_optimizer_spec)

        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            weights="variables:{}".format(self.policy.scope),
            deterministic=bool,
            preprocessed_states=preprocessed_state_space,
            rewards=reward_space,
            terminals=terminal_space,
            sequence_indices=BoolBox(add_batch_rank=True)
        ))

        # The merger to merge inputs into one record Dict going into the memory.
        self.merger = DictMerger("states", "actions", "rewards", "terminals")
        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer),\
            "ERROR: Actor-critic memory must be ring-buffer for episode-handling."
        # The splitter for splitting up the records coming from the memory.
        self.splitter = ContainerSplitter("states", "actions", "rewards", "terminals")

        self.loss_function = ActorCriticLossFunction(discount=self.discount, gae_lambda=gae_lambda,
                                                     weight_entropy=weight_entropy)

        # Add all our sub-components to the core.
        sub_components = [self.preprocessor, self.merger, self.memory, self.splitter, self.policy,
                          self.loss_function, self.optimizer, self.value_function, self.value_function_optimizer]
        self.root_component.add_components(*sub_components)

        # Define the Agent's (root-Component's) API.
        self.define_graph_api("value-function", "value-function-optimizer", self.policy.scope,
                              self.preprocessor.scope, self.optimizer.scope, *sub_components)

        if self.auto_build:
            self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                              batch_size=self.update_spec["batch_size"],
                              build_options=dict(vf_optimizer=self.value_function_optimizer))

            self.graph_built = True

    def define_graph_api(self, value_function_scope, vf_optimizer_scope, policy_scope, pre_processor_scope,
                         optimizer_scope, *sub_components):
        super(ActorCriticAgent, self).define_graph_api(policy_scope, pre_processor_scope)

        preprocessor, merger, memory, splitter, policy, loss_function, optimizer, value_function, \
            vf_optimizer = sub_components
        sample_episodes = self.sample_episodes

        # Reset operation (resets preprocessor).
        if self.preprocessing_required:
            @rlgraph_api(component=self.root_component)
            def reset_preprocessor(self):
                reset_op = preprocessor.reset()
                return reset_op

        # Act from preprocessed states.
        @rlgraph_api(component=self.root_component)
        def action_from_preprocessed_state(self, preprocessed_states, deterministic=False):
            out = policy.get_action(preprocessed_states, deterministic=deterministic)
            return preprocessed_states, out["action"]

        # State (from environment) to action with preprocessing.
        @rlgraph_api(component=self.root_component)
        def get_preprocessed_state_and_action(self, states, deterministic=False):
            preprocessed_states = preprocessor.preprocess(states)
            return self.action_from_preprocessed_state(preprocessed_states, deterministic)

        # Insert into memory.
        @rlgraph_api(component=self.root_component)
        def insert_records(self, preprocessed_states, actions, rewards, terminals):
            records = merger.merge(preprocessed_states, actions, rewards, terminals)
            return memory.insert_records(records)

        # Learn from memory.
        @rlgraph_api(component=self.root_component)
        def update_from_memory(self_):
            if sample_episodes:
                records = memory.get_episodes(self.update_spec["batch_size"])
            else:
                records = memory.get_records(self.update_spec["batch_size"])
            preprocessed_s, actions, rewards, terminals = splitter.split(records)
            sequence_indices = terminals
            return self_.update_from_external_batch(
                preprocessed_s, actions, rewards, terminals, sequence_indices
            )

        # Learn from an external batch.
        @rlgraph_api(component=self.root_component)
        def update_from_external_batch(self_, preprocessed_states, actions, rewards, terminals, sequence_indices):

            baseline_values = value_function.value_output(preprocessed_states)
            log_probs = policy.get_action_log_probs(preprocessed_states, actions)["action_log_probs"]
            entropy = policy.get_entropy(preprocessed_states)["entropy"]
            loss, loss_per_item, vf_loss, vf_loss_per_item = self_.get_sub_component_by_name(loss_function.scope).loss(
                log_probs, entropy, baseline_values, actions, rewards,
                terminals, sequence_indices
            )

            # Args are passed in again because some device strategies may want to split them to different devices.
            policy_vars = self_.get_sub_component_by_name(policy_scope)._variables()
            vf_vars = self_.get_sub_component_by_name(value_function_scope)._variables()

            step_op, loss, loss_per_item = optimizer.step(policy_vars, loss, loss_per_item)
            vf_step_op, vf_loss, vf_loss_per_item = vf_optimizer.step(vf_vars, vf_loss, vf_loss_per_item)

            return step_op, loss, loss_per_item, vf_step_op, vf_loss, vf_loss_per_item

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None):
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

    # TODO make next states optional in observe API.
    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None, sequence_indices=None):
        """
            sequence_indices (Optional[np.ndarray, list]): Sequence indices are used in multi-env batches where
                partial episode fragments may be concatenated within the trajectory. For a single env, these are equal
                to terminals. If None are given, terminals will be used as sequence indices. A sequence index is True
                where an episode fragment ends and False otherwise. The reason separate indices are necessary is so that
                e.g. in GAE discounting, correct boot-strapping is applied depending on whether a true terminal state
                was reached, or a partial episode fragment of an environment ended.

                Example: If env_1 has terminals [0 0 0] for an episode fragment and env_2 terminals = [0 0 1],
                    we may pass them in as one combined array [0 0 0 0 0 1] with sequence indices showing where each
                    episode ends: [0 0 1 0 0 1].
        """
        # step, loss, loss per item, vf step, vf loss, vf loss per item
        return_ops = [0, 1, 2, 3, 4, 5]
        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", None, return_ops))
        else:
            # No sequence indices means terminals are used in place.
            if sequence_indices is None:
                sequence_indices = batch["terminals"]
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"], sequence_indices]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input, return_ops))

        # [0] no-op, [1] loss, [2] loss per item
        return ret[1], ret[2]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.preprocessing_required and len(self.preprocessor.variables) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def __repr__(self):
        return "ActorCriticAgent"
