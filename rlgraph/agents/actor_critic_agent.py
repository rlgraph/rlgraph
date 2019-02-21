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

from rlgraph.agents import Agent
from rlgraph.components import Memory, ContainerMerger, ContainerSplitter, RingBuffer, ValueFunction, Optimizer
from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.components.loss_functions.actor_critic_loss_function import ActorCriticLossFunction
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.utils import util
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import strip_list


class ActorCriticAgent(Agent):
    """
    Basic actor-critic policy gradient architecture with generalized advantage estimation,
    and entropy regularization. Suitable for execution with A2C, A3C.
    """

    def __init__(self, gae_lambda=1.0, clip_rewards=0.0, sample_episodes=False,
                 weight_entropy=None, memory_spec=None, **kwargs):
        """
        Args:
            gae_lambda (float): Lambda for generalized advantage estimation.
            clip_rewards (float): Reward clip value. If not 0, rewards will be clipped into this range.
            sample_episodes (bool): If true, the update method interprets the batch_size as the number of
                episodes to fetch from the memory. If false, batch_size will refer to the number of time-steps. This
                is especially relevant for environments where episode lengths may vastly differ throughout training. For
                example, in CartPole, a losing episode is typically 10 steps, and a winning episode 200 steps.
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use. Should typically be
            a ring-buffer.
        """
        # Set policy to stochastic.
        if "policy_spec" in kwargs:
            policy_spec = kwargs.pop("policy_spec")
            policy_spec["deterministic"] = False
        else:
            policy_spec = dict(deterministic=False)
        super(ActorCriticAgent, self).__init__(
            policy_spec=policy_spec,
            name=kwargs.pop("name", "actor-critic-agent"), **kwargs
        )
        self.sample_episodes = sample_episodes

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)

        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            policy_weights="variables:{}".format(self.policy.scope),
            deterministic=bool,
            preprocessed_states=preprocessed_state_space,
            rewards=reward_space,
            terminals=terminal_space,
            sequence_indices=BoolBox(add_batch_rank=True)
        ))

        # The merger to merge inputs into one record Dict going into the memory.
        self.merger = ContainerMerger("states", "actions", "rewards", "terminals")
        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer), \
            "ERROR: Actor-critic memory must be ring-buffer for episode-handling."
        # The splitter for splitting up the records coming from the memory.
        self.splitter = ContainerSplitter("states", "actions", "rewards", "terminals")

        self.gae_function = GeneralizedAdvantageEstimation(gae_lambda=gae_lambda, discount=self.discount,
                                                           clip_rewards=clip_rewards)
        self.loss_function = ActorCriticLossFunction(weight_entropy=weight_entropy)

        # Add all our sub-components to the core.
        sub_components = [self.preprocessor, self.merger, self.memory, self.splitter, self.policy,
                          self.loss_function, self.optimizer, self.value_function, self.value_function_optimizer,
                          self.gae_function]
        self.root_component.add_components(*sub_components)

        # Define the Agent's (root-Component's) API.
        self.define_graph_api()
        self.build_options = dict(vf_optimizer=self.value_function_optimizer)

        if self.auto_build:
            self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                              batch_size=self.update_spec["batch_size"],
                              build_options=self.build_options)

            self.graph_built = True

    def define_graph_api(self):
        super(ActorCriticAgent, self).define_graph_api()

        agent = self
        sample_episodes = self.sample_episodes

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
            records = agent.merger.merge(preprocessed_states, actions, rewards, terminals)
            return agent.memory.insert_records(records)

        @rlgraph_api(component=self.root_component)
        def post_process(root, preprocessed_states, rewards, terminals, sequence_indices):
            baseline_values = agent.value_function.value_output(preprocessed_states)
            pg_advantages = agent.gae_function.calc_gae_values(baseline_values, rewards, terminals, sequence_indices)
            return pg_advantages

        # Learn from memory.
        @rlgraph_api(component=self.root_component)
        def update_from_memory(root):
            if sample_episodes:
                records = agent.memory.get_episodes(agent.update_spec["batch_size"])
            else:
                records = agent.memory.get_records(agent.update_spec["batch_size"])
            preprocessed_s, actions, rewards, terminals = agent.splitter.split(records)
            sequence_indices = terminals
            return root.post_process_and_update(
                preprocessed_s, actions, rewards, terminals, sequence_indices
            )

        # First post-process, then update (so we can separately update already post-processed data).
        @rlgraph_api(component=self.root_component)
        def post_process_and_update(root, preprocessed_states, actions, rewards, terminals, sequence_indices):
            rewards = root.post_process(preprocessed_states, rewards, terminals, sequence_indices)
            return root.update_from_external_batch(preprocessed_states, actions, rewards, terminals)

        # Learn from an external batch.
        @rlgraph_api(component=self.root_component)
        def update_from_external_batch(root, preprocessed_states, actions, rewards, terminals):

            baseline_values = agent.value_function.value_output(preprocessed_states)
            log_probs = agent.policy.get_action_log_probs(preprocessed_states, actions)["action_log_probs"]
            entropy = agent.policy.get_entropy(preprocessed_states)["entropy"]
            loss, loss_per_item, vf_loss, vf_loss_per_item = agent.loss_function.loss(
                log_probs, entropy, baseline_values, actions, rewards, terminals
            )

            # Args are passed in again because some device strategies may want to split them to different devices.
            policy_vars = agent.policy.variables()
            vf_vars = agent.value_function.variables()

            step_op, loss, loss_per_item = agent.optimizer.step(policy_vars, loss, loss_per_item)
            vf_step_op, vf_loss, vf_loss_per_item = agent.value_function_optimizer.step(
                vf_vars, vf_loss, vf_loss_per_item
            )

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
        return_ops = [0, 1] if "preprocessed_states" in extra_returns else [0]  # 1=preprocessed_states, 0=action
        ret = self.graph_executor.execute((
            call_method,
            [batched_states, not use_exploration],  # deterministic = not use_exploration
            return_ops
        ))
        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

    # TODO make next states optional in observe API.
    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None, sequence_indices=None, apply_postprocessing=True):
        """
        Args:
            batch (dict): Update batch.
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
        # [0] step_op, [1] loss, [2] loss_per_item, [3] vf_step_op, [4]vf_loss, [5]vf_loss_per_item
        return_ops = [0, 1, 2, 3, 4, 5]
        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", None, return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]
        else:
            # No sequence indices means terminals are used in place.
            if sequence_indices is None:
                sequence_indices = batch["terminals"]

            pps_dtype = self.preprocessed_state_space.dtype
            batch["states"] = np.asarray(batch["states"], dtype=util.convert_dtype(dtype=pps_dtype, to='np'))
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"], sequence_indices]

            # Execute post-processing or already post-processed by workers?
            if apply_postprocessing:
                ret = self.graph_executor.execute(("post_process_and_update", batch_input, return_ops))
                # Remove unnecessary return dicts (e.g. sync-op).
                if isinstance(ret, dict):
                    ret = ret["post_process_and_update"]
            else:
                ret = self.graph_executor.execute(("update_from_external_batch", batch_input, return_ops))
                # Remove unnecessary return dicts (e.g. sync-op).
                if isinstance(ret, dict):
                    ret = ret["update_from_external_batch"]

        # [1] loss, [2] loss per item
        return ret[1], ret[2]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.preprocessing_required and len(self.preprocessor.variable_registry) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def __repr__(self):
        return "ActorCriticAgent"
