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

from rlgraph import get_backend
from rlgraph.agents import Agent
from rlgraph.components import DictMerger, ContainerSplitter, Memory, RingBuffer, PPOLossFunction, ValueFunction,\
    Optimizer
from rlgraph.spaces import BoolBox, FloatBox
from rlgraph.utils.util import strip_list
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf


class PPOAgent(Agent):
    """
    Proximal policy optimization is a variant of policy optimization in which
    the likelihood ratio between updated and prior policy is constrained by clipping, and
    where updates are performed via repeated sub-sampling of the input batch.

    Paper: https://arxiv.org/abs/1707.06347
    """
    def __init__(self, value_function_spec, value_function_optimizer_spec=None,
                 clip_ratio=0.2, gae_lambda=1.0, standardize_advantages=False,
                 sample_episodes=True, weight_entropy=None, memory_spec=None, **kwargs):
        """
        Args:
            value_function_spec (list): Neural network specification for baseline.
            value_function_optimizer_spec (dict): Optimizer config for value function otpimizer. If None, the optimizer
                spec for the policy is used (same learning rate and optimizer type).
            clip_ratio (float): Clipping parameter for likelihood ratio.
            gae_lambda (float): Lambda for generalized advantage estimation.
            standardize_advantages (bool): If true, standardize advantage values in update.
            sample_episodes (bool): If true, the update method interprets the batch_size as the number of
                episodes to fetch from the memory. If false, batch_size will refer to the number of time-steps. This
                is especially relevant for environments where episode lengths may vastly differ throughout training. For
                example, in CartPole, a losing episode is typically 10 steps, and a winning episode 200 steps.
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use. Should typically be
            a ring-buffer.
        """
        super(PPOAgent, self).__init__(
            policy_spec=dict(deterministic=False),  # Set policy to stochastic.
            name=kwargs.pop("name", "ppo-agent"), **kwargs
        )
        self.sample_episodes = sample_episodes

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)

        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            weights="variables:policy",
            deterministic=bool,
            preprocessed_states=preprocessed_state_space,
            rewards=reward_space,
            terminals=terminal_space
        ))

        # The merger to merge inputs into one record Dict going into the memory.
        self.merger = DictMerger("states", "actions", "rewards", "terminals")
        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer), "ERROR: PPO memory must be ring-buffer for episode-handling!"

        # Make sure the python buffer is not larger than our memory capacity.
        assert self.observe_spec["buffer_size"] <= self.memory.capacity,\
            "ERROR: Buffer's size ({}) in `observe_spec` must be smaller or equal to the memory's capacity ({})!".\
            format(self.observe_spec["buffer_size"], self.memory.capacity)

        # The splitter for splitting up the records coming from the memory.
        self.splitter = ContainerSplitter("states", "actions", "rewards", "terminals")

        self.loss_function = PPOLossFunction(discount=self.discount, gae_lambda=gae_lambda, clip_ratio=clip_ratio,
                                             standardize_advantages=standardize_advantages,
                                             weight_entropy=weight_entropy)

        # TODO make network sharing optional.
        # Create non-shared baseline network.
        self.value_function = ValueFunction(network_spec=value_function_spec)

        # Cannot use the default scope for another optimizer again.
        if value_function_optimizer_spec is None:
            vf_optimizer_spec = self.optimizer_spec
        else:
            vf_optimizer_spec = value_function_optimizer_spec

        vf_optimizer_spec["scope"] = "value-function-optimizer"
        self.value_function_optimizer = Optimizer.from_spec(vf_optimizer_spec)

        self.iterations = self.update_spec["num_iterations"]
        self.sample_size = self.update_spec["sample_size"]
        self.batch_size = self.update_spec["batch_size"]

        # Add all our sub-components to the core.
        sub_components = [self.preprocessor, self.merger, self.memory, self.splitter, self.policy,
                          self.exploration, self.loss_function, self.optimizer,
                          self.value_function, self.value_function_optimizer]
        self.root_component.add_components(*sub_components)

        # Define the Agent's (root-Component's) API.
        self.define_graph_api(
            "value-function", "value-function-optimizer", "policy", "preprocessor-stack",
            self.optimizer.scope, *sub_components
        )

        if self.auto_build:
            self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                              batch_size=self.update_spec["batch_size"],
                              build_options=dict(vf_optimizer=self.value_function_optimizer))
            self.graph_built = True

    def define_graph_api(self, value_function_scope, vf_optimizer_scope, policy_scope, pre_processor_scope,
                         optimizer_scope, *sub_components):
        super(PPOAgent, self).define_graph_api(policy_scope, pre_processor_scope)

        preprocessor, merger, memory, splitter, policy, exploration, loss_function, optimizer, value_function, \
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

            # Route to external update method.
            return self_.update_from_external_batch(preprocessed_s, actions, rewards, terminals)

        # Learn from an external batch.
        @rlgraph_api(component=self.root_component)
        def update_from_external_batch(self_, preprocessed_states, actions, rewards, terminals):
            return self_._graph_fn_iterative_opt(preprocessed_states, actions, rewards, terminals)

        # N.b. this is here because the iterative_optimization would need policy/losses as sub-components, but
        # multiple parents are not allowed currently.
        @graph_fn(component=self.root_component)
        def _graph_fn_iterative_opt(self_, preprocessed_states, actions, rewards, terminals):
            """
            Calls iterative optimization by repeatedly sub-sampling.
            """
            if get_backend() == "tf":
                batch_size = tf.shape(preprocessed_states)[0]
                last_terminal = tf.expand_dims(terminals[-1], -1)

                # Ensure the very last entry is terminal so we don't connect different episodes when sampling
                # sub-episodes and wrapping, e.g. batch size 1000, sample 100, start 950: range [950, 50].
                terminals = tf.concat([terminals[:-1], tf.ones_like(last_terminal)], axis=0)

                def opt_body(index, loss, loss_per_item, vf_loss, vf_loss_per_item):
                    start = tf.random_uniform(shape=(1,), minval=0, maxval=batch_size - 1, dtype=tf.int32)[0]
                    indices = tf.range(start=start, limit=start + self.sample_size) % batch_size
                    sample_states = tf.gather(params=preprocessed_states, indices=indices)
                    sample_actions = tf.gather(params=actions, indices=indices)
                    sample_rewards = tf.gather(params=rewards, indices=indices)
                    sample_terminals = tf.gather(params=terminals, indices=indices)

                    policy_probs = self.policy.get_action_log_probs(sample_states, sample_actions)
                    baseline_values = self.value_function.value_output(sample_states)

                    loss, loss_per_item, vf_loss, vf_loss_per_item = self_.get_sub_component_by_name(loss_function.scope).loss(
                        policy_probs["action_log_probs"], baseline_values, actions, sample_rewards,
                        sample_terminals, policy_probs["logits"]
                    )

                    policy_vars = self_.get_sub_component_by_name(policy_scope)._variables()
                    vf_vars = self_.get_sub_component_by_name(value_function_scope)._variables()

                    step_op, loss, loss_per_item = self_.get_sub_component_by_name(optimizer_scope).step(
                        policy_vars, loss, loss_per_item)
                    loss.set_shape([0])
                    loss_per_item.set_shape((self.sample_size, ))

                    vf_step_op, vf_loss, vf_loss_per_item = self_.get_sub_component_by_name(vf_optimizer_scope).step(
                        vf_vars, vf_loss, vf_loss_per_item)
                    vf_loss.set_shape([0])
                    vf_loss_per_item.set_shape((self.sample_size, ))

                    with tf.control_dependencies([step_op, vf_step_op]):
                        return index + 1, loss, loss_per_item, vf_loss, vf_loss_per_item

                def cond(index, loss, loss_per_item, v_loss, v_loss_per_item):
                    return index < self.iterations

                index, loss, loss_per_item, vf_loss, vf_loss_per_item = tf.while_loop(
                    cond=cond,
                    body=opt_body,
                    loop_vars=[0,
                               tf.zeros(0, dtype=tf.float32),
                               tf.zeros(shape=(self.sample_size,)),
                               tf.zeros(0, dtype=tf.float32),
                               tf.zeros(shape=(self.sample_size,))],
                    parallel_iterations=1
                )
                return loss, loss_per_item, vf_loss, vf_loss_per_item

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

    def update(self, batch=None):

        # [0]=the loss; [1]=loss-per-item, [2]=vf-loss, [3] - vf-loss- per item
        return_ops = [0, 1, 2, 3]
        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", None, return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]
        else:
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"]]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input, return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_external_batch"]

        # [1]=the loss (0=update noop)
        # [2]=loss per item for external update, records for update from memory
        return ret[1], ret[2]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.preprocessing_required and len(self.preprocessor.variables) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def __repr__(self):
        return "PPOAgent()"
