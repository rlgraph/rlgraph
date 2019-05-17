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
from rlgraph.components.algorithms.sac_algorithm_component import SACAlgorithmComponent
from rlgraph.spaces import FloatBox, BoolBox, IntBox, ContainerSpace
from rlgraph.utils import RLGraphError
from rlgraph.utils.util import strip_list, force_list


class SACAgent(Agent):
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
        name="sac-agent",
        double_q=True,
        initial_alpha=1.0,
        gumbel_softmax_temperature=1.0,
        target_entropy=None,
        memory_spec=None,
        value_function_sync_spec=None
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
            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Spec-dict to specify `Agent.update()` settings.
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
            internal_states_space=internal_states_space,
            execution_spec=execution_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            auto_build=auto_build,
            name=name
        )

        self.double_q = double_q
        self.target_entropy = target_entropy
        self.initial_alpha = initial_alpha

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
        self.batch_size = self.update_spec["batch_size"]

        float_action_space = self.action_space.with_batch_rank().map(
            mapping=lambda flat_key, space: space.as_one_hot_float_space() if isinstance(space, IntBox) else space
        )

        self.input_spaces.update(dict(
            env_actions=self.action_space.with_batch_rank(),
            actions=float_action_space,
            preprocessed_states=preprocessed_state_space,
            rewards=reward_space,
            terminals=terminal_space,
            next_states=preprocessed_state_space,
            states=self.state_space.with_batch_rank(add_batch_rank=True),
            batch_size=int,
            importance_weights=FloatBox(add_batch_rank=True),
            deterministic=bool,
            weights="variables:{}".format(self.policy.scope)
        ))

        #self.memory = Memory.from_spec(memory_spec)
        #self.alpha_optimizer = self.optimizer.copy(scope="alpha-" + self.optimizer.scope) if self.target_entropy is not None else None

        self.root_component = SACAlgorithmComponent(
            agent=self,
            policy=policy_spec,
            network_spec=network_spec,
            value_function_spec=value_function_spec,  # q-functions
            preprocessing_spec=preprocessing_spec,
            memory_spec=memory_spec,
            discount=discount,
            initial_alpha=self.initial_alpha,
            target_entropy=target_entropy,
            gumbel_softmax_temperature=gumbel_softmax_temperature,
            optimizer_spec=optimizer_spec,
            value_function_optimizer_spec=value_function_optimizer_spec,
            #alpha_optimizer=self.alpha_optimizer,
            q_sync_spec=value_function_sync_spec,
            num_q_functions=2 if self.double_q is True else 1
        )

        #extra_optimizers = []
        #if self.alpha_optimizer is not None:
        #    extra_optimizers.append(self.alpha_optimizer)
        self.build_options = dict(optimizers=self.root_component.all_optimizers)

        if self.auto_build:
            self._build_graph(
                [self.root_component], self.input_spaces, optimizer=self.root_component.optimizer,
                batch_size=self.update_spec["batch_size"],
                build_options=self.build_options
            )
            self.graph_built = True

    def set_weights(self, policy_weights, value_function_weights=None):
        # TODO: Overrides parent but should this be policy of value function?
        return self.graph_executor.execute((self.root_component.set_policy_weights, policy_weights))

    def get_weights(self):
        return dict(policy_weights=self.graph_executor.execute(self.root_component.get_policy_weights))

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
            call_method = self.root_component.get_actions
            batched_states = self.state_space.force_batch(states)
        else:
            call_method = self.root_component.get_actions_from_preprocessed_states
            batched_states = states
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1

        # Increase timesteps by the batch size (number of states in batch).
        batch_size = len(batched_states)
        self.timesteps += batch_size

        # Control, which return value to "pull" (depending on `additional_returns`).
        return_ops = [0, 1] if "preprocessed_states" in extra_returns else [0]
        ret = force_list(self.graph_executor.execute((
            call_method,
            [batched_states, not use_exploration],  # deterministic = not use_exploration
            # 0=preprocessed_states, 1=action
            return_ops
        )))
        # Convert Gumble (relaxed one-hot) sample back into int type for all discrete composite actions.
        if isinstance(self.action_space, ContainerSpace):
            ret[0] = ret[0].map(
                mapping=lambda key, action: np.argmax(action, axis=-1).astype(action.dtype)
                if isinstance(self.flat_action_space[key], IntBox) else action
            )
        elif isinstance(self.action_space, IntBox):
            ret[0] = np.argmax(ret[0], axis=-1).astype(self.action_space.dtype)

        if remove_batch_rank:
            ret[0] = strip_list(ret[0])

        if "preprocessed_states" in extra_returns:
            return ret[0], ret[1]
        else:
            return ret[0]

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, next_states, terminals]))

    def update(self, batch=None):
        if batch is None:
            size = self.graph_executor.execute(self.root_component.get_memory_size)
            # TODO: is this necessary?
            if size < self.batch_size:
                return 0.0, 0.0, 0.0
            ret = self.graph_executor.execute((self.root_component.update_from_memory, [self.batch_size]))
        else:
            # No sequence indices means terminals are used in place.
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"], batch["next_states"]]
            ret = self.graph_executor.execute((self.root_component.update_from_external_batch, batch_input))

        return ret["actor_loss"], ret["actor_loss_per_item"], ret["critic_loss"], ret["alpha_loss"]

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
            self.double_q, self.initial_alpha, self.target_entropy
        )
