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
from rlgraph.components.algorithms.ppo_algorithm_component import PPOAlgorithmComponent
from rlgraph.spaces import BoolBox, FloatBox
from rlgraph.utils import util
from rlgraph.utils.util import strip_list


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
            clip_ratio (float): Clipping parameter for likelihood ratio.
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
            internal_states_space=internal_states_space,
            execution_spec=execution_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            name=name,
            auto_build=auto_build
        )
        if policy_spec is not None:
            policy_spec["deterministic"] = False
        else:
            policy_spec = dict(deterministic=False)

        # TODO: Have to manually set it here for multi-GPU synchronizer to know its number
        # TODO: of return values when calling _graph_fn_calculate_update_from_external_batch.
        # self.root_component.graph_fn_num_outputs["_graph_fn_update_from_external_batch"] = 4

        # Change our root-component to PPO.
        self.root_component = PPOAlgorithmComponent(
            agent=self, discount=discount, memory_spec=memory_spec, gae_lambda=gae_lambda, clip_rewards=clip_rewards,
            clip_ratio=clip_ratio, value_function_clipping=value_function_clipping, weight_entropy=weight_entropy,
            preprocessing_spec=preprocessing_spec, policy_spec=policy_spec, network_spec=network_spec,
            value_function_spec=value_function_spec,
            exploration_spec=None, optimizer_spec=optimizer_spec,
            value_function_optimizer_spec=value_function_optimizer_spec,
            sample_episodes=sample_episodes, standardize_advantages=standardize_advantages,
            batch_size=self.update_spec["batch_size"], sample_size=self.update_spec["sample_size"],
            num_iterations=self.update_spec["num_iterations"]
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
            num_records=int
        ))

        self.build_options = dict(vf_optimizer=self.root_component.value_function_optimizer)

        if self.auto_build:
            self._build_graph(
                [self.root_component], self.input_spaces, optimizer=self.root_component.optimizer,
                # Important: Use sample-size, not batch-size as the sub-samples (from a batch) are the ones that get
                # multi-gpu-split.
                batch_size=self.update_spec["sample_size"],
                build_options=self.build_options
            )
            self.graph_built = True

    # TODO: Move into Agent base class (it's always the same duplicate code).
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
            call_method = "get_actions"
            batched_states = self.state_space.force_batch(states)
        else:
            call_method = "get_actions_from_preprocessed_states"
            batched_states = states
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1

        # Increase timesteps by the batch size (number of states in batch).
        batch_size = len(batched_states)
        self.timesteps += batch_size

        # Control, which return value to "pull" (depending on `additional_returns`).
        return_ops = ["actions"] + list(extra_returns)
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

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals, **kwargs):
        sequence_indices = kwargs.pop("sequence_indices")
        self.graph_executor.execute(
            ("insert_records", [preprocessed_states, actions, rewards, terminals, sequence_indices])
        )

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

        # [0] = the loss; [1] = loss-per-item, [2] = vf-loss, [3] = vf-loss- per item
        return_ops = [0, 1, 2, 3]
        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", [True], return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]
        else:
            # No sequence indices means terminals are used in place.
            if sequence_indices is None:
                sequence_indices = batch["terminals"]

            pps_dtype = self.root_component.preprocessed_state_space.dtype
            batch["states"] = np.asarray(batch["states"], dtype=util.convert_dtype(dtype=pps_dtype, to='np'))
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"],
                           sequence_indices, apply_postprocessing]

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

        # [0/2] policy loss + vf loss, [1/3] losses per item
        return ret[0] + ret[2], ret[1] + ret[3]

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
