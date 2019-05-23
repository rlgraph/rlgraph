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
from rlgraph.components import Memory, RingBuffer
from rlgraph.components.algorithms.algorithm_component import AlgorithmComponent
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
            update_spec=None,
            summary_spec=None,
            saver_spec=None,
            auto_build=True,
            name="actor-critic-agent",
            gae_lambda=1.0,
            clip_rewards=0.0,
            sample_episodes=False,
            weight_pg=None,
            weight_vf=None,
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
            observe_spec (Optional[dict]): Obsoleted: Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Obsoleted: Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.
            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.
            name (str): Some name for this Agent object.
            gae_lambda (float): Lambda for generalized advantage estimation.
            clip_rewards (float): Reward clip value. If not 0, rewards will be clipped into this range.
            sample_episodes (bool): If true, the update method interprets the batch_size as the number of
                episodes to fetch from the memory. If false, batch_size will refer to the number of time-steps. This
                is especially relevant for environments where episode lengths may vastly differ throughout training. For
                example, in CartPole, a losing episode is typically 10 steps, and a winning episode 200 steps.
            weight_pg (float): The coefficient used for the policy gradient loss term (L[PG]).
            weight_vf (float): The coefficient used for the state value function loss term (L[V]).
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use. Should typically be
            a ring-buffer.
        """
        super(ActorCriticAgent, self).__init__(
            state_space=state_space,
            action_space=action_space,
            python_buffer_size=python_buffer_size,
            custom_python_buffers=custom_python_buffers,
            internal_states_space=internal_states_space,
            execution_spec=execution_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,  # Obsoleted
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            name=name
        )

        self.root_component = ActorCriticAlgorithmComponent(
            self, memory_batch_size=memory_batch_size, discount=discount,
            preprocessing_spec=preprocessing_spec, network_spec=network_spec, policy_spec=policy_spec,
            value_function_spec=value_function_spec, optimizer_spec=optimizer_spec,
            value_function_optimizer_spec=value_function_optimizer_spec,
            gae_lambda=gae_lambda, clip_rewards=clip_rewards,
            sample_episodes=sample_episodes,
            weight_pg=weight_pg, weight_vf=weight_vf, weight_entropy=weight_entropy,
            memory_spec=memory_spec
        )

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.root_component.preprocessed_state_space.with_batch_rank()
        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            policy_weights="variables:{}".format(self.root_component.policy.scope),
            deterministic=bool,
            preprocessed_states=preprocessed_state_space,
            rewards=FloatBox(add_batch_rank=True),
            advantages=FloatBox(add_batch_rank=True),
            terminals=BoolBox(add_batch_rank=True),
            sequence_indices=BoolBox(add_batch_rank=True)
        ))

        if auto_build is True:
            build_options = dict(vf_optimizer=self.root_component.value_function_optimizer)
            self.build(build_options=build_options)

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
        #if time_percentage is None:
        #    time_percentage = self.timesteps / (self.max_timesteps or 1e6)

        extra_returns = [extra_returns] if isinstance(extra_returns, str) else (extra_returns or list())
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

        ret = self.graph_executor.execute((
            call_method,
            [batched_states, not use_exploration],  # deterministic = not use_exploration
            # Control, which return value to "pull" (depending on `extra_returns`).
            ["actions"] + extra_returns
        ))
        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals, **kwargs):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None, time_percentage=None, sequence_indices=None, apply_postprocessing=True):
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
        if time_percentage is None:
            time_percentage = self.timesteps / (self.max_timesteps or 1e6)

        self.num_updates += 1

        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", [time_percentage]))
        else:
            # No sequence indices means terminals are used in place.
            if sequence_indices is None:
                sequence_indices = batch["terminals"]

            pps_dtype = self.root_component.preprocessed_state_space.dtype
            batch["states"] = np.asarray(batch["states"], dtype=util.convert_dtype(dtype=pps_dtype, to='np'))
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"], sequence_indices,
                           time_percentage]

            # Execute post-processing or already post-processed by workers?
            if apply_postprocessing:
                ret = self.graph_executor.execute(("post_process_and_update", batch_input))
            else:
                ret = self.graph_executor.execute(("update_from_external_batch", batch_input))

        return ret["loss"] + ret["vf_loss"], ret["loss_per_item"] + ret["vf_loss_per_item"]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.root_component.preprocessing_required and len(self.root_component.preprocessor.variable_registry) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def __repr__(self):
        return "ActorCriticAgent"


class ActorCriticAlgorithmComponent(AlgorithmComponent):

    def __init__(self, agent, memory_spec, gae_lambda, clip_rewards, weight_pg=None, weight_vf=None,
                 weight_entropy=None, sample_episodes=False, scope="actor-critic-algorithm-component", **kwargs):

        # Set policy to stochastic.
        policy_spec = kwargs.pop("policy_spec", None)
        if policy_spec is not None:
            policy_spec["deterministic"] = False
        else:
            policy_spec = dict(deterministic=False)

        super(ActorCriticAlgorithmComponent, self).__init__(agent, policy_spec=policy_spec, scope=scope, **kwargs)

        self.sample_episodes = sample_episodes

        self.memory = Memory.from_spec(memory_spec)
        assert isinstance(self.memory, RingBuffer), \
            "ERROR: Actor-critic memory must be ring-buffer for episode-handling."
        self.gae_function = GeneralizedAdvantageEstimation(gae_lambda=gae_lambda, discount=self.discount,
                                                           clip_rewards=clip_rewards)
        self.loss_function = ActorCriticLossFunction(
            weight_pg=weight_pg, weight_vf=weight_vf, weight_entropy=weight_entropy
        )

        # Add all our sub-components to the core.
        self.add_components(self.memory, self.gae_function, self.loss_function)

    ## Act from preprocessed states.
    #@rlgraph_api
    #def action_from_preprocessed_state(self, preprocessed_states, deterministic=False):
    #    out = self.policy.get_action(preprocessed_states, deterministic=deterministic)
    #    return dict(actions=out["action"], preprocessed_states=preprocessed_states)

    ## State (from environment) to action with preprocessing.
    #@rlgraph_api
    #def get_preprocessed_state_and_action(self, states, deterministic=False):
    #    preprocessed_states = self.preprocessor.preprocess(states)
    #    return self.action_from_preprocessed_state(preprocessed_states, deterministic)

    # Insert into memory.
    @rlgraph_api
    def insert_records(self, preprocessed_states, actions, rewards, terminals):
        records = dict(states=preprocessed_states, actions=actions, rewards=rewards, terminals=terminals)
        return self.memory.insert_records(records)

    @rlgraph_api
    def post_process(self, preprocessed_states, rewards, terminals, sequence_indices):
        state_values = self.value_function.value_output(preprocessed_states)
        pg_advantages = self.gae_function.calc_gae_values(state_values, rewards, terminals, sequence_indices)
        return pg_advantages

    # Learn from memory.
    @rlgraph_api
    def update_from_memory(self, time_percentage=None):
        if self.sample_episodes:
            records = self.memory.get_episodes(self.memory_batch_size)
        else:
            records = self.memory.get_records(self.memory_batch_size)
        sequence_indices = records["terminals"]  # TODO: return sequence_indeces from memory (optionally)
        return self.post_process_and_update(
            records["states"], records["actions"], records["rewards"], records["terminals"],
            sequence_indices, time_percentage
        )

    # First post-process, then update (so we can separately update already post-processed data).
    @rlgraph_api
    def post_process_and_update(self, preprocessed_states, actions, rewards, terminals, sequence_indices,
                                time_percentage=None):
        advantages = self.post_process(preprocessed_states, rewards, terminals, sequence_indices)
        return self.update_from_external_batch(preprocessed_states, actions, advantages, terminals, time_percentage)

    # Learn from an external batch.
    @rlgraph_api
    def update_from_external_batch(self, preprocessed_states, actions, advantages, terminals, time_percentage=None):
        baseline_values = self.value_function.value_output(preprocessed_states)
        log_probs = self.policy.get_log_likelihood(preprocessed_states, actions)["log_likelihood"]
        entropy = self.policy.get_entropy(preprocessed_states)["entropy"]
        loss, loss_per_item, vf_loss, vf_loss_per_item = self.loss_function.loss(
            log_probs, baseline_values, advantages, entropy, time_percentage
        )

        # Args are passed in again because some device strategies may want to split them to different devices.
        policy_vars = self.policy.variables()
        vf_vars = self.value_function.variables()

        step_op = self.optimizer.step(policy_vars, loss, loss_per_item, time_percentage)
        vf_step_op = self.value_function_optimizer.step(vf_vars, vf_loss, vf_loss_per_item, time_percentage)
        step_op = self._graph_fn_group(step_op, vf_step_op)

        return dict(
            step_op=step_op, loss=loss, loss_per_item=loss_per_item,
            vf_loss=vf_loss, vf_loss_per_item=vf_loss_per_item
        )

