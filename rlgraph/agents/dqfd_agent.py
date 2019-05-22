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
from rlgraph.components import Memory, PrioritizedReplay, ContainerMerger, ContainerSplitter, DQFDLossFunction
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.utils import RLGraphError
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import strip_list


class DQFDAgent(Agent):
    """
    Deep-Q-learning from demonstration is an extension compatible with Double/Dueling DQN which
    uses a supervised large-margin loss to pretrain an agent from expert demonstrations. Paper:

    https://arxiv.org/abs/1704.03732
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
        exploration_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        observe_spec=None,
        update_spec=None,
        summary_spec=None,
        saver_spec=None,
        auto_build=True,
        name="dqfd-agent",
        expert_margin=0.5,
        supervised_weight=1.0,
        double_q=True,
        dueling_q=True,
        huber_loss=False,
        n_step=1,
        shared_container_action_target=False,
        memory_spec=None,
        demo_memory_spec=None,
        demo_sample_ratio=0.2,
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
            exploration_spec (Optional[dict]): The spec-dict to create the Exploration Component.
            execution_spec (Optional[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.
            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.
            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.
            name (str): Some name for this Agent object.
            expert_margin (float): The expert margin enforces a distance in Q-values between expert action and
                all other actions.
            supervised_weight (float): Indicates weight of the expert loss.
            double_q (bool): Whether to use the double DQN loss function (see [2]).
            dueling_q (bool): Whether to use a dueling layer in the ActionAdapter  (see [3]).
            huber_loss (bool) : Whether to apply a Huber loss. (see [4]).
            n_step (Optional[int]): n-step adjustment to discounting.
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use.
            demo_memory_spec (Optional[dict,Memory]): The spec for the Demo-Memory to use.
        """
        # Fix action-adapter before passing it to the super constructor.
        # Use a DuelingPolicy (instead of a basic Policy) if option is set.
        if dueling_q is True:
            if policy_spec is None:
                policy_spec = {}
            policy_spec["type"] = "dueling-policy"
            # Give us some default state-value nodes.
            if "units_state_value_stream" not in policy_spec:
                policy_spec["units_state_value_stream"] = 128
        super(DQFDAgent, self).__init__(
            state_space=state_space,
            action_space=action_space,
            discount=discount,
            preprocessing_spec=preprocessing_spec,
            network_spec=network_spec,
            internal_states_space=internal_states_space,
            policy_spec=policy_spec,
            exploration_spec=exploration_spec,
            execution_spec=execution_spec,
            optimizer_spec=optimizer_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            auto_build=auto_build,
            name=name
        )
        # Assert that the synch interval is a multiple of the update_interval.
        if self.update_spec["sync_interval"] / self.update_spec["update_interval"] != \
                self.update_spec["sync_interval"] // self.update_spec["update_interval"]:
            raise RLGraphError(
                "ERROR: sync_interval ({}) must be multiple of update_interval "
                "({})!".format(self.update_spec["sync_interval"], self.update_spec["update_interval"])
            )

        self.double_q = double_q
        self.dueling_q = dueling_q
        self.huber_loss = huber_loss
        self.expert_margin = expert_margin

        self.batch_size = self.update_spec["batch_size"]
        self.default_margins = np.asarray([self.expert_margin] * self.batch_size)

        self.demo_batch_size = int(demo_sample_ratio * self.update_spec["batch_size"] / (1.0 - demo_sample_ratio))
        self.demo_margins = np.asarray([self.expert_margin] * self.demo_batch_size)
        self.shared_container_action_target = shared_container_action_target

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)
        weight_space = FloatBox(add_batch_rank=True)

        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            policy_weights="variables:{}".format(self.policy.scope),
            time_step=int,
            use_exploration=bool,
            demo_batch_size=int,
            apply_demo_loss=bool,
            preprocessed_states=preprocessed_state_space,
            rewards=reward_space,
            terminals=terminal_space,
            expert_margins=FloatBox(add_batch_rank=True),
            next_states=preprocessed_state_space,
            preprocessed_next_states=preprocessed_state_space,
            importance_weights=weight_space
        ))

        # The merger to merge inputs into one record Dict going into the memory.
        self.merger = ContainerMerger("states", "actions", "rewards", "next_states", "terminals")

        # The replay memory.
        self.memory = Memory.from_spec(memory_spec)
        # Cannot have same default name.
        demo_memory_spec["scope"] = "demo-memory"
        self.demo_memory = Memory.from_spec(demo_memory_spec)

        # The splitter for splitting up the records from the memories.
        self.splitter = ContainerSplitter("states", "actions", "rewards", "terminals", "next_states")

        # Copy our Policy (target-net), make target-net synchronizable.
        self.target_policy = self.policy.copy(scope="target-policy", trainable=False)
        # Number of steps since the last target-net synching from the main policy.
        self.steps_since_target_net_sync = 0

        self.use_importance_weights = isinstance(self.memory, PrioritizedReplay)
        self.loss_function = DQFDLossFunction(
            supervised_weight=supervised_weight,
            discount=self.discount, double_q=self.double_q, huber_loss=self.huber_loss,
            shared_container_action_target=shared_container_action_target,
            importance_weights=self.use_importance_weights, n_step=n_step
        )

        # Add all our sub-components to the core.
        self.root_component.add_components(
            self.preprocessor, self.merger, self.memory, self.demo_memory, self.splitter, self.policy,
            self.target_policy, self.exploration, self.loss_function, self.optimizer
        )

        # Define the Agent's (root-Component's) API.
        self.define_graph_api()

        if self.auto_build:
            self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                              batch_size=self.update_spec["batch_size"])
            self.graph_built = True

    def define_graph_api(self):
        super(DQFDAgent, self).define_graph_api()

        agent = self

        # Reset operation (resets preprocessor).
        if self.preprocessing_required:
            @rlgraph_api(component=self.root_component)
            def reset_preprocessor(root):
                reset_op = agent.preprocessor.reset()
                return reset_op

        # Act from preprocessed states.
        @rlgraph_api(component=self.root_component)
        def action_from_preprocessed_state(root, preprocessed_states, time_step=0, use_exploration=True):
            sample_deterministic = agent.policy.get_deterministic_action(preprocessed_states)
            actions = agent.exploration.get_action(sample_deterministic["action"], time_step, use_exploration)
            return actions, preprocessed_states

        # State (from environment) to action with preprocessing.
        @rlgraph_api(component=self.root_component)
        def get_preprocessed_state_and_action(root, states, time_step=0, use_exploration=True):
            preprocessed_states = agent.preprocessor.preprocess(states)
            return root.action_from_preprocessed_state(preprocessed_states, time_step, use_exploration)

        # Insert into memory.
        @rlgraph_api(component=self.root_component)
        def insert_records(root, preprocessed_states, actions, rewards, next_states, terminals):
            records = agent.merger.merge(preprocessed_states, actions, rewards, next_states, terminals)
            return agent.memory.insert_records(records)

        # Insert into demo memory.
        @rlgraph_api(component=self.root_component)
        def insert_demos(root, preprocessed_states, actions, rewards, next_states, terminals):
            records = agent.merger.merge(preprocessed_states, actions, rewards, next_states, terminals)
            return agent.demo_memory.insert_records(records)

        # Syncing target-net.
        @rlgraph_api(component=self.root_component)
        def sync_target_qnet(root):
            # If we are a multi-GPU root:
            # Simply feeds everything into the multi-GPU sync optimizer's method and return.
            if "multi-gpu-synchronizer" in root.sub_components:
                multi_gpu_syncer = root.sub_components["multi-gpu-synchronizer"]
                return multi_gpu_syncer.sync_target_qnets()
            # We could be the main root or a multi-GPU tower.
            else:
                policy_vars = root.get_sub_component_by_name(agent.policy.scope).variables()
                return root.get_sub_component_by_name(agent.target_policy.scope).sync(policy_vars)

        # Learn from online memory AND demo memory.
        @rlgraph_api(component=self.root_component)
        def update_from_memory(root, apply_demo_loss, expert_margins, time_percentage=None):
            # Sample from online memory.
            records, sample_indices, importance_weights = agent.memory.get_records(self.batch_size)
            preprocessed_s, actions, rewards, terminals, preprocessed_s_prime = agent.splitter.call(records)

            # Do not apply demo loss to online experience.
            online_step_op, loss, loss_per_item = root.update_from_external_batch(
                preprocessed_s, actions, rewards, terminals, preprocessed_s_prime, importance_weights, apply_demo_loss,
                expert_margins, time_percentage
            )

            if isinstance(agent.memory, PrioritizedReplay):
                update_pr_step_op = agent.memory.update_records(sample_indices, loss_per_item)
                return online_step_op, loss, loss_per_item, update_pr_step_op
            else:
                return online_step_op, loss, loss_per_item

        # Learn from demo-data.
        @rlgraph_api(component=self.root_component)
        def update_from_demos(root, demo_batch_size, apply_demo_loss, expert_margins, time_percentage=None):
            # Sample from demo memory.
            records, sample_indices, importance_weights = agent.demo_memory.get_records(demo_batch_size)
            preprocessed_s, actions, rewards, terminals, preprocessed_s_prime = agent.splitter.call(records)
            step_op, loss, loss_per_item, q_values_s = root.update_from_external_batch(
                preprocessed_s, actions, rewards, terminals, preprocessed_s_prime, importance_weights, apply_demo_loss,
                expert_margins, time_percentage
            )
            return step_op, loss, loss_per_item

        # Learn from an external batch - note the flag to apply demo loss.
        @rlgraph_api(component=self.root_component)
        def update_from_external_batch(
                root, preprocessed_states, actions, rewards, terminals, preprocessed_next_states, importance_weights,
                apply_demo_loss, expert_margins, time_percentage=None
        ):
            # If we are a multi-GPU root:
            # Simply feeds everything into the multi-GPU sync optimizer's method and return.
            if "multi-gpu-synchronizer" in root.sub_components:
                main_policy_vars = agent.policy.variables()
                # TODO: This may be called differently in other agents (replace by root-policy).
                grads_and_vars, loss, loss_per_item, q_values_s = \
                    root.sub_components["multi-gpu-synchronizer"].calculate_update_from_external_batch(
                        dict(policy=main_policy_vars), preprocessed_states, actions, rewards, terminals,
                        preprocessed_next_states, importance_weights, apply_demo_loss, expert_margins, time_percentage
                    )
                step_op = agent.optimizer.apply_gradients(grads_and_vars)
                # Increase the global training step counter.
                step_op = root._graph_fn_training_step(step_op)
                step_and_sync_op = root.sub_components["multi-gpu-synchronizer"].sync_variables_to_towers(
                    step_op, main_policy_vars
                )
                return step_and_sync_op, loss, loss_per_item

            # Get sub-components relative to the root (could be multi-GPU setup where root=some-tower).
            policy = root.get_sub_component_by_name(agent.policy.scope)
            target_policy = root.get_sub_component_by_name(agent.target_policy.scope)
            loss_function = root.get_sub_component_by_name(agent.loss_function.scope)
            optimizer = root.get_sub_component_by_name(agent.optimizer.scope)

            # Get the different Q-values.
            q_values_s = policy.get_adapter_outputs(preprocessed_states)["adapter_outputs"]
            qt_values_sp = target_policy.get_adapter_outputs(preprocessed_next_states)["adapter_outputs"]

            q_values_sp = None
            if self.double_q:
                q_values_sp = policy.get_adapter_outputs(preprocessed_next_states)["adapter_outputs"]

            loss, loss_per_item = loss_function.loss(
                q_values_s, actions, rewards, terminals, qt_values_sp, expert_margins, q_values_sp,
                importance_weights, apply_demo_loss, time_percentage
            )

            # Args are passed in again because some device strategies may want to split them to different devices.
            policy_vars = policy.variables()
            if hasattr(root, "is_multi_gpu_tower") and root.is_multi_gpu_tower is True:
                grads_and_vars = optimizer.calculate_gradients(policy_vars, loss, time_percentage)
                return grads_and_vars, loss, loss_per_item
            else:
                step_op = optimizer.step(policy_vars, loss, loss_per_item, time_percentage)
                # Increase the global training step counter.
                step_op = root._graph_fn_training_step(step_op)
                return step_op, loss, loss_per_item

        @rlgraph_api(component=self.root_component)
        def get_td_loss(root, preprocessed_states, actions, rewards,
                        terminals, preprocessed_next_states, importance_weights, apply_demo_loss, expert_margins,
                        time_percentage=None):

            policy = root.get_sub_component_by_name(agent.policy.scope)
            target_policy = root.get_sub_component_by_name(agent.target_policy.scope)
            loss_function = root.get_sub_component_by_name(agent.loss_function.scope)

            # Get the different Q-values.
            q_values_s = policy.get_adapter_outputs(preprocessed_states)["adapter_outputs"]
            qt_values_sp = target_policy.get_adapter_outputs(preprocessed_next_states)["adapter_outputs"]

            q_values_sp = None
            if self.double_q:
                q_values_sp = policy.get_adapter_outputs(preprocessed_next_states)["adapter_outputs"]

            loss, loss_per_item = loss_function.loss(
                q_values_s, actions, rewards, terminals, qt_values_sp, expert_margins,
                q_values_sp, importance_weights, apply_demo_loss, time_percentage
            )
            return loss, loss_per_item

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
            [batched_states, self.timesteps, use_exploration],
            return_ops
        ))
        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, next_states, terminals]))

    def update(self, batch=None, time_percentage=None, update_from_demos=False, expert_margins=None,
               apply_demo_loss_to_batch=False):
        """
        Updates from external batch or replay memory.
        Args:
            batch (Optional[dict]): Optional dict to use for updating. If false, samples from replay memory
            update_from_demos (bool): If true, also updates from demo memory by sampling demonstrations from
                demo memory. Default false (only updating from main replay memory).
            expert_margins (SingleDataOp): The expert margin enforces a distance in Q-values between expert action and
                all other actions.
            apply_demo_loss_to_batch (bool): If true and an external batch is given, demo loss is applied to this
                batch. Can be combined with external per-sample expert margins. The purpose of this is to update
                from external demonstration data where each sample has a different margin. If false and an
                external batch is given, the agent updates this with the normal Double/Dueling-q loss. Default
                false. Only valid if batch is not None.
        Returns:
            tuple: Loss and loss per item.
        """
        # TODO: Move update_spec to Worker. Agent should not hold these execution details.
        if time_percentage is None:
            time_percentage = self.timesteps / self.update_spec.get("max_timesteps", 1e6)

        # Should we sync the target net?
        self.steps_since_target_net_sync += self.update_spec["update_interval"]
        if self.steps_since_target_net_sync >= self.update_spec["sync_interval"]:
            sync_call = "sync_target_qnet"
            self.steps_since_target_net_sync = 0
        else:
            sync_call = None

        # Update from replay memory.  Potentially also update from demo memory with default margins.
        if batch is None:
            # Combine: Update from memory (apply_demo_loss=False), update_from_demo (apply=True).
            # Otherwise only update from online memory.
            if update_from_demos:
                # Use default margins whe sampling from memory.
                ret = self.graph_executor.execute(
                    ("update_from_memory", [False, self.default_margins, time_percentage]),
                    ("update_from_demos", [self.demo_batch_size, True, self.demo_margins, time_percentage]),
                    sync_call
                )
            else:
                ret = self.graph_executor.execute(
                    ("update_from_memory",  [False, self.default_margins, time_percentage]),
                    sync_call
                )

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]
        else:
            # Update from external batch, optionally applying demo loss. Also optionally
            # sample demos from separate demo memory.
            if expert_margins is None:
                # Default margins with correct len.
                expert_margins = np.asarray([self.expert_margin] * len(batch["terminals"]))

            # Apply demo loss to external flag depending on batch.
            # Expert margins only have effect if True.
            batch_input = [
                batch["states"], batch["actions"], batch["rewards"], batch["terminals"], batch["next_states"],
                batch["importance_weights"], apply_demo_loss_to_batch, expert_margins, time_percentage
            ]

            if update_from_demos:
                ret = self.graph_executor.execute(
                    ("update_from_external_batch", batch_input),
                    ("update_from_demos", [self.demo_batch_size, True, self.demo_margins, time_percentage]),
                    sync_call
                )
            else:
                # Only update from external batch (which may use demo loss depending on flag.
                ret = self.graph_executor.execute(
                    ("update_from_external_batch", batch_input),
                    sync_call
                )

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
        if self.preprocessing_required and len(self.preprocessor.variable_registry) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def update_from_demos(self, batch_size=None, time_percentage=None, num_updates=1):
        """
        Executes a number of updates by sampling from the expert memory.

        Args:
            num_updates (int): The number of samples to execute.
            batch_size (Optional[int]): Sampling batch size to use. If None, uses the demo
                batch size computed via the sampling ratio.
        """
        # TODO: Move update_spec to Worker. Agent should not hold these execution details.
        if time_percentage is None:
            time_percentage = self.timesteps / self.update_spec.get("max_timesteps", 1e6)

        if batch_size is None:
            batch_size = self.demo_batch_size
            margins = self.demo_margins
        else:
            margins = np.asarray([self.expert_margin] * batch_size)
        for _ in range(num_updates):
            self.graph_executor.execute(("update_from_demos", [batch_size, True, margins, time_percentage]))

    def observe_demos(self, preprocessed_states, actions, rewards, next_states, terminals):
        """
        Inserts observations into the demonstration memory.
        """
        self.graph_executor.execute(("insert_demos", [preprocessed_states, actions, rewards, next_states, terminals]))

    def __repr__(self):
        return "DQFDAgent(doubleQ={} duelingQ={}, expert margin={})".format(
            self.double_q, self.dueling_q, self.expert_margin
        )
