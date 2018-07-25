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

from rlgraph.agents.agent import Agent
from rlgraph.components.loss_functions.impala_loss_function import IMPALALossFunction
from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.spaces import FloatBox, BoolBox


class IMPALAAgent(Agent):
    """
    An Agent implementing the IMPALA algorithm described in [1]. The Agent contains both learner and explorer
    API-methods, which will be put into the graph depending on the type ().

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, **kwargs):
        """
        Keyword Args:
            type (str): One of "explorer" or "learner". Default: "explorer".
        """
        type_ = kwargs.pop("type", "explorer")
        assert type_ in ["explorer", "learner"]
        self.type = type_

        # Depending on the job-type, remove the pieces from the Agent-spec/graph we won't need.
        exploration_spec = kwargs.pop("exploration_spec", None),
        optimizer_spec = kwargs.pop("optimizer_spec", None)
        observe_spec = kwargs.pop("observe_spec", None)
        # Learners won't need  to explore (act) or observe (insert into Queue).
        if self.type == "learner":
            exploration_spec = None
            observe_spec = None
            update_spec = kwargs.pop("update_spec", None)
        # Explorers won't need to learn (no optimizer needed in graph).
        else:
            optimizer_spec = None
            update_spec = kwargs.pop("update_spec", dict(do_updates=False))

        # Now that we fixed the Agent's spec, call the super constructor.
        super(IMPALAAgent, self).__init__(
            exploration_spec=exploration_spec, optimizer_spec=optimizer_spec, observe_spec=observe_spec,
            update_spec=update_spec, name=kwargs.pop("name", "impala-{}-agent".format(self.type)), **kwargs
        )

        # Create our FIFOQueue (explorers will enqueue, learner(s) will dequeue).
        self.fifo_queue = FIFOQueue()

        # Extend input Space definitions to this Agent's specific API-methods.
        state_space = self.state_space.with_batch_rank()
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        action_space = self.action_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)
        self.input_spaces.update(dict(
            perform_n_steps_and_insert_into_fifo=[state_space, int, bool],  # state, time-step, use_exploration
        ))

        # Add all our sub-components to the core.
        if self.type == "explorer":
            self.loss_function = None
            sub_components = [self.preprocessor, self.policy, self.exploration, self.fifo_queue]
        else:
            # TODO: add loss func options here and to our ctor.
            self.loss_function = IMPALALossFunction()
            sub_components = [self.fifo_queue, self.policy, self.loss_function, self.optimizer]

        self.core_component.add_components(*sub_components)

        # Define the Agent's (core Component's) API.
        self.define_api_methods(*sub_components)

        # markup = get_graph_markup(self.graph_builder.core_component)
        # print(markup)
        self.build_graph(self.input_spaces, self.optimizer)

    def define_api_methods(self, *sub_components):
        super(IMPALAAgent, self).define_api_methods()

        if self.type == "explorer":
            self.define_api_methods_explorer(*sub_components)
        else:
            self.define_api_methods_learner(*sub_components)

    def define_api_methods_explorer(self, preprocessor, policy, exploration, fifo_queue):
        """
        Defines the API-methods used by an IMPALA explorer. Explorers only step through an environment (n-steps at
        a time), collect the results and push them into the FIFO queue. Results include: The actions actually
        taken, the discounted accumulated returns for each action, the probability of each taken action according to
        the behavior policy.

        Args:
            preprocessor (PreprocessorStack): The PreprocessorStack to preprocess all states coming from the env.
            policy (Policy): The Policy Component used to compute actions.
            exploration (Exploration): The Exploration
            fifo_queue (FIFOQueue): The FIFOQueue Component used to enqueue env sample runs (n-step).
        """
        # Perform n-steps in the env and insert the results into our FIFO-queue.
        def perform_n_steps_and_insert_into_fifo(self_, initial_states, initial_time_step, use_exploration=True):
            # TODO: use a new n-step Component to step through the env and collect results.
            preprocessed_states = self_.call(preprocessor.preprocess, states)
            sample_deterministic = self_.call(policy.sample_deterministic, preprocessed_states)
            sample_stochastic = self_.call(policy.sample_stochastic, preprocessed_states)
            actions = self_.call(exploration.get_action, sample_deterministic, sample_stochastic,
                                 time_step, use_exploration)
            return preprocessed_states, actions

        self.core_component.define_api_method(
            "perform_n_steps_and_insert_into_fifo", perform_n_steps_and_insert_into_fifo
        )

    def define_api_methods_learner(self, fifo_queue, policy, loss_function, optimizer):

        # Learn from memory.
        def update_from_fifo_queue(self_):
            records = self_.call(memory.get_records, self.update_spec["batch_size"])

            preprocessed_s, actions, rewards, terminals, preprocessed_s_prime = self_.call(splitter.split,
                                                                                           records)

            # Get the different Q-values.
            q_values_s = self_.call(policy.get_q_values, preprocessed_s)
            qt_values_sp = self_.call(target_policy.get_q_values, preprocessed_s_prime)
            q_values_sp = None
            if self.double_q:
                q_values_sp = self_.call(policy.get_q_values, preprocessed_s_prime)

            loss, loss_per_item = self_.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                                             qt_values_sp, q_values_sp)

            policy_vars = self_.call(policy._variables)
            step_op = self_.call(optimizer.step, policy_vars, loss)

            # TODO: For multi-GPU, the final-loss will probably have to come from the optimizer.
            return step_op, loss, loss_per_item, records, q_values_s

        self.core_component.define_api_method("update_from_fifo_queue", update_from_fifo_queue)

    def get_action(self, states, internals=None, use_exploration=True, extra_returns=None):
        """
        Args:
            extra_returns (Optional[Set[str],str]): Optional string or set of strings for additional return
                values (besides the actions). Possible values are:
                - 'preprocessed_states': The preprocessed states after passing the given states
                    through the preprocessor stack.
                - 'internal_states': The internal states returned by the RNNs in the NN pipeline.
                - 'used_exploration': Whether epsilon- or noise-based exploration was used or not.

        Returns:
            tuple or single value depending on `extra_returns`:
                - action
                - the preprocessed states
        """
        extra_returns = {extra_returns} if isinstance(extra_returns, str) else (extra_returns or set())

        batched_states = self.state_space.force_batch(states)
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1

        # Increase timesteps by the batch size (number of states in batch).
        self.timesteps += len(batched_states)

        # Control, which return value to "pull" (depending on `additional_returns`).
        return_ops = [1, 0] if "preprocessed_states" in extra_returns else [1]
        ret = self.graph_executor.execute((
            "get_preprocessed_state_and_action",
            [batched_states, self.timesteps, use_exploration],
            # 0=preprocessed_states, 1=action
            return_ops
        ))
        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None):
        # [0]=no-op step; [1]=the loss; [2]=loss-per-item, [3]=memory-batch (if pulled); [4]=q-values
        return_ops = [0, 1, 2]
        q_table = None

        if batch is None:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            if self.store_last_q_table is True:
                return_ops += [3, 4]  # 3=batch, 4=q-values
            elif self.store_last_memory_batch is True:
                return_ops += [3]  # 3=batch
            ret = self.graph_executor.execute(("update_from_memory", None, return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]

            # Store the last Q-table?
            if self.store_last_q_table is True:
                q_table = dict(
                    states=ret[3]["states"],
                    q_values=ret[4]
                )
        else:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            if self.store_last_q_table is True:
                return_ops += [3]  # 3=q-values

            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"],
                           batch["next_states"], batch["importance_weights"]]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input, return_ops))

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_external_batch"]

            # Store the last Q-table?
            if self.store_last_q_table is True:
                q_table = dict(
                    states=batch["states"],
                    q_values=ret[3]
                )

        # Store the latest pulled memory batch?
        if self.store_last_memory_batch is True and batch is None:
            self.last_memory_batch = ret[2]
        if self.store_last_q_table is True:
            self.last_q_table = q_table

        # [1]=the loss (0=update noop)
        # [2]=loss per item for external update, records for update from memory
        return ret[1], ret[2]

    def __repr__(self):
        return "IMPALAAgent()"

