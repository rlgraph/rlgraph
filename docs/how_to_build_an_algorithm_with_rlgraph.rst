F.. Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   ============================================================================

.. image:: images/rlcore-logo-full.png
   :scale: 25%
   :alt:

How to Write Our Own Agent?
===========================

In the following, we will build a the 2015 DQN Algorithm from scratch and code its logic into a DQNAgent class
using only RLgraph's already existing components and Agent base class.

`The entire DQNAgent can be seen here <https://github.com/rlgraph/rlgraph/blob/master/rlgraph/agents/dqn_agent.py>`_
(as it's already part of the RLgraph library).


Writing the Agent's Class Stub and Ctor
---------------------------------------

First let's create a new python file, name it `simple_dqn_agent.py` and start coding with a few crucial imports
and a class stub:

.. code-block:: python

   import numpy as np

   from rlgraph.agents import Agent
   from rlgraph.components import Synchronizable, Memory, DQNLossFunction, DictMerger, \
       ContainerSplitter
   from rlgraph.spaces import FloatBox, BoolBox
   from rlgraph.utils.decorators import rlgraph_api
   from rlgraph.utils.util import strip_list


   class SimpleDQNAgent(Agent):
       """
       A basic DQN algorithm published in:
       [1] Human-level control through deep reinforcement learning. Mnih, Kavukcuoglu, Silver et al. - 2015
       """
       def __init__(self, name="simple-dqn-agent", **kwargs):
           super(DQNAgent, self).__init__(name=name, **kwargs)




.. code-block:: python

     # Extend input Space definitions to this Agent's specific API-methods.
     preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
     reward_space = FloatBox(add_batch_rank=True)
     terminal_space = BoolBox(add_batch_rank=True)

     self.input_spaces.update(dict(
         actions=self.action_space.with_batch_rank(),
         weights="variables:policy",
         time_step=int,
         use_exploration=bool,
         preprocessed_states=preprocessed_state_space,
         rewards=reward_space,
         terminals=terminal_space,
         next_states=preprocessed_state_space,
         preprocessed_next_states=preprocessed_state_space,
     ))

     # The merger to merge inputs into one record Dict going into the memory.
     self.merger = DictMerger("states", "actions", "rewards", "next_states", "terminals")
     # The replay memory.
     self.memory = Memory.from_spec(memory_spec)
     # The splitter for splitting up the records coming from the memory.
     self.splitter = ContainerSplitter("states", "actions", "rewards", "terminals", "next_states")

     # Copy our Policy (target-net), make target-net synchronizable.
     self.target_policy = self.policy.copy(scope="target-policy", trainable=False)
     self.target_policy.add_components(Synchronizable(), expose_apis="sync")
     # Number of steps since the last target-net synching from the main policy.
     self.steps_since_target_net_sync = 0

     self.loss_function = DQNLossFunction(
         discount=self.discount, double_q=False, huber_loss=False
     )

     # Define the Agent's (root-Component's) API.
     self.define_api_methods("policy", "preprocessor-stack", self.optimizer.scope)

     # markup = get_graph_markup(self.graph_builder.root_component)
     # print(markup)
     if self.auto_build:
         self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                           batch_size=self.update_spec["batch_size"])
         self.graph_built = True


.. code-block:: python

    def define_graph_api(self, policy_scope, pre_processor_scope):
        super(DQNAgent, self).define_api_methods(policy_scope, pre_processor_scope)

        # Add all our sub-components to the core.
        sub_components = [self.preprocessor, self.merger, self.memory, self.splitter, self.policy,
                          self.target_policy, self.exploration, self.loss_function, self.optimizer]
        self.root_component.add_components(*sub_components)

        # The sub-components we will be working with in our API-methods.
        # Assign them to local variables for convenience.
        preprocessor, merger, memory, splitter, policy, target_policy, exploration, loss_function, optimizer = \
            sub_components


We will now, one by one, define the root component's API-methods for use in the Agent's `get_action`, `update`, etc..


.. code-block:: python

    # IMPORTANT NOTE: We are still in `define_graph_api`.

    # Insert into memory.
    @rlgraph_api(component=self.root_component)
    def insert_records(self, preprocessed_states, actions, rewards, next_states, terminals):
        records = merger.merge(preprocessed_states, actions, rewards, next_states, terminals)
        return memory.insert_records(records)

    # Syncing target-net.
    @rlgraph_api(component=self.root_component)
    def sync_target_qnet(self):
        # If we are a multi-GPU root:
        # Simply feeds everything into the multi-GPU sync optimizer's method and return.
        if "multi-gpu-synchronizer" in self.sub_components:
            multi_gpu_syncer = self.sub_components["multi-gpu-synchronizer"]
            return multi_gpu_syncer.sync_target_qnets()
        else:
            policy_vars = self.get_sub_component_by_name(policy_scope).variables()
            return self.get_sub_component_by_name("target-policy").sync(policy_vars)

    # Learn from memory.
    @rlgraph_api(component=self.root_component)
    def update_from_memory(self_):
        # Non prioritized memory will just return weight 1.0 for all samples.
        records, sample_indices, importance_weights = memory.get_records(self.update_spec["batch_size"])
        preprocessed_s, actions, rewards, terminals, preprocessed_s_prime = splitter.split(records)

        step_op, loss, loss_per_item, q_values_s = self_.update_from_external_batch(
            preprocessed_s, actions, rewards, terminals, preprocessed_s_prime, importance_weights
        )

        # TODO this is really annoying.. will be solved once we have dict returns.
        if isinstance(memory, PrioritizedReplay):
            update_pr_step_op = memory.update_records(sample_indices, loss_per_item)
            return step_op, loss, loss_per_item, records, q_values_s, update_pr_step_op
        else:
            return step_op, loss, loss_per_item, records, q_values_s

    # Learn from an external batch.
    @rlgraph_api(component=self.root_component)
    def update_from_external_batch(
            self_, preprocessed_states, actions, rewards, terminals, preprocessed_next_states, importance_weights
    ):
        # Get the different Q-values.
        q_values_s = self_.get_sub_component_by_name(policy_scope).get_logits_probabilities_log_probs(
            preprocessed_states
        )["logits"]
        qt_values_sp = self_.get_sub_component_by_name(target_policy.scope).get_logits_probabilities_log_probs(
            preprocessed_next_states
        )["logits"]

        q_values_sp = None
        if self.double_q:
            q_values_sp = self_.get_sub_component_by_name(policy_scope).get_logits_probabilities_log_probs(
                preprocessed_next_states
            )["logits"]

        loss, loss_per_item = self_.get_sub_component_by_name(loss_function.scope).loss(
            q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp, importance_weights
        )

        # Args are passed in again because some device strategies may want to split them to different devices.
        policy_vars = self_.get_sub_component_by_name(policy_scope).variables()

        step_op, loss, loss_per_item = optimizer.step(policy_vars, loss, loss_per_item)
        return step_op, loss, loss_per_item, q_values_s

.. code-block:: python

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None):
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
            [batched_states, self.timesteps, use_exploration],
            # 0=preprocessed_states, 1=action
            return_ops
        ))
        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

.. code-block:: python

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, next_states, terminals]))


.. code-block:: python

    def update(self, batch=None):
        # Should we sync the target net?
        self.steps_since_target_net_sync += self.update_spec["update_interval"]
        if self.steps_since_target_net_sync >= self.update_spec["sync_interval"]:
            sync_call = "sync_target_qnet"
            self.steps_since_target_net_sync = 0
        else:
            sync_call = None

        # [0]=no-op step; [1]=the loss; [2]=loss-per-item, [3]=memory-batch (if pulled); [4]=q-values
        return_ops = [0, 1, 2]
        q_table = None

        if batch is None:
            pass
        else:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            if self.store_last_q_table is True:
                return_ops += [3]  # 3=q-values

            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"],
                           batch["next_states"], batch["importance_weights"]]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input, return_ops), sync_call)

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
