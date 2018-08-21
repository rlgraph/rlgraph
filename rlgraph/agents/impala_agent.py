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
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.loss_functions.impala_loss_function import IMPALALossFunction
from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.components.papers.impala.large_impala_network import LargeIMPALANetwork
from rlgraph.spaces import FloatBox, BoolBox


class IMPALAAgent(Agent):
    """
    An Agent implementing the IMPALA algorithm described in [1]. The Agent contains both learner and explorer
    API-methods, which will be put into the graph depending on the type ().

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, fifo_queue_spec=None, **kwargs):
        """
        Args:
            fifo_queue_spec (Optional[dict,FIFOQueue]): The spec for the FIFOQueue to use for the IMPALA algorithm.

        Keyword Args:
            type (str): One of "explorer" or "learner". Default: "explorer".
        """
        type_ = kwargs.pop("type", "explorer")
        assert type_ in ["explorer", "learner"]
        self.type = type_

        # Network-spec by default is a large architecture IMPALA network.
        network_spec = kwargs.pop("network_spec", LargeIMPALANetwork())

        # Depending on the job-type, remove the pieces from the Agent-spec/graph we won't need.
        exploration_spec = kwargs.pop("exploration_spec", None)
        optimizer_spec = kwargs.pop("optimizer_spec", None)
        observe_spec = kwargs.pop("observe_spec", None)
        # Learners won't need  to explore (act) or observe (insert into Queue).
        if self.type == "learner":
            exploration_spec = None
            observe_spec = None
            update_spec = kwargs.pop("update_spec", None)
            environment_spec = None
        # Explorers won't need to learn (no optimizer needed in graph).
        else:
            optimizer_spec = None
            update_spec = kwargs.pop("update_spec", dict(do_updates=False))
            environment_spec = kwargs.pop("environment_spec", dict(
                    type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
                    frameskip=4
            ))
            # Add simple previous-action preprocessor (flatten) to env-specific preprocessor spec.
            preprocessing_spec = kwargs.pop("preprocessing_spec", dict(preprocessors=dict()))
            preprocessing_spec["preprocessors"]["previous_action"] = [
                dict(type="reshape", flatten=True, flatten_categories=kwargs.get("action_space").num_categories)
            ]
            kwargs["preprocessing_spec"] = preprocessing_spec

        # Now that we fixed the Agent's spec, call the super constructor.
        super(IMPALAAgent, self).__init__(
            network_spec=network_spec, exploration_spec=exploration_spec, optimizer_spec=optimizer_spec,
            observe_spec=observe_spec, update_spec=update_spec,
            name=kwargs.pop("name", "impala-{}-agent".format(self.type)), **kwargs
        )

        # Create our FIFOQueue (explorers will enqueue, learner(s) will dequeue).
        self.fifo_queue = FIFOQueue.from_spec(fifo_queue_spec, reuse_variable_scope="shared-fifo-queue")
        # Check whether we have an RNN.
        self.has_rnn = self.neural_network.has_rnn()

        # Add all our sub-components to the core.
        if self.type == "explorer":
            # Extend input Space definitions to this Agent's specific API-methods.
            self.input_spaces.update(dict(
                weights="variables:environment-stepper/actor-component/policy",
                internal_states=self.internal_states_space.with_batch_rank(),
                num_steps=int,
                time_step=int
            ))
            self.loss_function = None
            actor_component = ActorComponent(self.preprocessor, self.policy, self.exploration)
            state_space = self.state_space.with_batch_rank()
            dummy_flattener = ReShape(flatten=True)  # dummy Flattener to calculate action-probs space
            reward_space = float
            self.environment_stepper = EnvironmentStepper(
                environment_spec=environment_spec,
                actor_component_spec=actor_component,
                state_space=state_space,
                reward_space=reward_space,
                add_previous_action=True,
                add_previous_reward=True,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space)
            )
            sub_components = [self.environment_stepper, self.fifo_queue]
        else:
            # Extend input Space definitions to this Agent's specific API-methods.
            self.input_spaces.update(dict(
                # TODO: fill out learner's input spaces
            ))

            # TODO: add loss func options here and to our ctor.
            self.loss_function = IMPALALossFunction()
            self.environment_stepper = None
            sub_components = [self.fifo_queue, self.policy, self.loss_function, self.optimizer]

        # Add all the agent's sub-components to the root.
        self.root_component.add_components(*sub_components)

        # Define the Agent's (root Component's) API.
        self.define_api_methods(*sub_components)

        # markup = get_graph_markup(self.graph_builder.root_component)
        # print(markup)
        if self.auto_build:
            self._build_graph([self.root_component], self.input_spaces, self.optimizer)
            self.graph_built = True

    def define_api_methods(self, *sub_components):
        super(IMPALAAgent, self).define_api_methods(
            "environment-stepper/actor-component/policy", "environment-stepper/actor-component/dict-preprocessor-stack"
        )

        if self.type == "explorer":
            self.define_api_methods_explorer(*sub_components)
        else:
            self.define_api_methods_learner(*sub_components)

    def define_api_methods_explorer(self, env_stepper, fifo_queue):
        """
        Defines the API-methods used by an IMPALA explorer. Explorers only step through an environment (n-steps at
        a time), collect the results and push them into the FIFO queue. Results include: The actions actually
        taken, the discounted accumulated returns for each action, the probability of each taken action according to
        the behavior policy.

        Args:
            env_stepper (EnvironmentStepper): The EnvironmentStepper Component to setp through the Env n steps
                in a single op call.
            fifo_queue (FIFOQueue): The FIFOQueue Component used to enqueue env sample runs (n-step).
        """
        # Perform n-steps in the env and insert the results into our FIFO-queue.
        def perform_n_steps_and_insert_into_fifo(self, internal_states=None, num_steps=1, time_step=0):
            # Take n steps in the environment.
            step_op, step_results = self.call(env_stepper.step, internal_states, num_steps, time_step)
            # Dissect the results.
            #preprocessed_s, actions, rewards, returns, terminals, next_states, action_log_probs, internal_states = split(step_results)

            # Insert results into the FIFOQueue.
            insert_op = self.call(fifo_queue.insert_records, step_results)
            return step_op, insert_op

        self.root_component.define_api_method(
            "perform_n_steps_and_insert_into_fifo", perform_n_steps_and_insert_into_fifo
        )

    def define_api_methods_learner(self, fifo_queue, policy, loss_function, optimizer):
        # TODO: implement learner logic
        return

        # Learn from memory.
        def update_from_fifo_queue(self):
            records = self.call(fifo_queue.get_records, self.update_spec["batch_size"])

            preprocessed_s, actions, rewards, terminals, preprocessed_s_prime = self.call(splitter.split,
                                                                                           records)

            # Get the different Q-values.
            q_values_s = self.call(policy.get_q_values, preprocessed_s)
            qt_values_sp = self.call(target_policy.get_q_values, preprocessed_s_prime)
            q_values_sp = None
            if self.double_q:
                q_values_sp = self.call(policy.get_q_values, preprocessed_s_prime)

            loss, loss_per_item = self.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                                             qt_values_sp, q_values_sp)

            policy_vars = self.call(policy._variables)
            # Pass extra args for device strategy.
            step_op = self.call(optimizer.step, policy_vars, loss, q_values_s, actions, rewards, terminals,
                                             qt_values_sp, q_values_sp)

            # TODO: For multi-GPU, the final-loss will probably have to come from the optimizer.
            return step_op, loss, loss_per_item, records, q_values_s

        self.root_component.define_api_method("update_from_fifo_queue", update_from_fifo_queue)

    def get_action(self, states, internal_states=None, use_exploration=True, extra_returns=None):
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
        # TODO: implement logic
        return

    def __repr__(self):
        return "IMPALAAgent()"

