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

import copy

from rlgraph.utils import RLGraphError
from rlgraph.agents.agent import Agent
from rlgraph.components.common.dict_merger import DictMerger
from rlgraph.components.common.container_splitter import ContainerSplitter
from rlgraph.components.common.slice import Slice
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.helpers.softmax import SoftMax
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.loss_functions.impala_loss_function import IMPALALossFunction
from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.components.memories.queue_runner import QueueRunner
from rlgraph.components.papers.impala.impala_networks import LargeIMPALANetwork, SmallIMPALANetwork
from rlgraph.spaces import FloatBox, Dict, Tuple
from rlgraph.utils.util import default_dict


class IMPALAAgent(Agent):
    """
    An Agent implementing the IMPALA algorithm described in [1]. The Agent contains both learner and actor
    API-methods, which will be put into the graph depending on the type ().

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """

    standard_internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=False)

    def __init__(self, discount=0.99, architecture="large", fifo_queue_spec=None, environment_spec=None,
                 weight_pg=None, weight_baseline=None, weight_entropy=None, worker_sample_size=20, **kwargs):
        """
        Args:
            discount (float): The discount factor gamma.
            architecture (str): Which IMPALA architecture to use. One of "small" or "large". Will be ignored if
                `network_spec` is given explicitly in kwargs. Default: "large".
            fifo_queue_spec (Optional[dict,FIFOQueue]): The spec for the FIFOQueue to use for the IMPALA algorithm.
            environment_spec (dict): The spec for constructing an Environment object for an actor-type IMPALA agent.
            weight_pg (float): See IMPALALossFunction Component.
            weight_baseline (float): See IMPALALossFunction Component.
            weight_entropy (float): See IMPALALossFunction Component.
            worker_sample_size (int): How many steps the actor will perform in the environment each sample-run.

        Keyword Args:
            type (str): One of "single", "actor" or "learner". Default: "single".
            num_actors (int): If type is "single", how many actors should be run in separate threads.
        """
        type_ = kwargs.pop("type", "single")
        assert type_ in ["single", "actor", "learner"]
        self.type = type_
        if self.type == "single":
            self.num_actors = kwargs.pop("num_actors", 1)
        else:
            self.num_actors = 0
        self.worker_sample_size = worker_sample_size

        # Network-spec by default is a "large architecture" IMPALA network.
        network_spec = kwargs.pop(
            "network_spec", "rlgraph.components.papers.impala.impala_networks.{}IMPALANetwork".
            format("Large" if architecture == "large" else "Small")
        )
        action_adapter_spec = kwargs.pop("action_adapter_spec", dict(type="baseline-action-adapter"))

        # Depending on the job-type, remove the pieces from the Agent-spec/graph we won't need.
        exploration_spec = kwargs.pop("exploration_spec", None)
        optimizer_spec = kwargs.pop("optimizer_spec", None)
        observe_spec = kwargs.pop("observe_spec", None)

        # Run everything in a single process.
        if self.type == "single":
            environment_spec = environment_spec or dict(
                type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
                frameskip=4
            )
            update_spec = kwargs.pop("update_spec", None)
        # Actors won't need to learn (no optimizer needed in graph).
        elif self.type == "actor":
            optimizer_spec = None
            update_spec = kwargs.pop("update_spec", dict(do_updates=False))
            environment_spec = environment_spec or dict(
                type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
                frameskip=4
            )
        # Learners won't need to explore (act) or observe (insert into Queue).
        else:
            # Add prev-a/r to Dict state space.
            #state_space = copy.deepcopy(kwargs["state_space"])
            #state_space["previous_action"] = kwargs["action_space"]
            #state_space["previous_reward"] = FloatBox()
            #kwargs["state_space"] = state_space
            exploration_spec = None
            observe_spec = None
            update_spec = kwargs.pop("update_spec", None)
            environment_spec = None

        # Add previous-action/reward preprocessors to env-specific preprocessor spec.
        preprocessing_spec = kwargs.pop("preprocessing_spec", dict(preprocessors=dict()))
        # Flatten actions.
        preprocessing_spec["preprocessors"]["previous_action"] = [
            dict(type="reshape", flatten=True, flatten_categories=kwargs.get("action_space").num_categories)
        ]
        # Bump reward and convert to float32, so that it can be concatenated by the Concat layer.
        preprocessing_spec["preprocessors"]["previous_reward"] = [
            dict(type="reshape", new_shape=(1,)),
            dict(type="convert_type", to_dtype="float32")
        ]

        # Now that we fixed the Agent's spec, call the super constructor.
        super(IMPALAAgent, self).__init__(
            discount=discount,
            preprocessing_spec=preprocessing_spec,
            network_spec=network_spec,
            action_adapter_spec=action_adapter_spec,
            exploration_spec=exploration_spec,
            optimizer_spec=optimizer_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,
            name=kwargs.pop("name", "impala-{}-agent".format(self.type)),
            **kwargs
        )
        # Manually set the reuse_variable_scope for our policies (actor: mu, learner: pi).
        self.policy.propagate_subcomponent_properties(dict(reuse_variable_scope="shared"))
        # Always use 1st learner as the parameter server for all policy variables.
        if self.execution_spec["mode"] == "distributed" and self.execution_spec["distributed_spec"]["cluster_spec"]:
            self.policy.propagate_subcomponent_properties(dict(device=dict(variables="/job:learner/task:0/cpu")))

        # Check whether we have an RNN.
        self.has_rnn = self.neural_network.has_rnn()

        # Some FIFO-queue specs.
        self.fifo_queue_keys = ["preprocessed_states", "actions", "rewards", "terminals", "last_next_states",
                                "action_probs", "initial_internal_states"]
        self.fifo_record_space = fifo_queue_spec["record_space"] if "record_space" in fifo_queue_spec else Dict(
            {
                "preprocessed_states": self.preprocessor.get_preprocessed_space(
                    default_dict(copy.deepcopy(self.state_space), dict(
                        previous_action=self.action_space,
                        previous_reward=FloatBox()
                    ))
                ),
                "actions": self.action_space,
                "rewards": float,
                "terminals": bool,
                "last_next_states": default_dict(copy.deepcopy(self.state_space), dict(
                    previous_action=self.action_space,
                    previous_reward=FloatBox()
                )),
                "action_probs": FloatBox(shape=(self.action_space.num_categories,)),
                "initial_internal_states": self.internal_states_space
            }, add_batch_rank=False, add_time_rank=self.worker_sample_size
        )
        # Take away again time-rank from initial-states and last-next-state (these come in only for one time-step)
        self.fifo_record_space["last_next_states"] = self.fifo_record_space["last_next_states"].with_time_rank(1)
        self.fifo_record_space["initial_internal_states"] = \
            self.fifo_record_space["initial_internal_states"].with_time_rank(False)
        # Create our FIFOQueue (actors will enqueue, learner(s) will dequeue).
        self.fifo_queue = FIFOQueue.from_spec(
            fifo_queue_spec, reuse_variable_scope="shared-fifo-queue", only_insert_single_records=True,
            record_space=self.fifo_record_space,
            device="/job:learner/task:0" if self.execution_spec["mode"] == "distributed" and
                                            self.execution_spec["distributed_spec"]["cluster_spec"] else None
        )

        # Remove `states` key from input_spaces: not needed.
        del self.input_spaces["states"]

        # Add all our sub-components to the core.
        if self.type == "single":
            # Create a queue runner that takes care of pushing items into the queue from our actors.

            self.env_output_splitter = ContainerSplitter(tuple_length=8, scope="env-output-splitter")
            self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys, scope="fifo-output-splitter")

            # Slice some data from the EnvStepper (e.g only first internal states are needed).
            self.next_states_slicer = Slice(scope="next-states-slicer", squeeze=False)
            self.internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)

            self.transpose_actions = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                             scope="transpose-a", flatten_categories=False)
            self.transpose_rewards = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                             scope="transpose-r")
            self.transpose_terminals = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                               scope="transpose-t")
            self.transpose_action_probs = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                                  scope="transpose-a-probs-mu")

            self.softmax = SoftMax()

            # Create an IMPALALossFunction with some parameters.
            self.loss_function = IMPALALossFunction(
                discount=self.discount, weight_pg=weight_pg, weight_baseline=weight_baseline,
                weight_entropy=weight_entropy
            )

            # Merge back to insert into FIFO.
            self.fifo_input_merger = DictMerger(*self.fifo_queue_keys)

            dummy_flattener = ReShape(flatten=True)  # dummy Flattener to calculate action-probs space

            self.environment_steppers = [EnvironmentStepper(
                environment_spec=environment_spec,
                actor_component_spec=ActorComponent(preprocessor_spec=preprocessing_spec,
                                                    policy_spec=dict(neural_network=network_spec,
                                                                     action_space=self.action_space,
                                                                     reuse_variable_scope="shared/policy"),
                                                    exploration_spec=exploration_spec),
                state_space=self.state_space.with_batch_rank(),
                reward_space=float,  # TODO <- float64 for deepmind? may not work for other envs
                internal_states_space=self.internal_states_space,
                add_previous_action=True,
                add_previous_reward=True,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space),
                scope="env-stepper-{}".format(i)
            ) for i in range(self.num_actors)]

            # Create the QueueRunners (one for each env-stepper).
            # - Take return value 1 of API-method step as record to insert.
            self.queue_runner = QueueRunner(self.fifo_queue, "step", 1, self.env_output_splitter,
                                            self.fifo_input_merger, self.next_states_slicer,
                                            self.internal_states_slicer,
                                            *self.environment_steppers)
            self.softmax = SoftMax()

            # Create an IMPALALossFunction with some parameters.
            self.loss_function = IMPALALossFunction(
                discount=self.discount, weight_pg=weight_pg, weight_baseline=weight_baseline,
                weight_entropy=weight_entropy
            )

            sub_components = [
                self.fifo_output_splitter, self.fifo_queue, self.queue_runner, self.transpose_actions,
                self.transpose_rewards, self.transpose_terminals, self.transpose_action_probs, self.preprocessor,
                self.policy, self.softmax, self.loss_function, self.optimizer
            ]

        elif self.type == "actor":
            # No learning, no loss function.
            self.loss_function = None
            # A Dict Splitter to split things from the EnvStepper.
            self.env_output_splitter = ContainerSplitter(tuple_length=8)
            # Slice some data from the EnvStepper (e.g only first internal states are needed).
            self.next_states_slicer = Slice(scope="next-states-slicer", squeeze=False)
            self.internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)
            # Merge back to insert into FIFO.
            self.fifo_input_merger = DictMerger(*self.fifo_queue_keys)

            self.softmax = None

            dummy_flattener = ReShape(flatten=True)  # dummy Flattener to calculate action-probs space
            self.environment_stepper = EnvironmentStepper(
                environment_spec=environment_spec,
                actor_component_spec=ActorComponent(self.preprocessor, self.policy, self.exploration),
                state_space=self.state_space.with_batch_rank(),
                reward_space=float,  # TODO <- float64 for deepmind? may not work for other envs
                add_previous_action=True,
                add_previous_reward=True,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space)
            )
            sub_components = [
                self.environment_stepper, self.env_output_splitter, self.next_states_slicer,
                self.internal_states_slicer, self.fifo_input_merger, self.fifo_queue
            ]
        # Learner.
        else:
            self.environment_stepper = None

            # A Dict splitter to split up items from the queue.
            self.fifo_input_merger = None
            self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys)
            self.next_states_slicer = None
            self.internal_states_slicer = None

            self.softmax = SoftMax()

            # Create an IMPALALossFunction with some parameters.
            self.loss_function = IMPALALossFunction(
                discount=self.discount, weight_pg=weight_pg, weight_baseline=weight_baseline,
                weight_entropy=weight_entropy
            )

            sub_components = [self.fifo_queue, self.fifo_output_splitter, self.preprocessor, self.policy,
                              self.softmax, self.loss_function, self.optimizer]

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
        # TODO: Unify agents with/w/o synchronizable policy.
        # TODO: Unify Agents with/w/o get_action method (w/ env-stepper vs w/o).
        #global_scope_base = "environment-stepper/actor-component/" if self.type == "actor" else ""
        #super(IMPALAAgent, self).define_api_methods(
        #    global_scope_base+"policy",
        #    global_scope_base+"dict-preprocessor-stack"
        #)

        # Assemble the specific agent.
        if self.type == "single":
            self.define_api_methods_single(*sub_components)
        elif self.type == "actor":
            self.define_api_methods_actor(*sub_components)
        else:
            self.define_api_methods_learner(*sub_components)

    def define_api_methods_single(self, fifo_output_splitter, fifo_queue, queue_runner, transpose_actions,
                                  transpose_rewards, transpose_terminals, transpose_action_probs, preprocessor,
                                  policy, softmax, loss_function, optimizer):
        def setup_queue_runners(self):
            return self.call(queue_runner.setup)

        self.root_component.define_api_method("setup_queue_runners", setup_queue_runners)

        def get_queue_size(self):
            return self.call(fifo_queue.get_size)

        self.root_component.define_api_method("get_queue_size", get_queue_size)

        def update_from_memory(self_):
            # Pull n records from the queue.
            # Note that everything will come out as batch-major and must be transposed before the main-LSTM.
            # This is done by the network itself for all network inputs:
            # - preprocessed_s
            # - preprocessed_last_s_prime
            # But must still be done for actions, rewards, terminals here in this API-method via separate ReShapers.
            records = self_.call(fifo_queue.get_records, self.update_spec["batch_size"])

            preprocessed_s, actions, rewards, terminals, last_s_prime, action_probs_mu, \
                initial_internal_states = self_.call(fifo_output_splitter.split, records)

            preprocessed_last_s_prime = self_.call(preprocessor.preprocess, last_s_prime)

            # Flip actions, rewards, terminals to time-major.
            # TODO: Create components that are less input-space sensitive (those that have no variables should
            # TODO: be reused for any kind of processing)
            actions = self_.call(transpose_actions.apply, actions)
            rewards = self_.call(transpose_rewards.apply, rewards)
            terminals = self_.call(transpose_terminals.apply, terminals)
            action_probs_mu = self_.call(transpose_action_probs.apply, action_probs_mu)

            # Get the pi-action probs AND the values for all our states.
            state_values_pi, logits_pi, current_internal_states = \
                self_.call(policy.get_baseline_output, preprocessed_s, initial_internal_states)

            # And the values for the last states.
            bootstrapped_values, _, _ = \
                self_.call(policy.get_baseline_output, preprocessed_last_s_prime, current_internal_states)

            _, log_probabilities_pi = self_.call(softmax.get_probabilities_and_log_probs, logits_pi)

            # Calculate the loss.
            loss, loss_per_item = self_.call(
                loss_function.loss, log_probabilities_pi, action_probs_mu, state_values_pi, actions, rewards,
                terminals, bootstrapped_values
            )
            policy_vars = self_.call(policy._variables)

            # Pass vars and loss values into optimizer.
            step_op, loss, loss_per_item = self_.call(optimizer.step, policy_vars, loss, loss_per_item)

            # Return optimizer op and all loss values.
            return step_op, loss, loss_per_item

        self.root_component.define_api_method("update_from_memory", update_from_memory)

    def define_api_methods_actor(self, env_stepper, splitter, next_states_slicer, internal_states_slicer, merger,
                                 fifo_queue):
        """
        Defines the API-methods used by an IMPALA actor. Actors only step through an environment (n-steps at
        a time), collect the results and push them into the FIFO queue. Results include: The actions actually
        taken, the discounted accumulated returns for each action, the probability of each taken action according to
        the behavior policy.

        Args:
            env_stepper (EnvironmentStepper): The EnvironmentStepper Component to setp through the Env n steps
                in a single op call.
            fifo_queue (FIFOQueue): The FIFOQueue Component used to enqueue env sample runs (n-step).
        """
        # Perform n-steps in the env and insert the results into our FIFO-queue.
        def perform_n_steps_and_insert_into_fifo(self_, internal_states=None, time_step=0):
            # Take n steps in the environment.
            step_op, step_results = self_.call(
                env_stepper.step, internal_states, self.worker_sample_size, time_step
            )

            preprocessed_s, actions, rewards, returns, terminals, next_states, action_log_probs, \
                internal_states = self_.call(splitter.split, step_results)

            last_next_state = self_.call(next_states_slicer.slice, next_states, -1)
            initial_internal_states = self_.call(internal_states_slicer.slice, internal_states, 0)
            current_internal_states = self_.call(internal_states_slicer.slice, internal_states, -1)

            record = self_.call(
                merger.merge, preprocessed_s, actions, rewards, terminals, last_next_state,
                action_log_probs, initial_internal_states
            )

            # Insert results into the FIFOQueue.
            insert_op = self_.call(fifo_queue.insert_records, record)
            return step_op, insert_op, current_internal_states, returns, terminals

        self.root_component.define_api_method(
            "perform_n_steps_and_insert_into_fifo", perform_n_steps_and_insert_into_fifo
        )

        def reset(self):
            # Resets the environment running inside the agent.
            reset_op = self.call(env_stepper.reset)
            return reset_op

        self.root_component.define_api_method("reset", reset)

    def define_api_methods_learner(self, fifo_queue, splitter, preprocessor, policy, softmax, loss_function, optimizer):
        """
        Defines the API-methods used by an IMPALA learner. Its job is basically: Pull a batch from the
        FIFOQueue, split it up into its components and pass these through the loss function and into the optimizer for
        a learning update.

        Args:
            fifo_queue (FIFOQueue): The FIFOQueue Component used to enqueue env sample runs (n-step).
            splitter (ContainerSplitter): The DictSplitter Component to split up a batch from the queue along its
                items.
            policy (Policy): The Policy Component, which to update.
            loss_function (IMPALALossFunction): The IMPALALossFunction Component.
            optimizer (Optimizer): The optimizer that we use to calculate an update and apply it.
        """
        def update_from_memory(self_):
            records = self_.call(fifo_queue.get_records, self.update_spec["batch_size"])

            preprocessed_s, actions, rewards, terminals, last_s_prime, action_probs_mu, \
                initial_internal_states = self_.call(splitter.split, records)

            preprocessed_last_s_prime = self_.call(preprocessor.preprocess, last_s_prime)
            # TODO: should we concatenate preprocessed_s and preprocessed_last_s_prime?
            # Get the pi-action probs AND the values for all our states.
            state_values_pi, logits_pi, current_internal_states = \
                self_.call(policy.get_baseline_output, preprocessed_s, initial_internal_states)
            # And the values for the last states.
            bootstrapped_values, _, _ = \
                self_.call(policy.get_baseline_output, preprocessed_last_s_prime, current_internal_states)

            _, log_probabilities_pi = self_.call(softmax.get_probabilities_and_log_probs, logits_pi)

            # Calculate the loss.
            loss, loss_per_item = self_.call(
                loss_function.loss, log_probabilities_pi, action_probs_mu, state_values_pi, actions, rewards,
                terminals, bootstrapped_values
            )
            policy_vars = self_.call(policy._variables)
            # Pass vars and loss values into optimizer.
            step_op, loss, loss_per_item = self_.call(optimizer.step, policy_vars, loss, loss_per_item)

            # Return optimizer op and all loss values.
            return step_op, loss, loss_per_item

        self.root_component.define_api_method("update_from_memory", update_from_memory)

        #def insert_dummy_records(self_):
        #    insert_op = self_.call(fifo_queue.insert_dummy_records)
        #    return insert_op

        #self.root_component.define_api_method("insert_dummy_records", insert_dummy_records)

    def get_action(self, states, internal_states=None, use_exploration=True, extra_returns=None):
        pass

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None):
        if batch is None:
            return self.graph_executor.execute("update_from_memory")
        else:
            raise RLGraphError("Cannot call update-from-batch on an IMPALA Agent.")

    def __repr__(self):
        return "IMPALAAgent(type={})".format(self.type)

