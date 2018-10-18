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

from rlgraph import get_backend
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils import RLGraphError
from rlgraph.agents.agent import Agent
from rlgraph.components.common.dict_merger import DictMerger
from rlgraph.components.common.container_splitter import ContainerSplitter
from rlgraph.components.common.slice import Slice
from rlgraph.components.common.staging_area import StagingArea
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.layers.preprocessing.transpose import Transpose
from rlgraph.components.layers.preprocessing.concat import Concat
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.neural_networks.dynamic_batching_policy import DynamicBatchingPolicy
from rlgraph.components.loss_functions.impala_loss_function import IMPALALossFunction
from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.components.memories.queue_runner import QueueRunner
from rlgraph.spaces import FloatBox, Dict, Tuple
from rlgraph.utils.util import default_dict

if get_backend() == "tf":
    import tensorflow as tf


class IMPALAAgent(Agent):
    """
    An Agent implementing the IMPALA algorithm described in [1]. The Agent contains both learner and actor
    API-methods, which will be put into the graph depending on the type ().

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """

    default_internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=False)
    #default_internal_states_space = Tuple(FloatBox(shape=(266,)), FloatBox(shape=(266,)), add_batch_rank=False)

    default_environment_spec = dict(
        type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
        frameskip=4
    )

    # TODO: capacity FIFO Queue make configurable.
    def __init__(self, discount=0.99, fifo_queue_spec=None, architecture="large", environment_spec=None,
                 weight_pg=None, weight_baseline=None, weight_entropy=None, worker_sample_size=100,
                 dynamic_batching=False, **kwargs):
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
            dynamic_batching (bool): Whether to use the deepmind's custom dynamic batching op for wrapping the
                optimizer's step call. The batcher.so file must be compiled for this to work (see Docker file).
                Default: False.

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
        self.dynamic_batching = dynamic_batching

        # Network-spec by default is a "large architecture" IMPALA network.
        network_spec = default_dict(kwargs.pop(
            "network_spec", dict(type="rlgraph.components.papers.impala.impala_networks.{}IMPALANetwork".
                                 format("Large" if architecture == "large" else "Small"))
        ), dict(worker_sample_size=1 if self.type == "actor" else self.worker_sample_size + 1))
        action_adapter_spec = kwargs.pop("action_adapter_spec", dict(type="baseline-action-adapter"))

        # Depending on the job-type, remove the pieces from the Agent-spec/graph we won't need.
        exploration_spec = kwargs.pop("exploration_spec", None)
        optimizer_spec = kwargs.pop("optimizer_spec", None)
        observe_spec = kwargs.pop("observe_spec", None)

        # Run everything in a single process.
        if self.type == "single":
            environment_spec = environment_spec or self.default_environment_spec
            update_spec = kwargs.pop("update_spec", None)

        # Actors won't need to learn (no optimizer needed in graph).
        elif self.type == "actor":
            optimizer_spec = None
            update_spec = kwargs.pop("update_spec", dict(do_updates=False))
            environment_spec = environment_spec or self.default_environment_spec
        # Learners won't need to explore (act) or observe (insert into Queue).
        else:
            exploration_spec = None
            observe_spec = None
            update_spec = kwargs.pop("update_spec", None)
            environment_spec = None

        # Add previous-action/reward preprocessors to env-specific preprocessor spec.
        # TODO: remove this empty hard-coded preprocessor.
        kwargs.pop("preprocessing_spec", None)
        preprocessing_spec = dict(type="dict-preprocessor-stack", preprocessors=dict(
            # Flatten actions.
            previous_action=[
                dict(type="reshape", flatten=True, flatten_categories=kwargs.get("action_space").num_categories)
            ],
            # Bump reward and convert to float32, so that it can be concatenated by the Concat layer.
            previous_reward=[
                dict(type="reshape", new_shape=(1,))
                #dict(type="convert_type", to_dtype="float32")
            ]
        ))

        # Limit communication in distributed mode between each actor and the learner (never between actors).
        execution_spec = kwargs.pop("execution_spec", None)
        if execution_spec is not None and execution_spec.get("mode") == "distributed":
            execution_spec["session_config"] = dict(
                type="monitored-training-session",
                allow_soft_placement=True,
                log_device_placement=False,
                device_filters=["/job:learner/task:0"] + (
                    ["/job:actor/task:{}".format(execution_spec["distributed_spec"]["task_index"])] if
                    self.type == "actor" else ["/job:learner/task:0"]
                )
            )
            # If Actor, make non-chief in either case (even if task idx == 0).
            if self.type == "actor":
                execution_spec["distributed_spec"]["is_chief"] = False
                # Hard-set device to the CPU for actors.
                execution_spec["device_strategy"] = "custom"
                execution_spec["default_device"] = "/job:{}/task:{}/cpu".format(self.type, execution_spec["distributed_spec"]["task_index"])

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
            execution_spec=execution_spec,
            name=kwargs.pop("name", "impala-{}-agent".format(self.type)),
            **kwargs
        )

        # If we use dynamic batching, wrap the dynamic batcher around the policy's graph_fn that we
        # actually call below during our build.
        if self.dynamic_batching:
            self.policy = DynamicBatchingPolicy(policy_spec=self.policy, scope="")
        # Manually set the reuse_variable_scope for our policies (actor: mu, learner: pi).
        self.policy.propagate_sub_component_properties(dict(reuse_variable_scope="shared"))
        # Always use 1st learner as the parameter server for all policy variables.
        if self.execution_spec["mode"] == "distributed" and self.execution_spec["distributed_spec"]["cluster_spec"]:
            self.policy.propagate_sub_component_properties(dict(device=dict(variables="/job:learner/task:0/cpu")))

        # Check whether we have an RNN.
        self.has_rnn = self.neural_network.has_rnn()
        # Check, whether we are running with GPU.
        self.has_gpu = self.execution_spec["gpu_spec"]["gpus_enabled"] is True and \
            self.execution_spec["gpu_spec"]["num_gpus"] > 0

        # Some FIFO-queue specs.
        self.fifo_queue_keys = ["terminals", "states", "action_probs", "initial_internal_states"]
        self.fifo_record_space = Dict(
            {
                "terminals": bool,
                "states": default_dict(copy.deepcopy(self.state_space), dict(
                    previous_action=self.action_space,
                    previous_reward=FloatBox()
                )),
                "action_probs": FloatBox(shape=(self.action_space.num_categories,)),
                "initial_internal_states": self.internal_states_space
            }, add_batch_rank=False, add_time_rank=(self.worker_sample_size + 1)
        )
        # Take away again time-rank from initial-states (comes in only for one time-step).
        self.fifo_record_space["initial_internal_states"] = \
            self.fifo_record_space["initial_internal_states"].with_time_rank(False)
        # Create our FIFOQueue (actors will enqueue, learner(s) will dequeue).
        self.fifo_queue = FIFOQueue.from_spec(
            fifo_queue_spec or dict(capacity=1),
            reuse_variable_scope="shared-fifo-queue",
            only_insert_single_records=True,
            record_space=self.fifo_record_space,
            device="/job:learner/task:0/cpu" if self.execution_spec["mode"] == "distributed" and
            self.execution_spec["distributed_spec"]["cluster_spec"] else None
        )

        # Remove `states` key from input_spaces: not needed.
        del self.input_spaces["states"]

        # Add all our sub-components to the core.
        if self.type == "single":
            # Create a queue runner that takes care of pushing items into the queue from our actors.

            self.env_output_splitter = ContainerSplitter(tuple_length=8, scope="env-output-splitter")
            self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys, scope="fifo-output-splitter")

            self.staging_area = StagingArea(num_data=len(self.fifo_queue_keys))

            # Slice some data from the EnvStepper (e.g only first internal states are needed).
            self.internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)

            # TODO: add state transposer, remove action/rewards transposer (part of state).
            self.transpose_actions = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                             scope="transpose-a", flatten_categories=False)
            self.transpose_rewards = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                             scope="transpose-r")
            self.transpose_terminals = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                               scope="transpose-t")
            self.transpose_action_probs = ReShape(flip_batch_and_time_rank=True, time_major=True,
                                                  scope="transpose-a-probs-mu")

            self.concat = Concat(axis=1)  # 1=the time rank (at the time of the concat, we are still batch-major)

            # Create an IMPALALossFunction with some parameters.
            self.loss_function = IMPALALossFunction(
                discount=self.discount, weight_pg=weight_pg, weight_baseline=weight_baseline,
                weight_entropy=weight_entropy
            )

            # Merge back to insert into FIFO.
            self.fifo_input_merger = DictMerger(*self.fifo_queue_keys)

            dummy_flattener = ReShape(flatten=True)  # dummy Flattener to calculate action-probs space

            self.environment_steppers = list()
            for i in range(self.num_actors):
                policy_spec = dict(
                    network_spec=network_spec,
                    action_adapter_spec=dict(type="baseline-action-adapter"),
                    action_space=self.action_space
                )

                env_stepper = EnvironmentStepper(
                    environment_spec=environment_spec,
                    actor_component_spec=ActorComponent(
                        preprocessor_spec=preprocessing_spec,
                        policy_spec=policy_spec,
                        exploration_spec=exploration_spec
                    ),
                    state_space=self.state_space.with_batch_rank(),
                    reward_space=float,  # TODO <- float64 for deepmind? may not work for other envs
                    internal_states_space=self.internal_states_space,
                    num_steps=self.worker_sample_size,
                    add_previous_action=True,
                    add_previous_reward=True,
                    add_action_probs=True,
                    action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space),
                    scope="env-stepper-{}".format(i)
                )
                if self.dynamic_batching:
                    env_stepper.actor_component.policy.parent_component = None
                    env_stepper.actor_component.policy = DynamicBatchingPolicy(policy_spec=env_stepper.actor_component.policy, scope="")
                    env_stepper.actor_component.add_components(env_stepper.actor_component.policy)

                env_stepper.actor_component.policy.propagate_sub_component_properties(dict(reuse_variable_scope="shared"))
                self.environment_steppers.append(env_stepper)

            # Create the QueueRunners (one for each env-stepper).
            # - Take return value 1 of API-method step as record to insert.
            self.queue_runner = QueueRunner(self.fifo_queue, "step", 1, self.env_output_splitter,
                                            self.fifo_input_merger,
                                            self.internal_states_slicer,
                                            *self.environment_steppers)

            sub_components = [
                self.fifo_output_splitter, self.fifo_queue, self.queue_runner, self.transpose_actions,
                self.transpose_rewards, self.transpose_terminals, self.transpose_action_probs, self.preprocessor,
                self.staging_area, self.concat, self.policy, self.loss_function, self.optimizer
            ]

        elif self.type == "actor":
            # No learning, no loss function.
            self.loss_function = None
            # A Dict Splitter to split things from the EnvStepper.
            self.env_output_splitter = ContainerSplitter(tuple_length=4, scope="env-output-splitter")

            self.states_dict_splitter = ContainerSplitter("RGB_INTERLEAVED", "INSTR", "previous_action", "previous_reward",
                                                          scope="states-dict-splitter")

            # Slice some data from the EnvStepper (e.g only first internal states are needed).
            self.internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)
            # Merge back to insert into FIFO.
            self.fifo_input_merger = DictMerger(*self.fifo_queue_keys)

            dummy_flattener = ReShape(flatten=True)  # dummy Flattener to calculate action-probs space
            self.environment_stepper = EnvironmentStepper(
                environment_spec=environment_spec,
                actor_component_spec=ActorComponent(self.preprocessor, self.policy, self.exploration),
                state_space=self.state_space.with_batch_rank(),
                reward_space=float,  # TODO <- float64 for deepmind? may not work for other envs
                internal_states_space=self.internal_states_space,
                num_steps=self.worker_sample_size,
                add_previous_action=True,
                add_previous_reward=True,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space)
            )
            sub_components = [
                self.environment_stepper, self.env_output_splitter,
                self.internal_states_slicer, self.fifo_input_merger, self.states_dict_splitter,
                self.fifo_queue
            ]
        # Learner.
        else:
            self.environment_stepper = None

            # A Dict splitter to split up items from the queue.
            self.fifo_input_merger = None
            self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys, scope="fifo-output-splitter")
            self.states_dict_splitter = ContainerSplitter("INSTR", "RGB_INTERLEAVED", "previous_action",
                                                          "previous_reward", scope="states-dict-splitter")
            self.internal_states_slicer = None

            self.transpose_states = Transpose(
                scope="transpose-states",
                device=dict(ops="/job:learner/task:0/cpu")
            )
            self.transpose_terminals = Transpose(
                scope="transpose-terminals",
                device=dict(ops="/job:learner/task:0/cpu")
            )
            self.transpose_action_probs = Transpose(
                scope="transpose-a-probs-mu",
                device=dict(ops="/job:learner/task:0/cpu")
            )

            self.staging_area = StagingArea(num_data=len(self.fifo_queue_keys))

            # Create an IMPALALossFunction with some parameters.
            self.loss_function = IMPALALossFunction(
                discount=self.discount, weight_pg=weight_pg, weight_baseline=weight_baseline,
                weight_entropy=weight_entropy, device="/job:learner/task:0/gpu"
            )

            self.policy.propagate_sub_component_properties(
                dict(device=dict(variables="/job:learner/task:0/cpu", ops="/job:learner/task:0/gpu"))
            )
            for component in [self.staging_area, self.preprocessor, self.optimizer]:
                component.propagate_sub_component_properties(
                    dict(device="/job:learner/task:0/gpu")
                )

            sub_components = [
                self.fifo_output_splitter, self.fifo_queue, self.states_dict_splitter,
                self.transpose_states, self.transpose_terminals, self.transpose_action_probs,
                self.staging_area, self.preprocessor, self.policy,
                self.loss_function, self.optimizer
            ]

        # Add all the agent's sub-components to the root.
        self.root_component.add_components(*sub_components)

        # Define the Agent's (root Component's) API.
        self.define_graph_api(*sub_components)

        if self.auto_build:
            if self.type == "learner":
                build_options = dict(
                    build_device_context="/job:learner/task:0/cpu",
                    pin_global_variable_device="/job:learner/task:0/cpu"
                )
                self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                                  build_options=build_options)
            else:
                self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                                  build_options=None)

            self.graph_built = True

            if self.has_gpu:
                # Get 1st return op of API-method `stage` of sub-component `staging-area` (which is the stage-op).
                self.stage_op = self.root_component.sub_components["staging-area"].api_methods["stage"]. \
                    out_op_columns[0].op_records[0].op
                # Initialize the stage.
                self.graph_executor.monitored_session.run_step_fn(
                    lambda step_context: step_context.session.run(self.stage_op)
                )

                # TODO remove after full refactor.
                self.dequeue_op = self.root_component.sub_components["fifo-queue"].api_methods["get_records"]. \
                    out_op_columns[0].op_records[0].op
            if self.type == "actor":
                self.enqueue_op = self.root_component.sub_components["fifo-queue"].api_methods["insert_records"]. \
                    out_op_columns[0].op_records[0].op

    def define_graph_api(self, *sub_components):
        # TODO: Unify agents with/w/o synchronizable policy.
        # TODO: Unify Agents with/w/o get_action method (w/ env-stepper vs w/o).
        #global_scope_base = "environment-stepper/actor-component/" if self.type == "actor" else ""
        #super(IMPALAAgent, self).define_graph_api(
        #    global_scope_base+"policy",
        #    global_scope_base+"dict-preprocessor-stack"
        #)

        # Assemble the specific agent.
        if self.type == "single":
            self.define_graph_api_single(*sub_components)
        elif self.type == "actor":
            self.define_graph_api_actor(*sub_components)
        else:
            self.define_graph_api_learner(*sub_components)

    def define_graph_api_single(self, fifo_output_splitter, fifo_queue, queue_runner, transpose_actions,
                                  transpose_rewards, transpose_terminals, transpose_action_probs, preprocessor,
                                  staging_area, concat, policy, loss_function, optimizer):
        @rlgraph_api(component=self.root_component)
        def setup_queue_runner(self_):
            return queue_runner.setup()

        @rlgraph_api(component=self.root_component)
        def get_queue_size(self_):
            return fifo_queue.get_size()

        @rlgraph_api(component=self.root_component)
        def update_from_memory(self_):
            # Pull n records from the queue.
            # Note that everything will come out as batch-major and must be transposed before the main-LSTM.
            # This is done by the network itself for all network inputs:
            # - preprocessed_s
            # - preprocessed_last_s_prime
            # But must still be done for actions, rewards, terminals here in this API-method via separate ReShapers.
            records = fifo_queue.get_records(self.update_spec["batch_size"])

            preprocessed_s, actions, rewards, terminals, last_s_prime, action_probs_mu, \
                initial_internal_states = fifo_output_splitter.split(records)

            preprocessed_last_s_prime = preprocessor.preprocess(last_s_prime)

            # Append last-next-state to the rest before sending it through the network.
            preprocessed_s_all = concat.apply(preprocessed_s, preprocessed_last_s_prime)

            # Flip actions, rewards, terminals to time-major.
            # TODO: Create components that are less input-space sensitive (those that have no variables should
            # TODO: be reused for any kind of processing)
            actions = transpose_actions.apply(actions)
            rewards = transpose_rewards.apply(rewards)
            terminals = transpose_terminals.apply(terminals)
            action_probs_mu = transpose_action_probs.apply(action_probs_mu)

            # If we use a GPU: Put everything on staging area (adds 1 time step policy lag, but makes copying
            # data into GPU more efficient).
            if self.has_gpu:
                stage_op = staging_area.stage(
                    preprocessed_s_all, actions, rewards, terminals, action_probs_mu, initial_internal_states
                )
                # Get data from stage again and continue.
                preprocessed_s_all, actions, rewards, terminals, action_probs_mu, \
                    initial_internal_states = staging_area.unstage()
                # endif.
            else:
                # TODO: No-op component?
                stage_op = None

            # Get the pi-action probs AND the values for all our states.
            state_values_pi, logits_pi, probs_pi, log_probabilities_pi, current_internal_states = \
                policy.get_state_values_logits_probabilities_log_probs(
                    preprocessed_s_all, initial_internal_states
                )

            # Calculate the loss.
            loss, loss_per_item = loss_function.loss(log_probabilities_pi, action_probs_mu, state_values_pi, actions,
                                                     rewards, terminals)
            if self.dynamic_batching:
                policy_vars = queue_runner.data_producing_components[0].actor_component.policy._variables()
            else:
                policy_vars = policy._variables()

            # TODO: dynbatching tboard check
            #return loss, loss, loss, loss

            # Pass vars and loss values into optimizer.
            step_op, loss, loss_per_item = optimizer.step(policy_vars, loss, loss_per_item)

            # Return optimizer op and all loss values.
            # TODO: Make it possible to return None from API-method without messing with the meta-graph.
            return step_op, (stage_op if stage_op else step_op), loss, loss_per_item

    def define_graph_api_actor(self, env_stepper, env_output_splitter, internal_states_slicer, merger,
                                 states_dict_splitter, fifo_queue):
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
        @rlgraph_api(component=self.root_component)
        def perform_n_steps_and_insert_into_fifo(self_):  #, internal_states, time_step=0):
            # Take n steps in the environment.
            step_results = env_stepper.step()

            terminals, states, action_log_probs, internal_states = env_output_splitter.split(step_results)
            initial_internal_states = internal_states_slicer.slice(internal_states, 0)

            record = merger.merge(terminals, states, action_log_probs, initial_internal_states)

            # Insert results into the FIFOQueue.
            insert_op = fifo_queue.insert_records(record)

            return insert_op, terminals

        @rlgraph_api(component=self.root_component)
        def reset(self):
            # Resets the environment running inside the agent.
            reset_op = env_stepper.reset()
            return reset_op

    def define_graph_api_learner(
            self, fifo_output_splitter, fifo_queue, states_dict_splitter, transpose_states, transpose_terminals,
            transpose_action_probs, staging_area, preprocessor, policy, loss_function, optimizer
    ):
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
        @rlgraph_api(component=self.root_component)
        def update_from_memory(self_):
            # Pull n records from the queue.
            # Note that everything will come out as batch-major and must be transposed before the main-LSTM.
            # This is done by the network itself for all network inputs:
            # - preprocessed_s
            # - preprocessed_last_s_prime
            # But must still be done for actions, rewards, terminals here in this API-method via separate ReShapers.
            records = fifo_queue.get_records(self.update_spec["batch_size"])

            terminals, states, action_probs_mu, initial_internal_states = fifo_output_splitter.split(records)

            # Flip everything to time-major.
            # TODO: Create components that are less input-space sensitive (those that have no variables should
            # TODO: be reused for any kind of processing)
            states = transpose_states.apply(states)
            terminals = transpose_terminals.apply(terminals)
            action_probs_mu = transpose_action_probs.apply(action_probs_mu)

            # If we use a GPU: Put everything on staging area (adds 1 time step policy lag, but makes copying
            # data into GPU more efficient).
            if self.has_gpu:
                stage_op = staging_area.stage(states, terminals, action_probs_mu, initial_internal_states)
                # Get data from stage again and continue.
                states, terminals, action_probs_mu, initial_internal_states = staging_area.unstage()
            else:
                # TODO: No-op component?
                stage_op = None

            # Preprocess actions and rewards inside the state (actions: flatten one-hot, rewards: expand).
            states = preprocessor.preprocess(states)

            # state_values_pi, _, _, log_probabilities_pi, current_internal_states = \
            #     policy.get_state_values_logits_probabilities_log_probs(states, initial_internal_states)

            # Only retrieve logits and do faster sparse softmax in loss.
            state_values_pi, logits, _, _, current_internal_states = \
                policy.get_state_values_logits_probabilities_log_probs(states, initial_internal_states)

            # Isolate actions and rewards from states.
            _, _, actions, rewards = states_dict_splitter.split(states)

            # Calculate the loss.
            # step_op,\  <- DEBUG: fake step op
            loss, loss_per_item = loss_function.loss(logits, action_probs_mu, state_values_pi, actions,
                                                     rewards, terminals)
            policy_vars = policy._variables()

            # Pass vars and loss values into optimizer.
            step_op, loss, loss_per_item = optimizer.step(policy_vars, loss, loss_per_item)

            # Return optimizer op and all loss values.
            # TODO: Make it possible to return None from API-method without messing with the meta-graph.
            return step_op, (stage_op if stage_op else step_op), loss, loss_per_item

        @rlgraph_api(component=self.root_component)
        def get_queue_size(self_):
            return fifo_queue.get_size()

    def get_action(self, states, internal_states=None, use_exploration=True, extra_returns=None):
        pass

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None):
        if batch is None:
            # Include stage_op or not?
            if self.has_gpu:
                return self.graph_executor.execute("update_from_memory")
            else:
                return self.graph_executor.execute(("update_from_memory", None, ([0, 2, 3])))
        else:
            raise RLGraphError("Cannot call update-from-batch on an IMPALA Agent.")

    def __repr__(self):
        return "IMPALAAgent(type={})".format(self.type)



