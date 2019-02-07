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

import copy

from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils import RLGraphError
from rlgraph.agents.agent import Agent
from rlgraph.components.common.container_merger import ContainerMerger
from rlgraph.components.common.container_splitter import ContainerSplitter
from rlgraph.components.common.slice import Slice
from rlgraph.components.common.staging_area import StagingArea
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.layers.preprocessing.transpose import Transpose
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.policies.dynamic_batching_policy import DynamicBatchingPolicy
from rlgraph.components.loss_functions.impala_loss_function import IMPALALossFunction
from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.components.memories.queue_runner import QueueRunner
from rlgraph.spaces import FloatBox, Dict, Tuple
from rlgraph.utils.util import default_dict


class IMPALAAgent(Agent):
    """
    An Agent implementing the IMPALA algorithm described in [1]. The Agent contains both learner and actor
    API-methods, which will be put into the graph depending on the type ().

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """

    default_internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=False)
    default_environment_spec = dict(
        type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
        frameskip=4
    )

    def __init__(self, discount=0.99, fifo_queue_spec=None, architecture="large", environment_spec=None,
                 feed_previous_action_through_nn=True, feed_previous_reward_through_nn=True,
                 weight_pg=None, weight_baseline=None, weight_entropy=None, worker_sample_size=100,
                 **kwargs):
        """
        Args:
            discount (float): The discount factor gamma.
            architecture (str): Which IMPALA architecture to use. One of "small" or "large". Will be ignored if
                `network_spec` is given explicitly in kwargs. Default: "large".
            fifo_queue_spec (Optional[dict,FIFOQueue]): The spec for the FIFOQueue to use for the IMPALA algorithm.
            environment_spec (dict): The spec for constructing an Environment object for an actor-type IMPALA agent.
            feed_previous_action_through_nn (bool): Whether to add the previous action as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_action". Default: True.
            feed_previous_reward_through_nn (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: True.
            weight_pg (float): See IMPALALossFunction Component.
            weight_baseline (float): See IMPALALossFunction Component.
            weight_entropy (float): See IMPALALossFunction Component.
            worker_sample_size (int): How many steps the actor will perform in the environment each sample-run.

        Keyword Args:
            type (str): One of "single", "actor" or "learner". Default: "single".
        """
        type_ = kwargs.pop("type", "single")
        assert type_ in ["single", "actor", "learner"]
        self.type = type_
        self.worker_sample_size = worker_sample_size

        # Network-spec by default is a "large architecture" IMPALA network.
        self.network_spec = kwargs.pop(
            "network_spec",
            dict(type="rlgraph.components.neural_networks.impala.impala_networks.{}IMPALANetwork".
                 format("Large" if architecture == "large" else "Small"))
        )
        if isinstance(self.network_spec, dict) and "type" in self.network_spec and \
                "IMPALANetwork" in self.network_spec["type"]:
            self.network_spec = default_dict(
                self.network_spec,
                dict(worker_sample_size=1 if self.type == "actor" else self.worker_sample_size + 1)
            )

        # Depending on the job-type, remove the pieces from the Agent-spec/graph we won't need.
        self.exploration_spec = kwargs.pop("exploration_spec", None)
        optimizer_spec = kwargs.pop("optimizer_spec", None)
        observe_spec = kwargs.pop("observe_spec", None)

        self.feed_previous_action_through_nn = feed_previous_action_through_nn
        self.feed_previous_reward_through_nn = feed_previous_reward_through_nn

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
            observe_spec = None
            update_spec = kwargs.pop("update_spec", None)
            environment_spec = None

        # Add previous-action/reward preprocessors to env-specific preprocessor spec.
        # TODO: remove this empty hard-coded preprocessor.
        self.preprocessing_spec = kwargs.pop(
            "preprocessing_spec", dict(type="dict-preprocessor-stack", preprocessors=dict(
                # Flatten actions.
                previous_action=[
                    dict(type="reshape", flatten=True, flatten_categories=kwargs.get("action_space").num_categories)
                ],
                # Bump reward and convert to float32, so that it can be concatenated by the Concat layer.
                previous_reward=[
                    dict(type="reshape", new_shape=(1,))
                ]
            ))
        )

        # Limit communication in distributed mode between each actor and the learner (never between actors).
        execution_spec = kwargs.pop("execution_spec", None)
        if execution_spec is not None and execution_spec.get("mode") == "distributed":
            default_dict(execution_spec["session_config"], dict(
                type="monitored-training-session",
                allow_soft_placement=True,
                device_filters=["/job:learner/task:0"] + (
                    ["/job:actor/task:{}".format(execution_spec["distributed_spec"]["task_index"])] if
                    self.type == "actor" else ["/job:learner/task:0"]
                )
            ))
            # If Actor, make non-chief in either case (even if task idx == 0).
            if self.type == "actor":
                execution_spec["distributed_spec"]["is_chief"] = False
                # Hard-set device to the CPU for actors.
                execution_spec["device_strategy"] = "custom"
                execution_spec["default_device"] = "/job:{}/task:{}/cpu".format(self.type, execution_spec["distributed_spec"]["task_index"])

        self.policy_spec = kwargs.pop("policy_spec", dict())
        # TODO: Create some auto-setting based on LSTM inside the NN.
        default_dict(self.policy_spec, dict(
            type="shared-value-function-policy",
            deterministic=False,
            reuse_variable_scope="shared-policy",
            action_space=kwargs.get("action_space")
        ))

        # Now that we fixed the Agent's spec, call the super constructor.
        super(IMPALAAgent, self).__init__(
            discount=discount,
            preprocessing_spec=self.preprocessing_spec,
            network_spec=self.network_spec,
            policy_spec=self.policy_spec,
            exploration_spec=self.exploration_spec,
            optimizer_spec=optimizer_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,
            execution_spec=execution_spec,
            name=kwargs.pop("name", "impala-{}-agent".format(self.type)),
            **kwargs
        )
        # Always use 1st learner as the parameter server for all policy variables.
        if self.execution_spec["mode"] == "distributed" and self.execution_spec["distributed_spec"]["cluster_spec"]:
            self.policy.propagate_sub_component_properties(dict(device=dict(variables="/job:learner/task:0/cpu")))

        # Check whether we have an RNN.
        self.has_rnn = self.policy.neural_network.has_rnn()
        # Check, whether we are running with GPU.
        self.has_gpu = self.execution_spec["gpu_spec"]["gpus_enabled"] is True and \
            self.execution_spec["gpu_spec"]["num_gpus"] > 0

        # Some FIFO-queue specs.
        self.fifo_queue_keys = ["terminals", "states"] + \
                               (["actions"] if not self.feed_previous_action_through_nn else []) + \
                               (["rewards"] if not self.feed_previous_reward_through_nn else []) + \
                               ["action_probs"] + \
                               (["initial_internal_states"] if self.has_rnn else [])
        # Define FIFO record space.
        # Note that only states and internal_states (RNN) contain num-steps+1 items, all other sub-records only contain
        # num-steps items.
        self.fifo_record_space = Dict(
            {
                "terminals": bool,
                "action_probs": FloatBox(shape=(self.action_space.num_categories,)),
            }, add_batch_rank=False, add_time_rank=self.worker_sample_size
        )
        self.fifo_record_space["states"] = self.state_space.with_time_rank(self.worker_sample_size + 1)
        # Add action and rewards to state or do they have an extra channel?
        if self.feed_previous_action_through_nn:
            self.fifo_record_space["states"]["previous_action"] = \
                self.action_space.with_time_rank(self.worker_sample_size + 1)
        else:
            self.fifo_record_space["actions"] = self.action_space.with_time_rank(self.worker_sample_size)
        if self.feed_previous_action_through_nn:
            self.fifo_record_space["states"]["previous_reward"] = FloatBox(add_time_rank=self.worker_sample_size + 1)
        else:
            self.fifo_record_space["rewards"] = FloatBox(add_time_rank=self.worker_sample_size)

        if self.has_rnn:
            self.fifo_record_space["initial_internal_states"] = self.internal_states_space.with_time_rank(False)

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
            pass

        elif self.type == "actor":
            # No learning, no loss function.
            self.loss_function = None
            # A Dict Splitter to split things from the EnvStepper.
            self.env_output_splitter = ContainerSplitter(tuple_length=4, scope="env-output-splitter")

            self.states_dict_splitter = None

            # Slice some data from the EnvStepper (e.g only first internal states are needed).
            self.internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)
            # Merge back to insert into FIFO.
            self.fifo_input_merger = ContainerMerger(*self.fifo_queue_keys)

            # Dummy Flattener to calculate action-probs space.
            dummy_flattener = ReShape(flatten=True, flatten_categories=self.action_space.num_categories)
            self.environment_stepper = EnvironmentStepper(
                environment_spec=environment_spec,
                actor_component_spec=ActorComponent(self.preprocessor, self.policy, self.exploration),
                state_space=self.state_space.with_batch_rank(),
                reward_space=float,  # TODO <- float64 for deepmind? may not work for other envs
                internal_states_space=self.internal_states_space,
                num_steps=self.worker_sample_size,
                add_previous_action_to_state=True,
                add_previous_reward_to_state=True,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space)
            )
            sub_components = [
                self.environment_stepper, self.env_output_splitter,
                self.internal_states_slicer, self.fifo_input_merger,
                self.fifo_queue
            ]
        # Learner.
        else:
            self.environment_stepper = None

            # A Dict splitter to split up items from the queue.
            self.fifo_input_merger = None
            self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys, scope="fifo-output-splitter")
            self.states_dict_splitter = ContainerSplitter(
                *list(self.fifo_record_space["states"].keys()), scope="states-dict-splitter"
            )
            self.internal_states_slicer = None

            self.transposer = Transpose(
                scope="transposer",
                device=dict(ops="/job:learner/task:0/cpu")
            )
            self.staging_area = StagingArea(num_data=len(self.fifo_queue_keys))

            # Create an IMPALALossFunction with some parameters.
            self.loss_function = IMPALALossFunction(
                discount=self.discount, weight_pg=weight_pg, weight_baseline=weight_baseline,
                weight_entropy=weight_entropy,
                slice_actions=self.feed_previous_action_through_nn,
                slice_rewards=self.feed_previous_reward_through_nn,
                device="/job:learner/task:0/gpu"
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
                self.transposer,
                self.staging_area, self.preprocessor, self.policy,
                self.loss_function, self.optimizer
            ]

        if self.type != "single":
            # Add all the agent's sub-components to the root.
            self.root_component.add_components(*sub_components)

            # Define the Agent's (root Component's) API.
            self.define_graph_api(*sub_components)

        if self.type != "single" and self.auto_build:
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
            pass
        elif self.type == "actor":
            self.define_graph_api_actor(*sub_components)
        else:
            self.define_graph_api_learner(*sub_components)

    def define_graph_api_actor(self, env_stepper, env_output_splitter, internal_states_slicer, merger, fifo_queue):
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
        def perform_n_steps_and_insert_into_fifo(self_):
            # Take n steps in the environment.
            step_results = env_stepper.step()

            split_output = env_output_splitter.split(step_results)
            # Slice off the initial internal state (so the learner can re-feed-forward from that internal-state).
            initial_internal_states = internal_states_slicer.slice(split_output[-1], 0)  # -1=internal states
            to_merge = split_output[:-1] + (initial_internal_states,)
            record = merger.merge(*to_merge)

            # Insert results into the FIFOQueue.
            insert_op = fifo_queue.insert_records(record)

            return insert_op, split_output[0]  # 0=terminals

    def define_graph_api_learner(
            self, fifo_output_splitter, fifo_queue, states_dict_splitter,
            transposer, staging_area, preprocessor, policy, loss_function, optimizer
    ):
        """
        Defines the API-methods used by an IMPALA learner. Its job is basically: Pull a batch from the
        FIFOQueue, split it up into its components and pass these through the loss function and into the optimizer for
        a learning update.

        Args:
            fifo_output_splitter (ContainerSplitter): The ContainerSplitter Component to split up a batch from the queue
                along its items.

            fifo_queue (FIFOQueue): The FIFOQueue Component used to enqueue env sample runs (n-step).

            states_dict_splitter (ContainerSplitter): The ContainerSplitter Component to split the state components
                into its single parts.

            transposer (Transpose): A space-agnostic Transpose to flip batch- and time ranks of all state-components.
            staging_area (StagingArea): A possible GPU stating area component.

            preprocessor (PreprocessorStack): A preprocessing Component for the states (may be a DictPreprocessorStack
                as well).

            policy (Policy): The Policy Component, which to update.
            loss_function (IMPALALossFunction): The IMPALALossFunction Component.
            optimizer (Optimizer): The optimizer that we use to calculate an update and apply it.
        """
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

            split_record = fifo_output_splitter.split(records)
            actions = None
            rewards = None
            if self.feed_previous_action_through_nn and self.feed_previous_reward_through_nn:
                terminals, states, action_probs_mu, initial_internal_states = split_record
            else:
                terminals, states, actions, rewards, action_probs_mu, initial_internal_states = split_record

            # Flip everything to time-major.
            # TODO: Create components that are less input-space sensitive (those that have no variables should
            # TODO: be reused for any kind of processing)
            states = transposer.apply(states)
            terminals = transposer.apply(terminals)
            action_probs_mu = transposer.apply(action_probs_mu)
            if self.feed_previous_action_through_nn is False:
                actions = transposer.apply(actions)
            if self.feed_previous_reward_through_nn is False:
                rewards = transposer.apply(rewards)

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
            preprocessed_states = preprocessor.preprocess(states)

            # Only retrieve logits and do faster sparse softmax in loss.
            out = policy.get_state_values_logits_parameters_log_probs(preprocessed_states, initial_internal_states)
            state_values_pi = out["state_values"]
            logits = out["logits"]
            #current_internal_states = out["last_internal_states"]

            # Isolate actions and rewards from states.
            if self.feed_previous_action_through_nn or self.feed_previous_reward_through_nn:
                states_split = states_dict_splitter.split(states)
                actions = states_split[-2]
                rewards = states_split[-1]

            # Calculate the loss.
            loss, loss_per_item = loss_function.loss(
                logits, action_probs_mu, state_values_pi, actions, rewards, terminals
            )
            policy_vars = policy.variables()

            # Pass vars and loss values into optimizer.
            step_op, loss, loss_per_item = optimizer.step(policy_vars, loss, loss_per_item)

            # Return optimizer op and all loss values.
            # TODO: Make it possible to return None from API-method without messing with the meta-graph.
            return step_op, (stage_op if stage_op else step_op), loss, loss_per_item, records

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
                return self.graph_executor.execute(("update_from_memory", None, ([0, 2, 3, 4])))
        else:
            raise RLGraphError("Cannot call update-from-batch on an IMPALA Agent.")

    def __repr__(self):
        return "IMPALAAgent(type={})".format(self.type)


class SingleIMPALAAgent(IMPALAAgent):
    """
    An single IMPALAAgent, performing both experience collection and learning updates via multi-threading
    (queue runners).
    """
    def __init__(self, discount=0.99, fifo_queue_spec=None, architecture="large", environment_spec=None,
                 feed_previous_action_through_nn=True, feed_previous_reward_through_nn=True,
                 weight_pg=None, weight_baseline=None, weight_entropy=None,
                 num_workers=1, worker_sample_size=100,
                 dynamic_batching=False, visualize=False, **kwargs):
        """
        Args:
            discount (float): The discount factor gamma.
            architecture (str): Which IMPALA architecture to use. One of "small" or "large". Will be ignored if
                `network_spec` is given explicitly in kwargs. Default: "large".
            fifo_queue_spec (Optional[dict,FIFOQueue]): The spec for the FIFOQueue to use for the IMPALA algorithm.
            environment_spec (dict): The spec for constructing an Environment object for an actor-type IMPALA agent.
            feed_previous_action_through_nn (bool): Whether to add the previous action as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_action". Default: True.
            feed_previous_reward_through_nn (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: True.
            weight_pg (float): See IMPALALossFunction Component.
            weight_baseline (float): See IMPALALossFunction Component.
            weight_entropy (float): See IMPALALossFunction Component.
            num_workers (int): How many actors (workers) should be run in separate threads.
            worker_sample_size (int): How many steps the actor will perform in the environment each sample-run.
            dynamic_batching (bool): Whether to use the deepmind's custom dynamic batching op for wrapping the
                optimizer's step call. The batcher.so file must be compiled for this to work (see Docker file).
                Default: False.
            visualize (Union[int,bool]): Whether and how many workers to visualize.
                Default: False (no visualization).
        """
        # Now that we fixed the Agent's spec, call the super constructor.
        super(SingleIMPALAAgent, self).__init__(
            type="single",
            discount=discount,
            architecture=architecture,
            fifo_queue_spec=fifo_queue_spec,
            environment_spec=environment_spec,
            feed_previous_action_through_nn=feed_previous_action_through_nn,
            feed_previous_reward_through_nn=feed_previous_reward_through_nn,
            weight_pg=weight_pg,
            weight_baseline=weight_baseline,
            weight_entropy=weight_entropy,
            worker_sample_size=worker_sample_size,
            name=kwargs.pop("name", "impala-single-agent"),
            **kwargs
        )
        self.dynamic_batching = dynamic_batching
        self.num_workers = num_workers
        self.visualize = visualize

        # If we use dynamic batching, wrap the dynamic batcher around the policy's graph_fn that we
        # actually call below during our build.
        if self.dynamic_batching:
            self.policy = DynamicBatchingPolicy(policy_spec=self.policy, scope="")

        self.env_output_splitter = ContainerSplitter(
            tuple_length=3 if self.has_rnn is False else 4, scope="env-output-splitter"
        )
        self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys, scope="fifo-output-splitter")
        self.states_dict_splitter = ContainerSplitter(
            *list(self.fifo_record_space["states"].keys() if isinstance(self.state_space, Dict) else "dummy"),
            scope="states-dict-splitter"
        )

        self.staging_area = StagingArea(num_data=len(self.fifo_queue_keys))

        # Slice some data from the EnvStepper (e.g only first internal states are needed).
        if self.has_rnn:
            internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)
        else:
            internal_states_slicer = None

        self.transposer = Transpose(scope="transposer")

        # Create an IMPALALossFunction with some parameters.
        self.loss_function = IMPALALossFunction(
            discount=self.discount, weight_pg=weight_pg, weight_baseline=weight_baseline,
            weight_entropy=weight_entropy, slice_actions=self.feed_previous_action_through_nn,
            slice_rewards=self.feed_previous_reward_through_nn
        )

        # Merge back to insert into FIFO.
        self.fifo_input_merger = ContainerMerger(*self.fifo_queue_keys)

        # Dummy Flattener to calculate action-probs space.
        dummy_flattener = ReShape(flatten=True, flatten_categories=self.action_space.num_categories)

        self.environment_steppers = list()
        for i in range(self.num_workers):
            environment_spec_ = copy.deepcopy(environment_spec)
            if self.visualize is True or (isinstance(self.visualize, int) and i+1 <= self.visualize):
                environment_spec_["visualize"] = True

            # Force worker_sample_size for IMPALA NNs (LSTM) in env-stepper to be 1.
            policy_spec = copy.deepcopy(self.policy_spec)
            if isinstance(policy_spec, dict) and isinstance(policy_spec["network_spec"], dict) and \
                    "type" in policy_spec["network_spec"] and "IMPALANetwork" in policy_spec["network_spec"]["type"]:
                policy_spec["network_spec"]["worker_sample_size"] = 1

            env_stepper = EnvironmentStepper(
                environment_spec=environment_spec_,
                actor_component_spec=ActorComponent(
                    preprocessor_spec=self.preprocessing_spec,
                    policy_spec=policy_spec,
                    exploration_spec=self.exploration_spec
                ),
                state_space=self.state_space.with_batch_rank(),
                action_space=self.action_space.with_batch_rank(),
                reward_space=float,
                internal_states_space=self.internal_states_space,
                num_steps=self.worker_sample_size,
                add_action=not self.feed_previous_action_through_nn,
                add_reward=not self.feed_previous_reward_through_nn,
                add_previous_action_to_state=self.feed_previous_action_through_nn,
                add_previous_reward_to_state=self.feed_previous_reward_through_nn,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space),
                scope="env-stepper-{}".format(i)
            )
            if self.dynamic_batching:
                env_stepper.actor_component.policy.parent_component = None
                env_stepper.actor_component.policy = DynamicBatchingPolicy(
                    policy_spec=env_stepper.actor_component.policy, scope="")
                env_stepper.actor_component.add_components(env_stepper.actor_component.policy)

            self.environment_steppers.append(env_stepper)

        # Create the QueueRunners (one for each env-stepper).
        self.queue_runner = QueueRunner(
            self.fifo_queue, "step", -1,  # -1: Take entire return value of API-method `step` as record to insert.
            self.env_output_splitter,
            self.fifo_input_merger,
            internal_states_slicer,
            *self.environment_steppers
        )

        sub_components = [
            self.fifo_output_splitter, self.fifo_queue, self.queue_runner,
            self.transposer,
            self.staging_area, self.preprocessor, self.states_dict_splitter,
            self.policy, self.loss_function, self.optimizer
        ]

        # Add all the agent's sub-components to the root.
        self.root_component.add_components(*sub_components)

        # Define the Agent's (root Component's) API.
        self.define_graph_api()

        if self.auto_build:
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

    def define_graph_api(self):

        agent = self

        @rlgraph_api(component=self.root_component)
        def setup_queue_runner(root):
            return agent.queue_runner.setup()

        @rlgraph_api(component=self.root_component)
        def get_queue_size(root):
            return agent.fifo_queue.get_size()

        @rlgraph_api(component=self.root_component)
        def update_from_memory(root):
            # Pull n records from the queue.
            # Note that everything will come out as batch-major and must be transposed before the main-LSTM.
            # This is done by the network itself for all network inputs:
            # - preprocessed_s
            # - preprocessed_last_s_prime
            # But must still be done for actions, rewards, terminals here in this API-method via separate ReShapers.
            records = agent.fifo_queue.get_records(self.update_spec["batch_size"])

            out = agent.fifo_output_splitter.split_into_dict(records)
            terminals = out["terminals"]
            states = out["states"]
            action_probs_mu = out["action_probs"]
            initial_internal_states = None
            if self.has_rnn:
                initial_internal_states = out["initial_internal_states"]

            # Flip everything to time-major.
            # TODO: Create components that are less input-space sensitive (those that have no variables should
            # TODO: be reused for any kind of processing: already done, use space_agnostic feature. See ReShape)
            states = agent.transposer.apply(states)
            terminals = agent.transposer.apply(terminals)
            action_probs_mu = agent.transposer.apply(action_probs_mu)
            actions = None
            if not self.feed_previous_action_through_nn:
                actions = agent.transposer.apply(out["actions"])
            rewards = None
            if not self.feed_previous_reward_through_nn:
                rewards = agent.transposer.apply(out["rewards"])

            # If we use a GPU: Put everything on staging area (adds 1 time step policy lag, but makes copying
            # data into GPU more efficient).
            if self.has_gpu:
                if self.has_rnn:
                    stage_op = agent.staging_area.stage(states, terminals, action_probs_mu, initial_internal_states)
                    states, terminals, action_probs_mu, initial_internal_states = agent.staging_area.unstage()
                else:
                    stage_op = agent.staging_area.stage(states, terminals, action_probs_mu)
                    states, terminals, action_probs_mu = agent.staging_area.unstage()
            else:
                # TODO: No-op component?
                stage_op = None

            # Preprocess actions and rewards inside the state (actions: flatten one-hot, rewards: expand).
            if agent.preprocessing_required:
                states = agent.preprocessor.preprocess(states)

            # Get the pi-action probs AND the values for all our states.
            out = agent.policy.get_state_values_logits_parameters_log_probs(states, initial_internal_states)
            state_values_pi = out["state_values"]
            logits_pi = out["logits"]

            # Isolate actions and rewards from states.
            # TODO: What if only one of actions or rewards is fed through NN, but the other not?
            if self.feed_previous_reward_through_nn and self.feed_previous_action_through_nn:
                out = agent.states_dict_splitter.split(states)
                actions = out[-2]  # TODO: Are these always the correct slots for "previous_action" and "previous_reward"?
                rewards = out[-1]

            # Calculate the loss.
            loss, loss_per_item = agent.loss_function.loss(
                logits_pi, action_probs_mu, state_values_pi, actions, rewards, terminals
            )
            if self.dynamic_batching:
                policy_vars = agent.queue_runner.data_producing_components[0].actor_component.policy.variables()
            else:
                policy_vars = agent.policy.variables()

            # Pass vars and loss values into optimizer.
            step_op, loss, loss_per_item = agent.optimizer.step(policy_vars, loss, loss_per_item)

            # Return optimizer op and all loss values.
            # TODO: Make it possible to return None from API-method without messing with the meta-graph.
            return step_op, (stage_op if stage_op else step_op), loss, loss_per_item, records

    def __repr__(self):
        return "SingleIMPALAAgent()"
