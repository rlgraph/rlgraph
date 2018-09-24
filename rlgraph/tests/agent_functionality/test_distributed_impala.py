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

import logging
import numpy as np
import time
import unittest

from rlgraph.agents.impala_agent import IMPALAAgent
from rlgraph.environments import DeepmindLabEnv
from rlgraph.execution.distributed_tf.impala.impala_worker import IMPALAWorker
from rlgraph.spaces import *
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger


class TestDistributedIMPALA(unittest.TestCase):
    """
    Tests the LargeIMPALANetwork functionality and IMPALAAgent assembly on the RandomEnv.
    For details on the IMPALA algorithm, see [1]:

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    root_logger.setLevel(level=logging.INFO)

    # Use the exact same Spaces as in the IMPALA paper.
    action_space = IntBox(9, add_batch_rank=True, add_time_rank=True, time_major=True)
    action_probs_space = FloatBox(shape=(9,), add_batch_rank=True, add_time_rank=True, time_major=True)
    input_space = Dict(
        RGB_INTERLEAVED=FloatBox(shape=(96, 72, 3)),
        INSTR=TextBox(),
        previous_action=FloatBox(shape=(9,)),
        previous_reward=FloatBox(shape=(1,)),  # add the extra rank for proper concatenating with the other inputs.
        add_batch_rank=True,
        add_time_rank=True,
        time_major=False
    )
    internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True)
    cluster_spec = dict(learner=["localhost:22222"], actor=["localhost:22223"])
    #cluster_spec_cloud = dict(learner=[":22222"], actor=["35.204.92.67:22223"])

    def test_single_impala_agent_functionality(self):
        """
        Creates a single IMPALAAgent and runs it for a few steps in a DeepMindLab Env to test
        all steps of the actor and learning process.
        """
        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        environment_spec = dict(
            type="deepmind-lab", level_id="lt_hallway_slope", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4
        )
        env = DeepmindLabEnv.from_spec(environment_spec)

        agent = IMPALAAgent.from_spec(
            agent_config,
            type="single",
            architecture="large",
            environment_spec=environment_spec,
            state_space=env.state_space,
            action_space=env.action_space,
            # TODO: automate this (by lookup from NN).
            internal_states_space=IMPALAAgent.default_internal_states_space,
            execution_spec=dict(
                enable_profiler=False,
                profiler_frequency=1,
                mode="distributed",
                distributed_spec=dict(cluster_spec=None)
            ),
            # Summarize time-steps to have an overview of the env-stepping speed.
            summary_spec=dict(summary_regexp="time-step", directory="/home/rlgraph/"),
            dynamic_batching=False,
            num_actors=4
        )
        # Count items in the queue.
        print("Items in queue: {}".format(agent.call_api_method("get_queue_size")))

        updates = 50
        update_times = list()
        print("Updating from queue ...")
        for _ in range(updates):
            start_time = time.monotonic()
            print(agent.update())
            update_times.append(time.monotonic() - start_time)

        print("Updates per second (including waiting for enqueued items): {}/s".format(updates / np.sum(update_times)))
        #print("Env-steps per second: {}".format(agent.update_spec["batch_size"]*20*updates / np.sum(update_times)))

        agent.terminate()

    def test_isolated_impala_actor_agent_functionality(self):
        """
        Creates a IMPALAAgent and runs it for a few steps in a DeepMindLab Env to test
        all steps of the learning process.
        """
        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        environment_spec = dict(
            type="deepmind-lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
            frameskip=4
        )
        env = DeepmindLabEnv.from_spec(environment_spec)

        agent = IMPALAAgent.from_spec(
            agent_config,
            type="actor",
            architecture="large",
            environment_spec=environment_spec,
            state_space=env.state_space,
            action_space=env.action_space,
            # TODO: automate this (by lookup from NN).
            internal_states_space=IMPALAAgent.default_internal_states_space,
            execution_spec=dict(
                enable_profiler=False,
                profiler_frequency=1,
            )
        )
        agent.call_api_method("reset")
        time_start = time.perf_counter()
        steps = 50
        for _ in range(steps):
            agent.call_api_method("perform_n_steps_and_insert_into_fifo")
        time_total = time.perf_counter() - time_start
        print("Done running {}x{} steps in Deepmind Lab env using IMPALA network in {}sec ({} actions/sec).".format(
            steps, agent.worker_sample_size, time_total , agent.worker_sample_size * steps / time_total)
        )

    def test_distributed_impala_agent_functionality_actor_part(self):
        """
        Creates two IMPALAAgents (actor and learner) and runs it for a few steps in a DeepMindLab Env to test
        communication between the two processes.
        """
        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        environment_spec = dict(
            type="deepmind-lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
            frameskip=4
        )
        env = DeepmindLabEnv.from_spec(environment_spec)

        agent = IMPALAAgent.from_spec(
            agent_config,
            type="actor",
            architecture="large",
            environment_spec=environment_spec,
            state_space=env.state_space,
            action_space=env.action_space,
            # TODO: automate this (by lookup from NN).
            internal_states_space=IMPALAAgent.default_internal_states_space,
            # Setup distributed tf.
            execution_spec=dict(
                mode="distributed",
                distributed_spec=dict(job="actor", task_index=0, cluster_spec=self.cluster_spec),
                session_config=dict(
                    type="monitored-training-session",
                    #log_device_placement=True
                ),
                #enable_profiler=True,
                #profiler_frequency=1
            )
        )
        print("IMPALA actor compiled.")
        agent.call_api_method("reset")
        time_start = time.perf_counter()
        steps = 50
        for _ in range(steps):
            agent.call_api_method("perform_n_steps_and_insert_into_fifo")
        time_total = time.perf_counter() - time_start
        print("Done running {}x{} steps in Deepmind Lab env using IMPALA network in {}sec ({} actions/sec).".format(
            steps, agent.worker_sample_size, time_total, agent.worker_sample_size * steps / time_total)
        )

        #worker = IMPALAWorker(agent=agent)
        # Run a few steps to produce data and start filling up the FIFO.
        #out = worker.execute_timesteps(2000)
        #print("IMPALA actor produced some data:\n{}".format(out))
        agent.terminate()

    def test_distributed_impala_agent_functionality_learner_part(self):
        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        environment_spec = dict(
            type="deepmind-lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4
        )
        env = DeepmindLabEnv.from_spec(environment_spec)
        agent = IMPALAAgent.from_spec(
            agent_config,
            type="learner",
            architecture="large",
            state_space=env.state_space,
            action_space=env.action_space,
            # TODO: automate this (by lookup from NN).
            internal_states_space=IMPALAAgent.default_internal_states_space,
            # Setup distributed tf.
            execution_spec=dict(
                mode="distributed",
                gpu_spec=dict(
                    gpus_enabled=True,
                    max_usable_gpus=1,
                    num_gpus=1
                ),
                distributed_spec=dict(job="learner", task_index=0, cluster_spec=self.cluster_spec),
                session_config=dict(
                    type="monitored-training-session",
                    allow_soft_placement=True,
                    log_device_placement=True
                ),
                enable_timeline=True,
            )
        )
        print("IMPALA learner compiled.")

        # Take one batch from the filled up queue and run an update_from_memory with the learner.
        update_steps = 10
        time_start = time.perf_counter()
        for _ in range(update_steps):
            agent.call_api_method("update_from_memory")
        time_total = time.perf_counter() - time_start
        print("Done learning {}xbatch-of-{} in {}sec ({} updates/sec).".format(
            update_steps, agent.update_spec["batch_size"], time_total , update_steps / time_total)
        )

        agent.terminate()
