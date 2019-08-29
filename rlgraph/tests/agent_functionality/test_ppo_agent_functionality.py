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

from __future__ import absolute_import, division, print_function

import logging
import unittest

from rlgraph.agents import PPOAgent
from rlgraph.environments import OpenAIGymEnv, RandomEnv
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker
from rlgraph.spaces import Dict, FloatBox, BoolBox
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger


class TestPPOAgentFunctionality(unittest.TestCase):
    """
    Tests the PPO Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_post_processing(self):
        """
        Tests external batch post-processing for the PPO agent.
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/ppo_agent_for_pong.json")
        agent = PPOAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        num_samples = 200
        states = agent.preprocessed_state_space.sample(num_samples)
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)
        sequence_indices_space = BoolBox(add_batch_rank=True)

        # GAE is separately tested, just testing if this API method returns results.
        pg_advantages = agent.post_process(dict(
            states=states,
            rewards=reward_space.sample(num_samples),
            terminals=terminal_space.sample(num_samples, fill_value=0),
            sequence_indices=sequence_indices_space.sample(num_samples, fill_value=0)
        ))

    def test_ppo_on_container_state_and_action_spaces_and_very_large_rewards(self):
        """
        Tests stability of PPO on an extreme env producing strange container states and large rewards and requiring
        container actions.
        """
        env = RandomEnv(
            state_space=Dict({"F_position": FloatBox(shape=(2,), low=0.01, high=0.02)}),
            action_space=Dict({"F_direction_low-1.0_high1.0": FloatBox(shape=(), low=-1.0, high=1.0),
                               "F_forward_direction_low-1.0_high1.0": FloatBox(shape=(), low=-1.0, high=1.0),
                               "B_jump": BoolBox()
                               }),
            reward_space=FloatBox(low=-1000.0, high=-100000.0),  # hugely negative rewards
            terminal_prob=0.0000001
        )

        agent_config = config_from_path("configs/ppo_agent_for_random_env_with_container_spaces.json")
        agent = PPOAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            preprocessing_spec=None,
            worker_executes_preprocessing=True,
            #episode_finish_callback=lambda episode_return, duration, timesteps, env_num:
            #print("episode return {}; steps={}".format(episode_return, timesteps))
        )
        results = worker.execute_timesteps(num_timesteps=int(1e6), use_exploration=True)

        print(results)

