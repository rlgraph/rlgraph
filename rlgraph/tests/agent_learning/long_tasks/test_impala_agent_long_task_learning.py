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

import logging
import numpy as np
import time
import unittest

from rlgraph.environments import OpenAIGymEnv
from rlgraph.agents import IMPALAAgent
from rlgraph.spaces import FloatBox
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path


class TestIMPALAAgentLongTaskLearning(unittest.TestCase):
    """
    Tests whether the DQNAgent can learn in tough environments.
    """
    root_logger.setLevel(level=logging.INFO)

    #atari_preprocessed_state_space = FloatBox(shape=(80, 80, 4), add_batch_rank=True)
    #atari_preprocessing_spec = [
    #    dict(type="image_crop", x=0, y=25, width=160, height=160),
    #    dict(type="image_resize", width=80, height=80),
    #    dict(type="grayscale", keep_rank=True),
    #    dict(type="divide", divisor=255,),
    #    dict(type="sequence", sequence_length=4, batch_size=1, add_rank=False)
    #]

    def test_impala_on_outbreak(self):
        """
        Creates a DQNAgent and runs it via a Runner on an openAI Pong Env.
        """
        env = OpenAIGymEnv("Breakout-v0", frameskip=4, max_num_noops=30, episodic_life=True, visualize=False)
        config_ = config_from_path("configs/impala_agent_for_breakout.json")
        agent = IMPALAAgent.from_spec(
            config_,
            state_space=env.state_space,
            action_space=env.action_space,
        )

        learn_updates = 4000000
        mean_returns = []
        for i in range(learn_updates):
            ret = agent.update()
            mean_return = self._calc_mean_return(ret)
            mean_returns.append(mean_return)
            print("i={} Loss={:.4} Avg-reward={:.2}".format(i, float(ret[1]), mean_return))

        time.sleep(3)
        agent.terminate()
        time.sleep(3)

    @staticmethod
    def _calc_mean_return(records):
        size = records[3]["rewards"].size
        rewards = records[3]["rewards"].reshape((size,))
        terminals = records[3]["terminals"].reshape((size,))
        returns = list()
        return_ = 0.0
        for r, t in zip(rewards, terminals):
            return_ += r
            if t:
                returns.append(return_)
                return_ = 0.0

        return np.mean(returns)
