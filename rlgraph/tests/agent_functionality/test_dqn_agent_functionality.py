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
import unittest

from rlgraph.agents import Agent
import rlgraph.spaces as spaces
from rlgraph.components.loss_functions.dqn_loss_function import DQNLossFunction
from rlgraph.environments import GridWorld, RandomEnv
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger, one_hot
from rlgraph.tests.agent_test import AgentTest


class TestDQNAgentFunctionality(unittest.TestCase):
    """
    Tests the DQN Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_dqn_functionality(self):
        """
        Creates a DQNAgent and runs it for a few steps in a GridWorld to vigorously test
        all steps of the learning process.
        """
        env = GridWorld(world="2x2", save_mode=True)  # no holes, just fire
        agent = Agent.from_spec(  # type: DQNAgent
            config_from_path("configs/dqn_agent_for_functionality_test.json"),
            double_q=True,
            dueling_q=True,
            state_space=env.state_space,
            action_space=env.action_space,
            store_last_memory_batch=True,
            store_last_q_table=True,
            discount=0.95
        )
        worker = SingleThreadedWorker(env_spec=lambda: GridWorld(world="2x2", save_mode=True), agent=agent)
        test = AgentTest(worker=worker)

        # Helper python DQNLossFunc object.
        loss_func = DQNLossFunction(backend="python", double_q=True, discount=agent.discount)
        loss_func.when_input_complete(input_spaces=dict(
            loss_per_item=[
                spaces.FloatBox(shape=(4,), add_batch_rank=True),
                spaces.IntBox(4, add_batch_rank=True),
                spaces.FloatBox(add_batch_rank=True),
                spaces.BoolBox(add_batch_rank=True),
                spaces.FloatBox(shape=(4,), add_batch_rank=True),
                spaces.FloatBox(shape=(4,), add_batch_rank=True)
            ]
        ), action_space=env.action_space)

        matrix1_qnet = np.array([[0.9] * 2] * 4)
        matrix2_qnet = np.array([[0.8] * 5] * 2)
        matrix1_target_net = np.array([[0.9] * 2] * 4)
        matrix2_target_net = np.array([[0.8] * 5] * 2)

        a = self._calculate_action(0, matrix1_qnet, matrix2_qnet)

        # 1st step -> Expect insert into python-buffer.
        # action: up (0)
        test.step(1, reset=True)
        # Environment's new state.
        test.check_env("state", 0)
        # Agent's buffer.
        test.check_agent("states_buffer", [[1.0, 0.0, 0.0, 0.0]], key_or_index="env_0")  # <- prev state (preprocessed)
        test.check_agent("actions_buffer", [a],  key_or_index="env_0")
        test.check_agent("rewards_buffer", [-1.0], key_or_index="env_0")
        test.check_agent("terminals_buffer", [False], key_or_index="env_0")
        # Memory contents.
        test.check_var("replay-memory/index", 0)
        test.check_var("replay-memory/size", 0)
        test.check_var("replay-memory/memory/states", np.array([[0] * 4] * agent.memory.capacity))
        test.check_var("replay-memory/memory/actions", np.array([0] * agent.memory.capacity))
        test.check_var("replay-memory/memory/rewards", np.array([0] * agent.memory.capacity))
        test.check_var("replay-memory/memory/terminals", np.array([False] * agent.memory.capacity))
        # Check policy and target-policy weights (should be the same).
        test.check_var("policy/neural-network/hidden/dense/kernel", matrix1_qnet)
        test.check_var("target-policy/neural-network/hidden/dense/kernel", matrix1_qnet)
        test.check_var("policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_qnet)
        test.check_var("target-policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_qnet)

        # 2nd step -> expect insert into memory (and python buffer should be empty again).
        # action: up (0)
        # Also check the policy and target policy values (Should be equal at this point).
        test.step(1)
        test.check_env("state", 0)
        test.check_agent("states_buffer", [], key_or_index="env_0")
        test.check_agent("actions_buffer", [], key_or_index="env_0")
        test.check_agent("rewards_buffer", [], key_or_index="env_0")
        test.check_agent("terminals_buffer", [], key_or_index="env_0")
        test.check_var("replay-memory/index", 2)
        test.check_var("replay-memory/size", 2)
        test.check_var("replay-memory/memory/states", np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]] +
                                                               [[0.0, 0.0, 0.0, 0.0]] * (agent.memory.capacity - 2)))
        test.check_var("replay-memory/memory/actions", np.array([0, 0] + [0] * (agent.memory.capacity - 2)))
        test.check_var("replay-memory/memory/rewards", np.array([-1.0, -1.0] + [0.0] * (agent.memory.capacity - 2)))
        test.check_var("replay-memory/memory/terminals", np.array([False, True] + [False] * (agent.memory.capacity - 2)))
        # Check policy and target-policy weights (should be the same).
        test.check_var("policy/neural-network/hidden/dense/kernel", matrix1_qnet)
        test.check_var("target-policy/neural-network/hidden/dense/kernel", matrix1_qnet)
        test.check_var("policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_qnet)
        test.check_var("target-policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_qnet)

        # 3rd and 4th step -> expect another insert into memory (and python buffer should be empty again).
        # actions: down (2), up (0)  <- exploring is True = more random actions
        # Expect an update to the policy variables (leave target as is (no sync yet)).
        test.step(2, use_exploration=True)
        test.check_env("state", 0)
        test.check_agent("states_buffer", [], key_or_index="env_0")
        test.check_agent("actions_buffer", [], key_or_index="env_0")
        test.check_agent("rewards_buffer", [], key_or_index="env_0")
        test.check_agent("terminals_buffer", [], key_or_index="env_0")
        test.check_var("replay-memory/index", 4)
        test.check_var("replay-memory/size", 4)
        test.check_var("replay-memory/memory/states", np.array([[1.0, 0.0, 0.0, 0.0]] * 3 +
                                                               [[0.0, 1.0, 0.0, 0.0]] +
                                                               [[0.0, 0.0, 0.0, 0.0]] * (agent.memory.capacity - 4)))
        test.check_var("replay-memory/memory/actions", np.array([0, 0, 2, 0] + [0] * (agent.memory.capacity - 4)))
        test.check_var("replay-memory/memory/rewards", np.array([-1.0] * 4 +  # + [-3.0] +
                                                                [0.0] * (agent.memory.capacity - 4)))
        test.check_var("replay-memory/memory/terminals", np.array([False, True] * 2 +
                                                                  [False] * (agent.memory.capacity - 4)))
        # Get the latest memory batch.
        expected_batch = dict(
            states=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            actions=np.array([0, 1]),
            rewards=np.array([-1.0, -3.0]),
            terminals=np.array([False, True]),
            next_states=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        )
        test.check_agent("last_memory_batch", expected_batch)

        # Calculate the weight updates and check against actually update weights by the AgentDQN.
        mat_updated = self._helper_update_matrix(expected_batch, matrix1_qnet, matrix2_qnet, matrix1_target_net,
                                                 matrix2_target_net, agent, loss_func)
        # Check policy and target-policy weights (policy should be updated now).
        test.check_var("policy/neural-network/hidden/dense/kernel", mat_updated[0], decimals=4)
        test.check_var("target-policy/neural-network/hidden/dense/kernel", matrix1_target_net)
        test.check_var("policy/dueling-action-adapter/action-layer/dense/kernel", mat_updated[1], decimals=4)
        test.check_var("target-policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_target_net)

        matrix1_qnet = mat_updated[0]
        matrix2_qnet = mat_updated[1]

        # 5th step -> Another buffer update check.
        # action: down (2) (weights have been updated -> different actions)
        test.step(1)
        test.check_env("state", 3)
        test.check_agent("states_buffer", [], key_or_index="env_0")  # <- all empty b/c we reached end of episode (buffer gets force-flushed)
        test.check_agent("actions_buffer", [], key_or_index="env_0")
        test.check_agent("rewards_buffer", [], key_or_index="env_0")
        test.check_agent("terminals_buffer", [], key_or_index="env_0")
        test.check_agent("last_memory_batch", expected_batch)
        test.check_var("replay-memory/index", 5)
        test.check_var("replay-memory/size", 5)
        test.check_var("replay-memory/memory/states", np.array([[1.0, 0.0, 0.0, 0.0]] * 4 + [[0.0, 0.0, 1.0, 0.0]] +
                                                               [[0.0, 0.0, 0.0, 0.0]] * (agent.memory.capacity - 5)))
        test.check_var("replay-memory/memory/actions", np.array([0, 0, 0, 1, 2, 0]))
        test.check_var("replay-memory/memory/rewards", np.array([-1.0] * 3 + [-3.0, 1.0, 0.0]))
        test.check_var("replay-memory/memory/terminals", np.array([False, True] * 2 + [True, False]))
        test.check_var("policy/neural-network/hidden/dense/kernel", matrix1_qnet, decimals=4)
        test.check_var("target-policy/neural-network/hidden/dense/kernel", matrix1_target_net)
        test.check_var("policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_qnet, decimals=4)
        test.check_var("target-policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_target_net)

        # 6th/7th step (with exploration enabled) -> Another buffer update check.
        # action: up, down (0, 2)
        test.step(2, use_exploration=True)
        test.check_env("state", 1)
        test.check_agent("states_buffer", [], key_or_index="env_0")  # <- all empty again; flushed after 6th step (when buffer was full).
        test.check_agent("actions_buffer", [], key_or_index="env_0")
        test.check_agent("rewards_buffer", [], key_or_index="env_0")
        test.check_agent("terminals_buffer", [], key_or_index="env_0")
        test.check_agent("last_memory_batch", expected_batch)
        test.check_var("replay-memory/index", 1)  # index has been rolled over (memory capacity is 6)
        test.check_var("replay-memory/size", 6)
        test.check_var("replay-memory/memory/states", np.array([[1.0, 0.0, 0.0, 0.0]] * 4 +
                                                               [[0.0, 0.0, 1.0, 0.0]] +
                                                               [[1.0, 0.0, 0.0, 0.0]]))
        test.check_var("replay-memory/memory/actions", np.array([2, 0, 0, 1, 2, 0]))
        test.check_var("replay-memory/memory/rewards", np.array([-1.0] * 3 + [-3.0, 1.0, -1.0]))
        test.check_var("replay-memory/memory/terminals", np.array([True, True, False, True, True, False]))

        test.check_var("policy/neural-network/hidden/dense/kernel", matrix1_qnet, decimals=4)
        test.check_var("target-policy/neural-network/hidden/dense/kernel", matrix1_target_net)
        test.check_var("policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_qnet, decimals=4)
        test.check_var("target-policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_target_net)

        # 8th step -> Another buffer update check and weights update and sync.
        # action: down (2)
        test.step(1)
        test.check_env("state", 1)
        test.check_agent("states_buffer", [1], key_or_index="env_0")
        test.check_agent("actions_buffer", [2], key_or_index="env_0")
        test.check_agent("rewards_buffer", [-1.0], key_or_index="env_0")
        test.check_agent("terminals_buffer", [False], key_or_index="env_0")
        expected_batch = dict(
            states=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            actions=np.array([0, 1]),
            rewards=np.array([-1.0, -3.0]),
            terminals=np.array([True, True]),
            next_states=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # TODO: <- This is wrong and must be fixed (next-state of first item is from a previous insert and unrelated to first item)
        )
        test.check_agent("last_memory_batch", expected_batch)
        test.check_var("replay-memory/index", 1)
        test.check_var("replay-memory/size", 6)
        test.check_var("replay-memory/memory/states", np.array([[1.0, 0.0, 0.0, 0.0]] * 4 +
                                                               [[0.0, 0.0, 1.0, 0.0]] +
                                                               [[1.0, 0.0, 0.0, 0.0]]))
        test.check_var("replay-memory/memory/actions", np.array([2, 0, 0, 1, 2, 0]))
        test.check_var("replay-memory/memory/rewards", np.array([-1.0, -1.0, -1.0, -3.0, 1.0, -1.0]))
        test.check_var("replay-memory/memory/terminals", np.array([True, True, False, True, True, False]))

        # Assume that the sync happens first (matrices are already the same when updating).
        mat_updated = self._helper_update_matrix(expected_batch, matrix1_qnet, matrix2_qnet, matrix1_qnet,
                                                 matrix2_qnet, agent, loss_func)

        # Now target-net should be again 1 step behind policy-net.
        test.check_var("policy/neural-network/hidden/dense/kernel", mat_updated[0], decimals=2)
        test.check_var("target-policy/neural-network/hidden/dense/kernel", matrix1_qnet, decimals=2)  # again: old matrix
        test.check_var("policy/dueling-action-adapter/action-layer/dense/kernel", mat_updated[1], decimals=2)
        test.check_var("target-policy/dueling-action-adapter/action-layer/dense/kernel", matrix2_qnet, decimals=2)

    def _calculate_action(self, state, matrix1, matrix2):
        s = np.asarray([state])
        s_flat = one_hot(s, depth=4)
        q_values = self._helper_get_q_values(s_flat, matrix1, matrix2)
        # Assume greedy.
        return np.argmax(q_values)

    @staticmethod
    def _helper_get_q_values(input_, matrix1, matrix2):
        """
        Calculates the q-values for a given simple 1-hidden 1-action-layer (both linear w/o biases) setup.

        Args:
            input_ (np.ndarray): The input array (batch x in-nodes).
            matrix1 (np.ndarray): The weights matrix of the hidden layer.
            matrix2 (np.ndarray): The weights matrix of the action-layer.

        Returns:
            np.ndarray: The calculated q-values.
        """
        # Simple NN implementation.
        nn_output = np.matmul(np.matmul(input_, matrix1), matrix2)
        # Simple dueling layer implementation.
        state_values = np.expand_dims(nn_output[:, 0], axis=-1)
        q_values = state_values + nn_output[:, 1:] - np.mean(nn_output[:, 1:], axis=-1, keepdims=True)
        return q_values

    def _helper_update_matrix(self, expected_batch, matrix1_qnet, matrix2_qnet, matrix1_target_net, matrix2_target_net,
                              agent, loss_func):
        # Calculate gradient per weight based on the above batch.
        q_s = self._helper_get_q_values(expected_batch["states"], matrix1_qnet, matrix2_qnet)
        q_sp = self._helper_get_q_values(expected_batch["next_states"], matrix1_qnet, matrix2_qnet)
        qt_sp = self._helper_get_q_values(expected_batch["next_states"], matrix1_target_net, matrix2_target_net)

        # The loss without weight changes.
        loss = np.mean(loss_func._graph_fn_loss_per_item(
            q_s, expected_batch["actions"], expected_batch["rewards"], expected_batch["terminals"], qt_sp, q_sp
        ))

        # Calculate the dLoss/dw for all individual weights (w) and apply [- LR * dLoss/dw] to each weight.
        # Then check again against the actual, now optimized weights.
        mat_updated = list()
        for i, mat in enumerate([matrix1_qnet, matrix2_qnet]):
            mat_updated.append(mat.copy())
            for index in np.ndindex(mat.shape):
                mat_w_plus_d = mat.copy()
                mat_w_plus_d[index] += 0.0001
                if i == 0:
                    q_s_plus_d = self._helper_get_q_values(expected_batch["states"], mat_w_plus_d, matrix2_qnet)
                    q_sp_plus_d = self._helper_get_q_values(expected_batch["next_states"], mat_w_plus_d, matrix2_qnet)
                else:
                    q_s_plus_d = self._helper_get_q_values(expected_batch["states"], matrix1_qnet, mat_w_plus_d)
                    q_sp_plus_d = self._helper_get_q_values(expected_batch["next_states"], matrix1_qnet, mat_w_plus_d)

                loss_w_plus_d = np.mean(loss_func._graph_fn_loss_per_item(
                    q_s_plus_d,
                    expected_batch["actions"], expected_batch["rewards"], expected_batch["terminals"],
                    qt_sp, q_sp_plus_d
                ))
                dl_over_dw = (loss - loss_w_plus_d) / 0.0001

                # Apply the changes to our matrices, then check their actual values.
                mat_updated[i][index] += agent.optimizer.learning_rate * dl_over_dw

        return mat_updated
