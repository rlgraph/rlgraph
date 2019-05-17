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

import unittest

import numpy as np
from rlgraph.components.memories.ring_buffer import RingBuffer
from rlgraph.spaces import Dict, BoolBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import non_terminal_records, terminal_records, recursive_assert_almost_equal
from six.moves import xrange as range_


class TestRingBufferMemory(unittest.TestCase):
    """
    Tests the ring buffer. The ring buffer has very similar tests to
    the replay memory as it supports similar insertion and retrieval semantics,
    but needs additional tests on episode indexing and its latest semantics.
    """

    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        rewards=float,
        terminals=BoolBox(),
        add_batch_rank=True
    )
    # Generic memory variables.
    memory_variables = ["size", "index"]

    # Ring buffer variables
    ring_buffer_variables = ["size", "index", "num-episodes", "episode-indices"]
    capacity = 10

    input_spaces = dict(
        records=record_space,
        num_records=int,
        num_episodes=int
    )
    input_spaces_no_episodes = dict(
        records=record_space,
        num_records=int,
    )

    def test_capacity_with_episodes(self):
        """
        Tests if inserts of non-terminals work.

        Note that this does not test episode semantics itself, which are tested below.
        """
        ring_buffer = RingBuffer(capacity=self.capacity)
        test = ComponentTest(component=ring_buffer, input_spaces=self.input_spaces)
        # Internal memory variables.
        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        size_value = ring_buffer_variables["size"]
        index_value = ring_buffer_variables["index"]
        num_episodes_value = ring_buffer_variables["num-episodes"]
        episode_index_values = ring_buffer_variables["episode-indices"]

        # Assert indices 0 before insert.
        self.assertEqual(size_value, 0)
        self.assertEqual(index_value, 0)
        self.assertEqual(num_episodes_value, 0)
        self.assertEqual(np.sum(episode_index_values), 0)

        # Insert one more element than capacity. Note: this is different than
        # replay test because due to episode semantics, it matters if
        # these are terminal or not. This tests if episode index updating
        # causes problems if none of the inserted elements are terminal.
        observation = non_terminal_records(self.record_space, self.capacity + 1)
        test.test(("insert_records", observation), expected_outputs=None)

        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        size_value = ring_buffer_variables["size"]
        index_value = ring_buffer_variables["index"]
        num_episodes_value = ring_buffer_variables["num-episodes"]
        episode_index_values = ring_buffer_variables["episode-indices"]

        # Size should be equivalent to capacity when full.
        self.assertEqual(size_value, self.capacity)

        # Index should be one over capacity due to modulo.
        self.assertEqual(index_value, 1)
        self.assertEqual(num_episodes_value, 0)
        self.assertEqual(np.sum(episode_index_values), 0)

        # If we fetch n elements, we expect to see exactly the last n.
        for last_n in range(1, 6):
            batch = test.test(("get_records", last_n), expected_outputs=None)
            recursive_assert_almost_equal(batch["actions"]["action1"], observation["actions"]["action1"][-last_n:])
            recursive_assert_almost_equal(batch["states"]["state2"], observation["states"]["state2"][-last_n:])
            recursive_assert_almost_equal(batch["terminals"], observation["terminals"][-last_n:])

    def test_episode_indices_when_inserting(self):
        """
        Tests if episodes indices and counts are set correctly when inserting
        terminals.
        """
        ring_buffer = RingBuffer(capacity=self.capacity)
        test = ComponentTest(component=ring_buffer, input_spaces=self.input_spaces)

        # First, we insert a single terminal record.
        observation = terminal_records(self.record_space, 1)
        test.test(("insert_records", observation), expected_outputs=None)

        # Internal memory variables.
        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        num_episodes_value = ring_buffer_variables["num-episodes"]
        episode_index_values = ring_buffer_variables["episode-indices"]

        # One episode should be present.
        self.assertEqual(num_episodes_value, 1)
        # However, the index of that episode is 0, so we cannot fetch it.
        self.assertEqual(sum(episode_index_values), 0)

        # Next, we insert 1 non-terminal, then 1 terminal element.
        observation = non_terminal_records(self.record_space, 1)
        test.test(("insert_records", observation), expected_outputs=None)
        observation = terminal_records(self.record_space, 1)
        test.test(("insert_records", observation), expected_outputs=None)

        # Now, we expect to have 2 episodes with episode indices at 0 and 2.
        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        num_episodes_value = ring_buffer_variables["num-episodes"]
        episode_index_values = ring_buffer_variables["episode-indices"]

        print('Episode indices after = {}'.format(episode_index_values))
        self.assertEqual(num_episodes_value, 2)
        self.assertEqual(episode_index_values[1], 2)

    def test_only_terminal_with_episodes(self):
        """
        Edge case: What if only terminals are inserted when episode
        semantics are enabled?
        """
        ring_buffer = RingBuffer(capacity=self.capacity)
        test = ComponentTest(component=ring_buffer, input_spaces=self.input_spaces)
        observation = terminal_records(self.record_space, self.capacity)
        test.test(("insert_records", observation), expected_outputs=None)

        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        num_episodes_value = ring_buffer_variables["num-episodes"]
        episode_index_values = ring_buffer_variables["episode-indices"]

        self.assertEqual(num_episodes_value, self.capacity)
        # Every episode index should correspond to its position
        for i in range_(self.capacity):
            self.assertEqual(episode_index_values[i], i)

    def test_episode_fetching(self):
        """
        Test if we can accurately fetch most recent episodes.
        """
        ring_buffer = RingBuffer(capacity=self.capacity)
        test = ComponentTest(component=ring_buffer, input_spaces=self.input_spaces)

        # Insert 2 non-terminals, 1 terminal
        observation = non_terminal_records(self.record_space, 2)
        test.test(("insert_records", observation), expected_outputs=None)
        observation = terminal_records(self.record_space, 1)
        test.test(("insert_records", observation), expected_outputs=None)

        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        num_episodes_value = ring_buffer_variables["num-episodes"]
        episode_index_values = ring_buffer_variables["episode-indices"]

        # One episode.
        self.assertEqual(num_episodes_value, 1)
        expected_indices = [0] * self.capacity
        expected_indices[0] = 2
        recursive_assert_almost_equal(episode_index_values, expected_indices)

        # We should now be able to retrieve one episode of length 3.
        episode = test.test(("get_episodes", 1), expected_outputs=None)
        expected_terminals = [0, 0, 1]
        recursive_assert_almost_equal(episode["terminals"], expected_terminals)

        # We should not be able to retrieve two episodes, and still return just one.
        episode = test.test(("get_episodes", 2), expected_outputs=None)
        expected_terminals = [0, 0, 1]
        recursive_assert_almost_equal(episode["terminals"], expected_terminals)

        # Insert 7 non-terminals.
        observation = non_terminal_records(self.record_space, 7)
        test.test(("insert_records", observation), expected_outputs=None)

        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        index_value = ring_buffer_variables["index"]
        episode_index_values = ring_buffer_variables["episode-indices"]

        # Episode indices should not have changed.
        expected_indices[0] = 2
        recursive_assert_almost_equal(episode_index_values, expected_indices)
        # Inserted 2 non-terminal, 1 terminal, 7 non-terminal at capacity 10 -> should be at 0 again.
        self.assertEqual(index_value, 0)

        # Now inserting one terminal so the terminal buffer has layout [1 0 1 0 0 0 0 0 0 0]
        observation = terminal_records(self.record_space, 1)
        test.test(("insert_records", observation), expected_outputs=None)

        # Episode indices:
        ring_buffer_variables = test.get_variable_values(ring_buffer, self.ring_buffer_variables)
        num_episodes_value = ring_buffer_variables["num-episodes"]
        recursive_assert_almost_equal(num_episodes_value, 2)

        # # Check if we can fetch 2 episodes:
        episodes = test.test(("get_episodes", 2), expected_outputs=None)
        #
        # # We now expect to have retrieved:
        # # - 10 time steps
        # # - 2 terminal values 1
        # # - Terminal values spaced apart 1 index due to the insertion order
        self.assertEqual(len(episodes['terminals']), self.capacity)
        self.assertEqual(episodes['terminals'][0], True)
        self.assertEqual(episodes['terminals'][2], True)

    def test_latest_batch(self):
        """
        Tests if we can fetch latest steps.
        """
        ring_buffer = RingBuffer(capacity=self.capacity)
        test = ComponentTest(component=ring_buffer, input_spaces=self.input_spaces)

        # Insert 5 random elements.
        observation = non_terminal_records(self.record_space, 5)
        test.test(("insert_records", observation), expected_outputs=None)

        # First, test if the basic computation works.
        batch = test.test(("get_records", 5), expected_outputs=None)
        recursive_assert_almost_equal(batch, observation)

        # Next, insert capacity more elements:
        observation = non_terminal_records(self.record_space, self.capacity)
        test.test(("insert_records", observation), expected_outputs=None)

        # If we now fetch capacity elements, we expect to see exactly the last 10.
        batch = test.test(("get_records", self.capacity), expected_outputs=None)
        recursive_assert_almost_equal(batch, observation)

        # If we fetch n elements, we expect to see exactly the last n.
        for last_n in range(1, 6):
            batch = test.test(("get_records", last_n), expected_outputs=None)
            recursive_assert_almost_equal(batch["actions"]["action1"], observation["actions"]["action1"][-last_n:])
            recursive_assert_almost_equal(batch["states"]["state2"], observation["states"]["state2"][-last_n:])
            recursive_assert_almost_equal(batch["terminals"], observation["terminals"][-last_n:])

