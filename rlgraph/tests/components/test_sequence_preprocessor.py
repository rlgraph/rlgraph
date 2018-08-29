# Copyright 2018 The RLgraph authors, All Rights Reserved.
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

import numpy as np
from six.moves import xrange as range_
import unittest

from rlgraph.components.layers import Sequence
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal


class TestSequencePreprocessor(unittest.TestCase):

    def test_sequence_preprocessor(self):
        space = FloatBox(shape=(1,), add_batch_rank=True)
        sequencer = Sequence(sequence_length=3, add_rank=True)
        test = ComponentTest(component=sequencer, input_spaces=dict(preprocessing_inputs=space))

        vars = sequencer.get_variables("index", "buffer", global_scope=False)
        index, buffer = vars["index"], vars["buffer"]

        for _ in range_(3):
            test.test("reset")
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, -1)
            test.test(("apply", np.array([[0.1]])),
                      expected_outputs=np.array([[[0.1, 0.1, 0.1]]]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 0)
            test.test(("apply", np.array([[0.2]])),
                      expected_outputs=np.array([[[0.1, 0.1, 0.2]]]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 1)
            test.test(("apply", np.array([[0.3]])),
                      expected_outputs=np.array([[[0.1, 0.2, 0.3]]]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 2)
            test.test(("apply", np.array([[0.4]])),
                      expected_outputs=np.array([[[0.2, 0.3, 0.4]]]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 0)
            test.test(("apply", np.array([[0.5]])),
                      expected_outputs=np.array([[[0.3, 0.4, 0.5]]]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 1)

    # TODO: Make it irrelevent whether we test a python or a tf Component (API and handling should be 100% identical)
    def test_python_sequence_preprocessor(self):
        seq_len = 3
        space = FloatBox(shape=(1,), add_batch_rank=True)
        sequencer = Sequence(sequence_length=seq_len, batch_size=4, add_rank=True, backend="python")
        sequencer.create_variables(input_spaces=dict(preprocessing_inputs=space))

        #test = ComponentTest(component=sequencer, input_spaces=dict(apply=space))

        for _ in range_(3):
            sequencer._graph_fn_reset()
            self.assertEqual(sequencer.index, -1)
            input_ = np.asarray([[1.0], [2.0], [3.0], [4.0]])
            out = sequencer._graph_fn_apply(input_)
            self.assertEqual(sequencer.index, 0)
            recursive_assert_almost_equal(
                out, np.asarray([[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0]], [[4.0, 4.0, 4.0]]])
            )
            input_ = np.asarray([[1.1], [2.2], [3.3], [4.4]])
            out = sequencer._graph_fn_apply(input_)
            self.assertEqual(sequencer.index, 1)
            recursive_assert_almost_equal(
                out, np.asarray([[[1.0, 1.0, 1.1]], [[2.0, 2.0, 2.2]], [[3.0, 3.0, 3.3]], [[4.0, 4.0, 4.4]]])
            )
            input_ = np.asarray([[1.11], [2.22], [3.33], [4.44]])
            out = sequencer._graph_fn_apply(input_)
            self.assertEqual(sequencer.index, 2)
            recursive_assert_almost_equal(
                out, np.asarray([[[1.0, 1.1, 1.11]], [[2.0, 2.2, 2.22]], [[3.0, 3.3, 3.33]], [[4.0, 4.4, 4.44]]])
            )
            input_ = np.asarray([[10], [20], [30], [40]])
            out = sequencer._graph_fn_apply(input_)
            self.assertEqual(sequencer.index, 0)
            recursive_assert_almost_equal(
                out, np.asarray([[[1.1, 1.11, 10]], [[2.2, 2.22, 20]], [[3.3, 3.33, 30]], [[4.4, 4.44, 40]]])
            )

    def test_sequence_preprocessor_with_batch(self):
        space = FloatBox(shape=(2,), add_batch_rank=True)
        sequencer = Sequence(sequence_length=2, batch_size=3, add_rank=True)
        test = ComponentTest(component=sequencer, input_spaces=dict(preprocessing_inputs=space))

        vars = sequencer.get_variables("index", "buffer", global_scope=False)
        index, buffer = vars["index"], vars["buffer"]

        for _ in range_(3):
            test.test("reset")
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, -1)

            test.test(("apply", np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
                      expected_outputs=np.array([
                          [[1.0, 1.0], [2.0, 2.0]],
                          [[3.0, 3.0], [4.0, 4.0]],
                          [[5.0, 5.0], [6.0, 6.0]]
                      ]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 0)

            test.test(("apply", np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])),
                      expected_outputs=np.array([
                          [[1.0, 0.1], [2.0, 0.2]],
                          [[3.0, 0.3], [4.0, 0.4]],
                          [[5.0, 0.5], [6.0, 0.6]]
                      ]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 1)

            test.test(("apply", np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])),
                      expected_outputs=np.array([
                          [[0.1, 10.0], [0.2, 20.0]],
                          [[0.3, 30.0], [0.4, 40.0]],
                          [[0.5, 50.0], [0.6, 60.0]]
                      ]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 0)

            test.test(("apply", np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])),
                      expected_outputs=np.array([
                          [[10.0, 100.0], [20.0, 200.0]],
                          [[30.0, 300.0], [40.0, 400.0]],
                          [[50.0, 500.0], [60.0, 600.0]]
                      ]))
            index_value, buffer_value = test.read_variable_values(index, buffer)
            self.assertEqual(index_value, 1)

    def test_sequence_preprocessor_with_container_space(self):
        # Test with no batch rank.
        space = Tuple(
            FloatBox(shape=(1,)),
            FloatBox(shape=(2, 2)),
            add_batch_rank=False
        )

        component_to_test = Sequence(sequence_length=4, add_rank=False)
        test = ComponentTest(component=component_to_test, input_spaces=dict(preprocessing_inputs=space))

        for i in range_(3):
            test.test("reset")

            test.test(("apply", np.array([np.array([0.5]), np.array([[0.6, 0.7], [0.8, 0.9]])])),
                      expected_outputs=(np.array([0.5, 0.5, 0.5, 0.5]), np.array([[0.6, 0.7] * 4,
                                                                                  [0.8, 0.9] * 4])))
            test.test(("apply", np.array([np.array([0.6]), np.array([[1.1, 1.1], [1.1, 1.1]])])),
                      expected_outputs=(np.array([0.5, 0.5, 0.5, 0.6]), np.array([[0.6, 0.7, 0.6, 0.7,
                                                                                   0.6, 0.7, 1.1, 1.1],
                                                                                  [0.8, 0.9, 0.8, 0.9,
                                                                                   0.8, 0.9, 1.1, 1.1]])))
            test.test(("apply", np.array([np.array([0.7]), np.array([[2.0, 2.1], [2.2, 2.3]])])),
                      expected_outputs=(np.array([0.5, 0.5, 0.6, 0.7]), np.array([[0.6, 0.7, 0.6, 0.7,
                                                                                   1.1, 1.1, 2.0, 2.1],
                                                                                  [0.8, 0.9, 0.8, 0.9,
                                                                                   1.1, 1.1, 2.2, 2.3]])))

