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

import time
import unittest

import numpy as np

from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.spaces import FloatBox
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal
from rlgraph.tests.test_util import config_from_path


class TestTimeRankFoldingPerformance(unittest.TestCase):
    """
    Tests whether folding (and unfolding) of a time rank is better for certain NNs.
    """
    def test_time_rank_folding_for_large_dense_nn(self):
        vector_dim = 256
        input_space = FloatBox(shape=(vector_dim,), add_batch_rank=True, add_time_rank=True)
        base_config = config_from_path("configs/test_large_dense_nn.json")
        neural_net_wo_folding = NeuralNetwork.from_spec(base_config)

        test = ComponentTest(component=neural_net_wo_folding, input_spaces=dict(nn_input=input_space))

        # Pull a large batch+time ranked sample.
        sample_shape = (256, 200)
        inputs = input_space.sample(sample_shape)

        start = time.monotonic()
        runs = 10
        for _ in range(runs):
            print(".", flush=True, end="")
            test.test(("call", inputs), expected_outputs=None)
        runtime_wo_folding = time.monotonic() - start

        print("\nTesting large dense NN w/o time-rank folding: {}x pass through with {}-data took "
              "{}s".format(runs, sample_shape, runtime_wo_folding))

        neural_net_w_folding = NeuralNetwork.from_spec(base_config)

        # Folded space.
        input_space_folded = FloatBox(shape=(vector_dim,), add_batch_rank=True)
        inputs = input_space.sample(sample_shape[0] * sample_shape[1])

        test = ComponentTest(component=neural_net_w_folding, input_spaces=dict(nn_input=input_space_folded))

        start = time.monotonic()
        for _ in range(runs):
            print(".", flush=True, end="")
            test.test(("call", inputs), expected_outputs=None)
        runtime_w_folding = time.monotonic() - start

        print("\nTesting large dense NN w/ time-rank folding: {}x pass through with {}-data took "
              "{}s".format(runs, sample_shape, runtime_w_folding))

        recursive_assert_almost_equal(runtime_w_folding, runtime_wo_folding, decimals=0)

    def test_time_rank_folding_for_large_cnn_nn(self):
        width = 86
        height = 86
        time_rank = 20
        input_space = FloatBox(shape=(width, height, 3), add_batch_rank=True, add_time_rank=True, time_major=True)
        base_config = config_from_path("configs/test_3x_cnn_nn.json")
        base_config.insert(0, {"type": "reshape", "fold_time_rank": True})
        base_config.append({"type": "reshape", "unfold_time_rank": time_rank, "time_major": True})
        neural_net = NeuralNetwork.from_spec(base_config)

        test = ComponentTest(component=neural_net, input_spaces=dict(nn_input=input_space))

        # Pull a large batch+time ranked sample.
        sample_shape = (time_rank, 256)
        inputs = input_space.sample(sample_shape)

        out = test.test(("call", inputs), expected_outputs=None)["output"]

        self.assertTrue(out.shape == (time_rank, 256, 7 * 7 * 64))
        self.assertTrue(out.dtype == np.float32)
