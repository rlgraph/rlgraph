# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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

import unittest

import numpy as np
from rlgraph.components.neural_networks.variational_auto_encoder import VariationalAutoEncoder
from rlgraph.spaces import FloatBox
from rlgraph.tests.component_test import ComponentTest
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
from rlgraph.utils.numpy import dense_layer


class TestVariationalAutoEncoders(unittest.TestCase):
    """
    Tests for the VariationalAutoEncoder class.
    """
    def test_simple_variational_auto_encoder(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        input_spaces = dict(
            input_=FloatBox(shape=(3,), add_batch_rank=True), z_vector=FloatBox(shape=(1,), add_batch_rank=True)
        )

        variational_auto_encoder = VariationalAutoEncoder(
            z_units=1,
            encoder_network_spec=config_from_path("configs/test_vae_encoder_network.json"),
            decoder_network_spec=config_from_path("configs/test_vae_decoder_network.json")
        )

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=variational_auto_encoder, input_spaces=input_spaces)

        # Batch of size=3.
        input_ = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        global_scope = "variational-auto-encoder/"
        # Calculate output manually.
        var_dict = test.read_variable_values(variational_auto_encoder.variable_registry)

        encoder_network_out = dense_layer(
            input_, var_dict[global_scope+"encoder-network/encoder-layer/dense/kernel"],
            var_dict[global_scope+"encoder-network/encoder-layer/dense/bias"]
        )
        expected_mean = dense_layer(
            encoder_network_out, var_dict[global_scope+"mean-layer/dense/kernel"],
            var_dict[global_scope+"mean-layer/dense/bias"]
        )
        expected_stddev = dense_layer(
            encoder_network_out, var_dict[global_scope + "stddev-layer/dense/kernel"],
            var_dict[global_scope + "stddev-layer/dense/bias"]
        )
        out = test.test(("encode", input_), expected_outputs=None)
        recursive_assert_almost_equal(out["mean"], expected_mean, decimals=5)
        recursive_assert_almost_equal(out["stddev"], np.exp(expected_stddev), decimals=5)
        self.assertTrue(out["z_sample"].shape == (3, 1))

        test.terminate()

