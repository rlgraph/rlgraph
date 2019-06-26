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

from rlgraph import get_backend
from rlgraph.components.distributions.normal import Normal
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.utils.decorators import rlgraph_api, graph_fn

# import importlib

#_BACKEND_MOD = importlib.import_module(
#    "rlcore.components.neural_networks."+get_backend()+".variational_auto_encoder"
#)

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class VariationalAutoEncoder(NeuralNetwork):
    def __init__(self, z_units, encoder_network_spec, decoder_network_spec, **kwargs):
        """
        Args:
            z_units (int): Number of units of the latent (z) vectors that the encoder will produce.

            encoder_network_spec (Union[dict,NeuralNetwork]): Specification dict to construct an encoder
                NeuralNetwork object from or a NeuralNetwork Component directly.

            decoder_network_spec (Union[dict,NeuralNetwork]): Specification dict to construct a decoder
                NeuralNetwork object from or a NeuralNetwork Component directly.
        """
        super(VariationalAutoEncoder, self).__init__(scope="variational-auto-encoder", **kwargs)

        self.z_units = z_units

        # Create encoder and decoder networks.
        self.encoder_network = NeuralNetwork.from_spec(encoder_network_spec, scope="encoder-network")
        self.decoder_network = NeuralNetwork.from_spec(decoder_network_spec, scope="decoder-network")

        # Create the two Gaussian layers.
        self.mean_layer = DenseLayer(units=self.z_units, scope="mean-layer")
        self.stddev_layer = DenseLayer(units=self.z_units, scope="stddev-layer")

        # Create the Normal Distribution from which to sample.
        self.normal_distribution = Normal()

        # A concat layer to concat mean and stddev before passing it to the Normal distribution.
        # No longer needed: Pass Tuple (mean + stddev) into API-method instead of concat'd tensor.
        #self.concat_layer = ConcatLayer(axis=-1)

        # Add all sub-Components.
        self.add_components(
            self.encoder_network, self.decoder_network, self.mean_layer, self.stddev_layer,
            self.normal_distribution#, self.concat_layer
        )

    @rlgraph_api
    def call(self, input_):
        """
        Our custom `call` method.
        """
        encoder_out = self.encode(input_)
        decoder_out = self.decode(encoder_out["z_sample"])
        return decoder_out

    @rlgraph_api
    def encode(self, input_):
        # Get the encoder raw output.
        encoder_output = self.encoder_network.call(input_)
        # Push it through our two mean/std layers.
        mean = self.mean_layer.call(encoder_output)
        log_stddev = self.stddev_layer.call(encoder_output)
        stddev = self._graph_fn_exp(log_stddev)
        # Generate a Tuple to be passed into `sample_stochastic` as parameters of a Normal distribution.
        z_sample = self.normal_distribution.sample_stochastic(tuple([mean, stddev]))
        return dict(z_sample=z_sample, mean=mean, stddev=stddev)

    @rlgraph_api
    def decode(self, z_vector):
        return self.decoder_network.call(z_vector)

    @graph_fn
    def _graph_fn_exp(self, input_):
        if get_backend() == "tf":
            return tf.exp(input_)
        elif get_backend() == "pytorch":
            return torch.exp(input_)
