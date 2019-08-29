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

from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.utils.ops import flatten_op
from rlgraph.utils.decorators import rlgraph_api


class MultiInputStreamNeuralNetwork(NeuralNetwork):
    """
    A NeuralNetwork that takes n separate input-streams and feeds each of them separately through a different NN.
    The final outputs of these NNs are then all concatenated and fed further through an (optional) post-network.
    """
    def __init__(self, input_network_specs, post_network_spec=None, **kwargs):
        """
        Args:
            input_network_specs (Union[Dict[str,dict],Tuple[dict]]): A specification dict or tuple with values being
                the spec dicts for the single streams. The `call` method expects a dict input or a single tuple input
                (not as *args) in its first parameter.

            post_network_spec (Optional[]): The specification dict of the post-concat network or the post-concat
                network object itself.
        """
        super(MultiInputStreamNeuralNetwork, self).__init__(scope="multi-input-stream-nn", **kwargs)

        # Create all streams' networks.
        if isinstance(input_network_specs, dict):
            self.input_stream_nns = {}
            for i, (flat_key, nn_spec) in enumerate(flatten_op(input_network_specs).items()):
                self.input_stream_nns[flat_key] = NeuralNetwork.from_spec(nn_spec, scope="input-stream-nn-{}".format(i))
            # Create the concat layer to merge all streams.
            self.concat_layer = ConcatLayer(dict_keys=list(self.input_stream_nns.keys()), axis=-1)
        else:
            assert isinstance(input_network_specs, (list, tuple)),\
                "ERROR: `input_network_specs` must be dict or tuple/list!"
            self.input_stream_nns = []
            for i, nn_spec in enumerate(input_network_specs):
                self.input_stream_nns.append(NeuralNetwork.from_spec(nn_spec, scope="input-stream-nn-{}".format(i)))
            # Create the concat layer to merge all streams.
            self.concat_layer = ConcatLayer(axis=-1)

        # Create the post-network (after the concat).
        self.post_nn = NeuralNetwork.from_spec(post_network_spec, scope="post-concat-nn")  # type: NeuralNetwork

        # Add all sub-Components.
        self.add_components(
            self.post_nn, self.concat_layer,
            *list(self.input_stream_nns.values() if isinstance(input_network_specs, dict) else self.input_stream_nns)
        )

    @rlgraph_api
    def call(self, inputs):
        """
        Feeds all inputs through the sub networks' apply methods and concats their outputs and sends that
        concat'd output through the post-network.
        """
        # Feed all inputs through their respective NNs.
        if isinstance(self.input_stream_nns, dict):
            outputs = {}
            # TODO: Support last-timestep returning LSTMs in input-stream-networks.
            for input_stream_flat_key, input_stream_nn in self.input_stream_nns.items():
                outputs[input_stream_flat_key] = input_stream_nn.call(inputs[input_stream_flat_key])
            # Concat everything.
            concat_output = self.concat_layer.call(outputs)
        else:
            outputs = []
            # TODO: Support last-timestep returning LSTMs in input-stream-networks.
            for i, input_stream_nn in enumerate(self.input_stream_nns):
                outputs.append(input_stream_nn.call(inputs[i]))
            # Concat everything.
            concat_output = self.concat_layer.call(*outputs)

        # Send everything through post-network.
        post_nn_out = self.post_nn.call(concat_output)

        return post_nn_out

    def add_layer(self, layer_component):
        """
        Overwrite this by adding any new layer to the post-network (most obvious behavior).
        """
        return self.post_nn.add_layer(layer_component)
