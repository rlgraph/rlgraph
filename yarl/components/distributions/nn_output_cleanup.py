# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from math import log
import numpy as np
from six.moves import xrange

from yarl import backend, YARLError, SMALL_NUMBER
from yarl.components import Component
from yarl.components.layers import DenseLayer
from yarl.spaces import Space, IntBox, ContainerSpace

if backend == "tf":
    import tensorflow as tf


class NNOutputCleanup(Component):
    """
    A Component that cleans up neural network output and gets it ready for parameterizing a distribution Component.
    Cleanup includes reshaping (for the desired action space), adding a distribution bias, making sure probs are not
    0.0 or 1.0, etc..

    API:
    ins:
        nn_output (SingleDataOp): The raw neural net output to be cleaned up for further processing in a Distribution.
    outs:
        parameters (SingleDataOp): The cleaned up output (translated into Distribution-readable parameters).
    """
    def __init__(self, target_space, bias=None, scope="nn-output-cleanup", **kwargs):
        """
        Args:
            target_space (Space): The target Space that the NN output tries to reach (by sending out parameters
                for a distribution that matches this target Space).
            bias (any): An optional bias that will be added to the output of the network.
            TODO: For now, only IntBoxes -> Categorical are supported. We'll add support for continuous action spaces later
        """
        super(NNOutputCleanup, self).__init__(scope=scope, flatten_ops=kwargs.pop("flatten_ops", False), **kwargs)

        self.target_space = target_space

        if not isinstance(self.target_space, IntBox):
            raise YARLError("ERROR: `target_space` must be IntBox. Continuous target spaces will be supported later!")

        # Discrete action space. Make sure, all dimensions have the same bounds and the lower bound is 0.
        if self.target_space.global_bounds is False:
            raise YARLError("ERROR: `target_space` must not have individual lower and upper bounds!")
        elif self.target_space.num_categories == 0:
            raise YARLError("ERROR: `target_space` must have a `num_categories` of larger 0!")

        # Define our interface.
        self.define_inputs("nn_output")
        self.define_outputs("parameters")

        # If we have a bias layer, connect it before the actual cleanup.
        if bias is not None:
            bias_layer = DenseLayer(units=self.target_space.num_categories, biases_spec=bias if np.isscalar(bias) else
                                    [log(b) for _ in xrange(self.target_space.flat_dim) for b in bias])
            self.add_component(bias_layer, connections=dict(input="nn_output"))
            # Place our cleanup after the bias layer.
            self.add_graph_fn((bias_layer, "output"), "parameters", self._graph_fn_cleanup)
        # Place our cleanup directly after the nn-output.
        else:
            self.add_graph_fn("nn_output", "parameters", self._graph_fn_cleanup)

    def check_input_spaces(self, input_spaces, action_space):
        in_space = input_spaces["nn_output"]  # type: Space
        # a) Must not be  ContainerSpace (not supported yet for NNLayers, doesn't seem to make sense).
        assert not isinstance(in_space, ContainerSpace), "ERROR: Cannot handle container input Spaces " \
                                                         "in NNOutputCleanup '{}' (atm; may soon do)!".format(self.name)
        # b) All input Spaces need batch ranks (we are passing through NNs after all).
        assert in_space.has_batch_rank, "ERROR: Space in Socket 'input' to NNOutputCleanup '{}' must have a batch " \
                                        "rank (0th position)!".format(self.name)

    def _graph_fn_cleanup(self, nn_outputs_plus_bias):
        """
        Cleans up the output coming from a NN and gets it ready for some Distribution Component (creates distribution
        parameters from the NN-output).

        Args:
            nn_outputs_plus_bias (SingleDataOp): The already biased (optional) and flattened data coming from an NN.

        Returns:
            SingleDataOp: The parameters, ready to be passed to a Distribution object's in-Socket "parameters".
        """
        # Reshape logits to action shape
        shape = self.target_space.get_shape(with_batch_rank=-1, with_category_rank=True)
        if backend == "tf":
            logits = tf.reshape(tensor=nn_outputs_plus_bias, shape=shape)

            # Convert logits into probabilities and clamp them at SMALL_NUMBER.
            return tf.maximum(x=tf.nn.softmax(logits=logits, axis=-1), y=SMALL_NUMBER)

