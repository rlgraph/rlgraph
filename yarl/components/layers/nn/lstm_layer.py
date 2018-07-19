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

from yarl import get_backend

from yarl.utils.initializer import Initializer
from yarl.components.layers.nn.nn_layer import NNLayer
from yarl.components.layers.nn.activation_functions import get_activation_function

if get_backend() == "tf":
    import tensorflow as tf


class LSTMLayer(NNLayer):
    """
    An LSTM layer processing an initial internal state vector and a batch of sequences to produce
    a final internal state and a batch of output sequences.
    """
    def __init__(
            self, units, use_peepholes=False, cell_clip=None, weights_spec=None, forget_bias=1.0,
            sequence_length=None, parallel_iterations=32, swap_memory=False, time_major=False, **kwargs):
        """
        Args:
            units (int): The number of units in the LSTM cell.
            use_peepholes (bool): True to enable diagonal/peephole connections.
                Default: False.
            cell_clip (Optional[float]): If provided, the cell state is clipped by this value prior to the cell
                output activation. Default: None.
            weights_spec: A specifier for the weight-matrices' initializers.
                If None, use the default initializers.
            forget_bias (float): The forget gate bias to use. Default: 1.0.
            sequence_length (Optional[np.ndarray]): An int vector mapping each batch item to a sequence length
                such that the remaining time slots for each batch item are filled with zeros.
            parallel_iterations (int): The number of iterations to run in parallel.
                Default: 32.
            swap_memory (bool): Transparently swap the tensors produced in forward inference but needed for back
                prop from GPU to CPU. This allows training RNNs which would typically not fit on a single GPU,
                with very minimal (or no) performance penalty.
                Default: False.
            time_major (bool): Whether the time rank is the first rank (vs the batch rank).
                Default: False.
        """
        super(LSTMLayer, self).__init__(scope=kwargs.pop("scope", "lstm-layer"),
                                        activation=kwargs.pop("activation", "tanh"), **kwargs)

        self.units = units
        self.use_peepholes = use_peepholes
        self.cell_clip = cell_clip
        self.weights_spec = weights_spec
        self.weights_init = None
        self.forget_bias = forget_bias

        self.sequence_length = sequence_length
        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory
        self.time_major = time_major

        self.lstm_cell = None

    def create_variables(self, input_spaces, action_space):
        in_space = input_spaces["apply"][0]

        # Create weights.
        weights_shape = (in_space.shape[0] + self.units, self.units)  # [0] b/c Space.shape never includes batch-rank
        self.weights_init = Initializer.from_spec(shape=weights_shape, specification=self.weights_spec)

        # Wrapper for backend.
        if get_backend() == "tf":
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(
                num_units=self.units,
                use_peepholes=self.use_peepholes,
                cell_clip=self.cell_clip,
                initializer=self.weights_init.initializer,
                forget_bias=self.forget_bias,
                activation=get_activation_function(self.activation, *self.activation_params)
            )

            # Now build the layer so that its variables get created.
            self.lstm_cell.build(in_space.get_shape(with_batch_rank=True))
            # Register the generated variables with our registry.
            self.register_variables(*self.lstm_cell.variables)

    def _graph_fn_apply(self, input_, initial_state=None):
        if get_backend() == "tf":
            return tf.nn.dynamic_rnn(
                cell=self.lstm_cell, inputs=input_, sequence_length=self.sequence_length, initial_state=initial_state,
                parallel_iterations=self.parallel_iterations, swap_memory=self.swap_memory, time_major=self.time_major
            )
