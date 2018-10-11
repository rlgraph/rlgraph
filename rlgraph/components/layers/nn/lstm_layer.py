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

from rlgraph import get_backend

from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.spaces import Tuple
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils import PyTorchVariable
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import DataOpTuple

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch
    import torch.nn as nn


class LSTMLayer(NNLayer):
    """
    An LSTM layer processing an initial internal state vector and a batch of sequences to produce
    a final internal state and a batch of output sequences.
    """
    def __init__(
            self, units, use_peepholes=False, cell_clip=None, static_loop=False,
            forget_bias=1.0, parallel_iterations=32,
            swap_memory=False, time_major=False, **kwargs):  # weights_spec=None, dtype="float"
        """
        Args:
            units (int): The number of units in the LSTM cell.
            use_peepholes (bool): True to enable diagonal/peephole connections from the c-state into each of
                the layers. Default: False.
            cell_clip (Optional[float]): If provided, the cell state is clipped by this value prior to the cell
                output activation. Default: None.
            static_loop (Union[bool,int]): If an int, will perform a static RNN loop (with fixed sequence lengths
                of size `static_loop`) instead of a dynamic one (where the lengths for each input can be different).
                In this case, time_major must be set to True (as transposing for this case has not been automated yet).
                Default: False.
            #weights_spec: A specifier for the weight-matrices' initializers.
            #If None, use the default initializers.
            forget_bias (float): The forget gate bias to use. Default: 1.0.
            parallel_iterations (int): The number of iterations to run in parallel.
                Default: 32.
            swap_memory (bool): Transparently swap the tensors produced in forward inference but needed for back
                prop from GPU to CPU. This allows training RNNs which would typically not fit on a single GPU,
                with very minimal (or no) performance penalty.
                Default: False.
            #time_major (bool): Whether the time rank is the first rank (vs the batch rank).
            #    Default: False.
            #dtype (str): The dtype of this LSTM. Default: "float".
        """
        super(LSTMLayer, self).__init__(
            graph_fn_num_outputs=dict(_graph_fn_apply=2),  # LSTMs: unrolled output, final c_state, final h_state
            scope=kwargs.pop("scope", "lstm-layer"), activation=kwargs.pop("activation", "tanh"), **kwargs
        )

        self.units = units
        self.use_peepholes = use_peepholes
        self.cell_clip = cell_clip
        self.static_loop = static_loop
        assert self.static_loop is False or (self.static_loop > 0 and self.static_loop is not True), \
            "ERROR: `static_loop` in LSTMLayer must either be False or an int value (is {})!".format(self.static_loop)
        # self.weights_spec = weights_spec
        # self.weights_init = None
        self.forget_bias = forget_bias

        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory
        self.in_space = None

        # tf RNNCell
        # torch lstm and hidden state placeholder
        self.lstm = None
        self.hidden_state = None

    def check_input_spaces(self, input_spaces, action_space=None):
        super(LSTMLayer, self).check_input_spaces(input_spaces, action_space)

        # Check correct tuple-internal-states format (if not None, in which case we assume all 0.0s).
        if "internal_states" in input_spaces:
            sanity_check_space(input_spaces["internal_states"], allowed_types=[Tuple])
            assert len(input_spaces["internal_states"]) == 2,\
                "ERROR: If internal_states are provided (which is the case), an LSTMLayer requires the len of " \
                "this Tuple to be 2 (c- and h-states). Your Space is '{}'.".format(input_spaces["internal_states"])

        # Check for batch AND time-rank.
        self.in_space = input_spaces["inputs"]
        sanity_check_space(self.in_space, must_have_batch_rank=True, must_have_time_rank=True)

    def create_variables(self, input_spaces, action_space=None):
        self.in_space = input_spaces["inputs"]

        # Create one weight matrix: [input nodes + internal state nodes, 4 (4 internal layers) * internal state nodes]
        # weights_shape = (in_space.shape[0] + self.units, 4 * self.units)  # [0]=one past batch rank
        # self.weights_init = Initializer.from_spec(shape=weights_shape, specification=self.weights_spec)

        # Wrapper for backend.
        if get_backend() == "tf":
            self.lstm = tf.contrib.rnn.LSTMBlockCell(  #tf.nn.rnn_cell.LSTMCell(
                num_units=self.units,
                use_peephole=self.use_peepholes,
                cell_clip=self.cell_clip,
                forget_bias=self.forget_bias,
                name="lstm-cell"
                # TODO: self.trainable needs to be recognized somewhere here.

                # These are all not supported yet for LSTMBlockCell (only for the slower LSTMCell)
                # initializer=self.weights_init.initializer,
                # activation=get_activation_function(self.activation, *self.activation_params),
                # dtype=self.dtype,
            )

            # Now build the layer so that its variables get created.
            in_space_without_time_rank = list(self.in_space.get_shape(with_batch_rank=True))
            self.lstm.build(tf.TensorShape(in_space_without_time_rank))
            # Register the generated variables with our registry.
            self.register_variables(*self.lstm.variables)

        elif get_backend() == "pytorch":
            self.lstm = nn.LSTM(self.in_space, self.units)
            self.hidden_state = (torch.zeros(1, 1, self.units), torch.zeros(1, 1, self.units))
            self.register_variables(PyTorchVariable(name=self.global_scope, ref=self.lstm))

    @rlgraph_api
    def apply(self, inputs, initial_c_and_h_states=None, sequence_length=None):
        output, last_internal_states = self._graph_fn_apply(inputs, initial_c_and_h_states, sequence_length)
        return dict(output=output, last_internal_states=last_internal_states)

    @graph_fn
    def _graph_fn_apply(self, inputs, initial_c_and_h_states=None, sequence_length=None):
        """
        Args:
            inputs (SingleDataOp): The data to pass through the layer (batch of n items, m timesteps).
                Position of batch- and time-ranks in the input depend on `self.time_major` setting.
            initial_c_and_h_states (DataOpTuple): The initial cell- and hidden-states to use.
                None for the default behavior (TODO: describe here what default means: zero?)
                The cell-state in an LSTM is passed between cells from step to step and only affected by element-wise
                operations. The hidden state is identical to the output of the LSTM on the previous time step.
            sequence_length (Optional[SingleDataOp]): An int tensor mapping each batch item to a sequence length
                such that the remaining time slots for each batch item are filled with zeros.

        Returns:
            tuple:
                - The outputs over all timesteps of the LSTM.
                - DataOpTuple: The final cell- and hidden-states.
        """
        if get_backend() == "tf":
            # Convert to tf's LSTMStateTuple from DataOpTuple.
            if initial_c_and_h_states is not None:
                initial_c_and_h_states = tf.nn.rnn_cell.LSTMStateTuple(
                    initial_c_and_h_states[0], initial_c_and_h_states[1]
                )

            # We are running the LSTM as a dynamic while-loop.
            if self.static_loop is False:
                lstm_out, lstm_state_tuple = tf.nn.dynamic_rnn(
                    cell=self.lstm,
                    inputs=inputs,
                    sequence_length=sequence_length,
                    initial_state=initial_c_and_h_states,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=self.swap_memory,
                    time_major=self.in_space.time_major,
                    dtype="float"
                )
            # We are running with a fixed number of time steps (static unroll).
            else:
                output_list = list()
                lstm_state_tuple = initial_c_and_h_states
                # TODO: Add option to reset the internal state in the middle of this loop iff some reset signal
                # TODO: (e.g. terminal) is True during the loop.
                inputs.set_shape([self.static_loop] + inputs.shape.as_list()[1:])
                #for input_, terminal in zip(tf.unstack(inputs), tf.unstack(terminals)):
                for input_ in tf.unstack(inputs):
                    #input_ = inputs[i]
                    # If the episode ended, the core state should be reset before the next.
                    #core_state = nest.map_structure(functools.partial(tf.where, d),
                    #                                initial_core_state, core_state)
                    output, lstm_state_tuple = self.lstm(input_, lstm_state_tuple)
                    output_list.append(output)
                lstm_out = tf.stack(output_list)

            # Returns: Unrolled-outputs (time series of all encountered h-states), final c- and h-states.
            lstm_out._batch_rank = 0 if self.in_space.time_major is False else 1
            lstm_out._time_rank = 0 if self.in_space.time_major is True else 1
            return lstm_out, DataOpTuple(lstm_state_tuple)

        elif get_backend() == "pytorch":
            # TODO init hidden state has to be available at create variable time to use.
            inputs = torch.cat(inputs).view(len(inputs), 1, -1)
            out, self.hidden_state = self.lstm(inputs, self.hidden_state)
            return out, DataOpTuple(self.hidden_state)
