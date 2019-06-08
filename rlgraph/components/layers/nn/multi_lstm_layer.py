# Copyright 2018/2019 ducandu GmbH, All Rights Reserved.
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
from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.spaces import Tuple
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import DataOpTuple


class MultiLSTMLayer(NNLayer):
    """
    A multi-LSTM layer processing an initial internal state vector and a batch of sequences to produce
    a final internal state and a batch of output sequences.
    """
    def __init__(
            self, num_lstms, units, use_peepholes=False, cell_clip=None, static_loop=False,
            forget_bias=1.0, parallel_iterations=32, return_sequences=True,
            swap_memory=False, skip_connections=None, **kwargs):
        """
        Args:
            num_lstms (int): The number of LSTMs to stack deep.
            units (Union[List[int],int]): The number of units in the different LSTMLayers' cells.
            use_peepholes (Union[List[bool],bool]): True to enable diagonal/peephole connections from the c-state into
                each of the layers. Default: False.
            cell_clip (Optional[Union[List[float],float]]): If provided, the cell state is clipped by this value prior
                to the cell output activation. Default: None.
            static_loop (Union[bool,int]): If an int, will perform a static RNN loop (with fixed sequence lengths
                of size `static_loop`) instead of a dynamic one (where the lengths for each input can be different).
                In this case, time_major must be set to True (as transposing for this case has not been automated yet).
                Default: False.
            forget_bias (float): The forget gate bias to use. Default: 1.0.
            parallel_iterations (int): The number of iterations to run in parallel.
                Default: 32.
            return_sequences (bool): Whether to return one output for each input or only the last output.
                Default: True.
            swap_memory (bool): Transparently swap the tensors produced in forward inference but needed for back
                prop from GPU to CPU. This allows training RNNs which would typically not fit on a single GPU,
                with very minimal (or no) performance penalty.
                Default: False.
            skip_connections (Optional[List[List[bool]]]): An optional list of lists (2D) of bools indicating the skip
                connections for the input as well as outputs of each layer and whether these should be concatenated
                with the "regular" input for each layer. "Regular" here means the output from the previous layer.
                Example:
                A 4-layer LSTM:
                skip_connections=[
                    #   x    out0   out1   out2   out3    <- outputs (or x)
                                                          # layer 0 (never specified, only takes x as input)
                    [ True,  True, False, False, False],  # layer 1
                    True (for all outputs)                # layer 2
                    [ False, False, False, True, False],  # layer 3
                    ...
                ]
                0) Layer0 does not need to be specified (only takes x, obviously).
                1) Layer1 takes x concatenated with the output of layer0.
                2) Layer2 takes x and both out0 and out1, all concatenated.
                3) Layer3 takes only out2 as input.
                4) A missing sub-list in the main `skip_connections` list means that this layer only takes the previous
                    layer's output (no further skip connections for that layer).
        """
        super(MultiLSTMLayer, self).__init__(
            graph_fn_num_outputs=dict(_graph_fn_apply=2),  # LSTMs: unrolled output, final c_state, final h_state
            scope=kwargs.pop("scope", "multi-lstm-layer"), activation=kwargs.pop("activation", "tanh"), **kwargs
        )

        self.num_lstms = num_lstms
        assert self.num_lstms > 1, "ERROR: Must have more than 1 LSTM layer for MultiLSTMLayer Component!"
        self.units = units
        self.use_peepholes = use_peepholes
        self.cell_clip = cell_clip
        self.static_loop = static_loop
        assert self.static_loop is False or (self.static_loop > 0 and self.static_loop is not True), \
            "ERROR: `static_loop` in LSTMLayer must either be False or an int value (is {})!".format(self.static_loop)
        self.forget_bias = forget_bias

        self.parallel_iterations = parallel_iterations
        self.return_sequences = return_sequences
        self.swap_memory = swap_memory
        self.skip_connections = skip_connections or [[] for _ in range(num_lstms + 1)]

        self.in_space = None

        # tf RNNCell
        # torch lstm and hidden state placeholder
        self.lstms = []
        # The concat layers to concat together the different skip_connection outputs.
        self.concat_layers = []
        self.hidden_states = None

        for i in range(self.num_lstms):
            # Per layer or global settings?
            units = self.units[i] if isinstance(self.units, (list, tuple)) else self.units
            use_peepholes = self.use_peepholes[i] if isinstance(self.use_peepholes, (list, tuple)) else \
                self.use_peepholes
            cell_clip = self.cell_clip[i] if isinstance(self.cell_clip, (list, tuple)) else self.cell_clip
            forget_bias = self.forget_bias[i] if isinstance(self.forget_bias, (list, tuple)) else self.forget_bias
            activation = self.activation[i] if isinstance(self.activation, (list, tuple)) else self.activation

            # Generate the single layers.
            self.lstms.append(LSTMLayer(
                units=units,
                use_peepholes=use_peepholes,
                cell_clip=cell_clip,
                static_loop=self.static_loop,
                parallel_iterations=self.parallel_iterations,
                forget_bias=forget_bias,
                # Always return sequences except for last layer (there, return whatever the user wants).
                return_sequences=True if i < self.num_lstms - 1 else self.return_sequences,
                scope="lstm-layer-{}".format(i),
                swap_memory=self.swap_memory,
                activation=activation
            ))
            self.concat_layers.append(ConcatLayer(scope="concat-layer-{}".format(i)))

        self.add_components(*self.lstms)
        self.add_components(*self.concat_layers)

    def check_input_spaces(self, input_spaces, action_space=None):
        super(MultiLSTMLayer, self).check_input_spaces(input_spaces, action_space)

        # Check correct tuple-internal-states format (if not None, in which case we assume all 0.0s).
        if "internal_states" in input_spaces:
            # Check that main space is a Tuple (one item for each layer).
            sanity_check_space(
                input_spaces["internal_states"], allowed_types=[Tuple]
            )
            # Check that each layer gets a tuple of 2 values: c- and h-states.
            for i in range(self.num_lstms):
                sanity_check_space(
                    input_spaces["internal_states"][i], allowed_types=[Tuple], must_have_batch_rank=True,
                    must_have_time_rank=False
                )
                assert len(input_spaces["internal_states"][i]) == 2,\
                    "ERROR: If internal_states are provided (which is the case), an LSTMLayer requires the len of " \
                    "this Tuple to be 2 (c- and h-states). Your Space is '{}'.".\
                    format(input_spaces["internal_states"][i])

        # Check for batch AND time-rank.
        self.in_space = input_spaces["inputs"]
        sanity_check_space(self.in_space, must_have_batch_rank=True, must_have_time_rank=True)

    @rlgraph_api
    def apply(self, inputs, initial_c_and_h_states=None, sequence_length=None):
        output, last_internal_states = self._graph_fn_apply(
            inputs, initial_c_and_h_states=initial_c_and_h_states, sequence_length=sequence_length
        )
        return dict(output=output, last_internal_states=last_internal_states)

    @graph_fn
    def _graph_fn_apply(self, inputs, initial_c_and_h_states=None, sequence_length=None):
        """
        Args:
            inputs (SingleDataOp): The data to pass through the layer (batch of n items, m timesteps).
                Position of batch- and time-ranks in the input depend on `self.time_major` setting.

            initial_c_and_h_states (DataOpTuple): The initial cell- and hidden-states to use.
                None for the default behavior (all zeros).
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
            # Pass through all layers and concat with respective skip-connections each time.
            last_internal_states = []
            inputs_ = inputs
            outputs = [inputs]
            for i in range(self.num_lstms):
                output = self.lstms[i].call(
                    inputs_,
                    initial_c_and_h_states=initial_c_and_h_states[i] if initial_c_and_h_states is not None else None,
                    sequence_length=sequence_length
                )
                # Store all outputs for possible future skip_connections.
                outputs.append(output["output"])
                # Store current internal states for each layer.
                last_internal_states.append(output["last_internal_states"])

                # Concat with previous (skip-connection) outputs?
                skip_connections = self.skip_connections[i + 1] if len(self.skip_connections) > i + 1 else [True if j == i + 1 else False for j in range(self.num_lstms + 1)]
                if isinstance(skip_connections, bool):
                    skip_connections = [skip_connections for _ in range(self.num_lstms + 1)]
                concat_inputs = [outputs[j] for j, sc in enumerate(skip_connections) if sc is True]

                if len(concat_inputs) > 1:
                    next_input = self.concat_layers[i].call(*concat_inputs)
                else:
                    next_input = concat_inputs[0]
                # Set input for next layer.
                inputs_ = next_input

            return inputs_, DataOpTuple(last_internal_states)
