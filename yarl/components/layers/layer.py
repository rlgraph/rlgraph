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

from six.moves import xrange

from yarl.components.layers.stack import Stack


class Layer(Stack):
    """
    A Stack that has its own graph_fn logic (e.g. a NN layer) but - just like a Stack - can
    also be constructed via nested sub-components that are first automatically connected to each other
    (in the sequence they are given in this c'tor) and then connected to this component's graph_fn unit.
    The final interface will hence consist of the first sub-component's input(s)- and this layer's graph_fn's
    output-socket(s).
    """
    def __init__(self, *sub_components, **kwargs):
        """
        Args:
            sub_components (List[Component]): The sub-components to immediately place into this one and
                connect to each other. The last sub-component is connected to this one's graph_fn unit,
                which then provides the output Sockets of this component.

        Keyword Args:
            num_graph_fn_inputs (int): The number of parameters that our graph_fn takes.
            num_graph_fn_outputs (int): The number of output values that our graph_fn returns (as tuple)
                given that the number of input parameters is num_graph_inputs.
        """
        self.num_graph_fn_inputs = kwargs.pop("num_graph_fn_inputs", 1)
        self.num_graph_fn_outputs = kwargs.pop("num_graph_fn_outputs", 1)
        # By default, switch on splitting for all Layers.
        super(Layer, self).__init__(*sub_components, expose_outs=False,
                                    split_ops=kwargs.pop("split_ops", True), **kwargs)

        # No sub-components, just create empty in-Sockets.
        if len(sub_components) == 0:
            if self.num_graph_fn_inputs > 1:
                for in_ in xrange(self.num_graph_fn_inputs):
                    self.add_sockets("input{}".format(in_))
            else:
                self.add_sockets("input")

        # Create our output Sockets.
        if self.num_graph_fn_outputs > 1:
            for out_ in xrange(self.num_graph_fn_outputs):
                self.add_sockets("output{}".format(out_))
        else:
            self.add_sockets("output")

        # Connect our graph_fn from our input socket(s) (or last sub-component's output(s)) to our output
        # socket(s).
        # NOTE: Layers always split graph_fns on complex input Spaces.
        inputs = sub_components[-1].output_sockets if len(sub_components) > 0 else self.input_sockets
        inputs = [in_ for in_ in inputs if in_.name != "_variables"]
        outputs = [out_ for out_ in self.output_sockets if out_.name != "_variables"]
        self.add_graph_fn(inputs, outputs, "apply")

    def check_input_spaces(self, input_spaces, action_space):
        # Make sure the number of items in the connected input_space matches what we said about our num_graph_fn_inputs.
        assert len(input_spaces) == self.num_graph_fn_inputs, \
            "ERROR: `num_graph_fn_inputs` ({}) does not match the " \
            "number of items in `input_spaces` ({})!".format(self.num_graph_fn_inputs, len(input_spaces))

    def _graph_fn_apply(self, *inputs):
        """
        This is where the graph-logic of this layer goes.

        Args:
            *inputs (any): The input(s) to this layer. The number of inputs must match self.num_graph_fn_inputs.

        Returns:
            The output(s) of this layer. The number of elements in the returned tuple must match self.num_graph_fn_outputs.
        """
        return inputs  # optional

