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

import logging

from rlgraph.utils.ops import SingleDataOp
from rlgraph.components import Component


def component_print_out(component, phase=None):
    """
    Prints out an already built Component in the following way (example):

    COMPONENT: dummy-component
    in-sockets:
        'input': FloatBox((2,3))
        'input2': IntBox(5)
    graph_fns:
        'reset':
            'input' + 'input2' -> 'output'
    sub-components:
        'scale':
            'input' -> 'scale/input'
            'scale/output' -> 'output'
    out-sockets:
        'output': FloatBox(shape=(1,))

    Args:
        component (Component): The Component to print out at INFO logging level.
        phase (str): Some build-phase descriptor (e.g. "after assembly", "pre-build", etc.)
    """
    logger = logging.getLogger(__name__)

    txt = "COMPONENT{}: {}\n".format("("+phase+")" if phase is not None else "", component.name or "__core__")

    # Collect data for printout.
    txt += "in-sockets:\n"
    for in_sock in component.input_sockets:
        txt += "\t'{}': {} ({} ops)\n".format(in_sock.name, in_sock.space, len( in_sock.op_records))

    # Check all the component's graph_fns for input-completeness.
    if len(component.graph_fns) > 0:
        txt += "graph_fns:\n"
    for graph_fn in component.graph_fns:
        txt += "\t'{}':\n\t\t".format(graph_fn.name)
        if graph_fn.input_complete is False:
            txt += "(NOT INPUT COMPLETE) "
        for i, in_sock_rec in enumerate(graph_fn.input_sockets.values()):
            txt += "{}'{}'".format(" + " if i > 0 else "", in_sock_rec["socket"].name)
        txt += " -> "
        for i, out_sock in enumerate(graph_fn.output_sockets):
            txt += "{}'{}' ({} ops)".format(" + " if i > 0 else "", out_sock.name, len(out_sock.op_records))
        txt += "\n"

    if len(component.sub_components) > 0:
        txt += "sub-components:\n"

    # Check component's sub-components for input-completeness.
    for sub_component in component.sub_components.values():
        txt += "\t'{}':\n".format(sub_component.name)
        if sub_component.input_complete is False:
            txt += "\t\tNOT INPUT COMPLETE\n"
        txt += "\t\tins:\n"
        for in_sock in sub_component.input_sockets:  # type: Socket
            for in_coming in in_sock.incoming_connections:
                if hasattr(in_coming, "name"):
                    #labels = ""
                    #if isinstance(in_coming, Socket) and in_sock in in_coming.labels:
                    #    labels = " (label="+str(in_coming.labels[in_sock])[1:-1]+")"
                    txt += "\t\t'{}/{}' -> '{}' ({} ops)\n".format(
                        in_coming.component.name, in_coming.name, in_sock.name, len(in_sock.op_records)
                    )
                elif isinstance(in_coming, SingleDataOp):
                    txt += "\t\tconst({}) -> '{}'\n".format(in_coming.constant_value, in_sock.name)
                else:
                    txt += "\t\t'{}'\n".format(in_coming)
        txt += "\t\touts:\n"
        for out_sock in sub_component.output_sockets:
            for out_going in out_sock.outgoing_connections:
                if isinstance(out_going, Socket):
                    txt += "\t\t'{}' -> '{}/{}' ({} ops)\n".format(out_sock.name, out_going.component.name,
                                                          out_going.name, len(out_going.op_records))
                elif hasattr(out_going, "name"):
                    txt += "\t\t'{}' -> '{}/{}'\n".format(out_sock.name, out_going.component.name,
                                                          out_going.name)
                else:
                    txt += "\t\t'{}'\n".format(out_sock)

    txt += "out-sockets:\n"
    for out_sock in component.output_sockets:
        txt += "\t'{}': {} ({} ops)\n".format(out_sock.name, out_sock.space, len(out_sock.op_records))

    logger.info(txt)

