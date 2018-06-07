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

from yarl import YARLError
from yarl.components import Socket
from yarl.spaces import Space


def get_graph_markup(component, level=0):
    """
    Returns graph markup to be used for YARL metagraph plotting.

    Uses the (mermaid)[https://github.com/knsv/mermaid] markup language.

    Args:
        component (Component): Component to generate markup for.
        level (int): Indentation level. If >= 1, return this component as sub-component.

    Returns:
        str: graph markup string.
    """

    # Print (sub)graph declaration
    if level >= 1:
        markup = " " * 4 * level + "subgraph {}\n".format(component.name)
    elif level == 0:
        markup = "graph TD\n"
        markup += "classDef input_socket fill:#9ff,stroke:#333,stroke-width:2px;\n"
        markup += "classDef output_socket fill:#f9f,stroke:#333,stroke-width:2px;\n"
        markup += "classDef space fill:#999,stroke:#333,stroke-width:2px;\n"
        markup += "\n"
    else:
        raise YARLError("Invalid component indentation level {}".format(level))

    all_sockets = list()

    # Add input socket nodes with the following markup: in_socket_HASH(INPUT SOCKET NAME)
    markup_input_sockets = list()
    for input_socket in component.input_sockets:
        markup += " " * 4 * (level + 1) + "socket_{hash}(\"{name}\")\n".format(hash=hash(input_socket),
                                                                     name=input_socket.name)
        markup_input_sockets.append("socket_{hash}".format(hash=hash(input_socket)))
        all_sockets.append(input_socket)

    # Add output socket nodes with the following markup: out_socket_HASH(OUTPUT SOCKET NAME)
    markup_output_sockets = list()
    for output_socket in component.output_sockets:
        markup += " " * 4 * (level + 1) + "socket_{hash}(\"{name}\")\n".format(hash=hash(output_socket),
                                                                     name=output_socket.name)
        markup_output_sockets.append("socket_{hash}".format(hash=hash(output_socket)))
        all_sockets.append(output_socket)

    markup += "\n"

    # Collect connections by looping through all incoming connections.
    # All outgoing connections should be incoming connections of another socket, so we don't need to loop through them.
    connections = list()
    markup_spaces = list()
    for socket in all_sockets:
        for incoming_connection in socket.incoming_connections:
            if isinstance(incoming_connection, Socket):
                connections.append((
                    "socket_{}".format(hash(incoming_connection)),
                    "socket_{}".format(hash(socket))
                ))
            elif isinstance(incoming_connection, Space):
                # Add spaces to markup (we only know about them because of their connections).
                markup += " " * 4 * (level + 1) + "space_{hash}(\"{name}\")\n".format(hash=hash(incoming_connection),
                                                                                      name=str(incoming_connection))
                markup_spaces.append("space_{hash}".format(hash=hash(incoming_connection)))
                connections.append((
                    "space_{}".format(hash(incoming_connection)),
                    "socket_{}".format(hash(socket))
                ))

    # Add style class `input_socket` to the input sockets
    if markup_input_sockets:
        markup += " " * 4 * (level + 1) + "class {} input_socket;\n".format(','.join(markup_input_sockets))

    # Add style class `output_socket` to the output sockets
    if markup_output_sockets:
        markup += " " * 4 * (level + 1) + "class {} output_socket;\n".format(','.join(markup_output_sockets))

    # Add style class `space` to the spaces
    if markup_spaces:
        markup += " " * 4 * (level + 1) + "class {} space;\n".format(','.join(markup_spaces))

    markup += "\n"

    # Add subcomponents.
    for sub_component_name, sub_component in component.sub_components.items():
        markup += get_graph_markup(sub_component, level=level + 1)

    # Subgraphs (level >= 1) require an end statement.
    if level >= 1:
        markup += " " * 4 * level + "end\n"

    markup += "\n"

    # Connection are inserted after the graph
    for connection in connections:
        markup += " " * 4 * level + "{}-->{}\n".format(connection[0], connection[1])

    markup += "\n"

    return markup
