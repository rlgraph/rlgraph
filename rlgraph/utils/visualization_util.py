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

import requests

from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.spaces import Space


def get_graph_markup(component, level=0, draw_graph_fns=False):
    """
    Returns graph markup to be used for RLGraph metagraph plotting.

    Uses the (mermaid)[https://github.com/knsv/mermaid] markup language.

    Args:
        component (Component): Component to generate meta-graph markup for.
        level (int): Indentation level. If >= 1, return this component as sub-component.
        draw_graph_fns (bool): Include graph fns in plot.

    Returns:
        str: Meta-graph markup string.
    """

    # Print (sub)graph declaration
    if level >= 1:
        markup = " " * 4 * level + "subgraph {}\n".format(component.name)
    elif level == 0:
        markup = "graph TD\n"
        markup += "classDef input_socket fill:#9ff,stroke:#333,stroke-width:2px;\n"
        markup += "classDef output_socket fill:#f9f,stroke:#333,stroke-width:2px;\n"
        markup += "classDef space fill:#999,stroke:#333,stroke-width:2px;\n"
        markup += "classDef graph_fn fill:#ff9,stroke:#333,stroke-width:2px;\n"
        markup += "\n"
    else:
        raise RLGraphError("Invalid component indentation level {}".format(level))

    all_sockets = list()
    all_graph_fns = list()

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

    # Add graph function nodes with the following markup: graphfn_HASH(GRAPH FN NAME)
    markup_graph_fns = list()
    for graph_fn in component.graph_fns:
        markup += " " * 4 * (level + 1) + "graphfn_{hash}(\"{name}\")\n".format(hash=hash(graph_fn),
                                                                                name=graph_fn.name)
        markup_graph_fns.append("graphfn_{hash}".format(hash=hash(graph_fn)))
        all_graph_fns.append(graph_fn)

    # Collect connections by looping through all incoming connections.
    # All outgoing connections should be incoming connections of another socket, so we don't need to loop through them.
    connections = list()
    markup_spaces = list()
    for socket in all_sockets:
        for incoming_connection in socket.incoming_connections:
            if isinstance(incoming_connection, Socket):
                connections.append((
                    "socket_{}".format(hash(incoming_connection)),
                    "socket_{}".format(hash(socket)),
                    None
                ))
            elif isinstance(incoming_connection, Space):
                # Add spaces to markup (we only know about them because of their connections).
                markup += " " * 4 * (level + 1) + "space_{hash}(\"{name}\")\n".format(hash=hash(incoming_connection),
                                                                                      name=str(incoming_connection))
                markup_spaces.append("space_{hash}".format(hash=hash(incoming_connection)))
                connections.append((
                    "space_{}".format(hash(incoming_connection)),
                    "socket_{}".format(hash(socket)),
                    None
                ))
            elif isinstance(incoming_connection, GraphFunction):
                connections.append((
                    "graphfn_{}".format(hash(incoming_connection)),
                    "socket_{}".format(hash(socket)),
                    None
                ))

    # Collect graph fn connections by looping through all input sockets of the graph fns.
    # All output sockets should have been covered by the above collection of incoming connections to the sockets.
    for graph_fn in all_graph_fns:
        for input_socket_name, input_socket_dict in graph_fn.input_sockets.items():
            input_socket = input_socket_dict['socket']
            if isinstance(input_socket, Socket):
                connections.append((
                    "socket_{}".format(hash(input_socket)),
                    "graphfn_{}".format(hash(graph_fn)),
                    None
                ))
            else:
                raise ValueError("Not a valid input socket: {} ({})".format(input_socket, type(input_socket)))

    # Add style class `input_socket` to the input sockets
    if markup_input_sockets:
        markup += " " * 4 * (level + 1) + "class {} input_socket;\n".format(','.join(markup_input_sockets))

    # Add style class `output_socket` to the output sockets
    if markup_output_sockets:
        markup += " " * 4 * (level + 1) + "class {} output_socket;\n".format(','.join(markup_output_sockets))

    # Add style class `space` to the spaces
    if markup_spaces:
        markup += " " * 4 * (level + 1) + "class {} space;\n".format(','.join(markup_spaces))

    # Add style class `graph_fn` to the graph fns
    if markup_graph_fns:
        markup += " " * 4 * (level + 1) + "class {} graph_fn;\n".format(','.join(markup_graph_fns))

    markup += "\n"

    # Add sub-components.
    for sub_component_name, sub_component in component.sub_components.items():
        markup += get_graph_markup(sub_component, level=level + 1, draw_graph_fns=draw_graph_fns)

    # Subgraphs (level >= 1) require an end statement.
    if level >= 1:
        markup += " " * 4 * level + "end\n"

    markup += "\n"

    # Connection are inserted after the graph
    for connection in connections:
        if connection[2]:
            # Labeled connection
            markup += " " * 4 * level + "{}--{}-->{}\n".format(connection[0], connection[2], connection[1])
        else:
            # Unlabeled connection
            markup += " " * 4 * level + "{}-->{}\n".format(connection[0], connection[1])

    markup += "\n"

    return markup


def send_graph_markup(component, host='localhost', port=8080, token=None, ssl=False, path='markup',
                      level=0, draw_graph_fns=True, **kwargs):
    """
    Send graph markup to HTTP(s) host for pseudo-interactive plotting.

    Args:
        component (Component): component to plot.
        host (str): HTTP host to send markup to.
        port (int): Port on host to connect to.
        token (str): Optional token to identify at host.
        ssl (bool): Use HTTPS or not.
        path (str): Path to post data to (e.g. 'markup' for `host:port/markup`)
        level (int): level argument for `get_graph_markup()`.
        draw_graph_fns (bool): create markup for graph fns, argument for `get_graph_markup()`.
        **kwargs: Keyword arguments will be passed to the `requests.post()` function.

    Returns:
        bool: True if markup generation succeeded and server accepted the request, False otherwise

    """
    graph_markup = get_graph_markup(component, level=0, draw_graph_fns=draw_graph_fns)
    if not graph_markup:
        return False

    target_url = "{protocol}://{host}:{port}/{path}".format(
        protocol='https' if ssl else 'http',
        host=host,
        port=port,
        path=path
    )
    result = requests.post(target_url, data=dict(graph_markup=graph_markup, token=token), **kwargs)

    if result.status_code == 200:
        # Only return true if everything went well.
        return True

    # No, we do not follow redirects. Anything other than 200 leads to an error.
    return False
