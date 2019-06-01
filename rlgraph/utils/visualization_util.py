# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

import os
import re

import requests

from rlgraph import rl_graph_dir
from rlgraph.components.component import Component
from rlgraph.utils.op_records import DataOpRecord, DataOpRecordColumnIntoAPIMethod, DataOpRecordColumnFromAPIMethod, \
    DataOpRecordColumnIntoGraphFn, DataOpRecordColumnFromGraphFn

# Try importing graphviz.
try:
    from graphviz import Graph, Digraph
except ImportError as e:
    print(
        "Error when importing graphviz: Try the following steps to enable automated graphviz-backed debug drawings:\n"
        "1) pip install graphviz\n"
        "2) Install the GraphViz engine on your local machine: https://graphviz.gitlab.io/download/\n"
        "3) Add the GraphViz `bin` directory (created in step 2) when installing GraphViz) to your `PATH` "
        "environment variable\n"
        "4) Reopen your `python` shell and try again.\n\n"
    )
    raise e


def draw_meta_graph(component, *, apis=False, graph_fns=False, render=True, _graphviz_graph=None):
    """
    Creates and draws the GraphViz component graph for the given Component (and all its sub-components).
    - Each Component is a subgraph inside the graph, each one named exactly like the `global_scope` of its Component
      (with root being called `root` if no name was given to it) and the prefix "cluster_" (GraphViz needs this
      to allow styling of the subgraph).
    - A Component's APIs are subgraphs (within the Component's subgraph) and named `api:[global_scope]/[api_name]`.
    - An API's in- and out-columns are nodes and named after their `id` property, which is unique accross
        the entire graph.
    - graph_fns are subgraphs (within the Component's subgraph) and named `graph:[global_scope]/[graph_fn_name]`.
    - A graph_fn's in- and out-columns are nodes and named via their `id` property (see API in/out columns).

    Args:
        component (Component): The Component object to draw the component-graph for.
        apis (Union[bool,str]): Whether to include API methods. If "no-columns", do not draw in/out columns.
        graph_fns (Union[bool,str]): Whether to include GraphFns. If "no-columns", do not draw in/out columns.
        #meta_graph (MetaGraph): The meta-graph object to use. Only necessary if either `apis` is True or
        #    `graph_fns` is True.
        render (bool): Whether to show the resulting graph as pdf in the browser.
        _graphviz_graph (Digraph): Whether this is a recursive (sub-component) call.

    Returns:
        Digraph: The GraphViz Digraph object.
    """
    #assert (apis is False and graph_fns is False) or meta_graph is not False,\
    #    "ERROR: You must provide `meta_graph` if at least one of `apis` or `graph_fns` is True!"

    # _return is True if we are in the first original call (not one of the recursive ones).
    return_ = False
    if _graphviz_graph is None:
        _graphviz_graph = Digraph(name=component.scope if len(component.scope) > 0 else "root")
        _graphviz_graph.attr(label=_graphviz_graph.name)
        _graphviz_graph.attr(color="#999933")
        if apis is not False:
            for api_name in component.api_methods:
                _add_api_to_graph(
                    component, api_name, _graphviz_graph, draw_columns=(apis is True)
                )
        return_ = True

    for sub_component in component.sub_components.values():
        # Skip helper Components.
        if re.match(r'^\.helper-', sub_component.scope):
            continue
        with _graphviz_graph.subgraph(name="cluster_" + sub_component.global_scope) as sg:
            #_graphviz_graph._rlgraph_subgraphs["cluster_" + sub_component.global_scope] = sg
            #sg._rlgraph_subgraphs = {}
            # Set some attributes of the sub-graph for display purposes.
            color = "{},0.5,0.5".format(min(0.2 * sub_component.nesting_level, 1.0))
            sg.attr(color=color)
            sg.attr(style="filled")
            sg.attr(bb="1px")
            sg.attr(label=sub_component.scope)
            # Add all APIs of this sub-Component.
            if apis is not False:
                for api_name in sub_component.api_methods:
                    _add_api_to_graph(sub_component, api_name, sg, draw_columns=(apis is True))
            else:
                # Add a fake node. TODO: remove this necessity.
                sg.node(sub_component.global_scope, label="")
                sg.node_attr.update(shape="point")
            # Call recursively on sub-Components of this sub-Component.
            draw_meta_graph(
                sub_component, apis=apis, graph_fns=graph_fns, render=False, _graphviz_graph=sg
            )

    # We are in the first original call (not one of the recursive ones) -> return GraphViz Digraph here.
    if return_ is True:
        # Render the graph as pdf (in browser).
        if render is True:
            _graphviz_graph.render(os.path.join(rl_graph_dir, "rlgraph_debug_draw.gv"), view=True)
        return _graphviz_graph


def _add_api_to_graph(component, api_name, graphviz_graph, draw_columns=True):
    api_method_rec = component.api_methods[api_name]
    # Draw only when called at least once.
    num_calls = len(api_method_rec.out_op_columns)

    if num_calls > 0:
        # Create API as subgraph of graphviz_graph.
        # Make all connections coming into all in columns and out-columns of this API.
        with graphviz_graph.subgraph(name="cluster_api:" + component.global_scope + "/" + api_name) as sg:
            sg.attr(color="#ff0000", bb="2px", label=api_name)
            # Draw all in/out columns as nodes.
            if draw_columns is True:
                for call_id in reversed(range(len(component.api_methods[api_name].out_op_columns))):
                    out_col = component.api_methods[api_name].out_op_columns[call_id]
                    assert len(out_col.op_records) > 0
                    sg.node(out_col.id, label="out-" + str(call_id))
                    # Make all connections going into this column (all its op-recs).
                    for incoming_op_rec in out_col.op_records:
                        graphviz_graph.edge(incoming_op_rec.previous.column.id, out_col.id)
                # Draw in/out columns as one node (op-recs going in and coming out will be edges).
                for call_id in reversed(range(len(component.api_methods[api_name].in_op_columns))):
                    in_col = component.api_methods[api_name].in_op_columns[call_id]
                    if len(in_col.op_records) > 0:
                        sg.node("api-in:{}/{}/{}".format(component.global_scope, api_name, call_id),
                                label="in-" + str(call_id))
                    # Make all connections going into this column (all its op-recs).
                    for incoming_op_rec in in_col.op_records:
                        graphviz_graph.edge(incoming_op_rec.previous.column.id, in_col.id)
            # TODO: This is a hack to make the subgraph visible (needs at least one node). Creating a fake-node here. Try to solve this more elegantly.
            else:
                sg.node("api:" + component.global_scope + "/" + api_name, label="")
                sg.node_attr.update(shape="point")


def draw_sub_meta_graph_from_op_rec(op_rec, meta_graph):
    """
    Traces back an op-rec through the meta-graph drawing everything on the path til reaching the leftmost placeholders.

    Args:
        op_rec (DataOpRecord): The DataOpRecord object to trace back to the beginning of the graph.
        meta_graph (MetaGraph): The Meta-graph, of which `op_rec` is a part.

    Returns:
        str: Meta-graph markup string.
    """
    api_methods_and_graph_fns, connections = _backtrace_op_rec(op_rec)
    markup = get_component_markup(
        meta_graph.root_component,
        api_methods_and_graph_fns_filter=api_methods_and_graph_fns, connections_filter=connections
    )
    send_graph_markup(markup)


def get_component_markup(
        component, max_nesting_level=None, draw_graph_fns=False, api_methods_and_graph_fns_filter=None,
        connections_filter=None, _recursive=False, _indent=0
):
    """
    Returns graph markup to be used for meta-graph plotting.
    Uses the (mermaid)[https://github.com/knsv/mermaid] markup language.

    Args:
        component (Component): Component to generate meta-graph markup for.
        level (int): Indentation level. If >= 1, return this component as sub-component.
        draw_graph_fns (bool): Include graph fns in plot.

    Returns:
        str: Meta-graph markup string.
    """
    markup = ""
    # Define some styling classes.
    if component.nesting_level == 0:
        markup = "graph LR\n"
    #    markup += "classDef component fill:#33f,stroke:#000,stroke-width:2px;\n"
    #    markup += "classDef api fill:#f9f,stroke:#333,stroke-width:1px;\n"
    #    markup += "classDef graph_fn fill:#ff9,stroke:#333,stroke-width:1px;\n"
    #    markup += "classDef space fill:#9f9,stroke:#333,stroke-width:1px;\n"
    #    markup += "\n"

    #backend = get_backend()

    # The Component itself is a subgraph.
    markup += " " * _indent + "%% Component: {}\n".format(component.global_scope)
    markup += " " * _indent + "subgraph {}\n".format(component.scope)

    # Unique input-args.
    markup += " " * _indent + "%% Inputs\n"
    for input_name in component.api_method_inputs:
        markup += " " * _indent + " input:" + component.global_scope + "/" + input_name + "(" + input_name + ")\n"

    # API-methods as subgraph into the Component.
    markup += " " * _indent + "%% APIs\n"
    for api_method_rec in component.api_methods.values():
        # Check filter.
        name = component.global_scope + "/" + api_method_rec.name
        if api_methods_and_graph_fns_filter is None or name in api_methods_and_graph_fns_filter:
            markup += " " * _indent + " subgraph {}\n".format(name)
            # All input args for this API-method (as either simple arg-pos or kwarg).
            for input_name in api_method_rec.input_names:
                # Ignore args[].
                if re.search(r'\[(\d+)\]$', input_name):
                    continue
                markup += " " * _indent + "  " + component.global_scope + "/" + input_name + "(" + input_name +")\n"
            markup += " " * _indent + " end\n"

    # graph_fns as subgraph into the Component.
    markup += " " * _indent + "%% GraphFns\n"
    for graph_fn_rec in component.graph_fns.values():
        # Check filter.
        graph_fn_name = component.global_scope + "/" + graph_fn_rec.name
        if api_methods_and_graph_fns_filter is None or graph_fn_name in api_methods_and_graph_fns_filter:
            markup += " " * _indent + " subgraph {}\n".format(graph_fn_name)
            # Inputs.
            markup += " " * _indent + "  subgraph inputs\n"
            for input_slot in range(len(graph_fn_rec.in_op_columns[0].op_records)):
                markup += " " * _indent + "   " + graph_fn_name + "/{}(({}))\n".format(input_slot, input_slot)
            markup += " " * _indent + "  end\n"
            # Outputs.
            markup += " " * _indent + " subgraph outputs\n"
            for output_slot in range(len(graph_fn_rec.out_op_columns[0].op_records)):
                markup += " " * _indent + "  " + graph_fn_name + "/{}(({}))\n".format(output_slot, output_slot)
            markup += " " * _indent + "  end\n"
            markup += " " * _indent + " end\n"

    if _recursive is True and max_nesting_level is not None and component.nesting_level > max_nesting_level:
        markup += " " * _indent + "end\n"
        return markup

    # Add sub-components.
    for sub_component in component.sub_components.values():
        markup += get_component_markup(
            sub_component, max_nesting_level=max_nesting_level, draw_graph_fns=draw_graph_fns,
            api_methods_and_graph_fns_filter=api_methods_and_graph_fns_filter, connections_filter=connections_filter,
            _recursive=True, _indent=_indent + 1
        )

    # Connection are inserted after the graph
    #for connection in all_connections:
        #if connection[2]:
        #    # Labeled connection
        #    markup += " " * 4 * _indent + "{}--{}-->{}\n".format(connection[0], connection[2], connection[1])
        # Unlabeled connection
    #    markup += " " * _indent + "{}-->{}\n".format(connection[0], connection[1])

    markup += " " * _indent + "end\n"

    return markup


def send_graph_markup(markup, host="localhost", port=8080, token=None, ssl=False, path="markup",
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
    graph_markup = markup  #get_graph_markup(component, level=0, draw_graph_fns=draw_graph_fns)
    if not graph_markup:
        return False

    target_url = "{protocol}://{host}:{port}/{path}".format(
        protocol="https" if ssl else "http",
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


def _backtrace_op_rec(op_rec, _api_methods_and_graph_fns=None, _connections=None):
    """
    Returns all component/API-method/graph_fn strings in a set that lie on the way back of the given op-rec till
    the beginning of the meta-graph (placeholders).

    Args:
        op_rec (DataOpRecord): The DataOpRecord to backtrace.

        _api_methods_and_graph_fns (Optional[Set[str]]): Set of all APi-methods and graph_fns (plus global_scope of
            their Component) that lie on the backtraced path of the op-rec.

        _connections (Optional[Set[str]]): All connections that lie in the backtraced path.

    Returns:
        tuple:
            - Set[str]: The set of `global_scope/[API-method|graph_fn]-name` that lie on the backtraced path.
            - Set[str]: The set of connections between API-methods/graph_fns backtraced from the given op_rec.
    """
    if _api_methods_and_graph_fns is None:
        _api_methods_and_graph_fns = set()
    if _connections is None:
        _connections = set()

    # If column is None, we have reached the leftmost (placeholder) op-recs.
    while op_rec is not None and op_rec.column is not None:
        col = op_rec.column
        column_type, column_scope = _get_column_type_and_scope(col)
        _api_methods_and_graph_fns.add(column_scope)

        if op_rec.previous is None:
            # Graph_fn.
            if isinstance(op_rec.column, DataOpRecordColumnFromGraphFn):
                # Add all inputs recursively to our sets and stop here.
                for in_op_rec in op_rec.column.in_graph_fn_column.op_records:
                    _api_methods_and_graph_fns, _connections = _backtrace_op_rec(
                        in_op_rec, _api_methods_and_graph_fns, _connections
                    )
            # Leftmost placeholder.
            else:
                pass
        else:
            # Store the connection between op_rec and previous (two strings describing the two connected op-recs).
            prev_col = op_rec.previous.column
            if prev_col is not None:
                prev_col_type, prev_scope = _get_column_type_and_scope(prev_col)
                _connections.add((
                    prev_col_type + ":" + prev_scope + "/" + str(op_rec.previous.kwarg or op_rec.previous.position),
                    column_type + ":" + column_scope + "/" + str(op_rec.kwarg or op_rec.position)
                ))
            else:
                _connections.add((
                    "Placeholder:" + op_rec.previous.placeholder,
                    column_type + ":" + column_scope + "/" + str(op_rec.kwarg or op_rec.position)
                ))

        # Process next op-rec on left side.
        op_rec = op_rec.previous

    return _api_methods_and_graph_fns, _connections


def _get_column_type_and_scope(col):
    if isinstance(col, DataOpRecordColumnIntoAPIMethod):
        return "APIin:", col.component.global_scope + "/" + col.api_method_rec.name
    elif isinstance(col, DataOpRecordColumnFromAPIMethod):
        return "APIout:", col.component.global_scope + "/" + col.api_method_name
    elif isinstance(col, DataOpRecordColumnIntoGraphFn):
        return "GFin:", col.component.global_scope + "/" + col.graph_fn.__name__
    elif isinstance(col, DataOpRecordColumnFromGraphFn):
        return "GFout:", col.component.global_scope + "/" + col.graph_fn_name
    else:
        raise ValueError("Non supported column type: {}!".format(type(col).__name__))


if __name__ == "__main__":
    get_sub_graph_from_op_rec(None)


