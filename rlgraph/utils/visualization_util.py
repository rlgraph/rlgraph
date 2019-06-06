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

from __future__ import absolute_import, division, print_function

import os
import re

import graphviz

from rlgraph import rlgraph_dir
from rlgraph.components.component import Component
from rlgraph.spaces import IntBox, BoolBox, FloatBox, TextBox
from rlgraph.utils.op_records import DataOpRecord, DataOpRecordColumnIntoAPIMethod, DataOpRecordColumnFromAPIMethod, \
    DataOpRecordColumnIntoGraphFn, DataOpRecordColumnFromGraphFn

# Define some colors for the debug graphs.
_API_COLOR = "#f4c542"
_GRAPH_FN_COLOR = "#a06f35"
_COMPONENT_SUBGRAPH_FACTORS_RGB = [0.0, 0.666, 1.0]
_INTBOX_COLOR = "#325e3c"
_FLOATBOX_COLOR = "#1ff450"
_BOOLBOX_COLOR = "#930230"
_TEXTBOX_COLOR = "#3b6ddb"
_OTHER_TYPE_COLOR = "#777777"


def draw_meta_graph(component, *, apis=True, graph_fns=False, render=True,
                    component_filter=None, connection_filter=None,
                    _graphviz_graph=None, _max_nesting_level=None, _all_connections=None, highlighted_connections=None):
    """
    Creates and draws the GraphViz component graph for the given Component (and all its sub-components).
    - An (outermost) placeholder is represented by a node named `Placeholder_[arg-name]`.
    - Each Component is a subgraph inside the graph, each one named exactly like the `global_scope` of its Component
      (with root being called `root` if no name was given to it) and the prefix "cluster_" (GraphViz needs this
      to allow styling of the subgraph).
    - A Component's APIs are subgraphs (within the Component's subgraph) and named `api:[global_scope]/[api_name]`.
    - An API's in- and out-columns are nodes and named after their str(`id`) property, which is unique across
        the entire graph.
    - graph_fns are subgraphs (within the Component's subgraph) and named `graph:[global_scope]/[graph_fn_name]`.
    - A graph_fn's in- and out-columns are nodes and named via their `id` property (same global counter as
        API's in/out columns).

    Args:
        component (Component): The Component object to draw the component-graph for.
        apis (Union[bool,Set[str]]): Whether to include API methods. If Set of str: Only include APIs whose names are in
            the given set.
        graph_fns (Union[bool,str]): Whether to include GraphFns. If Set of str: Only include GraphFns whose names are
            in the given set.
        component_filter (Optional[Set[Component]]): If provided, only Components in the filter set will be rendered.
        connection_filter (Optional[Set[str]]): If provided, only connections in the filter set will be rendered.
        render (bool): Whether to show the resulting graph as pdf in the browser.
        _graphviz_graph (Digraph): Whether this is a recursive (sub-component) call.
        _all_connections (Optional[Set[Tuple[str]]]): An optional set of connection-defining tuples, only applied on
            the global level.

    Returns:
        Digraph: The GraphViz Digraph object.
    """
    # _return is True if we are in the first original call (not one of the recursive ones).
    return_ = False
    if _graphviz_graph is None:
        assert _all_connections is None and _max_nesting_level is None  # only for recursive calls
        _graphviz_graph, _all_connections, _max_nesting_level = \
            _init_meta_graph(component, apis, graph_fns, connection_filter, highlighted_connections)
        return_ = True

    for sub_component in component.sub_components.values():
        # Skip filtered components and all helper Components.
        if (component_filter is not None and sub_component not in component_filter) or \
                re.match(r'^\.helper-', sub_component.scope):
            continue
        with _graphviz_graph.subgraph(name="cluster_" + sub_component.global_scope) as sg:
            # Set some attributes of the sub-graph for display purposes.
            sg.attr(color=_get_api_subgraph_color(sub_component.nesting_level, _max_nesting_level))
            #sg.attr(style="")
            sg.attr(bb="3px")
            sg.attr(label=sub_component.scope)
            # Add all APIs of this sub-Component.
            if apis is not False and sub_component.graph_builder is not None:
                for api_name in sub_component.api_methods:
                    if apis is not False and \
                            (apis is True or "api:" + sub_component.global_scope + "/" + api_name in apis):
                        _add_api_or_graph_fn_to_graph(
                            sub_component, sg, api_name=api_name, apis=apis, graph_fns=graph_fns,
                            connection_filter=connection_filter, _connections=_all_connections,
                            highlighted_connections=highlighted_connections
                        )
            else:
                # Add a fake node. TODO: remove this necessity.
                sg.node(sub_component.global_scope, label="")
                sg.node_attr.update(shape="point")

            # Add all GraphFns of this sub-Component.
            if graph_fns is not False and sub_component.graph_builder is not None:
                for graph_fn_name in sub_component.graph_fns:
                    if graph_fns is not False and \
                            (graph_fns is True or "graph:" + sub_component.global_scope + "/" + graph_fn_name in graph_fns):
                        _add_api_or_graph_fn_to_graph(
                            sub_component, sg, graph_fn_name=graph_fn_name, apis=apis, graph_fns=graph_fns,
                            connection_filter=connection_filter, _connections=_all_connections,
                            highlighted_connections=highlighted_connections
                        )

            # Call recursively on sub-Components of this sub-Component.
            draw_meta_graph(
                sub_component, apis=apis, graph_fns=graph_fns, connection_filter=connection_filter,
                render=False, _graphviz_graph=sg, _all_connections=_all_connections,
                _max_nesting_level=_max_nesting_level
            )

    # We are in the first original call (not one of the recursive ones).
    if return_ is True:
        # Do all connections between nodes on the top-level graph.
        # Connections are encoded as tuples: (from, to, label, color)
        for connection in _all_connections:
            _graphviz_graph.edge(
                connection[0], connection[1], label=connection[2], _attributes=dict(color=connection[3])
            )

        # Render the graph as pdf (in browser).
        if render is True:
            _render(_graphviz_graph, os.path.join(rlgraph_dir, "rlgraph_debug_draw.gv"), view=True)


def draw_sub_meta_graph_from_op_rec(op_rec, meta_graph):
    """
    Traces back an op-rec through the meta-graph drawing everything on the path til reaching the leftmost placeholders.

    Args:
        op_rec (DataOpRecord): The DataOpRecord object to trace back to the beginning of the graph.
        meta_graph (MetaGraph): The Meta-graph, of which `op_rec` is a part.

    Returns:
        str: Meta-graph markup string.
    """
    components, api_methods, graph_fns, connections, highlighted_connection = _backtrace_op_rec(op_rec)
    draw_meta_graph(
        meta_graph.root_component,
        apis=api_methods, graph_fns=graph_fns,
        component_filter=components,
        connection_filter=connections, highlighted_connections={highlighted_connection}
    )


def _backtrace_op_rec(op_rec, _components=None, _api_methods=None, _graph_fns=None, _connections=None):
    """
    Returns all component/API-method/graph_fn strings in a set that lie on the way back of the given op-rec till
    the beginning of the meta-graph (placeholders).

    Args:
        op_rec (DataOpRecord): The DataOpRecord to backtrace.

        _api_methods (Optional[Set[str]]): Set of all API-methods (plus global_scope of their Component)
            that lie on the backtraced path of the op-rec.

        _graph_fns (Optional[Set[str]]): Set of all GraphFns (plus global_scope of their Component)
            that lie on the backtraced path of the op-rec.

        _connections (Optional[Set[str]]): All connections that lie in the backtraced path.

    Returns:
        tuple:
            - Set[str]: The set of `global_scope/[API-method|graph_fn]-name` that lie on the backtraced path.
            - Set[str]: The set of connections between API-methods/graph_fns backtraced from the given op_rec.
    """
    highlighted_connection = None

    if _components is None:
        _components = set()
        highlighted_connection = (op_rec.previous.column.id, op_rec.column.id)
    if _api_methods is None:
        _api_methods = set()
    if _graph_fns is None:
        _graph_fns = set()
    if _connections is None:
        _connections = set()

    # If column is None, we have reached the leftmost (placeholder) op-recs.
    while op_rec is not None and op_rec.column is not None:
        col = op_rec.column
        if col is not None:
            _components.add(col.component)
        column_type, column_scope = _get_column_type_and_scope(col)
        if column_type == "API":
            _api_methods.add(column_scope)
        else:
            assert column_type == "GF"
            _graph_fns.add(column_scope)

        if op_rec.previous is None:
            # Graph_fn.
            if isinstance(op_rec.column, DataOpRecordColumnFromGraphFn):
                # Add all inputs recursively to our sets and stop here.
                for in_op_rec in op_rec.column.in_graph_fn_column.op_records:
                    _components, _api_methods, _graph_fns, _connections, _ = _backtrace_op_rec(
                        in_op_rec, _components, _api_methods, _graph_fns, _connections
                    )
            # Leftmost placeholder.
            else:
                pass
        else:
            # Store the connection between op_rec and previous (two strings describing the two connected op-recs).
            prev_col = op_rec.previous.column
            if prev_col is not None:
                _connections.add((prev_col.id, col.id))
            else:
                _connections.add(("Placeholder_" + op_rec.previous.placeholder, col.id))

        # Process next op-rec on left side.
        op_rec = op_rec.previous

    return _components, _api_methods, _graph_fns, _connections, highlighted_connection


def _add_api_or_graph_fn_to_graph(
        component, graphviz_graph, *, api_name=None, graph_fn_name=None, draw_columns=True, apis=True, graph_fns=False,
        connection_filter=None, _connections=None, highlighted_connections=None
):
    """
    Args:
        component (Component):  The Component whose API method `api_name` needs to be added to the graph.
        api_name (str): The name of the API method to add to the graph.
        graphviz_graph (Digraph): The GraphViz Digraph object to add the API-method's nodes and edges to.
        draw_columns (bool): Whether to draw in/out-columns as nodes or not draw them at all.
        connection_filter (Optional[Set[str]]): If provided, only draw connections that are in this set.
        _connections (Set[Tuple[str]]): Set of connection tuples to add to. Connections will only be done on the
            top-level.
    """
    # One must be provided.
    assert api_name is not None or graph_fn_name is not None
    name = (api_name or graph_fn_name)
    type_ = "api" if api_name else "graph"

    record = component.api_methods[api_name] if api_name else component.graph_fns[graph_fn_name]
    # Draw only when called at least once.
    num_calls = len(record.out_op_columns)

    if num_calls > 0:
        # Create API as subgraph of graphviz_graph.
        # Make all connections coming into all in columns and out-columns of this API.
        with graphviz_graph.subgraph(name="cluster_{}:{}/{}".format(type_, component.global_scope, name)) as sg:
            sg.attr(color=_API_COLOR if api_name else _GRAPH_FN_COLOR, bb="2px", label=name)
            sg.attr(style="filled")
            # Draw all in/out columns as nodes.
            if draw_columns is True:
                # Draw in/out columns as one node (op-recs going in and coming out will be edges).
                # Draw Placeholders as nodes (named: "Placeholder:[arg-name]").
                for call_id in reversed(range(len(record.out_op_columns))):
                    out_col = record.out_op_columns[call_id]
                    assert len(out_col.op_records) > 0
                    sg.node(str(out_col.id), label="out-" + str(call_id))
                    # Make all connections going into this column (all its op-recs).
                    for op_rec in out_col.op_records:
                        # Column is None: graph-in to graph-out -> Use simple non-labeled, non-colored connector.
                        if op_rec.previous is None:
                            _connections.add((str(op_rec.column.in_graph_fn_column.id), str(out_col.id), "", None))
                            continue

                        # Possible filtering by API/graph-fn.
                        type_, scope = _get_column_type_and_scope(op_rec.previous.column)
                        if (type_ == "GF" and (graph_fns is True or (graph_fns is not False and scope in graph_fns))) \
                                or (type_ == "API" and (apis is True or (apis is not False and scope in apis))) \
                                or type_ == "":
                            # Check connection-filter as well.
                            from_to = (op_rec.previous.column.id, out_col.id)
                            if connection_filter is None or from_to in connection_filter:
                                color, shape = _get_color_and_shape_from_space(op_rec.space)
                                # Highlight this connection?
                                if highlighted_connections is not None and from_to in highlighted_connections:
                                    color = "#ff0000"
                                _connections.add((str(from_to[0]), str(from_to[1])) + (shape, color))

                # In columns.
                for call_id in reversed(range(len(record.in_op_columns))):
                    in_col = record.in_op_columns[call_id]
                    if len(in_col.op_records) > 0:
                        sg.node(str(in_col.id), label="in-" + str(call_id))
                    # Make all connections going into this column (all its op-recs).
                    for op_rec in in_col.op_records:
                        # Possible filtering by graph-fn.
                        prev_type, prev_scope = _get_column_type_and_scope(op_rec.previous.column)
                        if (prev_type == "GF" and (graph_fns is True or (graph_fns is not False and prev_scope in graph_fns))) \
                                or (prev_type == "API" and (apis is True or (apis is not False and prev_scope in apis))) \
                                or prev_type == "":
                            color, shape = _get_color_and_shape_from_space(op_rec.space)

                            # GraphFn -> Use input arg position as label.
                            if type_ == "graph":
                                label = str(op_rec.previous.position)
                            # API no kwarg -> Use input arg's name.
                            elif op_rec.kwarg is None:
                                label = op_rec.column.api_method_rec.input_names[op_rec.position]
                            # API kwarg -> Use kwarg.
                            else:
                                label = op_rec.kwarg

                            if op_rec.previous.placeholder is None:
                                from_ = op_rec.previous.column.id
                                label += " (" + shape + ")"
                            else:
                                from_ = "Placeholder_{}".format(op_rec.previous.placeholder)
                                label = ""  # Set label to None (placeholder already describes itself)

                            # Check connection-filter as well.
                            from_to = (from_, in_col.id)
                            if connection_filter is None or from_to in connection_filter:
                                # Highlight this connection?
                                if highlighted_connections is not None and from_to in highlighted_connections:
                                    color = "#ff0000"
                                _connections.add((str(from_to[0]), str(from_to[1])) + (label, color))

            # TODO: This is a hack to make the subgraph visible (needs at least one node). Creating a fake-node here. Try to solve this more elegantly.
            else:
                sg.node(type_ + ":" + component.global_scope + "/" + api_name, label="")
                sg.node_attr.update(shape="point")


def _init_meta_graph(component, apis, graph_fns, connection_filter, highlighted_connections=None):
    _all_connections = set()
    graphviz_graph = graphviz.Digraph(name=component.scope if len(component.scope) > 0 else "root")
    graphviz_graph.attr(label=graphviz_graph.name)
    graphviz_graph.attr(labelloc="t")
    graphviz_graph.attr(rankdir="LR")

    # If we need to draw all APIs (and are already in build-phase).
    if apis is not False and component.graph_builder is not None:
        placeholders = set()  # Already generated placeholders (only one per unique input-arg name).
        for api_name in component.api_methods:
            # If this is the root, add placeholder nodes to graphviz graph.
            if component.nesting_level == 0:
                for call_id in reversed(range(len(component.api_methods[api_name].in_op_columns))):
                    in_col = component.api_methods[api_name].in_op_columns[call_id]
                    # Make all connections going into this column (all its op-recs).
                    for incoming_op_rec in in_col.op_records:
                        if incoming_op_rec.previous.placeholder is not None:
                            placeholder = incoming_op_rec.previous.placeholder
                            if placeholder not in placeholders:
                                color, shape = _get_color_and_shape_from_space(incoming_op_rec.previous.space)
                                graphviz_graph.node(
                                    "Placeholder_" + placeholder, label=placeholder + "({})".format(shape),
                                    _attributes=dict(color=color, style="filled")
                                )
                                placeholders.add(incoming_op_rec.previous.placeholder)

            if apis is not False and (apis is True or "api:" + component.global_scope + "/" + api_name in apis):
                _add_api_or_graph_fn_to_graph(
                    component, graphviz_graph,  # draw_columns=(apis is True),
                    api_name=api_name, apis=apis, graph_fns=graph_fns, connection_filter=connection_filter,
                    _connections=_all_connections, highlighted_connections=highlighted_connections
                )
    # Calculate the max. nesting level of the Component and all its sub-components and subtract
    # by the Component's nesting_level.
    _max_nesting_level = 0
    for sub_component in component.get_all_sub_components(exclude_self=False):
        if (sub_component.nesting_level or 0) > _max_nesting_level:
            _max_nesting_level = sub_component.nesting_level
    _max_nesting_level -= (component.nesting_level or 0)

    # Now that we know the max nesting level, set this Component's subgraph color.
    graphviz_graph.attr(color=_get_api_subgraph_color((component.nesting_level or 0), _max_nesting_level))
    #graphviz_graph.attr(style="filled")

    return graphviz_graph, _all_connections, _max_nesting_level


def _get_column_type_and_scope(col):
    if isinstance(col, DataOpRecordColumnIntoAPIMethod):
        return "API", "api:" + col.component.global_scope + "/" + col.api_method_rec.name
    elif isinstance(col, DataOpRecordColumnFromAPIMethod):
        return "API", "api:" + col.component.global_scope + "/" + col.api_method_name
    elif isinstance(col, DataOpRecordColumnIntoGraphFn):
        return "GF", "graph:" + col.component.global_scope + "/" + col.graph_fn.__name__
    elif isinstance(col, DataOpRecordColumnFromGraphFn):
        return "GF", "graph:" + col.component.global_scope + "/" + col.graph_fn_name
    else:
        return "", ""


def _get_color_and_shape_from_space(space):
    """
    Returns a color code and shape description given some Space.

    Args:
        space (Space): The Space to derive the information from.

    Returns:
        tuple:
            - str: Color code.
            - str: Shape descriptor.
    """
    if space is None:
        return None, ""

    shape_str = "{}".format(list(space.shape)) + (" +B" if space.has_batch_rank else "") + \
        (" +T" if space.has_time_rank else "")

    # Int -> dark green.
    if isinstance(space, IntBox):
        return _INTBOX_COLOR, shape_str
    # Float -> light green
    elif isinstance(space, FloatBox):
        return _FLOATBOX_COLOR, shape_str
    # Bool -> red.
    elif isinstance(space, BoolBox):
        return _BOOLBOX_COLOR, shape_str
    # Text -> blue.
    elif isinstance(space, TextBox):
        return _TEXTBOX_COLOR, shape_str
    # Else -> gray.
    else:
        return _OTHER_TYPE_COLOR, shape_str


def _get_api_subgraph_color(nesting_level, max_nesting_level):
    # Get the ratio of nesting_level with respect to max_nesting_level and make lighter (towards 255)
    # the higher this ratio.
    # -> Higher level components are then darker than lower level ones.
    r = int(255 - 64 * min(nesting_level / max_nesting_level, 1.0))
    color = "#{:02X}{:02X}{:02X}".format(
        int(r * _COMPONENT_SUBGRAPH_FACTORS_RGB[0]),
        int(r * _COMPONENT_SUBGRAPH_FACTORS_RGB[1]),
        int(r * _COMPONENT_SUBGRAPH_FACTORS_RGB[2])
    )
    return color


def _render(graphviz_graph, file, view=False):
    # Try rendering with the installed GraphViz engine.
    try:
        graphviz_graph.render(file, view=view)
    except graphviz.backend.ExecutableNotFound as e:
        print(
            "Error: GraphViz (backend engine) executable missing: Try the following steps to enable automated "
            "graphviz-backed debug drawings:\n"
            "1) Install the GraphViz engine on your local machine: https://graphviz.gitlab.io/download/\n"
            "2) Add the GraphViz `bin` directory (created in step 2) when installing GraphViz) to your `PATH` "
            "environment variable\n"
            "3) Reopen your `python` shell and try again.\n\n"
        )
        print(graphviz_graph.source)  # debug printout
