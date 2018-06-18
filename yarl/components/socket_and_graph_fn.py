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

import logging
from collections import OrderedDict
import re

from yarl import YARLError
from yarl.spaces.space_utils import flatten_op, unflatten_op, get_space_from_op
from yarl.utils.ops import SingleDataOp, DataOpRecord, FlattenedDataOp

_logger = logging.getLogger(__name__)


class Socket(object):
    """
    A Socket object describes a connection to other Sockets, GraphFunctions, or Spaces inside and between ModelComponents.
    One Socket either carries:

    - a single op (e.g. some tensor)
    - a tuple of ops (nesting also supported)
    - a dict of ops (nesting also supported)

    Also, each of the above possibilities can have many parallel solutions. These splits happen e.g. if two Sockets
    connect to the same target Socket. In this case, the target Socket's inputs are treated as possible alternatives
    and the Socket then implicitly produces two outputs that it further passes on to the next Sockets/GraphFunctions.

    When connected to a GraphFunction object, a Socket always represents one of the input parameters to the graph_fn
    method. Also, each returned value of a graph_fn method corresponds to one Socket.
    """
    def __init__(self, name, component, type_="in"):
        """
        Args:
            name (str): The name of this Socket (as it will show in the final call interface).
            component (Component): The Component object that this Socket belongs to.
            type_ (str): The Socket type: "in" or "out".
        """
        # The name of this Socket.
        self.name = name
        # "in" or "out"
        self.type = type_
        # The Component that this Socket belongs to.
        self.component = component

        # Which Socket(s), Space(s), GraphFunction(s) are we connected to on the incoming and outgoing sides?
        # - Records in these lists have a parallel relationship (they are all alternatives to each other).
        self.incoming_connections = list()
        self.outgoing_connections = list()

        # The inferred Space coming into this Socket.
        self.space = None

        # A Socket ([from-sock]) takes a label for a specific outgoing connection ([to-sock]) via a call to:
        # `Component.connect([from-sock], [to-sock], label="lab1")`.
        # The following rules apply now:
        # - [from-sock] will take all incoming ops (unless the incoming sock has its own filtering/labelling going on).
        # - [from-sock] will only pass to [to-sock] (AND label with "lab1") those ops that either have no label yet or
        #   that already carry the label "lab1". All other ops will not be passed to [to-sock].
        # Also, when sent through a graph_fn, the resulting ops carry a union of all input ops' labels.
        # key=[to-sock]'s component/name string; values=set of labels (str).
        self.labels = dict()

        # The set of (alternative) DataOpRecords (op plus optional label(s)).
        self.op_records = set()

    def connect_to(self, to_):
        """
        Adds an outgoing connection to this Socket, either to a Space, a GraphFunction or to another Socket.
        This means that this Socket will pass its DataOps over to `to_` during build time.

        Args:
            to_ (Union[Socket,GraphFunction]): The Socket/GraphFunction that we are connecting to.
        """
        if to_ not in self.outgoing_connections:
            self.outgoing_connections.append(to_)

    def disconnect_to(self, to_):
        """
        Equivalent to `self.connect_to`.
        """
        if to_ in self.outgoing_connections:
            self.outgoing_connections.remove(to_)

    def connect_from(self, from_, label=None):
        """
        Adds an incoming connection to this Socket, either from a Space, a GraphFunction or from another Socket.
        This means that this Socket will receive `from_`'s ops during build time.
        
        Args:
            from_ (Union[Socket,Space,GraphFunction]): The Socket/Space/GraphFunction that we are connected from.
            label (Optional[str]): A possible label to give to `from_`. This label will be passed along with
                `from_`'s ops during build time.
        """
        if from_ not in self.incoming_connections:
            # Constant value SingleDataOp -> Push op_record right away.
            if isinstance(from_, SingleDataOp):
                self.space = get_space_from_op(from_)
                self.op_records.add(DataOpRecord(from_))
            # Socket: Add the label for ops passed to this Socket.
            elif label is not None:
                assert isinstance(from_, Socket), "ERROR: No `label` ({}) allowed if `from_` ({}) is not a Socket " \
                                                  "object!".format(label, str(from_))
                if self not in from_.labels:
                    from_.labels[self] = set()
                from_.labels[self].add(label)
            self.incoming_connections.append(from_)

    def disconnect_from(self, from_):
        """
        Equivalent to `self.connect_from`.
        """
        if from_ in self.incoming_connections:
            # TODO: what if from_ has a label to this socket? Got to remove that as well? Maybe not important.

            # Constant value SingleDataOp -> Remove space and op_record.
            if isinstance(from_, SingleDataOp):
                self.space = None
                self.op_records = set()
            self.incoming_connections.remove(from_)

    def __str__(self):
        return "{}-Socket('{}/{}'{})".format(self.type, self.component.scope, self.name,
                                             " dev='{}'".format(self.component.device) if self.component.device else "")


class GraphFunction(object):
    """
    Class describing a segment of the graph defined by a graph_fn-method inside a Component.
    A GraphFunction is connected to incoming Sockets (these are the input parameters to the _graph-func) and to
    outgoing Sockets (these are the return values of the graph_fn).
    Implements the update_from_input method which checks whether all necessary inputs to a graph_fn
    are given and - if yes - starts producing output ops from these inputs and the graph_fn to be passed
    on further to the outgoing Sockets.
    """
    def __init__(self, method, component, input_sockets, output_sockets,
                 flatten_ops=True, split_ops=True,
                 add_auto_key_as_first_param=False, unflatten_ops=True):
        """
        Args:
            method (Union[str,callable]): The method of the graph_fn (must be the name (w/o _graph prefix)
                of a method in `component` or directly a callable.
            component (Component): The Component object that this GraphFunction belongs to.
            input_sockets (List[Socket]): The required input Sockets to be passed as parameters into the
                graph_fn. In the order of graph_fn's parameters.
            output_sockets (List[socket]): The Sockets associated with the return values coming from the graph_fn.
                In the order of the returned values.
            flatten_ops (Union[bool,Set[str]]): Whether to flatten all or some DataOps by creating
                a FlattenedDataOp (with automatic key names).
                Can also be a set of in-Socket names to flatten explicitly (True for all).
                (default: True).
            split_ops (Union[bool,Set[str]]): Whether to split all or some of the already flattened DataOps
                and send the SingleDataOps one by one through the graph_fn.
                Example: in-Sockets=A=Dict (container), B=int (primitive)
                    The graph_fn should then expect for each primitive Space in A:
                        _graph_fn(primitive-in-A (Space), B (int))
                        NOTE that B will be the same in all calls for all primitive-in-A's.
                (default: True).
            add_auto_key_as_first_param (bool): If `split_ops` is not False, whether to send the
                automatically generated flat key as the very first parameter into each call of the graph_fn.
                Example: in-Sockets=A=float (primitive), B=Tuple (container)
                    The graph_fn should then expect for each primitive Space in B:
                        _graph_fn(key, A (float), primitive-in-B (Space))
                        NOTE that A will be the same in all calls for all primitive-in-B's.
                        The key can now be used to index into variables equally structured as B.
                Has no effect if `split_ops` is False.
                (default: False).
            unflatten_ops (bool): Whether to re-establish a nested structure of DataOps
                for graph_fn-returned FlattenedDataOps.
                (default: True)

        Raises:
            YARLError: If a graph_fn with the given name cannot be found in the component.
        """

        # The component object that the method belongs to.
        self.component = component

        self.flatten_ops = flatten_ops
        self.split_ops = split_ops
        self.add_auto_key_as_first_param = add_auto_key_as_first_param
        self.unflatten_ops = unflatten_ops

        if isinstance(method, str):
            self.name = method
            self.method = getattr(self.component, "_graph_fn_" + method, None)
            if not self.method:
                raise YARLError("ERROR: No `_graph_fn_...` method with name '{}' found!".format(method))
        else:
            self.method = method
            self.name = re.sub(r'^_graph_fn_', "", method.__name__)

        # Dict-records for input-sockets (by name) to keep information on their position and "op-completeness".
        self.input_sockets = OrderedDict()
        for i, in_sock in enumerate(input_sockets):
            self.input_sockets[in_sock.name] = dict(socket=in_sock, pos=i)
        # Just a list of Socket objects.
        self.output_sockets = output_sockets

        # Whether we have all necessary input-sockets for passing at least one input-op combination through
        # our computation method. As long as this is False, we return prematurely and wait for more ops to come in
        # (through other Sockets).
        self.input_complete = False
        # Registry for which incoming Sockets' op-records we have already passed through the graph_fn to generate
        # which output op-records.
        # key=tuple of input-op-records (len==number of input params).
        # value=list of generated output op-records (len==number of return values).
        self.in_out_records_map = dict()

    def check_input_completeness(self):
        """
        Checks whether this GraphFunction is "input-complete" and stores the result in self.input_complete.
        Input-completeness is reached (only once and then it stays that way) if all in-Sockets to this computation
        have at least one op defined in their Socket.op_records set.
        """
        if not self.input_complete:
            # Check, whether we are input-complete now (whether all in-Sockets have at least one op defined).
            self.input_complete = True
            for in_sock_rec in self.input_sockets.values():
                if len(in_sock_rec["socket"].op_records) == 0:
                    self.input_complete = False
                    return False
        return True

    def flatten_input_ops(self, *ops):
        """
        Flattens all DataOps in ops into FlattenedDataOp with auto-key generation.
        Ops whose Sockets are not in self.flatten_ops (if its a set)
        will be ignored.

        Args:
            *ops (DataOp): The items to flatten.

        Returns:
            tuple: All *ops as FlattenedDataOp.
        """
        # The returned sequence of output ops.
        ret = []
        in_socket_names = self.input_sockets.keys()
        for i, op in enumerate(ops):
            # self.flatten_ops cannot be False here.
            if self.flatten_ops is True or (isinstance(self.flatten_ops, set) and
                                            in_socket_names[i] in self.flatten_ops):
                ret.append(flatten_op(op))
            else:
                ret.append(op)

        # Always return a tuple for indexing into the return values.
        return tuple(ret)

    @staticmethod
    def unflatten_output_ops(*ops):
        """
        Re-creates the originally nested input structure (as DataOpDict/DataOpTuple) of the given output ops.
        Process all FlattenedDataOp with auto-generated keys, and leave the others untouched.

        Args:
            *ops (DataOp): The ops that need to be re-nested (only process the FlattenedDataOp
                amongst these and ignore all others).

        Returns:
            Tuple[DataOp]: A tuple containing the ops as they came in, except that all FlattenedDataOp
                have been un-flattened (re-nested) into their original ContainerDataOp structures.
        """
        # The returned sequence of output ops.
        ret = []

        for i, op in enumerate(ops):
            # A FlattenedDataOp: Try to re-nest it and then compare it to input_template_op's structure.
            if isinstance(op, FlattenedDataOp):
                ret.append(unflatten_op(op))
            # All others are left as-is.
            else:
                ret.append(op)

        # Always return a tuple for indexing into the return values.
        return tuple(ret)

    def __str__(self):
        return "{}('{}' in=[{}] out=[{}])". \
            format(type(self).__name__, self.name, str(self.input_sockets), str(self.output_sockets))
