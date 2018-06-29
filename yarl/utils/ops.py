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

from collections import OrderedDict
import numpy as np


class DataOp(object):
    """
    The basic class for any Socket-held operation or variable, or collection thereof.
    Each Socket (in or out) holds either one DataOp or a set of alternative DataOps.
    """
    pass


class SingleDataOp(DataOp):
    """
    A placeholder class for a simple (non-container) Tensor going into a GraphFunction or coming out of a GraphFunction,
    or a tf.no_op-like item.
    """
    def __init__(self, constant_value=None):
        """
        Args:
            constant_value (any): A constant value this SingleDataOp holds instead of an actual op.
                This value is always converted into a numpy array (even if it's a scalar python primitive).
        """
        # Numpy'ize scalar values (tf doesn't sometimes like python primitives).
        if isinstance(constant_value, (float, int, bool)):
            constant_value = np.array(constant_value)
        self.constant_value = constant_value


class ContainerDataOp(DataOp):
    """
    A placeholder class for any DataOp that's not a SingleDataOp, but a (possibly nested) container structure
    containing SingleDataOps as leave nodes.
    """
    pass


class DataOpDict(ContainerDataOp, dict):
    """
    A hashable dict that's used to make (possibly nested) dicts of SingleDataOps hashable, so that
    we can store them in sets and use them as lookup keys in other dicts.
    Dict() Spaces produce DataOpDicts when methods like `get_tensor_variables` are called on them.
    """
    def __hash__(self):
        """
        Hash based on sequence of sorted items (keys are all strings, values are always other DataOps).
        """
        return hash(tuple(sorted(self.items())))


class DataOpTuple(ContainerDataOp, tuple):
    """
    A simple wrapper for a (possibly nested) tuple that contains other DataOps.
    """
    def __new__(cls, *components):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]

        return tuple.__new__(cls, components)


class FlattenedDataOp(DataOp, OrderedDict):
    """
    An OrderedDict-type placeholder class that only contains str as keys and SingleDataOps
    (as opposed to ContainerDataOps) as values.
    """
    # TODO: enforce str as keys?
    pass


class OBSOLETE_DataOpRecord(object):
    """
    A simple wrapper class for a DataOp carrying the op itself and some additional information about it.
    """
    def __init__(self, op=None, group=None, filter_by_socket_name=None, socket=None):
        """
        Args:
            op (Optional[DataOp]): The actual DataOp carried by this record. May be left empty at construction time.
            group (Optional[int]): A group ID to guide the op (together with others with the same label) through the
                Components and GraphFns.
            filter_by_socket_name (Optional[str]): If given, signals that only ops coming from the Socket with that name,
                can be placed into this record. This record must have an empty op for this.
            socket (Socket): The Socket object that will hold this record.
        """
        self.op = op
        self.group = group
        # Filtering tools to accept only raw ops from certain sources.
        self.filter_by_socket_name = filter_by_socket_name
        # If given, tells us into which succeeding record to push our primitive op to.
        self.push_op_into_recs = set()

        # The Socket object that holds this record (optional).
        self.socket = socket

    def __hash__(self):
        # The following properties will not change (self.op might, which is why we can't use it here).
        return hash(self.group) + hash(self.filter_by_socket_name)


class DataOpRecord(object):
    """
    A simple wrapper class for a DataOp carrying the op itself and some additional information about it.
    """
    #_ID = -1

    def __init__(self, op=None, column=None):
        #self.id = self.get_id()
        self.op = op

        # Set of (op-col ID, slot) tuples that are connected from this one.
        self.next = set()
        # Link back to the column we belong to.
        self.column = column
        # This op record's Component object.
        #self.component = None

    #def get_id(self):
    #    self._ID += 1
    #    return self._ID

    #def __hash__(self):
    #    return hash(self.id)


class DataOpRecordColumn(object):
    _ID = -1

    def __init__(self, op_records, component):
        self.id = self.get_id()

        if not isinstance(op_records, int):
            self.op_records = [op_records] if isinstance(op_records, DataOpRecord) else list(op_records)
            # For graph_fn and convenience reasons, give a pointer to the column to each op in it.
            for op_rec in self.op_records:
                op_rec.column = self
        else:
            self.op_records = [DataOpRecord(op=None, column=self)] * op_records

        self.component = component

    def is_complete(self):
        for op_rec in self.op_records:
            if op_rec.op is None:
                return False
        return True

    def get_id(self):
        self._ID += 1
        return self._ID

    def __hash__(self):
        return hash(self.id)


class DataOpRecordColumnIntoGraphFn(DataOpRecordColumn):
    def __init__(self, op_records, component, graph_fn, out_graph_fn_column, flatten_ops=False,
                 split_ops=False, add_auto_key_as_first_param=False):
        super(DataOpRecordColumnIntoGraphFn, self).__init__(op_records=op_records, component=component)

        self.graph_fn = graph_fn

        # The column after passing this one through the graph_fn.
        self.out_graph_fn_column = out_graph_fn_column

        self.flatten_op = flatten_ops
        self.split_ops = split_ops
        self.add_auto_key_as_first_param = add_auto_key_as_first_param


class DataOpRecordColumnFromGraphFn(DataOpRecordColumn):
    pass


class DataOpRecordColumnIntoAPIMethod(DataOpRecordColumn):
    def __init__(self, op_records, component, api_method_rec):
        super(DataOpRecordColumnIntoAPIMethod, self).__init__(op_records=op_records, component=component)

        self.api_method_rec = api_method_rec


class DataOpRecordColumnFromAPIMethod(DataOpRecordColumn):
    pass


class APIMethodRecord(object):
    def __init__(self, method, component, must_be_complete=True):
        self.method = method
        self.component = component
        self.must_be_complete = must_be_complete

        self.spaces = None
        self.in_op_columns = list()
        self.out_op_columns = list()


class GraphFnRecord(object):
    def __init__(self, graph_fn, component):
        self.graph_fn = graph_fn
        self.component = component

        self.in_op_columns = list()
        self.out_op_columns = list()
