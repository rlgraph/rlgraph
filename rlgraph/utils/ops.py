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

from collections import OrderedDict
import numpy as np
import re

from rlgraph.utils.rlgraph_error import RLGraphError

# Defines how to generate auto-keys for flattened Tuple-Space items.
# _T\d+_
FLAT_TUPLE_OPEN = "_T"
FLAT_TUPLE_CLOSE = "_"


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
    pass


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
    Dict() Spaces produce DataOpDicts when methods like `get_variable` are called on them.
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


class DataOpRecord(object):
    """
    A simple wrapper class for a DataOp carrying the op itself and some additional information about it.
    """
    _ID = -1

    def __init__(self, op=None, column=None, position=None, kwarg=None, space=None):
        """
        Args:
            op (Optional[DataOp]): The optional DataOp to already store in this op-rec.
            column (DataOpRecordColumn): The DataOpRecordColumn to which this op-rec belongs.
            position (Optional[int]): An optional position (index) for this op inside `column`.
            kwarg (Optional[str]): The keyword with which to call the API-method if this op-rec is not a positional
                arg.
            space (Optional[Space]): The Space of `op` if already known at construction time. Will be poulated
                later (during build phase) if not.
        """
        self.id = self.get_id()
        self.op = op
        # Whether the op in this record is one of the last in the graph (a core API-method returned op).
        self.is_terminal_op = False

        self.column = column
        self.position = position
        self.kwarg = kwarg

        # The inferred Space of this op.
        self.space = space

        # Set of (op-col ID, slot) tuples that are connected from this one.
        self.next = set()
        # The previous op that lead to this one.
        self.previous = None

    @staticmethod
    def get_id():
        DataOpRecord._ID += 1
        return DataOpRecord._ID

    def __str__(self):
        return "DataOpRec(id={} {}{})".format(
            self.id,"pos="+str(self.position) if self.kwarg is None else "kwarg="+self.kwarg,
            "" if self.column is None else " in "+str(self.column)
        )

    def __hash__(self):
        return hash(self.id)


class DataOpRecordColumn(object):
    _ID = -1

    def __init__(self, op_records, component, kwarg_names=None):
        """
        Args:
            op_records (int): The number of individual op_records to create for this column.
            component (Component): The Component to which this column belongs.
            kwarg_names (Optional[List[str]]): Optional (but complete!) list of already known
                kwarg-names for some of the op-recs (op-recs w/o kwarg should have `None` in this list).
        """
        self.id = self.get_id()

        self.op_records = list()
        for i in range(op_records):
            kwarg = kwarg_names[i] if kwarg_names is not None else None
            self.op_records.append(DataOpRecord(op=None, column=self, position=i, kwarg=kwarg))

        # For __str__ purposes.
        self.op_id_list = [o.id for o in self.op_records]

        # The component this column belongs to.
        self.component = component

    def is_complete(self):
        for op_rec in self.op_records:
            if op_rec.op is None:
                return False
        return True

    @staticmethod
    def get_id():
        DataOpRecordColumn._ID += 1
        return DataOpRecordColumn._ID

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        raise NotImplementedError


class DataOpRecordColumnIntoGraphFn(DataOpRecordColumn):
    """
    An array of input parameters (DataOpRecord objects) that will go in a single call into a graph_fn.

    GraphFns are called only at build-time. During assembly time, empty DataOpRecordColumns are created on both
    side of the graph_fn (input=DataOpRecordColumnIntoGraphFn and return values=DataOpRecordColumnFromGraphFn).

    Keeps a link to the graph_fn and also specifies options on how to call the graph_fn.
    The call of the graph_fn will result in another column (return values) of DataOpRecords that this record points
    to.
    """
    def __init__(self, op_records, component, graph_fn, kwarg_names=None, flatten_ops=False,
                 split_ops=False, add_auto_key_as_first_param=False):
        super(DataOpRecordColumnIntoGraphFn, self).__init__(
            op_records=op_records, component=component, kwarg_names=kwarg_names
        )

        # The graph_fn that our ops come from.
        self.graph_fn = graph_fn

        self.flatten_ops = flatten_ops
        self.split_ops = split_ops
        self.add_auto_key_as_first_param = add_auto_key_as_first_param

        # The column after passing this one through the graph_fn.
        self.out_graph_fn_column = None

        # Whether this column has already been sent through the graph_fn.
        self.already_sent = False

    def flatten_input_ops(self, *ops, **kwarg_ops):
        """
        Flattens all DataOps in ops into FlattenedDataOp with auto-key generation.
        Ops whose Sockets are not in self.flatten_ops (if its a set)
        will be ignored.

        Args:
            *ops (op): The primitive ops to flatten.
            **kwarg_ops (op): More primitive ops to flatten (but by named key).

        Returns:
            Tuple[DataOp]: A new tuple with all ops (or those specified by `flatten_ops` as FlattenedDataOp.
        """
        assert all(op is not None for op in ops)  # just make sure

        # The returned sequence of output ops.
        ret = []
        for i, op in enumerate(ops):
            if self.flatten_ops is True or (isinstance(self.flatten_ops, set) and i in self.flatten_ops):
                ret.append(flatten_op(op))
            else:
                ret.append(op)

        # Process kwargs, if given.
        kwarg_ret = dict()
        if len(kwarg_ops) > 0:
            for key, op in kwarg_ops.items():
                if self.flatten_ops is True or (isinstance(self.flatten_ops, set) and key in self.flatten_ops):
                    kwarg_ret[key] = flatten_op(op)
                else:
                    kwarg_ret[key] = op

        # Always return a tuple for indexing into the return values.
        return tuple(ret), kwarg_ret

    def split_flattened_input_ops(self, *ops, **kwarg_ops):
        """
        Splits any FlattenedDataOp in *ops and **kwarg_ops into its SingleDataOps and collects them to be passed
        one by one through some graph_fn. If more than one FlattenedDataOp exists in *ops and **kwarg_ops,
        these must have the exact same keys.
        If `add_auto_key_as_first_param` is True: Add auto-key as very first parameter in each
        returned parameter tuple.

        Args:
            *ops (op): The primitive ops to split.
            **kwarg_ops (op): More primitive ops to split (but by named key).

        Returns:
            Union[FlattenedDataOp,Tuple[DataOp]]: The sorted parameter tuples (by flat-key) to use as api_methods in the
                calls to the graph_fn.
                If no FlattenedDataOp is in ops, returns ops as-is.

        Raises:
            RLGraphError: If there are more than 1 flattened ops in ops and their keys don't match 100%.
        """
        assert all(op is not None for op in ops)  # just make sure

        # Collect FlattenedDataOp for checking their keys (must match).
        flattened = [op.items() for op in ops if len(op) > 1 or "" not in op]
        # If it's more than 1, make sure they match. If they don't match: raise Error.
        if len(flattened) > 1:
            # Loop through the non-first ones and make sure all keys match vs the first one.
            for other in flattened[1:]:
                iter_ = iter(other)
                for key, value in flattened[0]:
                    k_other, v_other = next(iter_)
                    if key != k_other:  # or get_shape(v_other) != get_shape(value):
                        raise RLGraphError("ERROR: Flattened ops have a key mismatch ({} vs {})!".format(key, k_other))

        # We have one or many (matching) ContainerDataOps: Split the calls.
        if len(flattened) > 0:
            # The first op that is a FlattenedDataOp.
            guide_op = next(op for op in ops if len(op) > 1 or "" not in op)
            # Re-create our iterators.
            collected_call_params = FlattenedDataOp()
            # Do the single split calls to our computation func.
            for key in guide_op.keys():
                # Prep input params for a single call.
                params = [key] if self.add_auto_key_as_first_param is True else []
                for op in ops:
                    params.append(op[key] if key in op else op[""])
                # Add kwarg_ops
                for kwarg_key, kwarg_op in kwarg_ops:
                    params.append(tuple([
                        kwarg_key,
                        kwarg_ops[kwarg_key][key] if key in kwarg_ops[kwarg_key] else kwarg_ops[kwarg_key][""]
                    ]))
                # Now do the single call.
                collected_call_params[key] = params
            return collected_call_params
        # We don't have any container ops: No splitting possible. Return args and kwargs as is.
        else:
            return tuple(([""] if self.add_auto_key_as_first_param is True else []) + [op[""] for op in ops]),\
                   {key: value[""] for key, value in kwarg_ops}

    @staticmethod
    def unflatten_output_ops(*ops):
        """
        Re-creates the originally nested input structure (as DataOpDict/DataOpTuple) of the given op-record column.
        Process all FlattenedDataOp with auto-generated keys, and leave the others untouched.

        Args:
            ops (DataOp): The ops that need to be unflattened (only process the FlattenedDataOp
                amongst these and ignore all others).

        Returns:
            Tuple[DataOp]: A tuple containing the ops as they came in, except that all FlattenedDataOp
                have been un-flattened (re-nested) into their original structures.
        """
        # The returned sequence of output ops.
        ret = []

        for i, op in enumerate(ops):
            # A FlattenedDataOp: Try to re-nest it.
            if isinstance(op, FlattenedDataOp):
                ret.append(unflatten_op(op))
            # All others are left as-is.
            else:
                ret.append(op)

        # Always return a tuple for indexing into the return values.
        return tuple(ret)

    def __str__(self):
        return "OpRecCol(ops: {})->GraphFn('{}')".format(self.op_id_list, self.graph_fn.__name__)


class DataOpRecordColumnFromGraphFn(DataOpRecordColumn):
    """
    An array of return values from a graph_fn pass through.
    """
    def __init__(self, op_records, component, graph_fn_name, in_graph_fn_column):
        """
        Args:
            graph_fn_name (str): The name of the graph_fn that returned the ops going into `self.op_records`.
        """
        super(DataOpRecordColumnFromGraphFn, self).__init__(op_records, component)
        # The graph_fn that our ops come from.
        self.graph_fn_name = graph_fn_name
        # The column after passing this one through the graph_fn.
        self.in_graph_fn_column = in_graph_fn_column

    def __str__(self):
        return "GraphFn('{}')->OpRecCol(ops: {})".format(self.graph_fn_name, self.op_id_list)


class DataOpRecordColumnIntoAPIMethod(DataOpRecordColumn):
    """
    An array of input parameters (DataOpRecord objects) that will go in a single call into an API-method.

    API-methods are called and run through during meta-graph assembly time.

    Stores the api method record and all DataOpRecords used for the call.
    """
    def __init__(self, op_records, component, api_method_rec):
        self.api_method_rec = api_method_rec
        super(DataOpRecordColumnIntoAPIMethod, self).__init__(op_records=op_records, component=component)

    def __str__(self):
        return "OpRecCol(ops: {})->APIMethod('{}')".format(self.op_id_list, self.api_method_rec.method.__name__)


class DataOpRecordColumnFromAPIMethod(DataOpRecordColumn):
    """
    An array of return values from an API-method pass through.
    """
    def __init__(self, op_records, component, api_method_name):
        self.api_method_name = api_method_name
        super(DataOpRecordColumnFromAPIMethod, self).__init__(op_records, component)

    def __str__(self):
        return "APIMethod('{}')->OpRecCol(ops: {})".format(self.api_method_name, self.op_id_list)


class APIMethodRecord(object):
    def __init__(self, method, component, must_be_complete=True, is_graph_fn_wrapper=False,
                 add_auto_key_as_first_param=False):  #, callable_anytime=False):
        """
        Args:
            method (callable): The actual API-method (callable).
            component (Component): The Component this API-method belongs to.
            must_be_complete (bool): Whether the Component can only be input-complete if at least one
                input op-record column is complete.
            #callable_anytime (bool): Whether this API-method can be called even before the Component is input-complete.
        """
        self.method = method
        self.name = self.method.__name__
        self.component = component
        self.must_be_complete = must_be_complete

        self.is_graph_fn_wrapper = is_graph_fn_wrapper
        self.add_auto_key_as_first_param = add_auto_key_as_first_param

        # List of the input-parameter names (str) of this API-method.
        self.input_names = None
        #self.in_spaces = None

        self.in_op_columns = list()
        self.out_op_columns = list()

    def __str__(self):
        return "APIMethodRecord({} {} called {}x)".format(self.name, self.input_names, len(self.in_op_columns))


class GraphFnRecord(object):
    def __init__(self, graph_fn, component):
        self.graph_fn = graph_fn
        self.name = self.graph_fn.__name__
        self.component = component

        self.in_op_columns = list()
        self.out_op_columns = list()


def flatten_op(op, scope_="", list_=None):
    """
    Flattens a single ContainerDataOp or a native python dict/tuple into a FlattenedDataOp with auto-key generation.

    Args:
        op (Union[ContainerDataOp,dict,tuple]): The item to flatten.
        scope_ (str): The recursive scope for auto-key generation.
        list_ (list): The list of tuples (key, value) to be converted into the final FlattenedDataOp.

    Returns:
        FlattenedDataOp: The flattened representation of the op.
    """
    ret = False

    # Are we in the non-recursive (first) call?
    if list_ is None:
        # Flatten a SingleDataOp -> return FlattenedDataOp with only-key=""
        if not isinstance(op, (ContainerDataOp, dict, tuple)):
            return FlattenedDataOp([("", op)])
        list_ = list()
        ret = True

    if isinstance(op, dict):
        scope_ += "/"
        for key in sorted(op.keys()):
            # Make sure we have no double slashes from flattening an already FlattenedDataOp.
            flatten_op(op[key], scope_=(scope_[:-1] if len(key) == 0 or key[0] == "/" else scope_) + key, list_=list_)
    elif isinstance(op, tuple):
        scope_ += "/" + FLAT_TUPLE_OPEN
        for i, c in enumerate(op):
            flatten_op(c, scope_=scope_ + str(i) + FLAT_TUPLE_CLOSE, list_=list_)
    else:
        assert not isinstance(op, (dict, tuple))
        list_.append((scope_, op))

    # Non recursive (first) call -> Return the final FlattenedDataOp.
    if ret:
        return FlattenedDataOp(list_)


def unflatten_op(op):
    """
    Takes a FlattenedDataOp with auto-generated keys and returns the corresponding
    unflattened DataOp.
    If the only key in the input FlattenedDataOp is "", it returns the SingleDataOp under
    that key.

    Args:
        op (dict): The item to be unflattened (re-nested) into any DataOp. Usually a FlattenedDataOp, but can also
            be a plain dict.

    Returns:
        DataOp: The unflattened (re-nested) item.
    """
    # Special case: FlattenedDataOp with only 1 SingleDataOp (key="").
    if len(op) == 1 and "" in op:
        return op[""]

    # Normal case: FlattenedDataOp that came from a ContainerItem.
    base_structure = None

    for op_name, op_val in op.items():
        parent_structure = None
        parent_key = None
        current_structure = None
        type_ = None

        op_key_list = op_name[1:].split("/")  # skip 1st char (/)
        for sub_key in op_key_list:
            mo = re.match(r'^{}(\d+){}$'.format(FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE), sub_key)
            if mo:
                type_ = list
                idx = int(mo.group(1))
            else:
                type_ = DataOpDict
                idx = sub_key

            if current_structure is None:
                if base_structure is None:
                    base_structure = [None] if type_ == list else DataOpDict()
                current_structure = base_structure
            elif parent_key is not None:
                if isinstance(parent_structure, list) and parent_structure[parent_key] is None or \
                        isinstance(parent_structure, DataOpDict) and parent_key not in parent_structure:
                    current_structure = [None] if type_ == list else DataOpDict()
                    parent_structure[parent_key] = current_structure
                else:
                    current_structure = parent_structure[parent_key]
                    if type_ == list and len(current_structure) == idx:
                        current_structure.append(None)

            parent_structure = current_structure
            parent_key = idx
            if isinstance(parent_structure, list) and len(parent_structure) == parent_key:
                parent_structure.append(None)

        if type_ == list and len(current_structure) == parent_key:
            current_structure.append(None)
        current_structure[parent_key] = op_val

    # Deep conversion from list to tuple.
    return deep_tuple(base_structure)


def deep_tuple(x):
    """
    Converts an input list of list (of list, etc..) into the respective nested DataOpTuple.

    Args:
        x (list): The input list to be converted into a tuple.

    Returns:
        tuple: The corresponding tuple to x.
    """
    # A list -> convert to DataOpTuple.
    if isinstance(x, list):
        return DataOpTuple(list(map(deep_tuple, x)))
    # A dict -> leave type as is and keep converting recursively.
    elif isinstance(x, dict):
        # type(x) b/c x could be DataOpDict as well.
        return type(x)(dict(map(lambda i: (i[0], deep_tuple(i[1])), x.items())))
    # A primitive -> keep as is.
    else:
        return x

