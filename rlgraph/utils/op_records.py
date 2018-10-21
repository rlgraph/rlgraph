# Copyright 2018 The RLgraph Authors, All Rights Reserved.
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

import inspect
import numpy as np

from rlgraph.spaces.space_utils import get_space_from_op
from rlgraph.utils import dtype
from rlgraph.utils.rlgraph_errors import RLGraphError, RLGraphAPICallParamError
from rlgraph.utils.ops import FlattenedDataOp, flatten_op, unflatten_op, is_constant


class DataOpRecord(object):
    """
    A simple wrapper class for a DataOp carrying the op itself and some additional information about it.
    """
    _ID = -1

    def __init__(self, op=None, column=None, position=None, kwarg=None, space=None, previous=None, next_=None):
        """
        Args:
            op (Optional[DataOp]): The optional DataOp to already store in this op-rec.
            column (DataOpRecordColumn): The DataOpRecordColumn to which this op-rec belongs.
            position (Optional[int]): An optional position (index) for this op inside `column`.
            kwarg (Optional[str]): The keyword with which to call the API-method if this op-rec is not a positional
                arg.
            space (Optional[Space]): The Space of `op` if already known at construction time. Will be poulated
                later (during build phase) if not.
            next\_ (Optional(Set[DataOpRecord],DataOpRecord)): The next op-record or set of op-records.
            previous (Optional(DataOpRecord)): The previous op-record.
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
        self.next = next_ if isinstance(next_, set) else ({next_} if next_ is not None else set())
        # The previous op that lead to this one.
        self.previous = previous

    @staticmethod
    def get_id():
        DataOpRecord._ID += 1
        return DataOpRecord._ID

    @staticmethod
    def reset():
        DataOpRecord._ID = -1

    def __str__(self):
        return "DataOpRec(id={} {}{})".format(
            self.id,"pos="+str(self.position) if self.kwarg is None else "kwarg="+self.kwarg,
            "" if self.column is None else " in "+str(self.column)
        )

    def __hash__(self):
        return hash(self.id)


class DataOpRecordColumn(object):
    _ID = -1

    def __init__(self, component, num_op_records=None, args=None, kwargs=None):
        """
        Args:
            component (Component): The Component to which this column belongs.
        """
        self.id = self.get_id()

        if num_op_records is None:
            self.op_records = []
            if args is not None:
                for i in range(len(args)):
                    if args[i] is None:
                        continue
                    op_rec = DataOpRecord(op=None, column=self, position=i)
                    # If incoming is an op-rec -> Link them.
                    if isinstance(args[i], DataOpRecord):
                        op_rec.previous = args[i]
                        op = args[i].op
                        if op is not None:
                            op_rec.op = op
                            op_rec.space = get_space_from_op(op)
                        args[i].next.add(op_rec)
                    # Do constant value assignment here.
                    elif args[i] is not None:
                        op = args[i]
                        if is_constant(op):
                            op = np.array(op, dtype=dtype(type(op), "np"))
                        op_rec.op = op
                        op_rec.space = get_space_from_op(op)
                        component.constant_op_records.add(op_rec)
                    self.op_records.append(op_rec)

            if kwargs is not None:
                for key in sorted(kwargs.keys()):
                    value = kwargs[key]
                    if value is None:
                        continue
                    op_rec = DataOpRecord(op=None, column=self, kwarg=key)
                    # If incoming is an op-rec -> Link them.
                    if isinstance(value, DataOpRecord):
                        op_rec.previous = value
                        op_rec.op = value.op  # assign op if any
                        value.next.add(op_rec)
                    # Do constant value assignment here.
                    elif value is not None:
                        op = value
                        if is_constant(op):
                            op = np.array(op, dtype=dtype(type(op), "np"))
                        op_rec.op = op
                        op_rec.space = get_space_from_op(op)
                        component.constant_op_records.add(op_rec)
                    self.op_records.append(op_rec)
        else:
            self.op_records = [DataOpRecord(op=None, column=self, position=i) for i in range(num_op_records)]

        # For __str__ purposes.
        self.op_id_list = [o.id for o in self.op_records]

        # The component this column belongs to.
        self.component = component

    def is_complete(self):
        for op_rec in self.op_records:
            if op_rec.op is None:
                return False
        return True

    def get_args_and_kwargs(self):
        args = []
        kwargs = {}
        for op_rec in self.op_records:
            if op_rec.kwarg is None:
                args.append(op_rec)
            else:
                kwargs[op_rec.kwarg] = op_rec
        return tuple(args), kwargs

    def get_args_and_kwargs_as_list(self):
        args, kwargs = self.get_args_and_kwargs()
        return [(i, a) for i, a in enumerate(args)] + [(k, v) for k, v in sorted(kwargs.items())]

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
    def __init__(self, component, graph_fn, flatten_ops=False,
                 split_ops=False, add_auto_key_as_first_param=False, args=None, kwargs=None):
        super(DataOpRecordColumnIntoGraphFn, self).__init__(
            component=component, args=args, kwargs=kwargs
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
        kwarg_ret = {}
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
    def __init__(self, num_op_records, component, graph_fn_name, in_graph_fn_column):
        """
        Args:
            graph_fn_name (str): The name of the graph_fn that returned the ops going into `self.op_records`.
        """
        super(DataOpRecordColumnFromGraphFn, self).__init__(
            num_op_records=num_op_records, component=component
        )
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
    def __init__(self, component, api_method_rec, args=None, kwargs=None):
        self.api_method_rec = api_method_rec
        super(DataOpRecordColumnIntoAPIMethod, self).__init__(component=component, args=args, kwargs=kwargs)

    def __str__(self):
        return "OpRecCol(ops: {})->APIMethod('{}')".format(self.op_id_list, self.api_method_rec.name)


class DataOpRecordColumnFromAPIMethod(DataOpRecordColumn):
    """
    An array of return values from an API-method pass through.
    """
    def __init__(self, component, api_method_name, args=None, kwargs=None):
        self.api_method_name = api_method_name
        super(DataOpRecordColumnFromAPIMethod, self).__init__(component, args=args, kwargs=kwargs)

    def __str__(self):
        return "APIMethod('{}')->OpRecCol(ops: {})".format(self.api_method_name, self.op_id_list)


class APIMethodRecord(object):
    def __init__(self, func, wrapper_func, name,
                 component=None, must_be_complete=True, ok_to_overwrite=False,
                 is_graph_fn_wrapper=False, is_class_method=True,
                 flatten_ops=False, split_ops=False, add_auto_key_as_first_param=False):
        """
        Args:
            func (callable): The actual API-method (callable).
            component (Component): The Component this API-method belongs to.
            must_be_complete (bool): Whether the Component can only be input-complete if at least one
                input op-record column is complete.
            TODO: documentation.
        """
        self.func = func
        self.wrapper_func = wrapper_func
        self.name = name
        self.component = component
        self.must_be_complete = must_be_complete
        self.ok_to_overwrite = ok_to_overwrite

        self.is_class_method = is_class_method

        self.is_graph_fn_wrapper = is_graph_fn_wrapper
        self.flatten_ops = flatten_ops
        self.split_ops = split_ops
        self.add_auto_key_as_first_param = add_auto_key_as_first_param

        # List of the input-parameter names (str) of this API-method.
        self.input_names = []
        # Name of the *args arg (usually "args") w/o the *.
        self.args_name = None
        # Name of the **kwargs arg (usually "kwargs") w/o the *.
        self.kwargs_name = None
        # List of the names of all non-arg/non-kwarg arguments (the ones that come first in the signature).
        self.non_args_kwargs = []
        # List of args that have a default value.
        self.default_args = []

        self.in_op_columns = []
        self.out_op_columns = []

        # Update the api_method_inputs dict (with empty Spaces if not defined yet).
        skip_args = 1
        skip_args += (self.is_graph_fn_wrapper and self.add_auto_key_as_first_param)
        param_list = list(inspect.signature(self.func).parameters.values())[skip_args:]

        for param in param_list:
            # This param has a default value.
            if param.default != inspect.Parameter.empty:
                self.non_args_kwargs.append(param.name)
                self.default_args.append(param.name)
            # *args
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                self.args_name = param.name
            # **kwargs
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                self.kwargs_name = param.name
            # Normal, non-default.
            else:
                self.non_args_kwargs.append(param.name)

    def __str__(self):
        return "APIMethodRecord({} {} called {}x)".format(self.name, self.input_names, len(self.in_op_columns))


class GraphFnRecord(object):
    def __init__(self, func, wrapper_func, component=None, is_class_method=True,
                 flatten_ops=False, split_ops=False, add_auto_key_as_first_param=False):
        self.func = func
        self.wrapper_func = wrapper_func
        self.name = self.func.__name__
        self.component = component

        self.is_class_method = is_class_method

        self.flatten_ops = flatten_ops
        self.split_ops = split_ops
        self.add_auto_key_as_first_param = add_auto_key_as_first_param

        self.in_op_columns = []
        self.out_op_columns = []


def get_call_param_name(op_rec):
    api_method_rec = op_rec.column.api_method_rec  # type: APIMethodRecord
    pos_past_normals = op_rec.position - len(api_method_rec.non_args_kwargs)

    # There are *args in the signature.
    if api_method_rec.args_name is not None:
        # op_rec has no name -> Can only be normal arg or one of *args.
        if op_rec.kwarg is None:
            # Position is higher than number of "normal" args -> must be one of *args.
            if pos_past_normals >= 0:
                param_name = api_method_rec.args_name + "[{}]".format(pos_past_normals)
            # Normal arg (not part of *args).
            else:
                param_name = api_method_rec.input_names[op_rec.position]
        # op_rec has name -> Can only be normal arg of one of **kwargs (if any).
        else:
            if op_rec.kwarg in api_method_rec.non_args_kwargs:
                param_name = op_rec.kwarg
            else:
                if api_method_rec.kwargs_name is None:
                    raise RLGraphAPICallParamError(
                        "ERROR: API-method '{}' has no **kwargs, but op-rec {} indicates that it does!".
                        format(api_method_rec.name, op_rec.id)
                    )
                param_name = api_method_rec.kwargs_name + "[{}]".format(op_rec.kwarg)
    # There are *kwargs in the signature.
    elif api_method_rec.kwargs_name is not None:
        # op_rec has no name -> Can only be a normal arg.
        if op_rec.kwarg is None:
            # Position is higher than number of "normal" args -> ERROR.
            if pos_past_normals >= 0:
                raise RLGraphAPICallParamError(
                    "Op-rec '{}' has no kwarg, but its position ({}) indicates that it's part "
                    "of {}'s **kwargs!".format(op_rec.id, op_rec.position, api_method_rec.name)
                )
            # Normal arg (by position).
            else:
                param_name = api_method_rec.input_names[op_rec.position]
        # op_rec has name -> Can only be normal arg of one of **kwargs.
        else:
            if op_rec.kwarg in api_method_rec.non_args_kwargs:
                param_name = op_rec.kwarg
            else:
                param_name = api_method_rec.kwargs_name + "[{}]".format(op_rec.kwarg)
    else:
        # op_rec has no name -> Can only be normal arg.
        if op_rec.kwarg is None:
            # Position is higher than number of "normal" args -> ERROR.
            if pos_past_normals >= 0:
                raise RLGraphAPICallParamError(
                    "Op-rec {}'s position ({}) is higher than {}'s number of args!".
                    format(op_rec.id, op_rec.position, api_method_rec.name)
                )
            # Normal arg (by position).
            else:
                param_name = api_method_rec.input_names[op_rec.position]
        # op_rec has name -> Can only be normal arg.
        else:
            if op_rec.kwarg in api_method_rec.non_args_kwargs:
                param_name = op_rec.kwarg
            else:
                raise RLGraphAPICallParamError(
                    "Op-rec's kwarg ({}) is not an parameter of API-method {}'s signature!".
                    format(op_rec.kwarg, api_method_rec.name)
                )

    return param_name
