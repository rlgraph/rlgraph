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
import re

from yarl.utils.util import YARLError, dtype, get_shape, deep_tuple
from yarl.utils.dictop import DictOp
from .containers import Dict, Tuple, _TUPLE_OPEN, _TUPLE_CLOSE
from .continuous import Continuous
from .intbox import IntBox
from .bool_space import Bool
from .discrete import Discrete


def get_space_from_op(op):
    """
    Tries to re-create a Space object given some op (dictop, tuple, op).
    This is useful for shape inference when passing a Socket's ops through a Computation and
    auto-inferring the resulting shape/Space.

    Args:
        op (Union[dictop,tuple,op]): The op to create a corresponding Space for.

    Returns:
        Space: The inferred Space object.
    """
    # a Dict
    if isinstance(op, DictOp):
        spec = dict()
        add_batch_rank = False
        for k, v in op.items():
            spec[k] = get_space_from_op(v)
            if spec[k].has_batch_rank:
                add_batch_rank = True
        return Dict(spec, add_batch_rank=add_batch_rank)
    # a Tuple
    elif isinstance(op, tuple):
        spec = list()
        add_batch_rank = False
        for i in op:
            spec.append(get_space_from_op(i))
            if spec[-1].has_batch_rank:
                add_batch_rank = True
        return Tuple(spec, add_batch_rank=add_batch_rank)
    # primitive Space -> infer from op dtype and shape
    else:
        # No Space: e.g. the tf.no_op.
        if hasattr(op, "dtype") is False:
            return 0
        else:
            shape = get_shape(op)
            add_batch_rank = False
            if shape is not () and shape[0] is None:
                shape = shape[1:]
                add_batch_rank = True

            # a Continuous
            if op.dtype == dtype("float"):
                return Continuous(shape=shape, add_batch_rank=add_batch_rank)
            # an IntBox/Discrete
            elif op.dtype == dtype("int"):
                # TODO: How do we distinguish these two by a tensor (name/type)?
                # TODO: Solution: Merge Discrete and IntBox into `IntBox`. Discrete is a special IntBox with shape=() and low=0 and high=n-1
                if len(shape) == 1:
                    return Discrete(n=shape[0], add_batch_rank=add_batch_rank)
                else:
                    return IntBox(low=0, high=255, shape=shape, add_batch_rank=add_batch_rank)  # low, high=dummy values
            # a Bool
            elif op.dtype == dtype("bool"):
                return Bool(add_batch_rank=add_batch_rank)
            else:
                raise YARLError("ERROR: Cannot derive Space from op '{}' (unknown type?)!".format(op))


def flatten_op(op, scope_="", list_=None):
    """
    Flattens a single ContainerSpace (op) into a python OrderedDict with auto-key generation.

    Args:
        op (Union[dictop,tuple]): The op to flatten. This can only be a tuple or a dictop.
        scope_ (str): The recursive scope for auto-key generation.
        list_ (list): The list of tuples (key, value) to be converted into the final OrderedDict.

    Returns:
        OrderedDict: The flattened representation of the op.
    """
    ret = False
    # Are we in the non-recursive (first) call?
    if list_ is None:
        assert isinstance(op, (dict, tuple)), "ERROR: Can only flatten container (dictop/tuple) ops!"
        list_ = list()
        ret = True

    if isinstance(op, tuple):
        scope_ += "/" + _TUPLE_OPEN
        for i, c in enumerate(op):
            flatten_op(c, scope_=scope_ + str(i) + _TUPLE_CLOSE, list_=list_)
    elif isinstance(op, dict):
        scope_ += "/"
        for key in sorted(op.keys()):
            flatten_op(op[key], scope_=scope_ + key, list_=list_)
    else:
        list_.append((scope_, op))

    # Non recursive (first) call -> Return the final OrderedDict.
    if ret:
        return OrderedDict(list_)


def re_nest_op(op):
    base_structure = None

    for k, v in op.items():
        parent_structure = None
        parent_key = None
        current_structure = None
        type_ = None

        keys = k[1:].split("/")  # skip 1st char (/)
        for key in keys:
            mo = re.match(r'^{}(\d+){}$'.format(_TUPLE_OPEN, _TUPLE_CLOSE), key)
            if mo:
                type_ = list
                idx = int(mo.group(1))
            else:
                type_ = DictOp
                idx = key

            if current_structure is None:
                if base_structure is None:
                    base_structure = [None] if type_ == list else DictOp()
                current_structure = base_structure
            elif parent_key is not None:
                if isinstance(parent_structure, list) and parent_structure[parent_key] is None or \
                        isinstance(parent_structure, DictOp) and parent_key not in parent_structure:
                    current_structure = [None] if type_ == list else DictOp()
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
        current_structure[parent_key] = v

    # Deep conversion from list to tuple.
    return deep_tuple(base_structure)


