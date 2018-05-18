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
from yarl.utils.util import DictOp, dtype, get_shape, get_rank

from .space import Space
from .containers import Dict, Tuple
from .continuous import Continuous
from .intbox import IntBox
from .discrete import Discrete
from .bool_space import Bool


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

