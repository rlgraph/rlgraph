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

from yarl.utils.util import DictOp, dtype, get_shape, get_rank

from .space import Space
from .containers import Dict, Tuple
from .continuous import Continuous
from .intbox import IntBox
from .discrete import Discrete
from .bool import Bool


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
        for k, v in op.items():
            spec[k] = get_space_from_op(v)
        return Dict(spec)
    # a Tuple
    elif isinstance(op, tuple):
        spec = list()
        for i in op:
            spec.append(get_space_from_op(i))
        return Tuple(spec)
    # primitive Space -> infer from op dtype and shape
    else:
        # a Bool
        if op.dtype == dtype("bool"):
            return Bool()
        # a Continuous
        elif op.dtype == dtype("float"):
            return Continuous(shape=get_shape(op))
        # an IntBox/Discrete
        # TODO: see whether we can unite these two into a MultiDiscrete Space
        elif op.dtype == dtype("int"):
            if get_rank(op) == 1:
                return Discrete(n=get_shape(op)[0])
            else:
                return IntBox(low=0, high=255, shape=get_shape(op))  # low, high=dummy values

