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

import numpy as np

from collections import OrderedDict


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


class DataOpRecord(object):
    def __init__(self, op, labels=None):
        """
        Args:
            op (DataOp): The actual DataOp carried by this record.
            labels (Set[str]): A set of string-labels that are associated with the DataOp.
        """
        self.op = op
        self.labels = set(labels or [])

    def __hash__(self):
        return self.op
