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
from cached_property import cached_property

from yarl import YARLError
from .space import Space


class Dict(Space, OrderedDict):
    """
    A Dict space (an ordered and keyed combination of n other spaces).
    Supports nesting of other Dict spaces (or any other Space types) inside itself.
    """
    def __init__(self, spec=None, **kwargs):
        # Allow for any spec or already constructed Space to be passed in as values in the python-dict.
        # Spec may be part of kwargs.
        if spec is None:
            spec = kwargs

        dict_ = OrderedDict()
        for key in sorted(spec.keys()):
            value = spec[key]
            # value is already a Space -> keep it
            if isinstance(value, Space):
                dict_[key] = value
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                dict_[key] = Tuple(value)
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                dict_[key] = Space.from_spec(value)
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                dict_[key] = Dict(value)

        if len(dict_) == 0:
            raise YARLError("ERROR: Dict() c'tor needs a non-empty spec!")
        OrderedDict.__init__(self, dict_)

    @cached_property
    def shape(self):
        return tuple([self[key].flat_dim for key in self.keys()])

    @cached_property
    def rank(self):
        return tuple([self[key].rank for key in self.keys()])

    @cached_property
    def flat_dim(self):
        return int(np.sum([c.flat_dim for c in self.values()]))

    @cached_property
    def dtype(self):
        return OrderedDict([(key, subspace.dtype()) for key, subspace in self.items()])

    def __repr__(self):
        return "Dict({})".format([(key, self[key].__repr__()) for key in self.keys()])

    def __eq__(self, other):
        if not isinstance(other, Dict):
            return False
        return OrderedDict(self) == OrderedDict(other)

    def get_tensor_variable(self, name, is_input_feed=False, **kwargs):
        return OrderedDict([(key, subspace.get_tensor_variable(key+"/"+name, is_input_feed, **kwargs))
                            for key, subspace in self.items()])

    def sample(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return OrderedDict([(key, subspace.sample()) for key, subspace in self.items()])

    def contains(self, x):
        return isinstance(x, (OrderedDict, dict)) and all(self[key].contains(x[key]) for key in self.keys())

    @staticmethod
    def map(container, wrapper, func):
        """
        Loops through this Dict and maps all values through the given funcreturns a dict
        Args:
            container ():
            func ():

        Returns:

        """
        return dict(map(lambda item: (item[0], wrapper(item[1], func)), container.items()))


class Tuple(Space, tuple):
    def __new__(cls, *components):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]

        # Allow for any spec or already constructed Space to be passed in as values in the python-list/tuple.
        list_ = []
        for value in components:
            # value is already a Space -> keep it
            if isinstance(value, Space):
                list_.append(value)
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                list_.append(Tuple(value))
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                list_.append(Space.from_spec(value))
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                list_.append(Dict(value))

        return tuple.__new__(cls, list_)

    @cached_property
    def shape(self):
        return tuple([c.flat_dim for c in self])

    @cached_property
    def rank(self):
        return tuple([c.rank for c in self])

    @cached_property
    def flat_dim(self):
        return np.sum([c.flat_dim for c in self.components])

    @cached_property
    def dtype(self):
        return tuple([c.dtype for c in self])

    def __repr__(self):
        return "Tuple({})".format(tuple([cmp.__repr__() for cmp in self]))

    def __eq__(self, other):
        return tuple.__eq__(self, other)

    def get_tensor_variable(self, name, is_input_feed=False, **kwargs):
        return tuple([subspace.get_tensor_variable(name+"/"+i, is_input_feed, **kwargs)
                      for i, subspace in enumerate(self)])

    def sample(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return tuple(x.sample() for x in self)

    def contains(self, x):
        return isinstance(x, (tuple, list)) and len(self) == len(x) and all(c.contains(xi) for c, xi in zip(self, x))

    @staticmethod
    def map(container, wrapper, func):
        return tuple(map(lambda c: wrapper(c, func), container))


