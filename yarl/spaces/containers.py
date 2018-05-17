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
from cached_property import cached_property
import re

from yarl import YARLError
from yarl.utils.dictop import DictOp
from .space import Space


class ContainerSpace(Space):
    """
    A simple placeholder class for Spaces that contain other Spaces.
    """
    pass


class Dict(ContainerSpace, dict):
    """
    A Dict space (an ordered and keyed combination of n other spaces).
    Supports nesting of other Dict/Tuple spaces (or any other Space types) inside itself.
    """
    def __init__(self, spec=None, add_batch_rank=False, **kwargs):
        ContainerSpace.__init__(self, add_batch_rank=add_batch_rank)

        # Allow for any spec or already constructed Space to be passed in as values in the python-dict.
        # Spec may be part of kwargs.
        if spec is None:
            spec = kwargs

        dict_ = dict()
        for key in sorted(spec.keys()):
            # Keys must be strings.
            if not isinstance(key, str):
                raise YARLError("ERROR: No non-str keys allowed in a Dict-Space!")
            # Prohibit reserved characters (for flattened syntax).
            if re.search(r'[/\[\]]', key):
                raise YARLError("ERROR: Keys to Dict() must not contain '/', '[' or ']' characters!")

            value = spec[key]
            # Value is already a Space: Keep it, but maybe add batch-rank.
            if isinstance(value, Space):
                if add_batch_rank is True:
                    value.add_batch_rank(add_batch_rank=True)
                dict_[key] = value
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                dict_[key] = Tuple(*value, add_batch_rank=add_batch_rank)
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                dict_[key] = Space.from_spec(value, add_batch_rank=add_batch_rank)
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                dict_[key] = Dict(value, add_batch_rank=add_batch_rank)

        if len(dict_) == 0:
            raise YARLError("ERROR: Dict() c'tor needs a non-empty spec!")
        dict.__init__(self, dict_)

    @cached_property
    def shape(self):
        return tuple([self[key].shape for key in sorted(self.keys())])

    @cached_property
    def shape_with_batch_rank(self):
        return tuple([self[key].shape_with_batch_rank for key in sorted(self.keys())])

    @cached_property
    def rank(self):
        return tuple([self[key].rank for key in sorted(self.keys())])

    @cached_property
    def flat_dim(self):
        return int(np.sum([c.flat_dim for c in self.values()]))

    @cached_property
    def dtype(self):
        return DictOp([(key, subspace.dtype) for key, subspace in self.items()])

    def get_tensor_variable(self, name, is_input_feed=False, add_batch_rank=None, **kwargs):
        return DictOp([(key, subspace.get_tensor_variable(name + "/" + key, is_input_feed, add_batch_rank, **kwargs))
                       for key, subspace in self.items()])

    def _flatten(self, mapping, scope_, list_):
        # Iterate through this Dict.
        scope_ += "/"
        for key in sorted(self.keys()):
            self[key].flatten(mapping, scope_ + key, list_)

    #def get_initializer(self, specification):
    #    return dict([(key, subspace.get_initializer(specification)) for key, subspace in self.items()])

    def __repr__(self):
        return "Dict({})".format([(key, self[key].__repr__()) for key in self.keys()])

    def __eq__(self, other):
        if not isinstance(other, Dict):
            return False
        return dict(self) == dict(other)

    def sample(self, size=None, seed=None, horizontal=False):
        """
        Args:
            horizontal (bool): False: Sample within this container each child-space `size` times.
                True: Produce `size` single containers in an np.array of len `size`.
        """
        if seed is not None:
            np.random.seed(seed)
        if horizontal:
            return np.array([dict([(key, subspace.sample()) for key, subspace in self.items()])] * (size or 1))
        else:
            return dict([(key, subspace.sample(size=size)) for key, subspace in self.items()])

    def contains(self, sample):
        return isinstance(sample, dict) and all(self[key].contains(sample[key]) for key in self.keys())

    def add_batch_rank(self, add_batch_rank=True):
        for v in self.values():
            v.add_batch_rank(add_batch_rank)


class Tuple(ContainerSpace, tuple):
    """
    A Tuple space (an ordered sequence of n other spaces).
    Supports nesting of other container (Dict/Tuple) spaces inside itself.
    """
    def __new__(cls, *components, **kwargs):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]

        add_batch_rank = kwargs.get("add_batch_rank", False)

        # Allow for any spec or already constructed Space to be passed in as values in the python-list/tuple.
        list_ = list()
        for value in components:
            # value is already a Space -> keep it
            if isinstance(value, Space):
                if add_batch_rank is True:
                    value.add_batch_rank(add_batch_rank=True)
                list_.append(value)
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                list_.append(Tuple(*value, add_batch_rank=add_batch_rank))
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                list_.append(Space.from_spec(value, add_batch_rank=add_batch_rank))
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                list_.append(Dict(value, add_batch_rank=add_batch_rank))

        return tuple.__new__(cls, list_)

    def __init__(self, *components, **kwargs):
        add_batch_rank = kwargs.get("add_batch_rank", False)
        super(Tuple, self).__init__(add_batch_rank=add_batch_rank)

    @cached_property
    def shape(self):
        return tuple([c.shape for c in self])

    @cached_property
    def shape_with_batch_rank(self):
        return tuple([c.shape_with_batch_rank for c in self])

    @cached_property
    def rank(self):
        return tuple([c.rank for c in self])

    @cached_property
    def flat_dim(self):
        return np.sum([c.flat_dim for c in self])

    @cached_property
    def dtype(self):
        return tuple([c.dtype for c in self])

    def get_tensor_variable(self, name, is_input_feed=False, add_batch_rank=None, **kwargs):
        return tuple([subspace.get_tensor_variable(name+"/"+str(i), is_input_feed, add_batch_rank, **kwargs)
                      for i, subspace in enumerate(self)])

    def _flatten(self, mapping, scope_, list_):
        # Iterate through this Tuple.
        scope_ += "/["
        for i, component in enumerate(self):
            component.flatten(mapping, scope_ + str(i) + "]", list_)

    #def get_initializer(self, specification):
    #    return tuple([subspace.get_initializer(specification) for subspace in self])

    def __repr__(self):
        return "Tuple({})".format(tuple([cmp.__repr__() for cmp in self]))

    def __eq__(self, other):
        return tuple.__eq__(self, other)

    def sample(self, size=None, seed=None, horizontal=False):
        """
        Args:
            horizontal (bool): False: Sample within this container each child-space `size` times.
                True: Produce `size` single containers in an np.array of len `size`.
        """
        if seed is not None:
            np.random.seed(seed)
        if horizontal:
            return np.array([tuple(subspace.sample() for subspace in self)] * (size or 1))
        else:
            return tuple(x.sample(size=size) for x in self)

    def contains(self, sample):
        return isinstance(sample, (tuple, list, np.ndarray)) and len(self) == len(sample) and \
               all(c.contains(xi) for c, xi in zip(self, sample))

    def add_batch_rank(self, add_batch_rank=True):
        for v in self:
            v.add_batch_rank(add_batch_rank)

