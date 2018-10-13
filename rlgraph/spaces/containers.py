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

import numpy as np
import re

from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.utils.ops import DataOpDict, DataOpTuple, FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE
from rlgraph.spaces.space import Space


class ContainerSpace(Space):
    """
    A simple placeholder class for Spaces that contain other Spaces.
    """
    def sample(self, size=None, horizontal=False):
        """
        Child classes must overwrite this one again with support for the `horizontal` parameter.

        Args:
            horizontal (bool): False: Within this container, sample each child-space `size` times.
                True: Produce `size` single containers in an np.array of len `size`.
        """
        raise NotImplementedError


class Dict(ContainerSpace, dict):
    """
    A Dict space (an ordered and keyed combination of n other spaces).
    Supports nesting of other Dict/Tuple spaces (or any other Space types) inside itself.
    """
    def __init__(self, spec=None, **kwargs):
        add_batch_rank = kwargs.pop("add_batch_rank", False)
        add_time_rank = kwargs.pop("add_time_rank", False)
        time_major = kwargs.pop("time_major", False)

        ContainerSpace.__init__(self, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major)

        # Allow for any spec or already constructed Space to be passed in as values in the python-dict.
        # Spec may be part of kwargs.
        if spec is None:
            spec = kwargs

        dict_ = {}
        for key in sorted(spec.keys()):
            # Keys must be strings.
            if not isinstance(key, str):
                raise RLGraphError("ERROR: No non-str keys allowed in a Dict-Space!")
            # Prohibit reserved characters (for flattened syntax).
            if re.search(r'/|{}\d+{}'.format(FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE), key):
                raise RLGraphError("ERROR: Key to Dict must not contain '/' or '{}\d+{}'! Is {}.".
                                   format(FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE, key))
            value = spec[key]
            # Value is already a Space: Copy it (to not affect original Space) and maybe add/remove batch/time-ranks.
            if isinstance(value, Space):
                w_batch_w_time = value.with_extra_ranks(add_batch_rank, add_time_rank, time_major)
                dict_[key] = w_batch_w_time
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                dict_[key] = Tuple(
                    *value, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major
                )
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                dict_[key] = Space.from_spec(
                    value, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major
                )
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                dict_[key] = Dict(
                    value, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major
                )

        dict.__init__(self, dict_)

    def _add_batch_rank(self, add_batch_rank=False):
        super(Dict, self)._add_batch_rank(add_batch_rank)
        for v in self.values():
            v._add_batch_rank(add_batch_rank)

    def _add_time_rank(self, add_time_rank=False, time_major=False):
        super(Dict, self)._add_time_rank(add_time_rank, time_major)
        for v in self.values():
            v._add_time_rank(add_time_rank, time_major)

    def force_batch(self, samples):
        return dict([(key, self[key].force_batch(samples[key])) for key in sorted(self.keys())])

    @property
    def shape(self):
        return tuple([self[key].shape for key in sorted(self.keys())])

    def get_shape(self, with_batch_rank=False, with_time_rank=False, time_major=None, with_category_rank=False):
        return tuple([self[key].get_shape(
            with_batch_rank=with_batch_rank, with_time_rank=with_time_rank, time_major=time_major,
            with_category_rank=with_category_rank
        ) for key in sorted(self.keys())])

    @property
    def rank(self):
        return tuple([self[key].rank for key in sorted(self.keys())])

    @property
    def flat_dim(self):
        return int(np.sum([c.flat_dim for c in self.values()]))

    @property
    def dtype(self):
        return DataOpDict([(key, subspace.dtype) for key, subspace in self.items()])

    def get_variable(self, name, is_input_feed=False, add_batch_rank=None, add_time_rank=None, time_major=None,
                     **kwargs):
        return DataOpDict(
            [(key, subspace.get_variable(
                name + "/" + key, is_input_feed=is_input_feed, add_batch_rank=add_batch_rank,
                add_time_rank=add_time_rank, time_major=time_major, **kwargs
            )) for key, subspace in self.items()]
        )

    def _flatten(self, mapping, custom_scope_separator, scope_separator_at_start, scope_, list_):
        # Iterate through this Dict.
        scope_ += custom_scope_separator if len(scope_) > 0 or scope_separator_at_start else ""
        for key in sorted(self.keys()):
            self[key].flatten(mapping, custom_scope_separator, scope_separator_at_start, scope_ + key, list_)

    def sample(self, size=None, horizontal=False):
        if horizontal:
            return np.array([{key: self[key].sample() for key in sorted(self.keys())}] * (size or 1))
        else:
            return {key: self[key].sample(size=size) for key in sorted(self.keys())}

    def zeros(self, size=None):
        return DataOpDict([(key, subspace.zeros(size=size)) for key, subspace in self.items()])

    def contains(self, sample):
        return isinstance(sample, dict) and all(self[key].contains(sample[key]) for key in self.keys())

    def __repr__(self):
        return "Dict({})".format([(key, self[key].__repr__()) for key in self.keys()])

    def __eq__(self, other):
        if not isinstance(other, Dict):
            return False
        return dict(self) == dict(other)


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
        add_time_rank = kwargs.get("add_time_rank", False)
        time_major = kwargs.get("time_major", False)

        # Allow for any spec or already constructed Space to be passed in as values in the python-list/tuple.
        list_ = list()
        for value in components:
            # Value is already a Space: Copy it (to not affect original Space) and maybe add/remove batch-rank.
            if isinstance(value, Space):
                list_.append(value.with_extra_ranks(add_batch_rank, add_time_rank, time_major))
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                list_.append(
                    Tuple(*value, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major)
                )
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                list_.append(Space.from_spec(
                    value, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major
                ))
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                list_.append(Dict(
                    value, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major
                ))

        return tuple.__new__(cls, list_)

    def __init__(self, *components, **kwargs):
        add_batch_rank = kwargs.get("add_batch_rank", False)
        add_time_rank = kwargs.get("add_time_rank", False)
        time_major = kwargs.get("time_major", False)
        super(Tuple, self).__init__(add_batch_rank=add_batch_rank, add_time_rank=add_time_rank, time_major=time_major)

    def _add_batch_rank(self, add_batch_rank=False):
        super(Tuple, self)._add_batch_rank(add_batch_rank)
        for v in self:
            v._add_batch_rank(add_batch_rank)

    def _add_time_rank(self, add_time_rank=False, time_major=False):
        super(Tuple, self)._add_time_rank(add_time_rank, time_major)
        for v in self:
            v._add_time_rank(add_time_rank, time_major)

    def force_batch(self, samples):
        return tuple([c.force_batch(samples[i]) for i, c in enumerate(self)])

    @property
    def shape(self):
        return tuple([c.shape for c in self])

    def get_shape(self, with_batch_rank=False, with_time_rank=False, time_major=None, with_category_rank=False):
        return tuple([c.get_shape(
            with_batch_rank=with_batch_rank, with_time_rank=with_time_rank, time_major=time_major,
            with_category_rank=with_category_rank
        ) for c in self])

    @property
    def rank(self):
        return tuple([c.rank for c in self])

    @property
    def flat_dim(self):
        return np.sum([c.flat_dim for c in self])

    @property
    def dtype(self):
        return DataOpTuple([c.dtype for c in self])

    def get_variable(self, name, is_input_feed=False, add_batch_rank=None, add_time_rank=None, time_major=None,
                     **kwargs):
        return DataOpTuple(
            [subspace.get_variable(
                name+"/"+str(i), is_input_feed=is_input_feed, add_batch_rank=add_batch_rank,
                add_time_rank=add_time_rank, time_major=time_major, **kwargs
            ) for i, subspace in enumerate(self)]
        )

    def _flatten(self, mapping, custom_scope_separator, scope_separator_at_start, scope_, list_):
        # Iterate through this Tuple.
        scope_ += (custom_scope_separator if len(scope_) > 0 or scope_separator_at_start else "") + FLAT_TUPLE_OPEN
        for i, component in enumerate(self):
            component.flatten(
                mapping, custom_scope_separator, scope_separator_at_start, scope_ + str(i) + FLAT_TUPLE_CLOSE, list_
            )

    def sample(self, size=None, horizontal=False):
        if horizontal:
            return np.array([tuple(subspace.sample() for subspace in self)] * (size or 1))
        else:
            return tuple(x.sample(size=size) for x in self)

    def zeros(self, size=None):
        return tuple([c.zeros(size=size) for i, c in enumerate(self)])

    def contains(self, sample):
        return isinstance(sample, (tuple, list, np.ndarray)) and len(self) == len(sample) and \
               all(c.contains(xi) for c, xi in zip(self, sample))

    def __repr__(self):
        return "Tuple({})".format(tuple([cmp.__repr__() for cmp in self]))

    def __eq__(self, other):
        return tuple.__eq__(self, other)
