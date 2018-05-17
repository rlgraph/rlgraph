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

from yarl import Specifiable, backend, YARLError
from yarl.utils.util import force_tuple
# TODO: make this backend-dependent
import tensorflow as tf


class Space(Specifiable):
    """
    Space class (based and compatible with openAI).
    Provides a classification for state-, action- and reward spaces.
    """
    def __init__(self, add_batch_rank=False):
        """
        Args:
            add_batch_rank (bool): Whether to always add a batch rank at the 0th position when creating
                variables from this Space.
        """
        self.has_batch_rank = None
        self.batch_rank_tuple = None
        self.add_batch_rank(add_batch_rank)

    @property
    def shape(self):
        """
        Returns:
            tuple: The shape of this Space as a tuple.
        """
        raise NotImplementedError

    @property
    def shape_with_batch_rank(self):
        """
        Returns:
            tuple: The shape of this Space as a tuple including a possible batch_rank as the 0th rank.
        """
        raise NotImplementedError

    @property
    def rank(self):
        """
        Returns:
            int: The rank of the Space (e.g. 3 for a space with shape=(10, 7, 5)).
        """
        return len(self.shape)

    @property
    def flat_dim(self):
        """
        Returns:
            int: The dimension of the flattened vector of the tensor representation.
        """
        raise NotImplementedError

    @property
    def dtype(self):
        """
        Returns:
            str: The dtype (as string) of this Space.
                Can be converted to tf/np/python dtypes via the misc.utils.dtype function.
        """
        raise NotImplementedError

    def get_tensor_variable(self, name, is_input_feed=False, add_batch_rank=None, **kwargs):
        """
        Returns a backend-specific variable/placeholder that matches the space's shape.

        Args:
            name (str): The name for the variable.
            is_input_feed (bool): Whether the returned object should be an input placeholder,
                instead of a full variable.
            add_batch_rank (Optional[bool,int]): If True, will add a 0th rank (None) to
                the created variable. If it is an int, will add that int (-1 means None).
                If None, will use the Space's default value: `self.add_batch_rank`.
                Default: None.

        Keyword Args:
            To be passed on to backend-specific methods.

        Returns:
            any: A Tensor Variable/Placeholder.
        """
        add_batch_rank = self.has_batch_rank if add_batch_rank is None else add_batch_rank
        batch_rank = () if add_batch_rank is False else (None,) if add_batch_rank is True else (add_batch_rank,)
        shape = tuple(batch_rank + self.shape)
        if backend() == "tf":
            if is_input_feed:
                return tf.placeholder(dtype=self.dtype, shape=shape, name=name)
            else:
                return tf.get_variable(name, shape=shape, dtype=self.dtype, **kwargs)
        else:
            raise YARLError("ERROR: Pytorch not supported yet!")

    #def get_initializer(self, specification):
    #    """
    #    Returns a backend-specific initializer object that matches the space's shape.
    #
    #    Args:
    #        specification (any): The specification to be passed into
    #
    #    Returns:
    #        An Initializer object.
    #    """
    #    return None  # optional

    def flatten(self, mapping=None, scope_=None, list_=None):
        """
        A mapping function to flatten this Space into a flat OrderedDict whose only values are
        primitive (non-container) Spaces. The keys are created automatically from Dict keys and
        Tuple indexes.

        Args:
            mapping (Optional[callable]): A mapping function that takes a primitive Space and converts it
                to something else. Default is pass through.
            scope_ (Optional[str]): For recursive calls only. Used for automatic key generation.
            list_ (Optional[list]): For recursive calls only. The list so far.

        Returns:
            OrderedDict: The flattened OrderedDict containing only primitive Spaces.
        """
        # default: no mapping
        if mapping is None:
            def mapping(key, x):
                return x

        # Are we in the non-recursive (first) call?
        ret = False
        if list_ is None:
            list_ = list()
            ret = True
            scope_ = ""

        self._flatten(mapping, scope_, list_)

        # Non recursive (first) call -> Return the final OrderedDict.
        if ret:
            return OrderedDict(list_)

    def _flatten(self, mapping, scope_, list_):
        """
        Base implementation. May be overridden by ContainerSpace classes.
        Simply sends self through the mapping function.

        Args:
            mapping (callable): The mapping function to use on a primitive (non-container) Space.
            scope_ (str): The key to use to store the mapped result in list_ (which will be converted into
                an OrderedDict at the very end).
            list_ (list): The list to append the mapped results to (under key=`scope_`).
        """
        list_.append(tuple([scope_, mapping(scope_, self)]))

    def __repr__(self):
        return "Space(shape=" + str(self.shape) + ")"

    def __eq__(self, other):
        raise NotImplementedError

    def sample(self, size=None, seed=None):
        """
        Uniformly randomly samples an element from this space. This is more for testing purposes, e.g. to simulate
        a random environment.

        Args:
            size (Union[int,iterable,tuple]): Output shape of the sample.
            seed (Optional[int]): The random seed to use.
        Returns:
            any: The sampled element.
        """
        raise NotImplementedError

    def _get_np_shape(self, num_samples=None):
        """
        Helper to determine, which shape one should pass to the numpy random funcs for sampling from a Space.
        Depends on num_samples, the shape of this Space and the add_batch_rank setting.

        Args:
            num_samples (Optional[int]): Number of samples to pull. If None or 0, pull 1 sample, but without batch rank
                (no matter what the value of self.add_batch_rank is).

        Returns:
            Tuple[int]: Shape to use for numpy random sampling.
        """
        # No batch rank.
        if not num_samples or (num_samples == 1 and not self.add_batch_rank):
            if len(self.shape) == 0:
                return None
            else:
                return self.shape
        # With batch rank.
        else:
            return tuple((num_samples,) + self.shape)

    def contains(self, sample):
        """
        Checks whether this space contains the given sample. This is more for testing purposes.

        Args:
            sample: The element to check.

        Returns:
            bool: Whether sample is a valid member of this space.
        """
        raise NotImplementedError

    def add_batch_rank(self, add_batch_rank=True):
        """
        Helper method for ContainerSpaces to add this feature later (after this Space has been constructed).

        Args:
            add_batch_rank (bool): See c'tor.
        """
        self.has_batch_rank = add_batch_rank
        self.batch_rank_tuple = (None,) if self.has_batch_rank else ()

