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
import copy

from rlgraph.utils.specifiable import Specifiable


class Space(Specifiable):
    """
    Space class (based on and compatible with openAI Spaces).
    Provides a classification for state-, action-, reward- and other spaces.
    """
    def __init__(self, add_batch_rank=False, add_time_rank=False, time_major=False):
        """
        Args:
            add_batch_rank (bool): Whether to always add a batch rank at the 0th (or 1st) position when creating
                variables from this Space.
            add_time_rank (bool): Whether to always add a time rank at the 1st (or 0th) position when creating
                variables from this Space.
            time_major (bool): Whether the time rank should come before the batch rank. Not important if one
                of these ranks (or both) does not exist.
        """
        super(Space, self).__init__()

        self._shape = None

        self.has_batch_rank = None
        self.has_time_rank = None
        self.time_major = None

        self._add_batch_rank(add_batch_rank)
        self._add_time_rank(add_time_rank, time_major)

    def _add_batch_rank(self, add_batch_rank=False):
        """
        Changes the add_batch_rank property of this Space (and of all child Spaces in a ContainerSpace).

        Args:
            add_batch_rank (bool): Whether this Space (and all child Spaces in a ContainerSpace) should have a
                batch rank.
        """
        self.has_batch_rank = add_batch_rank

    def _add_time_rank(self, add_time_rank=False, time_major=False):
        """
        Changes the add_time_rank property of this Space (and of all child Spaces in a ContainerSpace).

        Args:
            add_time_rank (bool): Whether this Space (and all child Spaces in a ContainerSpace) should have a
                time rank.
            time_major (bool): Whether the time rank should come before the batch rank. Not important if no batch rank
                exists.
        """
        self.has_time_rank = add_time_rank
        self.time_major = time_major

    def with_extra_ranks(self, add_batch_rank=True, add_time_rank=True, time_major=False):
        """
        Returns a deepcopy of this Space, but with `has_batch_rank` and `has_time_rank`
        set to the provided value. Use None to leave whatever value this Space has already.

        Args:
            add_batch_rank (Optional[bool]): If True or False, set the `has_batch_rank` property of the new Space
                to this value. Use None to leave the property as is.
            add_time_rank (Optional[bool]): If True or False, set the `has_time_rank` property of the new Space
                to this value. Use None to leave the property as is.
            time_major (Optional[bool]): Whether the time-rank should be the 0th rank (instead of the 1st by default).
                Not important if either batch_rank or time_rank are not set. Use None to leave the property as is.

        Returns:
            Space: The deepcopy of this Space, but with `has_batch_rank` set to True.
        """
        ret = copy.deepcopy(self)
        ret._add_batch_rank(add_batch_rank if add_batch_rank is not None else self.has_batch_rank)
        ret._add_time_rank(
            add_time_rank if add_time_rank is not None else self.has_time_rank,
            time_major if time_major is not None else self.time_major
        )
        return ret

    def with_batch_rank(self, add_batch_rank=True):
        """
        Returns a deepcopy of this Space, but with `has_batch_rank` set to the provided value.

        Args:
            add_batch_rank (Union[bool,int]): The fixed size of the batch-rank or True or False.

        Returns:
            Space: The deepcopy of this Space, but with `has_batch_rank` set to True.
        """
        return self.with_extra_ranks(add_batch_rank=add_batch_rank, add_time_rank=None)

    def with_time_rank(self, add_time_rank=True):
        """
        Returns a deepcopy of this Space, but with `has_time_rank` set to the provided value.

        Args:
            add_time_rank (Union[bool,int]): The fixed size of the time-rank or True or False.

        Returns:
            Space: The deepcopy of this Space, but with `has_time_rank` set to True.
        """
        return self.with_extra_ranks(add_batch_rank=None, add_time_rank=add_time_rank)

    def force_batch(self, samples):
        """
        Makes sure that `samples` is always returned with a batch rank no matter whether
        it already has one or not (in which case this method returns a batch of 1) or
        whether this Space has a batch rank or not.

        Args:
            samples (any): The samples to be batched. If already batched, return as-is.

        Returns:
            any: The batched sample.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """
        Returns:
            tuple: The shape of this Space as a tuple. Without batch or time ranks.
        """
        return self._shape

    def get_shape(self, with_batch_rank=False, with_time_rank=False, time_major=None, **kwargs):
        """
        Returns the shape of this Space as a tuple with certain additional ranks at the front (batch) or the back
        (e.g. categories).

        Args:
            with_batch_rank (Union[bool,int]): Whether to include a possible batch-rank as `None` at 0th (or 1st)
                position. If `with_batch_rank` is an int (e.g. -1), the possible batch-rank is returned as that number
                (instead of None) at the 0th (or 1st if time_major is True) position.
                Default: False.

            with_time_rank (Union[bool,int]): Whether to include a possible time-rank as `None` at 1st (or 0th)
                position. If `with_time_rank` is an int, the possible time-rank is returned as that number
                (instead of None) at the 1st (or 0th if time_major is True) position.
                Default: False.

            time_major (bool): Overwrites `self.time_major` if not None. Default: None (use `self.time_major`).

        Returns:
            tuple: The shape of this Space as a tuple.
        """
        raise NotImplementedError

    @property
    def rank(self):
        """
        Returns:
            int: The rank of the Space not including batch- or time-ranks
            (e.g. 3 for a space with shape=(10, 7, 5)).
        """
        return len(self.shape)

    @property
    def flat_dim(self):
        """
        Returns:
            int: The length of a flattened vector derived from this Space.
        """
        raise NotImplementedError

    def get_variable(self, name, is_input_feed=False, add_batch_rank=None, add_time_rank=None,
                     time_major=False, is_python=False, local=False, **kwargs):
        """
        Returns a backend-specific variable/placeholder that matches the space's shape.

        Args:
            name (str): The name for the variable.

            is_input_feed (bool): Whether the returned object should be an input placeholder,
                instead of a full variable.

            add_batch_rank (Optional[bool,int]): If True, will add a 0th (or 1st) rank (None) to
                the created variable. If it is an int, will add that int (-1 means None).
                If None, will use the Space's default value: `self.has_batch_rank`.
                Default: None.

            add_time_rank (Optional[bool,int]): If True, will add a 1st (or 0th) rank (None) to
                the created variable. If it is an int, will add that int (-1 means None).
                If None, will use the Space's default value: `self.has_time_rank`.
                Default: None.

            time_major (bool): Only relevant if both `add_batch_rank` and `add_time_rank` are True.
                Will make the time-rank the 0th rank and the batch-rank the 1st rank.
                Otherwise, batch-rank will be 0th and time-rank will be 1st.
                Default: False.

            is_python (bool): Whether to create a python-based variable (list) or a backend-specific one.

            local (bool): Whether the variable must not be shared across the network.
                Default: False.

        Keyword Args:
            To be passed on to backend-specific methods (e.g. trainable, initializer, etc..).

        Returns:
            any: A Tensor Variable/Placeholder.
        """
        raise NotImplementedError

    def flatten(self, mapping=None, custom_scope_separator='/', scope_separator_at_start=False,
                scope_=None, list_=None):
        """
        A mapping function to flatten this Space into an OrderedDict whose only values are
        primitive (non-container) Spaces. The keys are created automatically from Dict keys and
        Tuple indexes.

        Args:
            mapping (Optional[callable]): A mapping function that takes a flattened auto-generated key and a primitive
                Space and converts the primitive Space to something else. Default is pass through.

            custom_scope_separator (str): The separator to use in the returned dict for scopes.
                Default: '/'.

            scope_separator_at_start (bool): Whether to add the scope-separator also at the beginning.
                Default: False.

            scope\_ (Optional[str]): For recursive calls only. Used for automatic key generation.

            list\_ (Optional[list]): For recursive calls only. The list so far.

        Returns:
            OrderedDict: The OrderedDict using auto-generated keys and containing only primitive Spaces
                (or whatever the mapping function maps the primitive Spaces to).
        """
        # default: no mapping
        if mapping is None:
            def mapping(key, x):
                return x

        # Are we in the non-recursive (first) call?
        ret = False
        if list_ is None:
            list_ = []
            ret = True
            scope_ = ""

        self._flatten(mapping, custom_scope_separator, scope_separator_at_start, scope_, list_)

        # Non recursive (first) call -> Return the final FlattenedDataOp.
        if ret:
            return OrderedDict(list_)

    def _flatten(self, mapping, custom_scope_separator, scope_separator_at_start, scope_, list_):
        """
        Base implementation. May be overridden by ContainerSpace classes.
        Simply sends `self` through the mapping function.

        Args:
            mapping (callable): The mapping function to use on a primitive (non-container) Space.

            custom_scope_separator (str): The separator to use in the returned dict for scopes.
                Default: '/'.

            scope_separator_at_start (bool): Whether to add the scope-separator also at the beginning.
                Default: False.

            scope\_ (str): The flat-key to use to store the mapped result in list_.
            list\_ (list): The list to append the mapped results to (under key=`scope_`).
        """
        list_.append(tuple([scope_, mapping(scope_, self)]))

    def __repr__(self):
        return "Space(shape=" + str(self.shape) + ")"

    def __eq__(self, other):
        raise NotImplementedError

    def sample(self, size=None, fill_value=None):
        """
        Uniformly randomly samples an element from this space. This is for testing purposes, e.g. to simulate
        a random environment.

        Args:
            size (Optional[int]): The number of samples or batch size to sample.
                If size is > 1: Returns a batch of size samples with the 0th rank being the batch rank
                (even if `self.has_batch_rank` is False).
                If size is None or (1 and self.has_batch_rank is False): Returns a single sample w/o batch rank.
                If size is 1 and self.has_batch_rank is True: Returns a single sample w/ the batch rank.

            fill_value (Optional[any]): The number or initializer specifier to fill the sample. Can be used to create
                a (non-random) sample with a certain fill value in all elements.
                TODO: support initializer spec-strings like 'normal', 'truncated_normal', etc..

        Returns:
            any: The sampled element(s).
        """
        raise NotImplementedError

    def zeros(self, size=None):
        """
        Args:
            size (Optional): Same as `Space.sample()`.

        Returns:
            np.ndarray: `size` zero samples where all values are zero and have the correct type.
        """
        raise NotImplementedError

    def _get_np_shape(self, num_samples=None):
        """
        Helper to determine, which shape one should pass to the numpy random funcs for sampling from a Space.
        Depends on `num_samples`, the `shape` of this Space and the `self.has_batch_rank/has_time_rank` settings.

        Args:
            num_samples (Optional[int,Tuple[int,int]]): Number of samples to pull. If None or 0, pull 1 sample, but
                without batch/time rank (no matter what the value of `self.has_batch_rank` is).
                If tuple given, use the given values as time/batch ranks.

        Returns:
            Tuple[int]: Shape to use for numpy random sampling.
        """
        # No extra batch/time rank.
        if num_samples is None or (
                num_samples == () or num_samples == 1 and not self.has_batch_rank and not self.has_time_rank
        ):
            if len(self.shape) == 0:
                return None
            else:
                return self.shape
        # With one extra rank.
        elif isinstance(num_samples, int):
            return (num_samples,) + self.shape
        # With two extra ranks (given as list or tuple).
        else:
            assert isinstance(num_samples, (tuple, list)) and len(num_samples) == 2,\
                "ERROR: num_samples must be int or tuple/list of two ints, but is '{}'!".format(num_samples)
            return tuple(num_samples) + self.shape

    def contains(self, sample):
        """
        Checks whether this space contains the given sample. This is more for testing purposes.

        Args:
            sample: The element to check.

        Returns:
            bool: Whether sample is a valid member of this space.
        """
        raise NotImplementedError
