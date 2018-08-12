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

from rlgraph import RLGraphError
from rlgraph.backend_system import get_backend
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.spaces import IntBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.ops import flatten_op, unflatten_op

if get_backend() == "tf":
    import tensorflow as tf


class ReShape(PreprocessLayer):
    """
    A reshaping preprocessor that takes an input and reshapes it into a new shape.
    Also supports special options for time/batch rank manipulations and complete flattening
    (including IntBox categories).
    """
    def __init__(self, new_shape=None, flatten=False, flatten_categories=True, fold_time_rank=False,
                 unfold_time_rank=False, scope="reshape", **kwargs):
        """
        Args:
            new_shape (Optional[Dict[str,Tuple[int]],Tuple[int]]): A dict of str/tuples or a single tuple
                specifying the new-shape(s) to use (for each auto key in case of a Container input Space).
                At most one of the ranks in any new_shape may be -1 to indicate flexibility in that dimension.
                NOTE: Shape does not include batch- or time-ranks. If you want to manipulate these directly, use
                the fold_time_rank/unfold_time_rank options.
            flatten (bool): Whether to simply flatten the input Space into a single rank. This does not include
                batch- or time-ranks. These can be processed separately by the other ctor options.
                If flatten is True, new_shape must be None.
            flatten_categories (bool): If `flatten` is True, whether to also flatten IntBox categories.
                Default: True.
            fold_time_rank (bool): Whether to fold the time rank into a single batch rank.
                E.g. from (None, None, 2, 3) to (None, 2, 3). Providing both `fold_time_rank` and `new_shape` is
                allowed.
            unfold_time_rank (Union[bool,int]): The size of the time rank (unfolded from the batch rank or False
                for no unfolding taking place. Providing both `unfold_time_rank` as int and `new_shape` is
                allowed. A value of True is not allowed (ReShape wouldn't know how many time steps there are to
                unfold from the batch rank).
        """
        super(ReShape, self).__init__(scope=scope, add_auto_key_as_first_param=True, **kwargs)

        assert flatten is False or new_shape is None, "ERROR: If `flatten` is True, `new_shape` must be None!"
        assert flatten_categories is False or flatten is True,\
            "ERROR: Cannot set `flatten_categories` to True and leave `flatten` as False!"
        assert unfold_time_rank is not True,\
            "ERROR: `unfold_time_rank` must not be True! Only False or an int value are allowed."
        assert fold_time_rank is False or unfold_time_rank is False
        assert isinstance(unfold_time_rank, int) or unfold_time_rank is False,\
            "ERROR: `unfold_time_rank` must be an int or False (but is {})!".format(unfold_time_rank)

        # The new shape specifications.
        self.new_shape = new_shape
        self.flatten = flatten
        self.flatten_categories = flatten_categories
        self.fold_time_rank = fold_time_rank
        self.unfold_time_rank = unfold_time_rank

        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

        # Stores the number of categories in IntBoxes.
        self.num_categories = dict()

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, single_space in space.flatten().items():
            class_ = type(single_space)

            # Determine the actual shape (not batch/time ranks).
            if self.flatten is True:
                new_shape = tuple(
                    [single_space.flat_dim_with_categories if
                     self.flatten_categories is True and type(single_space) == IntBox else single_space.flat_dim]
                )
            else:
                new_shape = self.new_shape[key] if isinstance(self.new_shape, dict) else self.new_shape

            # Check the batch/time rank options.
            if self.fold_time_rank is True:
                assert single_space.has_time_rank is True,\
                    "ERROR: ReShape trying to fold time-rank into batch-rank, but space '{}' has no time-rank!".\
                    format(single_space)
                ret[key] = class_(
                    shape=single_space.shape if new_shape is None else new_shape,
                    add_batch_rank=single_space.has_batch_rank, add_time_rank=False
                )
            # Time rank should be unfolded from batch rank with the given dimension.
            elif type(self.unfold_time_rank) == int:
                assert single_space.has_time_rank is False,\
                    "ERROR: ReShape trying to unfold time-rank from batch-rank, but space '{}' already has time-rank!".\
                    format(single_space)
                ret[key] = class_(
                    shape=single_space.shape if new_shape is None else new_shape,
                    add_batch_rank=single_space.has_batch_rank, add_time_rank=True
                )
            # Only change the actual shape (leave batch/time ranks as is).
            else:
                ret[key] = class_(shape=new_shape, add_batch_rank=single_space.has_batch_rank,
                                  add_time_rank=single_space.has_time_rank)
        return unflatten_op(ret)

    def check_input_spaces(self, input_spaces, action_space=None):
        super(ReShape, self).check_input_spaces(input_spaces, action_space)

        # Check whether our input space has-batch or not and store this information here.
        in_space = input_spaces["preprocessing_inputs"]  # type: Space

        if isinstance(in_space, IntBox):
            sanity_check_space(in_space, must_have_categories=True, num_categories=(2, 10000))

        # Store the mapped output Spaces (per flat key).
        self.output_spaces = flatten_op(self.get_preprocessed_space(in_space))

        # Check whether we have to flatten the incoming categories of an IntBox into a FloatBox with additional
        # rank (categories rank). Store the dimension of this additional rank in the `self.num_categories` dict.
        if self.flatten_categories is True:
            def mapping_func(key, space):
                if isinstance(space, IntBox):
                    # Must have global bounds (bounds valid for all axes).
                    if space.num_categories is False:
                        raise RLGraphError("ERROR: Cannot flatten categories if one of the IntBox spaces ({}={}) does "
                                           "not have global bounds (its `num_categories` is False)!".format(key, space))
                    return space.num_categories
                # No categories. Keep as is.
                return 1
            self.num_categories = in_space.flatten(mapping=mapping_func)

    def _graph_fn_apply(self, key, preprocessing_inputs):
        """
        Reshapes the input to the specified new shape.

        Args:
            preprocessing_inputs (SingleDataOp): The input to reshape.

        Returns:
            SingleDataOp: The reshaped input.
        """
        # Create a one-hot axis for the categories at the end?
        if self.num_categories[key] > 1:
            preprocessing_inputs = tf.one_hot(indices=preprocessing_inputs, depth=self.num_categories[key], axis=-1)

        new_shape = self.output_spaces[key].get_shape(
            with_batch_rank=-1, with_time_rank=self.unfold_time_rank if type(self.unfold_time_rank) == int else -1
        )
        # Dynamic workaround: If both batch and time rank must be left alone, get these two dynamically.
        if len(new_shape) > 2 and new_shape[0] == -1 and new_shape[1] == -1:
            dynamic_shape = tf.shape(preprocessing_inputs)
            new_shape = (dynamic_shape[0], dynamic_shape[1]) + new_shape[2:]

        if self.backend == "python" or get_backend() == "python":
            return np.reshape(preprocessing_inputs, newshape=new_shape)
        elif get_backend() == "tf":
            return tf.reshape(tensor=preprocessing_inputs, shape=new_shape, name="reshaped")
