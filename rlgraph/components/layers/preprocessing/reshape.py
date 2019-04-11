# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.components.layers.preprocessing.preprocess_layer import PreprocessLayer
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.spaces.space_utils import sanity_check_space, get_space_from_op
from rlgraph.utils import pytorch_one_hot
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.numpy import one_hot
from rlgraph.utils.ops import unflatten_op, FLATTEN_SCOPE_PREFIX

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class ReShape(PreprocessLayer):
    """
    A reshaping preprocessor that takes an input and reshapes it into a new shape.
    Also supports special options for time/batch rank manipulations and complete flattening
    (including IntBox categories).
    """
    def __init__(self, new_shape=None, flatten=False, flatten_categories=None, fold_time_rank=False,
                 unfold_time_rank=False, time_major=None, scope=None, **kwargs):
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

            flatten_categories (Union[Dict[str,int],int]): Only important if `flatten` is True and incoming space is
                an IntBox. Specifies, how to also flatten IntBox categories by giving the exact number of int
                categories generally or by flat-dict key.
                Default: None.

            fold_time_rank (bool): Whether to fold the time rank into a single batch rank.
                E.g. from (None, None, 2, 3) to (None, 2, 3). Providing both `fold_time_rank` (True) and
                `new_shape` is allowed.

            unfold_time_rank (Union[bool,int]): Whether to unfold the time rank from a currently common batch+time-rank.
                The exact size of the time rank to unfold is either directly provided or determined automatically via
                the original sample.
                Providing both `unfold_time_rank` (True) and `new_shape` is allowed.

            time_major (Optional[bool]): Only used if not None and if unfold_time_rank is True. Specifies whether the
                time rank should come before the batch rank after unfolding.
        """
        scope = scope or (
            "reshape-fold" if fold_time_rank else "reshape-unfold" if unfold_time_rank else
            "reshape-flatten" if flatten is True else "reshape"
        )
        super(ReShape, self).__init__(space_agnostic=True, scope=scope, **kwargs)

        assert flatten is False or new_shape is None, "ERROR: If `flatten` is True, `new_shape` must be None!"
        assert not fold_time_rank or not unfold_time_rank,\
            "ERROR: Can only either fold or unfold the time-rank! Both `fold_time_rank` and `unfold_time_rank` " \
            "cannot be True at the same time."

        # The new shape specifications.
        self.new_shape = new_shape
        self.flatten = flatten
        self.flatten_categories = flatten_categories
        self.fold_time_rank = fold_time_rank
        self.unfold_time_rank = unfold_time_rank
        self.time_major = time_major

    def get_preprocessed_space(self, space):
        ret = {}
        for key, single_space in space.flatten().items():
            class_ = type(single_space)

            # Determine the actual shape (not batch/time ranks).
            if self.flatten is True:
                if type(single_space) == IntBox and self.flatten_categories is not False:
                    assert self.flatten_categories is not None,\
                        "ERROR: `flatten_categories` must not be None if `flatten` is True and input is IntBox!"
                    new_shape = (self.get_num_categories(key, single_space),)
                    class_ = FloatBox
                else:
                    new_shape = (single_space.flat_dim,)
            else:
                new_shape = self.new_shape[key] if isinstance(self.new_shape, dict) else self.new_shape

            # Check the batch/time rank options.
            if self.fold_time_rank is True:
                sanity_check_space(single_space, must_have_batch_rank=True, must_have_time_rank=True)
                ret[key] = class_(
                    shape=single_space.shape if new_shape is None else new_shape,
                    add_batch_rank=True, add_time_rank=False
                )
            # Time rank should be unfolded from batch rank with the given dimension.
            elif self.unfold_time_rank:
                sanity_check_space(single_space, must_have_batch_rank=True, must_have_time_rank=False)
                ret[key] = class_(
                    shape=single_space.shape if new_shape is None else new_shape,
                    add_batch_rank=True, add_time_rank=True,
                    time_major=self.time_major if self.time_major is not None else False
                )
            # Only change the actual shape (leave batch/time ranks as is).
            else:
                time_major = single_space.time_major
                ret[key] = class_(shape=single_space.shape if new_shape is None else new_shape,
                                  add_batch_rank=single_space.has_batch_rank,
                                  add_time_rank=single_space.has_time_rank, time_major=time_major)
        ret = unflatten_op(ret)
        return ret

    def get_num_categories(self, key, single_space):
        if self.flatten_categories is True and isinstance(single_space, IntBox):
            num_categories = single_space.flat_dim_with_categories
        elif isinstance(self.flatten_categories, dict):
            if key.startswith(FLATTEN_SCOPE_PREFIX):
                key = key[1:]
            num_categories = self.flatten_categories.get(key, 1)
        else:
            num_categories = self.flatten_categories
        return num_categories

    @rlgraph_api(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_call(self, key, inputs, input_before_time_rank_folding=None):
        """
        Reshapes the input to the specified new shape.

        Args:
            inputs (SingleDataOp): The input to reshape.
            input_before_time_rank_folding (Optional[SingleDataOp]): The original input (before!) the time-rank had
                been folded (this was done in a different ReShape Component).
                Used to figure out the exact time-rank dimension to unfold iff `self.unfold_time_rank` is True.

        Returns:
            SingleDataOp: The reshaped input.
        """
        assert self.unfold_time_rank is not True or input_before_time_rank_folding is not None

        if self.backend == "python" or get_backend() == "python":
            # Create a one-hot axis for the categories at the end?
            num_categories = self.get_num_categories(key, get_space_from_op(inputs))
            if num_categories and num_categories > 1:
                inputs = one_hot(inputs, depth=num_categories)

            if self.unfold_time_rank:
                new_shape = [-1, -1] + list(inputs.shape[1:])
                if type(self.unfold_time_rank) == int:
                    new_shape[0 if self.time_major else 1] = self.unfold_time_rank
                new_shape = tuple(new_shape)
            elif self.fold_time_rank:
                new_shape = (-1,) + inputs.shape[2:]
            else:
                new_shape = self.get_preprocessed_space(get_space_from_op(inputs)).get_shape(
                    with_batch_rank=-1, with_time_rank=-1
                )

            # Dynamic new shape inference:
            # If both batch and time rank must be left alone OR the time rank must be unfolded from a currently common
            # batch+time 0th rank, get these two dynamically.
            if len(inputs.shape) > 2 and new_shape[0] == -1 and new_shape[1] == -1:
                # Time rank unfolding. Get the time rank from original input.
                if self.unfold_time_rank is True:
                    original_shape = input_before_time_rank_folding.shape
                    new_shape = (original_shape[0], original_shape[1]) + new_shape[2:]
                # No time-rank unfolding, but we do have both batch- and time-rank.
                else:
                    input_shape = inputs.shape
                    # Batch and time rank stay as is.
                    new_shape = (input_shape[0], input_shape[1]) + new_shape[2:]

            return np.reshape(inputs, newshape=new_shape)

        elif get_backend() == "pytorch":
            # Create a one-hot axis for the categories at the end?
            num_categories = self.get_num_categories(key, get_space_from_op(inputs))
            if num_categories and num_categories > 1:
                inputs = pytorch_one_hot(inputs, depth=num_categories)

            if self.unfold_time_rank:
                new_shape = [-1, -1] + list(inputs.shape[1:])
                if type(self.unfold_time_rank) == int:
                    new_shape[0 if self.time_major else 1] = self.unfold_time_rank
                new_shape = tuple(new_shape)
            elif self.fold_time_rank:
                new_shape = (-1,) + inputs.shape[2:]
            else:
                new_shape = self.get_preprocessed_space(get_space_from_op(inputs)).get_shape(
                    with_batch_rank=-1, with_time_rank=-1
                )

            # Dynamic new shape inference:
            # If both batch and time rank must be left alone OR the time rank must be unfolded from a currently common
            # batch+time 0th rank, get these two dynamically.
            if len(new_shape) > 2 and new_shape[0] == -1 and new_shape[1] == -1:
                # Time rank unfolding. Get the time rank from original input.
                if self.unfold_time_rank is True:
                    original_shape = input_before_time_rank_folding.shape
                    new_shape = (original_shape[0], original_shape[1]) + new_shape[2:]
                # No time-rank unfolding, but we do have both batch- and time-rank.
                else:
                    input_shape = inputs.shape
                    # Batch and time rank stay as is.
                    new_shape = (input_shape[0], input_shape[1]) + new_shape[2:]

            # print("Reshaping input of shape {} to new shape {} (flatten = {})".format(inputs.shape,
            #                                                                           new_shape, self.flatten))

            old_size = np.prod(list(inputs.shape))
            new_size = np.prod(new_shape)

            # The problem here is the following: Input has dim e.g. [4, 256, 1, 1]
            # -> If shape inference in spaces failed, output dim is not correct -> reshape will attempt
            # something like reshaping to [256].
            if self.flatten and inputs.dim() > 1:
                flattened_shape_without_batchrank = np.prod(inputs.shape[1:])
                flattened_shape = (inputs.shape[0],) + (flattened_shape_without_batchrank,)
                return torch.reshape(inputs, flattened_shape)
            # If new shape does not fit into old shape, batch inference failed -> try to restore:
            # Equal except batch rank -> return as is:
            elif old_size != new_size:
                if tuple(inputs.shape[1:]) == new_shape:
                    return inputs
                else:
                    # Attempt to rescue reshape by combining new shape with batch dim.
                    full_new_shape = (inputs.shape[0],) + new_shape
                    return torch.reshape(inputs, full_new_shape)
            else:
                return torch.reshape(inputs, new_shape)

        elif get_backend() == "tf":
            # Create a one-hot axis for the categories at the end?
            space = get_space_from_op(inputs)
            num_categories = self.get_num_categories(key, space)
            if num_categories and num_categories > 1:
                inputs_ = tf.one_hot(
                    inputs, depth=num_categories, axis=-1, dtype="float32"
                )
                if hasattr(inputs, "_batch_rank"):
                    inputs_._batch_rank = inputs._batch_rank
                if hasattr(inputs, "_time_rank"):
                    inputs_._time_rank = inputs._time_rank
                inputs = inputs_

            if self.fold_time_rank:
                new_shape = (-1,) + tuple(inputs.shape.as_list()[2:])
            else:
                time_rank = -1
                if type(self.unfold_time_rank) == int:
                    time_rank = self.unfold_time_rank

                new_shape = self.get_preprocessed_space(get_space_from_op(inputs)).get_shape(
                    with_batch_rank=-1, with_time_rank=time_rank
                )

            # Dynamic new shape inference:
            # If both batch and time rank must be left alone OR the time rank must be unfolded from a currently common
            # batch+time 0th rank, get these two dynamically.
            if len(new_shape) >= 2 and new_shape[0] == -1 and new_shape[1] == -1:
                # Time rank unfolding. Get the time rank from original input.
                if self.unfold_time_rank is True:
                    original_shape = tf.shape(input_before_time_rank_folding)
                    new_shape = (original_shape[0], original_shape[1]) + new_shape[2:]
                # No time-rank unfolding, but we do have both batch- and time-rank.
                else:
                    input_shape = tf.shape(inputs)
                    # Batch and time rank stay as is.
                    new_shape = (input_shape[0], input_shape[1]) + new_shape[2:]

            reshaped = tf.reshape(tensor=inputs, shape=new_shape, name="reshaped")

            # Have to place the time rank back in as unknown (for the auto Space inference).
            if type(self.unfold_time_rank) == int:
                reshaped._batch_rank = 1 if self.time_major is True else 0
                reshaped._time_rank = 0 if self.time_major is True else 1
                return reshaped
            else:
                # TODO: add other cases of reshaping and fix batch/time rank hints.
                if self.fold_time_rank:
                    reshaped._batch_rank = 0
                elif self.unfold_time_rank:
                    reshaped._batch_rank = 1 if self.time_major is True else 0
                    reshaped._time_rank = 0 if self.time_major is True else 1
                else:
                    if space.has_batch_rank is True:
                        if space.time_major is False:
                            reshaped._batch_rank = 0
                        else:
                            reshaped._time_rank = 0
                            reshaped._batch_rank = 1
                    if space.has_time_rank is True:
                        reshaped._time_rank = 0 if space.time_major is True else 1

                return reshaped
