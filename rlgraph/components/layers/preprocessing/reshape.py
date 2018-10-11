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

from rlgraph import get_backend
from rlgraph.utils import pytorch_one_hot
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.utils.numpy import one_hot

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
    def __init__(self, new_shape=None, flatten=False, flatten_categories=True, fold_time_rank=False,
                 unfold_time_rank=False, time_major=None, flip_batch_and_time_rank=False, scope="reshape", **kwargs):
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
            flatten_categories (Union[bool,int]): Only important if `flatten` is True and incoming space is an IntBox.
                Specifies, whether to also flatten IntBox categories OR specifies the exact number of int categories to
                use for flattening.
                Default: True.
            fold_time_rank (bool): Whether to fold the time rank into a single batch rank.
                E.g. from (None, None, 2, 3) to (None, 2, 3). Providing both `fold_time_rank` (True) and
                `new_shape` is allowed.
            unfold_time_rank (bool): Whether to unfold the time rank from a currently common batch+time-rank.
                The exact size of the time rank to unfold is determined automatically via the original sample.
                Providing both `unfold_time_rank` (True) and `new_shape` is
                allowed.
            time_major (Optional[bool]): Only used if not None. Specifies whether the time rank should come before
                the batch rank after(!) reshaping.
                If `new_shape` is specified: Can be used to flip batch and time rank with respect to the input Space.
                If `unfold_time_rank` is True: Specifies whether the shape after reshaping is time-major or not.
            flip_batch_and_time_rank (bool): Whether to flip batch and time rank during the reshape.
                Default: False.
        """
        super(ReShape, self).__init__(scope=scope, **kwargs)

        assert flatten is False or new_shape is None, "ERROR: If `flatten` is True, `new_shape` must be None!"
        assert fold_time_rank is False or unfold_time_rank is False,\
            "ERROR: Can only either fold or unfold the time-rank! Both `fold_time_rank` and `unfold_time_rank` cannot " \
            "be True at the same time."

        # The new shape specifications.
        self.new_shape = new_shape
        self.flatten = flatten
        self.flatten_categories = flatten_categories
        self.fold_time_rank = fold_time_rank
        self.unfold_time_rank = unfold_time_rank
        self.time_major = time_major
        self.flip_batch_and_time_rank = flip_batch_and_time_rank

        # The input space.
        self.in_space = None

        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

        # Stores the number of categories in IntBoxes.
        self.num_categories = dict()
        # Stores the `time_major` settings of incoming Spaces.
        self.in_space_time_majors = dict()

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, single_space in space.flatten().items():
            class_ = type(single_space)

            # Determine the actual shape (not batch/time ranks).
            if self.flatten is True:
                if self.flatten_categories is not False and type(single_space) == IntBox:
                    if self.flatten_categories is True:
                        num_categories = single_space.flat_dim_with_categories
                    else:
                        num_categories = self.flatten_categories
                    new_shape = (num_categories,)
                else:
                    new_shape = (single_space.flat_dim,)

                if self.flatten_categories is not False and type(single_space) == IntBox:
                    class_ = FloatBox
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
            elif self.unfold_time_rank is True:
                sanity_check_space(single_space, must_have_batch_rank=True, must_have_time_rank=False)
                ret[key] = class_(
                    shape=single_space.shape if new_shape is None else new_shape,
                    add_batch_rank=True, add_time_rank=True,
                    time_major=self.time_major if self.time_major is not None else False
                )
            # Only change the actual shape (leave batch/time ranks as is).
            else:
                # Do we flip batch and time ranks?
                time_major = single_space.time_major if self.flip_batch_and_time_rank is False else \
                    not single_space.time_major

                ret[key] = class_(shape=single_space.shape if new_shape is None else new_shape,
                                  add_batch_rank=single_space.has_batch_rank,
                                  add_time_rank=single_space.has_time_rank, time_major=time_major)
        ret = unflatten_op(ret)
        return ret

    def check_input_spaces(self, input_spaces, action_space=None):
        super(ReShape, self).check_input_spaces(input_spaces, action_space)

        # Check whether our input space has-batch or not and store this information here.
        in_space = input_spaces["preprocessing_inputs"]  # type: Space

        if self.flatten is True and isinstance(in_space, IntBox) and self.flatten_categories is True:
            sanity_check_space(in_space, must_have_categories=True, num_categories=(2, 10000))

        #if input_spaces["input_before_time_rank_folding"] != "flex":
        #    assert self.time_major is None or \
        #           (self.flip_batch_and_time_rank is False and
        #            self.time_major == input_spaces["input_before_time_rank_folding"].time_major) or \
        #           (self.flip_batch_and_time_rank is True and
        #            self.time_major != input_spaces["input_before_time_rank_folding"].time_major), \
        #           "ERROR: Space of `input_before_time_rank_folding` to ReShape ('{}') has time-major={}, but " \
        #           "ReShape has time-major={}!".format(self.global_scope, in_space.time_major, self.time_major)

    def create_variables(self, input_spaces, action_space=None):
        self.in_space = input_spaces["preprocessing_inputs"]  # type: Space

        # Store the mapped output Spaces (per flat key).
        self.output_spaces = flatten_op(self.get_preprocessed_space(self.in_space))
        # Store time_major settings of incoming spaces.
        self.in_space_time_majors = self.in_space.flatten(mapping=lambda key, space: space.time_major)

        # Check whether we have to flatten the incoming categories of an IntBox into a FloatBox with additional
        # rank (categories rank). Store the dimension of this additional rank in the `self.num_categories` dict.
        if self.flatten is True:
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
                self.num_categories = self.in_space.flatten(mapping=mapping_func)
            elif self.flatten_categories is not False:
                # TODO: adjust for input ContainerSpaces. For now only support single space (flat-key=="")
                self.num_categories = {"": self.flatten_categories}

    @rlgraph_api(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_apply(self, key, preprocessing_inputs, input_before_time_rank_folding=None):
        """
        Reshapes the input to the specified new shape.

        Args:
            preprocessing_inputs (SingleDataOp): The input to reshape.
            input_before_time_rank_folding (Optional[SingleDataOp]): The original input (before!) the time-rank had
                been folded (this was done in a different ReShape Component). Serves if `self.unfold_time_rank` is True
                to figure out the exact time-rank dimension to unfold.

        Returns:
            SingleDataOp: The reshaped input.
        """
        assert self.unfold_time_rank is False or input_before_time_rank_folding is not None

        #preprocessing_inputs = tf.Print(preprocessing_inputs, [tf.shape(preprocessing_inputs)], summarize=1000,
        #                                message="input shape for {} (key={}): {}".format(preprocessing_inputs.name, key, self.scope))

        if self.backend == "python" or get_backend() == "python":
            # Create a one-hot axis for the categories at the end?
            if self.num_categories.get(key, 0) > 1:
                preprocessing_inputs = one_hot(preprocessing_inputs, depth=self.num_categories[key])

            new_shape = self.output_spaces[key].get_shape(
                with_batch_rank=-1, with_time_rank=-1, time_major=self.time_major
            )
            # Dynamic new shape inference:
            # If both batch and time rank must be left alone OR the time rank must be unfolded from a currently common
            # batch+time 0th rank, get these two dynamically.
            # Note: We may still flip the two, if input space has a different `time_major` than output space.
            if len(new_shape) > 2 and new_shape[0] == -1 and new_shape[1] == -1:
                # Time rank unfolding. Get the time rank from original input.
                if self.unfold_time_rank is True:
                    original_shape = input_before_time_rank_folding.shape
                    new_shape = (original_shape[0], original_shape[1]) + new_shape[2:]
                # No time-rank unfolding, but we do have both batch- and time-rank.
                else:
                    input_shape = preprocessing_inputs.shape
                    # Batch and time rank stay as is.
                    if self.time_major is None or self.time_major is self.in_space_time_majors[key]:
                        new_shape = (input_shape[0], input_shape[1]) + new_shape[2:]
                    # Batch and time rank need to be flipped around: Do a transpose.
                    else:
                        preprocessing_inputs = np.transpose(preprocessing_inputs, axes=(1, 0) + input_shape[2:])
                        new_shape = (input_shape[1], input_shape[0]) + new_shape[2:]

            return np.reshape(preprocessing_inputs, newshape=new_shape)
        elif get_backend() == "pytorch":
            # Create a one-hot axis for the categories at the end?
            if self.num_categories.get(key, 0) > 1:
                preprocessing_inputs = pytorch_one_hot(preprocessing_inputs, depth=self.num_categories[key])
            new_shape = self.output_spaces[key].get_shape(
                with_batch_rank=-1, with_time_rank=-1, time_major=self.time_major
            )
            # Dynamic new shape inference:
            # If both batch and time rank must be left alone OR the time rank must be unfolded from a currently common
            # batch+time 0th rank, get these two dynamically.
            # Note: We may still flip the two, if input space has a different `time_major` than output space.
            if len(new_shape) > 2 and new_shape[0] == -1 and new_shape[1] == -1:
                # Time rank unfolding. Get the time rank from original input.
                if self.unfold_time_rank is True:
                    original_shape = input_before_time_rank_folding.shape
                    new_shape = (original_shape[0], original_shape[1]) + new_shape[2:]
                # No time-rank unfolding, but we do have both batch- and time-rank.
                else:
                    input_shape = preprocessing_inputs.shape
                    # Batch and time rank stay as is.
                    if self.time_major is None or self.time_major is self.in_space_time_majors[key]:
                        new_shape = (input_shape[0], input_shape[1]) + new_shape[2:]
                    # Batch and time rank need to be flipped around: Do a transpose.
                    else:
                        preprocessing_inputs = torch.transpose(preprocessing_inputs, (1, 0) + input_shape[2:])
                        new_shape = (input_shape[1], input_shape[0]) + new_shape[2:]

            # print("Reshaping input of shape {} to new shape {} ".format(preprocessing_inputs.shape, new_shape))

            # The problem here is the following: Input has dim e.g. [4, 256, 1, 1]
            # -> If shape inference in spaces failed, output dim is not correct -> reshape will attempt
            # something like reshaping to [256].
            if self.flatten or (preprocessing_inputs.size(0) > 1 and preprocessing_inputs.dim() > 1):
                return preprocessing_inputs.squeeze()
            else:
                return torch.reshape(preprocessing_inputs, new_shape)

        elif get_backend() == "tf":
            # Create a one-hot axis for the categories at the end?
            if self.num_categories.get(key, 0) > 1:
                preprocessing_inputs = tf.one_hot(
                    preprocessing_inputs, depth=self.num_categories[key], axis=-1, dtype="float32"
                )

            new_shape = self.output_spaces[key].get_shape(
                with_batch_rank=-1, with_time_rank=-1, time_major=self.time_major
            )
            # Dynamic new shape inference:
            # If both batch and time rank must be left alone OR the time rank must be unfolded from a currently common
            # batch+time 0th rank, get these two dynamically.
            # Note: We may still flip the two, if input space has a different `time_major` than output space.
            flip_after_reshape = False
            if len(new_shape) >= 2 and new_shape[0] == -1 and new_shape[1] == -1:
                # Time rank unfolding. Get the time rank from original input (and maybe flip).
                if self.unfold_time_rank is True:
                    original_shape = tf.shape(input_before_time_rank_folding)
                    new_shape = (original_shape[0], original_shape[1]) + new_shape[2:]
                    flip_after_reshape = self.flip_batch_and_time_rank
                # No time-rank unfolding, but we do have both batch- and time-rank.
                else:
                    input_shape = tf.shape(preprocessing_inputs)
                    # Batch and time rank stay as is.
                    if self.time_major is None or self.time_major is self.in_space_time_majors[key]:
                        new_shape = (input_shape[0], input_shape[1]) + new_shape[2:]
                    # Batch and time rank need to be flipped around: Do a transpose.
                    else:
                        assert self.flip_batch_and_time_rank is True
                        preprocessing_inputs = tf.transpose(
                            preprocessing_inputs, perm=(1, 0) + tuple(i for i in range(
                                2, input_shape.shape.as_list()[0]
                            )), name="transpose-flip-batch-time-ranks"
                        )
                        new_shape = (input_shape[1], input_shape[0]) + new_shape[2:]

            reshaped = tf.reshape(tensor=preprocessing_inputs, shape=new_shape, name="reshaped")

            if flip_after_reshape and self.flip_batch_and_time_rank:
                reshaped = tf.transpose(reshaped, (1, 0) + tuple(i for i in range(2, len(new_shape))), name="transpose-flip-batch-time-ranks-after-reshape")

            #reshaped = tf.Print(reshaped, [tf.shape(reshaped)], summarize=1000,
            #                    message="output shape for {} (key={}): {}".format(reshaped, key, self.scope))

            # Have to place the time rank back in as unknown (for the auto Space inference).
            if type(self.unfold_time_rank) == int:
                # TODO: replace placeholder with default value by _batch_rank/_time_rank properties.
                return tf.placeholder_with_default(reshaped, shape=(None, None) + new_shape[2:])
            else:
                # TODO: add other cases of reshaping and fix batch/time rank hints.
                if self.fold_time_rank:
                    reshaped._batch_rank = 0
                elif self.unfold_time_rank or self.flip_batch_and_time_rank:
                    reshaped._batch_rank = 0 if self.time_major is False else 1
                    reshaped._time_rank = 0 if self.time_major is True else 1

                return reshaped
