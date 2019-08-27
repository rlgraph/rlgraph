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

from __future__ import absolute_import, division, print_function

import re

import numpy as np
from six.moves import xrange as range_
from rlgraph import get_backend
from rlgraph.spaces.bool_box import BoolBox
from rlgraph.spaces.box_space import BoxSpace
from rlgraph.spaces.containers import ContainerSpace, Dict, Tuple
from rlgraph.spaces.float_box import FloatBox
from rlgraph.spaces.int_box import IntBox
from rlgraph.spaces.text_box import TextBox
from rlgraph.utils.rlgraph_errors import RLGraphError, RLGraphSpaceError
from rlgraph.utils.util import convert_dtype, get_shape, LARGE_INTEGER, force_tuple

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


# TODO: replace completely by `Component.get_variable` (python-backend)
def get_list_registry(from_space, capacity=None, initializer=0, flatten=True, add_batch_rank=False):
    """
    Creates a list storage for a space by providing an ordered dict mapping space names
    to empty lists.

    Args:
        from_space: Space to create registry from.
        capacity (Optional[int]): Optional capacity to initalize list.
        initializer (Optional(any)): Optional initializer for list if capacity is not None.
        flatten (bool): Whether to produce a FlattenedDataOp with auto-keys.

        add_batch_rank (Optional[bool,int]): If from_space is given and is True, will add a 0th rank (None) to
            the created variable. If it is an int, will add that int instead of None.
            Default: False.

    Returns:
        dict: Container dict mapping spaces to empty lists.
    """
    if flatten:
        if capacity is not None:
            var = from_space.flatten(
                custom_scope_separator="-", scope_separator_at_start=False,
                mapping=lambda k, primitive: [initializer for _ in range_(capacity)]
            )
        else:
            var = from_space.flatten(
                custom_scope_separator="-", scope_separator_at_start=False,
                mapping=lambda k, primitive: []
            )
    else:
        if capacity is not None:
            var = [initializer for _ in range_(capacity)]
        else:
            var = []
    return var


def get_space_from_op(op, read_key_hints=False, dtype=None, low=None, high=None):
    """
    Tries to re-create a Space object given some DataOp (e.g. a tf op).
    This is useful for shape inference on returned ops after having run through a graph_fn.

    Args:
        op (DataOp): The op to create a corresponding Space for.

        read_key_hints (bool): If True, tries to read type- and low/high-hints from the pattern of the Dict keys (str).
            - Preceding "I_": IntBox, "F_": FloatBox, "B_": BoolBox.
            - Succeeding "_low=0.0": Low value.
            - Succeeding "_high=1.0": High value.
            E.g. Dict key "F_somekey_low=0.0_high=2.0" indicates a FloatBox with low=0.0 and high=2.0.
                 Dict key "I_somekey" indicates an intbox with no limits.
                 Dict key "I_somekey_high=5" indicates an intbox with high=5 (values 0-4).

            Default: False.

        dtype (Optional[str]): An optional indicator, what the `dtype` of a BoxSpace should be.
        low (Optional[int,float]): An optional indicator, what the `low` property for a BoxSpace should be.
        high (Optional[int,float]): An optional indicator, what the `high` property for a BoxSpace should be.

    Returns:
        Space: The inferred Space object.
    """
    # a Dict
    if isinstance(op, dict):  # DataOpDict
        spec = {}
        add_batch_rank = False
        add_time_rank = False
        for key, value in op.items():
            # Try to infer hints from the key.
            if read_key_hints is True:
                dtype, low, high = get_space_hints_from_dict_key(key)
            spec[key] = get_space_from_op(value, dtype=dtype, low=low, high=high)
            # Return
            if spec[key] == 0:
                return 0
            if spec[key].has_batch_rank:
                add_batch_rank = True
            if spec[key].has_time_rank:
                add_time_rank = True
        return Dict(spec, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank)
    # a Tuple
    elif isinstance(op, tuple):  # DataOpTuple
        spec = []
        add_batch_rank = False
        add_time_rank = False
        for i in op:
            space = get_space_from_op(i)
            if space == 0:
                return 0
            spec.append(space)
            if spec[-1].has_batch_rank:
                add_batch_rank = True
            if spec[-1].has_time_rank:
                add_time_rank = True
        return Tuple(spec, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank)

    # primitive Space -> infer from op dtype and shape
    else:
        low_high = {}
        if high is not None:
            low_high["high"] = high
        if low is not None:
            low_high["low"] = low
        # Op itself is a single value, simple python type.
        if isinstance(op, (bool, int, float)):
            return BoxSpace.from_spec(spec=(dtype or type(op)), shape=(), **low_high)
        elif isinstance(op, str):
            raise RLGraphError("Cannot derive Space from non-allowed op ({})!".format(op))
        # A single numpy array.
        elif isinstance(op, np.ndarray):
            return BoxSpace.from_spec(spec=convert_dtype(str(op.dtype), "np"), shape=op.shape, **low_high)
        elif isinstance(op, list):
            return try_space_inference_from_list(op, dtype=dtype, **low_high)
        # No Space: e.g. the tf.no_op, a distribution (anything that's not a tensor).
        # PyTorch Tensors do not have get_shape so must check backend.
        elif hasattr(op, "dtype") is False or (get_backend() == "tf" and not hasattr(op, "get_shape")):
            return 0
        # Some tensor: can be converted into a BoxSpace.
        else:
            shape = get_shape(op)
            # Unknown shape (e.g. a cond op).
            if shape is None:
                return 0
            add_batch_rank = False
            add_time_rank = False
            time_major = False
            new_shape = list(shape)

            # New way: Detect via op._batch_rank and op._time_rank properties where these ranks are.
            if hasattr(op, "_batch_rank") and isinstance(op._batch_rank, int):
                add_batch_rank = True
                new_shape[op._batch_rank] = -1

            # elif get_backend() == "pytorch":
            #     if isinstance(op, torch.Tensor):
            #         if op.dim() > 1 and shape[0] == 1:
            #             add_batch_rank = True
            #             new_shape[0] = 1
            if hasattr(op, "_time_rank") and isinstance(op._time_rank, int):
                add_time_rank = True
                if op._time_rank == 0:
                    time_major = True
                new_shape[op._time_rank] = -1
            shape = tuple(n for n in new_shape if n != -1)

            # Old way: Detect automatically whether the first rank(s) are batch and/or time rank.
            if add_batch_rank is False and add_time_rank is False and shape != () and shape[0] is None:
                if len(shape) > 1 and shape[1] is None:
                    #raise RLGraphError(
                    #    "ERROR: Cannot determine time-major flag if both batch- and time-ranks are in an op w/o saying "
                    #    "which rank goes to which position!"
                    #)
                    shape = shape[2:]
                    add_time_rank = True
                else:
                    shape = shape[1:]
                add_batch_rank = True

            # TODO: If op._batch_rank and/or op._time_rank are not set, set them now.

            base_dtype = op.dtype.base_dtype if hasattr(op.dtype, "base_dtype") else op.dtype
            # PyTorch does not have a bool type
            if get_backend() == "pytorch":
                if op.dtype is torch.uint8:
                    base_dtype = bool
            base_dtype_str = str(base_dtype)

            # FloatBox
            if "float" in base_dtype_str:
                return FloatBox(shape=shape, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank,
                                time_major=time_major, dtype=convert_dtype(base_dtype, "np"))
            # IntBox
            elif "int" in base_dtype_str:
                high_ = high or getattr(op, "_num_categories", None)
                return IntBox(high_, shape=shape, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank,
                              time_major=time_major, dtype=convert_dtype(base_dtype, "np"))
            # a BoolBox
            elif "bool" in base_dtype_str:
                return BoolBox(shape=shape, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank,
                               time_major=time_major)
            # a TextBox
            elif "string" in base_dtype_str:
                return TextBox(shape=shape, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank,
                               time_major=time_major)

    raise RLGraphError("ERROR: Cannot derive Space from op '{}' (unknown type?)!".format(op))


def get_space_hints_from_dict_key(key):
    """
    Args:
        key (str): The Dict key to analyze for type- and low/high hints.

    Returns:
        dtype (str), low (numeric), high (numeric): Some of these may be None if no hints were found in the key.
    """
    # Look for dtype hint.
    dtype = None
    mo = re.search(r'^([IFB])_', key)
    if mo is not None:
        dtype = mo.group(1)
        dtype = "float" if dtype == "F" else "int" if dtype == "I" else "bool"
    if dtype == "bool":
        return dtype, None, None

    # Look for low hint.
    low = None
    mo = re.search(r'_low([\-\d\.\d]+)', key)
    if mo is not None:
        low = mo.group(1)
        low = float(low) if dtype == "float" else int(low)

    # Look for high hint.
    high = None
    mo = re.search(r'_high([\-\d\.\d]+)', key)
    if mo is not None:
        high = mo.group(1)
        high = float(high) if dtype == "float" else int(high)

    return dtype, low, high


def sanity_check_space(
        space, allowed_types=None, allowed_sub_types=None, non_allowed_types=None, non_allowed_sub_types=None,
        must_have_batch_rank=None, must_have_time_rank=None, must_have_batch_or_time_rank=False,
        must_have_categories=None, num_categories=None,
        must_have_lower_limit=None, must_have_upper_limit=None,
        rank=None, shape=None
):
    """
    Sanity checks a given Space for certain criteria and raises exceptions if they are not met.

    Args:
        space (Space): The Space object to check.
        allowed_types (Optional[List[type]]): A list of types that this Space must be an instance of.

        allowed_sub_types (Optional[List[type]]): For container spaces, a list of sub-types that all
            flattened sub-Spaces must be an instance of.

        non_allowed_types (Optional[List[type]]): A list of type that this Space must not be an instance of.

        non_allowed_sub_types (Optional[List[type]]): For container spaces, a list of sub-types that all
            flattened sub-Spaces must not be an instance of.

        must_have_batch_rank (Optional[bool]): Whether the Space must (True) or must not (False) have the
            `has_batch_rank` property set to True. None, if it doesn't matter.

        must_have_time_rank (Optional[bool]): Whether the Space must (True) or must not (False) have the
            `has_time_rank` property set to True. None, if it doesn't matter.

        must_have_batch_or_time_rank (Optional[bool]): Whether the Space must (True) or must not (False) have either
            the `has_batch_rank` or the `has_time_rank` property set to True.

        must_have_categories (Optional[bool]): For IntBoxes, whether the Space must (True) or must not (False) have
            global bounds with `num_categories` > 0. None, if it doesn't matter.

        num_categories (Optional[int,tuple]): An int or a tuple (min,max) range within which the Space's
            `num_categories` rank must lie. Only valid for IntBoxes.
            None if it doesn't matter.

        must_have_lower_limit (Optional[bool]): If not None, whether this Space must have a lower limit.
        must_have_upper_limit (Optional[bool]): If not None, whether this Space must have an upper limit.

        rank (Optional[int,tuple]): An int or a tuple (min,max) range within which the Space's rank must lie.
            None if it doesn't matter.

        shape (Optional[tuple[int]]): A tuple of ints specifying the required shape. None if it doesn't matter.

    Raises:
        RLGraphSpaceError: If any of the conditions is not met.
    """
    flattened_space = space.flatten()

    # Check the types.
    if allowed_types is not None:
        if not isinstance(space, force_tuple(allowed_types)):
            raise RLGraphSpaceError(
                space, "ERROR: Space ({}) is not an instance of {}!".format(space, allowed_types)
            )

    if allowed_sub_types is not None:
        for flat_key, sub_space in flattened_space.items():
            if not isinstance(sub_space, force_tuple(allowed_sub_types)):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: sub-Space '{}' ({}) is not an instance of {}!".
                    format(flat_key, sub_space, allowed_sub_types)
                )

    if non_allowed_types is not None:
        if isinstance(space, force_tuple(non_allowed_types)):
            raise RLGraphSpaceError(
                space,
                "ERROR: Space ({}) must not be an instance of {}!".format(space, non_allowed_types)
            )

    if non_allowed_sub_types is not None:
        for flat_key, sub_space in flattened_space.items():
            if isinstance(sub_space, force_tuple(non_allowed_sub_types)):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: sub-Space '{}' ({}) must not be an instance of {}!".
                    format(flat_key, sub_space, non_allowed_sub_types)
                )

    if must_have_batch_or_time_rank is True:
        if space.has_batch_rank is False and space.has_time_rank is False:
            raise RLGraphSpaceError(
                space,
                "ERROR: Space ({}) does not have a batch- or a time-rank, but must have either one of "
                "these!".format(space)
            )

    if must_have_batch_rank is not None:
        if (space.has_batch_rank is False and must_have_batch_rank is True) or \
                (space.has_batch_rank is not False and must_have_batch_rank is False):
            # Last chance: Check for rank >= 2, that would be ok as well.
            if must_have_batch_rank is True and len(space.get_shape(with_batch_rank=True)) >= 2:
                pass
            # Something is wrong.
            elif space.has_batch_rank is not False:
                raise RLGraphSpaceError(
                    space,
                    "ERROR: Space ({}) has a batch rank, but is not allowed to!".format(space)
                )
            else:
                raise RLGraphSpaceError(
                    space,
                    "ERROR: Space ({}) does not have a batch rank, but must have one!".format(space)
                )

    if must_have_time_rank is not None:
        if (space.has_time_rank is False and must_have_time_rank is True) or \
                (space.has_time_rank is not False and must_have_time_rank is False):
            # Last chance: Check for rank >= 3, that would be ok as well.
            if must_have_time_rank is True and len(space.get_shape(with_batch_rank=True, with_time_rank=True)) >= 2:
                pass
            # Something is wrong.
            elif space.has_time_rank is not False:
                raise RLGraphSpaceError(
                    space,
                    "ERROR: Space ({}) has a time rank, but is not allowed to!".format(space)
                )
            else:
                raise RLGraphSpaceError(
                    space,
                    "ERROR: Space ({}) does not have a time rank, but must have one!".format(space)
                )

    if must_have_categories is not None:
        for flat_key, sub_space in flattened_space.items():
            if not isinstance(sub_space, IntBox):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) is not an IntBox. Only IntBox Spaces can have categories!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )
            elif sub_space.global_bounds is False:
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must have categories (globally valid value bounds)!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )

    if must_have_lower_limit is not None:
        for flat_key, sub_space in flattened_space.items():
            low = sub_space.low
            if must_have_lower_limit is True and (low == -LARGE_INTEGER or low == float("-inf")):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must have a lower limit, but has none!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )
            elif must_have_lower_limit is False and (low != -LARGE_INTEGER and low != float("-inf")):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must not have a lower limit, but has one ({})!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space, low)
                )

    if must_have_upper_limit is not None:
        for flat_key, sub_space in flattened_space.items():
            high = sub_space.high
            if must_have_upper_limit is True and (high != LARGE_INTEGER and high != float("inf")):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must have an upper limit, but has none!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )
            elif must_have_upper_limit is False and (high == LARGE_INTEGER or high == float("inf")):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must not have a upper limit, but has one ({})!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space, high)
                )

    if rank is not None:
        if isinstance(rank, int):
            for flat_key, sub_space in flattened_space.items():
                if sub_space.rank != rank:
                    raise RLGraphSpaceError(
                        sub_space,
                        "ERROR: A Space (flat-key={}) of '{}' has rank {}, but must have rank "
                        "{}!".format(flat_key, space, sub_space.rank, rank)
                    )
        else:
            for flat_key, sub_space in flattened_space.items():
                if not ((rank[0] or 0) <= sub_space.rank <= (rank[1] or float("inf"))):
                    raise RLGraphSpaceError(

                        sub_space,
                        "ERROR: A Space (flat-key={}) of '{}' has rank {}, but its rank must be between {} and "
                        "{}!".format(flat_key, space, sub_space.rank, rank[0], rank[1])
                    )

    if shape is not None:
        for flat_key, sub_space in flattened_space.items():
            if sub_space.shape != shape:
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: A Space (flat-key={}) of '{}' has shape {}, but its shape must be "
                    "{}!".format(flat_key, space, sub_space.get_shape(), shape)
                )

    if num_categories is not None:
        for flat_key, sub_space in flattened_space.items():
            if not isinstance(sub_space, IntBox):
                raise RLGraphSpaceError(
                    sub_space,
                    "ERROR: A Space (flat-key={}) of '{}' is not an IntBox. Only IntBox Spaces can have "
                    "categories!".format(flat_key, space)
                )
            elif isinstance(num_categories, int):
                if sub_space.num_categories != num_categories:
                    raise RLGraphSpaceError(
                        sub_space,
                        "ERROR: A Space (flat-key={}) of '{}' has `num_categories` {}, but must have {}!".
                        format(flat_key, space, sub_space.num_categories, num_categories)
                    )
            elif not ((num_categories[0] or 0) <= sub_space.num_categories <= (num_categories[1] or float("inf"))):
                raise RLGraphSpaceError(sub_space,
                    "ERROR: A Space (flat-key={}) of '{}' has `num_categories` {}, but this value must be between "
                    "{} and {}!".format(flat_key, space, sub_space.num_categories, num_categories[0], num_categories[1])
                )


def check_space_equivalence(space1, space2):
    """
    Compares the two input Spaces for equivalence and returns the more generic Space of the two.
    The more generic  Space  is the one that has the properties has_batch_rank and/or has _time_rank set (instead of
    hard values in these ranks).
    E.g.: FloatBox((64,)) is equivalent with FloatBox((), +batch-rank). The latter will be returned.

    NOTE: FloatBox((2,)) and FloatBox((3,)) are NOT equivalent.

    Args:
        space1 (Space): The 1st Space to compare.
        space2 (Space): The 2nd Space to compare.

    Returns:
        Union[Space,False]: False is the two spaces are not equivalent. The more generic Space of the two if they are
            equivalent.
    """
    # Spaces are the same: Return one of them.
    if space1 == space2:
        return space1
    # One has batch-rank, the other doesn't, but has one more rank.
    elif space1.has_batch_rank and not space2.has_batch_rank and \
            (np.asarray(space1.rank) == np.asarray(space2.rank) - 1).all():
        return space1
    elif space2.has_batch_rank and not space1.has_batch_rank and \
            (np.asarray(space2.rank) == np.asarray(space1.rank) - 1).all():
        return space2
    elif get_backend() == "pytorch":
        if not space1.has_batch_rank and not space2.has_batch_rank and \
                (np.asarray(space1.rank) == np.asarray(space2.rank)).all():
            return space1

        # TODO problem is that batch ranks are principally not handled correctly here ->
        # e.g. (batch_size, 256), (256,) can both be valid between layers
        if not space1.has_batch_rank and not space2.has_batch_rank and space1.rank > space2.rank:
            return space2
        if not space1.has_batch_rank and not space2.has_batch_rank and space1.rank < space2.rank:
            return space1
    # TODO: time rank?

    return False


def try_space_inference_from_list(list_op, dtype=None, **low_high):
    """
    Attempts to infer shape space from a list op. A list op may be the result of fetching state from a Python
    memory.

    Args:
        list_op (list): List with arbitrary sub-structure.

    Returns:
        Space: Inferred Space object represented by list.
    """
    shape = len(list_op)
    if shape > 0:
        # Try to infer more things by looking inside list.
        elem = list_op[0]
        if (get_backend() == "pytorch" and isinstance(elem, torch.Tensor)) or \
                get_backend() == "tf" and isinstance(elem, tf.Tensor):
            list_type = dtype or elem.dtype
            inner_shape = elem.shape
            return BoxSpace.from_spec(spec=convert_dtype(list_type, "np"), shape=(shape,) + inner_shape,
                                      add_batch_rank=True, **low_high)
        elif isinstance(elem, list):
            inner_shape = len(elem)
            return BoxSpace.from_spec(spec=convert_dtype(dtype or float, "np"), shape=(shape, inner_shape),
                                      add_batch_rank=True, **low_high)
        # IntBox -> elem must be int and dtype hint must match (or None).
        elif isinstance(elem, int) and (dtype is None or dtype == "int"):
            # In case of missing comma values, check all other items in list for float.
            # If one float in there -> FloatBox, otherwise -> IntBox.
            has_floats = any(isinstance(el, float) for el in list_op)
            if has_floats is False:
                return IntBox.from_spec(shape=(shape,), add_batch_rank=True, **low_high)
            else:
                return FloatBox.from_spec(shape=(shape,), add_batch_rank=True, **low_high)
        # FloatBox -> elem must be float (or int) and dtype hint must match (or None).
        elif isinstance(elem, (float, int)) and (dtype is None or dtype == "float"):
            return FloatBox.from_spec(shape=(shape,), add_batch_rank=True, **low_high)

    # Most general guess is a Float box.
    return FloatBox(shape=(shape,), **low_high)


def get_default_distribution_from_space(
        space, bounded_distribution_type="beta", discrete_distribution_type="categorical",
        gumbel_softmax_temperature=1.0
):
    """
    Args:
        space (Space): The primitive Space for which to derive a default distribution spec.
        bounded_distribution_type (str): The lookup class string for a bounded FloatBox distribution.
            Default: "beta".
        discrete_distribution_type(str): The class of distributions to use for discrete action spaces. For options
            check the components.distributions package. Default: categorical. Agents requiring reparameterization
            may require a GumbelSoftmax distribution instead.
        gumbel_softmax_temperature (float): Temperature parameter for the Gumbel-Softmax distribution used
            for discrete actions.

    Returns:
        Dict: A Spec dict, from which a valid default distribution object can be created.
    """
    # IntBox: Categorical.
    if isinstance(space, IntBox):
        if discrete_distribution_type == "gumbel_softmax":
            return dict(type="gumbel-softmax", temperature=gumbel_softmax_temperature)
        else:
            return dict(type=discrete_distribution_type)
    # BoolBox: Bernoulli.
    elif isinstance(space, BoolBox):
        return dict(type="bernoulli")
    # Continuous action space: Normal/Beta/etc. distribution.
    elif isinstance(space, FloatBox):
        # Unbounded -> Normal distribution.
        if not is_bounded_space(space):
            return dict(type="normal")
        # Bounded -> according to the bounded_distribution parameter.
        else:
            return dict(type=bounded_distribution_type, low=space.low, high=space.high)
    # Container Space.
    elif isinstance(space, ContainerSpace):
        return dict(type="joint-cumulative", distribution_specs=dict(
            {k: get_default_distribution_from_space(s) for k, s in space.flatten().items()}
        ))
    else:
        raise RLGraphError("No distribution defined for space {}!".format(space))


def is_bounded_space(box_space):
    if not isinstance(box_space, FloatBox):
        return False
    # Unbounded.
    if box_space.low == float("-inf") and box_space.high == float("inf"):
        return False
    # Bounded.
    elif box_space.low != float("-inf") and box_space.high != float("inf"):
        return True
    # TODO: Semi-bounded -> Exponential distribution.
    else:
        raise RLGraphError(
            "Semi-bounded spaces for distribution-generation are not supported yet! You passed in low={} high={}.".
            format(box_space.low, box_space.high)
        )


def horizontalize_space_sample(space, sample, batch_size=1):
    # For Dicts, we have to treat each key as an array with batch-rank at index 0.
    # The dict is then translated into a list of dicts where each dict contains the original data
    # but without the batch-rank.
    # E.g. {'A': array([0, 1]), 'B': array([2, 3])} -> [{'A': 0, 'B': 2}, {'A': 1, 'B': 3}]
    if isinstance(space, Dict):
        some_key = next(iter(sample))
        assert isinstance(sample, dict) and isinstance(sample[some_key], np.ndarray), \
            "ERROR: Cannot flip Dict batch with dict keys if returned value is not a dict OR " \
            "values of returned value are not np.ndarrays!"
        # TODO: What if actions come as nested dicts (more than one level deep)?
        # TODO: Use DataOpDict/Tuple's new `map` method.
        if hasattr(sample[some_key], "__len__"):
            result = [{key: value[i] for key, value in sample.items()} for i in range(len(sample[some_key]))]
        else:
            # Action was not array type.
            result = [{key: value for key, value in sample.items()}]
    # Tuple:
    # E.g. Tuple(array([0, 1]), array([2, 3])) -> [(0, 2), (1, 3)]
    elif isinstance(space, Tuple):
        assert isinstance(sample, tuple) and isinstance(sample[0], np.ndarray), \
            "ERROR: Cannot flip tuple batch if returned value is not a tuple OR " \
            "values of returned value are not np.ndarrays!"
        # TODO: Use DataOpDict/Tuple's new `map` method.
        result = [tuple(value[i] for _, value in enumerate(sample)) for i in range(len(sample[0]))]
    # No container batch-flipping necessary.
    else:
        result = sample
        if batch_size == 1 and result.shape == ():
            result = [result]

    return result
