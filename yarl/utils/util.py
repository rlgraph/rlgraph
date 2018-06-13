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

from functools import partial
import numpy as np
import itertools
from six.moves import xrange

import logging
import sys
import tensorflow as tf

from yarl.backend_system import backend
from yarl.utils.yarl_error import YARLError
from yarl.utils.ops import DataOpTuple

if backend == "tf":
    import tensorflow as be
else:
    import pytorch as be

SMALL_NUMBER = 1e-6
LARGE_INTEGER = 100000000

# Logging config for testing.
logging_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%y-%m-%d %H:%M:%S')
root_logger = logging.getLogger('')
root_logger.setLevel(level=logging.DEBUG)
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(level=logging.DEBUG)

print_logging_handler = logging.StreamHandler(stream=sys.stdout)
print_logging_handler.setFormatter(logging_formatter)
print_logging_handler.setLevel(level=logging.DEBUG)
root_logger.addHandler(print_logging_handler)


def dtype(dtype_, to="tf"):
    """
    Translates any type (tf, numpy, python, etc..) into the respective tensorflow/numpy data type.

    Args:
        dtype_ (any): String describing a numerical type (e.g. 'float'), numpy data type, tf dtype,
            pytorch data-type, or python numerical type.
        to (str): Either one of 'tf' (tensorflow), 'pt' (pytorch), or 'np' (numpy). Default="tf".

    Returns:
        TensorFlow or Numpy data type (depending on `to` parameter).
    """
    # bool: tensorflow
    if backend == "tf":
        if dtype_ in ["bool", bool, np.bool_, be.bool]:
            return np.bool_ if to == "np" else be.bool
    # bool: pytorch backend (not supported)
    elif dtype_ in ["bool", bool, np.bool_]:
        return np.bool_ if to == "np" else tf.bool

    # generic backend
    if dtype_ in ["float", "float32", float, np.float32, be.float32]:
        return np.float32 if to == "np" else tf.float32
    if dtype_ in ["float64", np.float64, be.float64]:
        return np.float64 if to == "np" else tf.float64
    elif dtype_ in ["int", "int32", int, np.int32, be.int32]:
        return np.int32 if to == "np" else tf.int32
    elif dtype_ in ["int64", np.int64]:
        return np.int64 if to == "np" else tf.int64

    raise YARLError("Error: Type conversion to '{}' for type '{}' not supported.".format(to, str(dtype_)))


def get_rank(tensor):
    """
    Returns the rank (as a single int) of an input tensor.

    Args:
        tensor (any): The input tensor.

    Returns:
        The rank of the given tensor.
    """
    if backend == "tf":
        return tensor.get_shape().ndims


def get_shape(op, flat=False, no_batch=False):
    """
    Returns the shape of the tensor as a tuple.

    Args:
        op (DataOp): The input op.
        flat (bool): Whether to return the flattened shape (the product of all ints in the shape tuple).
            Default: False.
        no_batch (bool): Whether to exclude a possible 0th batch rank (None) from the returned shape.

    Returns:
        tuple: The shape of the given op.
        int: The flattened dim of the given op (if flat=True).
    """
    # dict
    if isinstance(op, dict):
        shape = tuple([get_shape(op[key]) for key in sorted(op.keys())])
    # tuple-op
    elif isinstance(op, tuple):
        shape = tuple([get_shape(i) for i in op])
    # primitive op (e.g. tensorflow)
    else:
        op_shape = op.get_shape()
        # Unknown shape (e.g. a cond op).
        if op_shape.ndims is None:
            return None
        shape = tuple(op_shape.as_list())

    # Remove batch rank?
    if no_batch is True and shape[0] is None:
        shape = shape[1:]

    if flat is False:
        return shape
    else:
        return np.prod(shape)


def force_list(elements, to_tuple=False):
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single item, already a list, or a tuple.

    Args:
        elements (Optional[any]): The input single item, list, or tuple to be converted into a list/tuple.
            If None, returns empty list/tuple.
        to_tuple (bool): Whether to use tuple (instead of list).

    Returns:
        Union[list,tuple]: All given elements in a list/tuple depending on `to_tuple`'s value.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return ctor() if elements is None else ctor(elements) \
        if isinstance(elements, (list, tuple)) and not isinstance(elements, DataOpTuple) \
        else ctor([elements])


force_tuple = partial(force_list, to_tuple=True)


def default_dict(original, defaults):
    """
    Updates the original dict with values from `defaults`, but only for those keys that
    do not exist yet in `original`.
    Changes `original` in place, but leaves `defaults` as is.

    Args:
        original (Optional[dict]): The dict to (soft)-update. If None, return `defaults`.
        defaults (dict): The dict to update from.
    """
    if original is None:
        return defaults

    for key in defaults:
        if key not in original:
            original[key] = defaults[key]
    return original


def all_combinations(input_list, descending_length=False):
    """
    Returns a list containing tuples of all possible combinations of all possible length of the elements
    in the input_list (without repeating elements inside each tuple and without resorting the input_list).
    The returned list is, however, sorted by the length of the different combination-tuples
    (in descending (longest first) or ascending (shortest first) order).
    Examples:
         input_list=[1, 2, 3] returns: [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

         NOTE: We do not re-sort input_list (5 still before 1):
         input_list=[5, 1]    returns: [(5,), (1,), (5, 1)]

    Args:
        input_list (list): The list to get all combinations for.
        descending_length (bool): Whether to sort the tuples longest first (default: shortest first).

    Returns:
        The list of tuples of possible combinations.
    """
    return sorted(list(itertools.chain.from_iterable(
        itertools.combinations(input_list, i+1) for i in xrange(len(input_list)))),
        key=len, reverse=descending_length)


def clamp(x, min_, max_):
    """
    Clamps x between min_ and max_.

    Args:
        x (float): The input to be clamped.
        min_ (float): The min value for x.
        max_ (float): The max value for x.

    Returns:
        float: The clamped value.
    """
    return max(min_, min(x, max_))
