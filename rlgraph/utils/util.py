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

from functools import partial
import logging
import numpy as np
import inspect
import re
import sys
import tensorflow as tf

from rlgraph import get_backend
from rlgraph.utils.rlgraph_error import RLGraphError

if get_backend() == "tf":
    import tensorflow as backend
else:
    import torch as backend

SMALL_NUMBER = 1e-6
LARGE_INTEGER = 100000000

# Logging config for testing.
logging_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%y-%m-%d %H:%M:%S')
root_logger = logging.getLogger('')
root_logger.setLevel(level=logging.INFO)
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(level=logging.INFO)

print_logging_handler = logging.StreamHandler(stream=sys.stdout)
print_logging_handler.setFormatter(logging_formatter)
print_logging_handler.setLevel(level=logging.INFO)
root_logger.addHandler(print_logging_handler)


def dtype(dtype_, to="tf"):
    """
    Translates any type (tf, numpy, python, etc..) into the respective tensorflow/numpy data type.

    Args:
        dtype_ (any): String describing a numerical type (e.g. 'float'), numpy data type, tf dtype,
            pytorch data-type, or python numerical type.
        to (str): Either one of 'tf' (tensorflow), 'pt' (pytorch), 'np' (numpy), 'str' (string).
            Default="tf".

    Returns:
        TensorFlow, Numpy, pytorch or string, representing a data type (depending on `to` parameter).
    """
    # Bool: tensorflow.
    if get_backend() == "tf":
        if dtype_ in ["bool", bool, np.bool_, backend.bool]:
            return np.bool_ if to == "np" else backend.bool
    # Bool: All others.
    elif dtype_ in ["bool", bool, np.bool_]:
        return np.bool_ if to == "np" else tf.bool

    # Generic backend.
    if dtype_ in ["float", "float32", float, np.float32, backend.float32]:
        return np.float32 if to == "np" else tf.float32
    if dtype_ in ["float64", np.float64, backend.float64]:
        return np.float64 if to == "np" else tf.float64
    elif dtype_ in ["int", "int32", int, np.int32, backend.int32]:
        return np.int32 if to == "np" else tf.int32
    elif dtype_ in ["int64", np.int64]:
        return np.int64 if to == "np" else tf.int64
    elif dtype_ in ["uint8", np.uint8]:
        return np.uint8 if to == "np" else tf.uint8
    elif dtype_ in ["str", np.str_]:
        return np.unicode_ if to == "np" else tf.string
    elif dtype_ in ["int16", np.int16]:
        return np.int16 if to == "np" else tf.int16

    raise RLGraphError("Error: Type conversion to '{}' for type '{}' not supported.".format(to, str(dtype_)))


def get_rank(tensor):
    """
    Returns the rank (as a single int) of an input tensor.

    Args:
        tensor (Union[tf.Tensor,np.ndarray]): The input tensor.

    Returns:
        int: The rank of the given tensor.
    """
    if isinstance(tensor, np.ndarray) or get_backend() == "python":
        return tensor.ndim
    elif get_backend() == "tf":
        return tensor.get_shape().ndims


def get_shape(op, flat=False, no_batch=False):
    """
    Returns the (static) shape of a given DataOp as a tuple.

    Args:
        op (DataOp): The input op.
        flat (bool): Whether to return the flattened shape (the product of all ints in the shape tuple).
            Default: False.
        no_batch (bool): Whether to exclude a possible 0th batch rank from the returned shape.
            Default: False.

    Returns:
        tuple: The shape of the given op.
        int: The flattened dim of the given op (if flat=True).
    """
    # Dict.
    if isinstance(op, dict):
        shape = tuple([get_shape(op[key]) for key in sorted(op.keys())])
    # Tuple-op.
    elif isinstance(op, tuple):
        shape = tuple([get_shape(i) for i in op])
    # Numpy ndarrays.
    elif isinstance(op, np.ndarray):
        shape = op.shape
    # Primitive op (e.g. tensorflow)
    else:
        op_shape = op.get_shape()
        # Unknown shape (e.g. a cond op).
        if op_shape.ndims is None:
            return None
        shape = tuple(op_shape.as_list())

    # Remove batch rank?
    if no_batch is True and shape[0] is None:
        shape = shape[1:]

    # Return as-is or as flat shape?
    if flat is False:
        return shape
    else:
        return int(np.prod(shape))


def get_batch_size(tensor):
    """
    Returns the (dynamic) batch size (dim of 0th rank) of an input tensor.

    Args:
        tensor (SingleDataOp): The input tensor.

    Returns:
        SingleDataOp: The op holding the batch size information of the given tensor.
    """
    if get_backend() == "tf":
        return tf.shape(tensor)[0]


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
        if type(elements) in [list, tuple] else ctor([elements])


force_tuple = partial(force_list, to_tuple=True)


def strip_list(elements):
    """
    Loops through elements if it's a tuple, otherwise processes elements as follows:
    If a list (or np.ndarray) of length 1, extracts that single item, otherwise leaves
    the list/np.ndarray untouched.

    Args:
        elements (any): The input single item, list, or np.ndarray to be converted into
            single item(s) (if length is 1).

    Returns:
        any: Single element(s) (the only one in input) or the original input list.
    """
    # `elements` is a tuple (e.g. from a function return). Process each element separately.
    if isinstance(elements, tuple):
        ret = []
        for el in elements:
            ret.append(el[0] if isinstance(el, (np.ndarray, list)) and len(el) == 1 else el)
        return tuple(ret)
    # `elements` is not a tuple: Process only `elements`.
    else:
        return elements[0] if isinstance(elements, (np.ndarray, list)) and len(elements) == 1 else \
            elements


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


def clip(x, min_, max_):
    """
    Clips x between min_ and max_.

    Args:
        x (float): The input to be clipped.
        min_ (float): The min value for x.
        max_ (float): The max value for x.

    Returns:
        float: The clipped value.
    """
    return max(min_, min(x, max_))


def get_method_type(method):
    """
    Returns either "graph_fn" OR "api" OR "other" depending on which method (and method
    name) is passed in.

    Args:
        method (callable): The actual method to analyze.

    Returns:
        Union[str,None]: "graph_fn", "api" or "other". None if method is not a callable.
    """
    # Not a callable: Return None.
    if not callable(method):
        return None
    # Simply recognize graph_fn by their name.
    elif method.__name__[:9] == "_graph_fn":
        return "graph_fn"
    # Figure out API-methods by their source code: Have calls to `Component.call`.
    elif method.__name__[0] != "_" and method.__name__ != "define_api_method" and \
            re.search(r'\.call\(', strip_source_code(method)):
        return "api"
    else:
        return "unknown"


def does_method_call_graph_fns(method):
    """
    Inspects the source code of a method and returns whether this method calls graph_fns inside its code.

    Args:
        method (callable): The method to inspect.

    Returns:
        bool: Whether at least one graph_fn is called somewhere in the source code.
    """
    src = strip_source_code(method)

    mo = re.search(r'\.call\([^,\)]+\._graph_fn_', src)
    return mo is not None


def get_num_return_values(method):
    """
    Does a regexp-based source code inspection and tries to figure out the number of values that the method
    will return (assuming, that the method always returns the same number of values, regardless of its inputs).

    Args:
        method (callable): The method to get the number of returns  values for.

    Returns:
        int: The number of return values of `method`.
    """
    src = strip_source_code(method)

    mo = re.search(r'.*\breturn (.+)', src, flags=re.DOTALL)
    if mo:
        return_code = mo.group(1)
        # TODO: raise error if "tuple()" -> means we don't know how many return value we will have
        # TODO: in this case, the Component needs to specify how many output records it produces.
        # Resolve tricky things (parentheses, etc..).
        while re.search(r'[\(\)"]', return_code):
            return_code = re.sub(r'\([^\(\)]*\)', '', return_code)
            return_code = re.sub(r'"[^"]*"', '', return_code)
            # Probably have to add more resolution code here.
            # ...

        # Count the commas and return.
        num_return_values = len(return_code.split(","))
        return num_return_values
    else:
        return 0


def strip_source_code(method, remove_nested_functions=True):
    """
    Strips the source code of a method by everything that's "not important", like comments.

    Args:
        method (Union[callable,str]): The method to strip the source code for (or the source code directly as string).
        remove_nested_functions (bool): Whether to remove nested functions from the source code as well.
            These usually confuse the analysis of a method's source code as they comprise a method within a method.

    Returns:
        str: The stripped source code.
    """
    if callable(method):
        src = inspect.getsource(method)
    else:
        src = method

    # Resolve '\' at end of lines.
    src = re.sub(r'\\\s*\n', "", src)
    # Remove single line comments.
    src = re.sub(r'\n\s*#.+', "", src)
    # Remove multi-line comments.
    src = re.sub(r'"""(.|\n)*?"""', "", src)
    # Remove nested functions.
    if remove_nested_functions is True:
        src = re.sub(r'\n(\s*)def \w+\((.|\n)+?\n\1[^\s\n]', "", src)

    return src


# TODO: Solve this problem by always returning a single dict from any API-method, which carries named return values.
def unify_nn_and_rnn_api_output(nn_output, return_values_wo_internal_state=1):
    """
    Makes sure always two elements are returned, no matter whether nn_output is a single DataOpRec with
    only simple NN output in it or a tuple of DataOpRecs (one NN output, one last internal state(s) of the RNN(s)).

    Args:
        nn_output (Union[Tuple[DataOpRecord,DataOpRecord],DataOpRecord]): The output coming from a call to a
            `NeuralNetwork.apply()` API-method.
        return_values_wo_internal_state (int): Number of return value if no internal_state is expected to be returned.

    Returns:
        tuple:
            - The actual NN output DataOpRec.
            - The actual NN internal state DataOpRec or None (if no RNN).
    """
    # Single output -> return plus None
    if not isinstance(nn_output, tuple):
        return nn_output, None
    # Output and internal-state.
    elif len(nn_output) == (return_values_wo_internal_state + 1):
        return nn_output
    # Output without internal-state.
    else:
        return nn_output + (None,)
