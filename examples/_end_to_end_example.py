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

from tensorflow.contrib import autograph
import tensorflow as tf
import numpy as np

from yarl.spaces import Continuous, Tuple, Dict


def precomputation_gray(*inputs):
    # In the precomputation step, we can calculate (in pure python) what we need for each primitive Space
    # input (e.g. the reshaped weights below for the grayscale on arbitrarily sized incoming images).

    # asserts work as well (we are in python!)
    assert len(inputs) == 1
    # sort out incoming parameters
    image = inputs[0]

    # We can do this step already here (statically), instead of moving this into the graph
    # (as it's done in tensorforce).
    shape = tuple(1 for _ in range(image.get_shape().ndims - 1)) + (3,)
    weights = np.reshape((0.5, 0.25, 0.25), newshape=shape)

    # now return the args that will be passed into the autographd func.
    return image, weights


def computation_gray(primitive_in, reshaped_weights):
    # This is the autograph function:
    # Only think "Tensors" in here. No asserts, no control flow
    # that should not go into the graph, no tuple generation, etc..

    # Unfortunately, numpy doesn't work here. Maybe that should be a tf issue (if native python lists
    # with tf.stack work, then why not np arrays)?
    return tf.reduce_sum(reshaped_weights * primitive_in, axis=-1, keepdims=False)


def generic_wrapper(pre, compute, *ops):
    # This is a generic wrapper that should go into the base Components class (and probably does not need to be overwritten ever).

    # TODO: e.g. what if ops contains 2 dicts that have the exact same structure (ops[0]=dict, ops[1]=dict, ops[2]=primitive space or nothing)?
    # TODO: We should then pass each key alongside each other into `pre`. Same for 2 tuples, 3 dicts, 3 tuples, etc..
    # TODO: If there are more than 1 containers in ops and their structures don't align -> ERROR.

    # simple case: assume only one input
    op = ops[0]  # TODO: make this more generic

    if isinstance(op, dict):
        return dict(map(lambda item: (item[0], generic_wrapper(pre, compute, item[1])), op.items()))
    elif isinstance(op, tuple):
        return tuple(map(lambda c: generic_wrapper(pre, compute, c), op))
    else:
        # Get args for autograph.
        returns = pre(op)
        # And call autograph with these.
        return compute(*returns)


# TODO: create memory example for Michael to start.
def computation_func_insert(primitive_in, memory_var):
    pass


# TODO: Sven will check: This can probably be done smarter using Space classes?
def get_feed_dict(feed_dict, complex_sample, placeholders):
    if isinstance(complex_sample, dict):
        for k in complex_sample:
            get_feed_dict(feed_dict, complex_sample[k], placeholders[k])
    elif isinstance(complex_sample, tuple):
        for sam, ph in zip(complex_sample, placeholders):
            get_feed_dict(feed_dict, sam, ph)
    else:
        feed_dict[placeholders] = complex_sample


with tf.Session() as sess:
    # Create a messed up, complex Space:
    input_space = Dict(a=Tuple(Continuous(shape=(2,2,3)), Continuous(shape=(3,3,3))),
                       b=Dict(c=Continuous(shape=(1,1,3))))
    print(autograph.to_code(computation_gray))
    # Only needs be done once upon Computation object creation:
    computation_gray_autographd = autograph.to_graph(computation_gray, verbose=True)
    # This is now very cool with Spaces.
    input_ = input_space.get_tensor_variable(name="placeholder")
    # input_ is now a native dict that corresponds to the structure of input_space.

    # Let the wrapper do everything.
    # The wrapper will live in Component.py and should not need to be overwritten ever (I think).
    # We can call it something else, but component will use it all under the hood, automatically.
    gray_ops = generic_wrapper(precomputation_gray, computation_gray_autographd, input_)

    # Test the pipeline.
    sample = input_space.sample()
    feed_dict = {}
    get_feed_dict(feed_dict, sample, input_)

    # Fetch a complex opts-dict.
    gray_outs = sess.run(gray_ops, feed_dict=feed_dict)
    print(gray_outs)



