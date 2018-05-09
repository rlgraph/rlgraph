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

#from tensorflow.contrib import autograph
import tensorflow as tf
import numpy as np

from yarl.spaces import Continuous, Tuple, Dict

#class Op(object):
#    def __init__(self, primitive, shape):
#        self.primitive = primitive

#    @property
#    def shape(self):
#        return self.primitive.shape



def _computation_func_gray(primitive_in):
    shape = tuple(1 for _ in range(primitive_in.get_shape().ndims - 1)) + (3,)
    weights = np.reshape((0.5, 0.25, 0.25), newshape=shape)
    return tf.reduce_sum(weights * primitive_in, axis=-1, keepdims=False)


def _computation_func_insert(primitive_in, which_var="states"):
    pass


def _wrapper_func(op, func_):
    if isinstance(op, dict):
        return dict(map(lambda item: (item[0], _wrapper_func(item[1], func_)), op.items()))
    elif isinstance(op, tuple):
        return tuple(map(lambda c: _wrapper_func(c, func_), op))
    else:
        #shape = tuple(1 for _ in range(in_.get_shape().ndims - 1)) + (3,)
        return func_(op)


#def shape_test(in_shape):
#    global weights
#    global raw_weights
#    shape = tuple(1 for _ in range(len(in_shape) - 1)) + (3,)
#    weights = np.reshape(raw_weights, newshape=shape)

def get_random_feed_dict(feed_dict, complex_sample, placeholders):
    if isinstance(complex_sample, (dict, tuple)):
        for k in complex_sample:
            get_random_feed_dict(feed_dict, complex_sample[k], placeholders[k])
    else:
        feed_dict[placeholders] = complex_sample


with tf.Session() as sess:
    input_space = Dict({"a": Continuous(shape=(2,2,3)), "b": Dict({"c": Continuous(shape=(1,1,3))})})
    #input_space = Continuous(shape=(2, 2, 3))

    #shape_test(input_space.shape)
    #print(autograph.to_code(func))
    #func_autographd = autograph.to_graph(_computation_func, verbose=True)
    input_ = input_space.get_tensor_variable(name="placeholder")

    gray_ops = _wrapper_func(input_, _computation_func_gray)
    sample = input_space.sample()
    feed_dict = {}
    get_random_feed_dict(feed_dict, sample, input_)
    #feed_dict = {input_["a"]: sample["a"], input_["b"]["c"]: sample["b"]["c"]}
    gray_outs = sess.run(gray_ops, feed_dict=feed_dict)
    print(gray_outs)



