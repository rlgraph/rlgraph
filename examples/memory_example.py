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

from yarl.components import Component
from yarl.spaces import Continuous, Tuple, Dict


class Memory(Component):
    def __init__(self, *args, **kwargs):
        super(Memory, self).__init__(*args, **kwargs)
        self.record_space = Dict(
            state=Dict(state1=float, state2=float),
            reward=float
        )
        self.memory = self.get_variable("memory", trainable=False, from_space=self.record_space)
        self.index = self.get_variable("index", trainable=False, dtype=int)
        self.capacity = 5
        # TODO: put this into splitter component:
        self.var_list = [self.memory["state"]["state1"], self.memory["state"]["state2"], self.memory["reward"]]

        # TODO: this is normally done automatically by Component (see commented-out code below)
        self.insert = autograph.to_graph(self.computation_insert, verbose=True, partial_types={int})  #, partial_types={str, tuple})
        #self.define_inputs("record")
        #self.define_outputs("insert")
        #self.add_computation("record", "insert")

    def computation_insert(self, *records):
        num_elements = 1  #tf.shape(records[0])[0]
        num_primitives = tf.shape(records)[0]
        print(num_primitives)
        index_range = tf.range(start=self.index, limit=(self.index + num_elements)) % self.capacity
        #print("index_range={}".format(index_range))
        print(records[0])
        for n in range(num_primitives):  #, record in enumerate(list(records)):
            print(n)
        #    tf.scatter_update(
        #        ref=self.var_list[n],
        #        indices=index_range,
        #        updates=records[n]
        #    )

        tf.assign(ref=self.index, value=(self.index + num_elements) % self.capacity)
        return tf.no_op()


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
    memory = Memory()
    input_ = memory.record_space.get_tensor_variable(name="placeholder")

    # The wrapper will live in Component.py and should not need to be overwritten ever (I think).
    # We can call it something else, but component will use it all under the hood, automatically.

    # Test the pipeline.
    sample = memory.record_space.sample()
    feed_dict = {}
    get_feed_dict(feed_dict, sample, input_)

    sess.run(tf.global_variables_initializer())
    outputs = sess.run(memory.insert(memory, input_["state"]["state1"], input_["state"]["state2"], input_["reward"]), feed_dict=feed_dict)
    print("Sess.run="+str(outputs))
