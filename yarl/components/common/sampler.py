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

from yarl import get_backend
from yarl.components import Component

if get_backend() == "tf":
    import tensorflow as tf


class Sampler(Component):
    """
    A Sampling component can be used to sample entries from an input op, e.g.
    to repeatedly perform sub-sampling.
    """
    def __init__(self, input_space, sampling_strategy="uniform", scope="sampler", **kwargs):
        """
        Args:
            input_space (Space): The input space to sample on.
            # TODO potentially pass in distribution?
            sampling_strategy (str): Sampling strategy.

        """
        super(Sampler, self).__init__(scope=scope, **kwargs)
        self.sampling_strategy = sampling_strategy
        self.input_names = list(input_space.flatten().keys())
        self.define_inputs("sample_size")
        self.define_inputs(*self.input_names)
        # Returns the same ops.
        output_names = ['out_{}'.format(name) for name in self.input_names]
        self.define_outputs(output_names)

        self.input_names.append("sample_size")
        self.add_graph_fn(self.input_names, output_names, self._graph_fn_sample)

    def _graph_fn_sample(self, sample_size, *inputs):
        """
        Takes a set of inputs and subsamples.

        Args:
            sample_size (int): Subsample size.
            *inputs(DataOp): Input tensors to subsample from. Should all be the same size.

        Returns:
            tuple: The sub-sampled inputs.
        """
        batch_size = tf.shape(input=inputs[0])[0]

        if get_backend() == "tf":
            sample_indices = tf.random_uniform(
                shape=(sample_size,),
                maxval=batch_size,
                dtype=tf.int32
            )
            ret = list()
            for tensor in enumerate(inputs):
                ret.append(tf.gather(params=tensor, indices=sample_indices))
            return ret




