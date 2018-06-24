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
from yarl.utils.ops import FlattenedDataOp
from yarl.utils.util import get_batch_size

if get_backend() == "tf":
    import tensorflow as tf


class Sampler(Component):
    """
    A Sampling component can be used to sample entries from an input op, e.g.
    to repeatedly perform sub-sampling.
    """
    def __init__(self, sampling_strategy="uniform", scope="sampler", **kwargs):
        """
        Args:
            # TODO potentially pass in distribution?
            sampling_strategy (str): Sampling strategy.
        """
        super(Sampler, self).__init__(scope=scope, **kwargs)
        self.sampling_strategy = sampling_strategy

        # Define our interface.
        self.define_inputs("sample_size", "inputs")
        self.define_outputs("sample")
        # Connect the graph_fn, only flatten the incoming sample, not sample_size.
        self.add_graph_fn(["sample_size", "inputs"], "sample", self._graph_fn_sample,
                          flatten_ops={"inputs"})

    def _graph_fn_sample(self, sample_size, inputs):
        """
        Takes a set of input tensors and uniformly samples a subset of the
        specified size from them.

        Args:
            sample_size (SingleDataOp[int]): Subsample size.
            inputs (FlattenedDataOp): Input tensors (in a FlattenedDataOp) to sample from.
                All values (tensors) should all be the same size.

        Returns:
            FlattenedDataOp: The sub-sampled inputs (will be unflattened automatically).
        """
        batch_size = get_batch_size(next(iter(inputs.values())))

        if get_backend() == "tf":
            sample_indices = tf.random_uniform(
                shape=(sample_size,),
                maxval=batch_size,
                dtype=tf.int32
            )
            sample = FlattenedDataOp()
            for key, tensor in inputs.items():
                sample[key] = tf.gather(params=tensor, indices=sample_indices)
            return sample
