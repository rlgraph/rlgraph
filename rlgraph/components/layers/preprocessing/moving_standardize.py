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
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import SMALL_NUMBER

if get_backend() == "tf":
    import tensorflow as tf


class MovingStandardize(PreprocessLayer):
    """
    Standardizes inputs using a moving estimate of mean and std.
    """
    def __init__(self, scope=", moving-standardize", **kwargs):
        super(MovingStandardize, self).__init__(scope=scope, **kwargs)
        self.sample_count = None

        # Current estimate of state mean.
        self.mean_est = None

        # Current estimate of sum of stds.
        self.std_sum_est = None
        self.output_spaces = None

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["preprocessing_inputs"]
        self.output_spaces = in_space

        if self.backend == "python" or get_backend() == "python":
            self.sample_count = 0
            self.mean_est = np.zeros(in_space.shape)
            self.std_sum_est = np.zeros(in_space.shape)
        elif get_backend() == "tf":
            self.sample_count = self.get_variable(name="sample-count", dtype="int", initializer=0, trainable=False)
            self.mean_est = self.get_variable(
                name="mean-est", trainable=False, from_space=in_space,
                add_batch_rank=in_space.has_batch_rank)
            self.std_sum_est = self.get_variable(
                name="std-sum-est", trainable=False, from_space=in_space,
                add_batch_rank=in_space.has_batch_rank)

    @rlgraph_api
    def _graph_fn_apply(self, preprocessing_inputs):
        if self.backend == "python" or get_backend() == "python" or get_backend() == "pytorch":
            # https: // www.johndcook.com / blog / standard_deviation /
            preprocessing_inputs = np.asarray(preprocessing_inputs)
            prev_count = self.sample_count
            self.sample_count += 1
            if self.sample_count == 1:
                self.mean_est = preprocessing_inputs
            else:
                update = preprocessing_inputs - self.mean_est
                self.mean_est += update / self.sample_count
                self.std_sum_est += update * update * prev_count / self.sample_count

            # Subtract mean.
            standardized = preprocessing_inputs - self.mean_est

            # Estimate variance via sum of variance.
            if self.sample_count > 1:
                var_estimate = self.std_sum_est / (self.sample_count - 1)
            else:
                var_estimate = np.square(self.mean_est)
            std = np.sqrt(var_estimate) + SMALL_NUMBER

            return standardized / std

        elif get_backend() == "tf":
            updates = []
            # 1. Update vars

            # 2. Compute var estimate after update.
            with tf.control_dependencies(updates):
                pass


