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

import tensorflow as tf

from yarl import backend
from yarl.components.common.noise_components.noise import NoiseComponent


class GaussianNoise(NoiseComponent):
    """
    Simple Gaussian noise component.
    """
    def __init__(self, mean=0.0, sd=1.0, scope="gaussian_noise", **kwargs):
        super(GaussianNoise, self).__init__(scope=scope, **kwargs)

        self.mean = mean
        self.sd = sd

    def noise(self):
        if backend == "tf":
            return tf.random_normal(
                shape=(1,) + self.action_space.shape,
                mean=self.mean,
                stddev=self.sd,
                dtype=self.action_space.dtype
            )
