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


class OrnsteinUhlenbeckNoise(NoiseComponent):
    """
    Ornstein-Uhlenbeck noise component emitting a mean-reverting time-correlated stochastic noise.
    """
    def __init__(self, sigma=0.3, mu=0.0, theta=0.15, scope="ornstein_uhlenbeck_noise", **kwargs):
        """

        Args:
            sigma:
            mu: Mean reversion level
            theta: Mean reversion rate
            scope:
            **kwargs:
        """
        super(OrnsteinUhlenbeckNoise, self).__init__(scope=scope, **kwargs)

        self.sigma = sigma
        self.mu = mu
        self.theta = theta

    def noise(self):
        if backend == "tf":
            standard_noise = tf.random_normal(
                shape=self.action_space.shape,
                mean=0.0,
                stddev=1.0,
                dtype=self.action_space.dtype
            )
            ou_state = self.get_variable(
                name="ou_state",
                from_space=self.action_space,
                add_batch_rank=False,
                initializer=self.mu
            )

            drift = self.theta * (self.mu - ou_state)
            diffusion = self.sigma * standard_noise

            delta = drift + diffusion

            return tf.assign_add(ref=ou_state, value=delta)
