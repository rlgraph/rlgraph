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


class NoiseComponent(Component):
    """
    A base class Component that takes an action input and outputs some noise value.

    API:
    ins:
        action (float): The action value input.
    outs:
        noise (float): The noise value to be added to the action.
    """
    def __init__(self, action_space, scope="noise", **kwargs):
        """

        Args:
            action_space: The action space.
        """
        super(NoiseComponent, self).__init__(scope=scope, **kwargs)

        self.action_space = action_space

        # Our interface.
        self.define_outputs("noise")
        self.add_graph_fn(None, "noise", self._graph_fn_value)

    def noise(self):
        """
        The function that returns the DataOp to actually compute the noise.

        Returns:
            DataOp: The noise value.
        """
        return tf.constant(0.0)

    def _graph_fn_value(self):
        """
        Returns:
            DataOp: The noise value.
        """
        return self.noise()


class ConstantNoise(NoiseComponent):
    """
    Simple constant noise component.
    """
    def __init__(self, value=0.0, scope="constant_noise", **kwargs):
        super(ConstantNoise, self).__init__(scope=scope, **kwargs)

        self.value = value

    def noise(self):
        if get_backend() == "tf":
            return tf.constant(self.value)


class GaussianNoise(NoiseComponent):
    """
    Simple Gaussian noise component.
    """
    def __init__(self, mean=0.0, sd=1.0, scope="gaussian_noise", **kwargs):
        super(GaussianNoise, self).__init__(scope=scope, **kwargs)

        self.mean = mean
        self.sd = sd

    def noise(self):
        if get_backend() == "tf":
            return tf.random_normal(
                shape=(1,) + self.action_space.shape,
                mean=self.mean,
                stddev=self.sd,
                dtype=self.action_space.dtype
            )


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
        if get_backend() == "tf":
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
