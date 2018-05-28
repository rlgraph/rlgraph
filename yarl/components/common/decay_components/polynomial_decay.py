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

from functools import partial

from yarl import backend
from .decay_component import DecayComponent


class PolynomialDecay(DecayComponent):
    """
    Component that takes a time input and outputs a linearly decaying value (using init-, and final values).
    The formula is:
    out = (t/T) * (from - to) + to
    where
    - t=time step (counting from the decay start-time, which is not necessarily 0)
    - T=the number of timesteps over which to decay.
    - from=start value
    - to=end value
    """
    def __init__(self, power=1.0, scope="polynomial-decay", **kwargs):
        """
        Args:
            power (float): The polynomial power to use (e.g. 1.0 for linear).

        Keyword Args:
            see DecayComponent
        """
        super(PolynomialDecay, self).__init__(scope=scope, **kwargs)

        self.power = power

    def decay(self, time_steps_in_decay_window):
        if backend() == "tf":
            import tensorflow as tf
            return tf.train.polynomial_decay(self.from_, time_steps_in_decay_window, self.num_timesteps,
                                             self.to_, power=self.power)


LinearDecay = partial(PolynomialDecay, power=1.0)
