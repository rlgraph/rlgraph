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

from yarl import backend

from .decay_component import DecayComponent


class ExponentialDecay(DecayComponent):
    """
    Component that takes a time input and outputs an exponentially decaying value (using a half-life parameter and
    init-, and final values).
    The formula is:
    out = 2exp(-t/h) * (from - to) + to
    where
    - t=time step (counting from the decay start-time, which is not necessarily 0)
    - h=the number of timesteps over which the decay is 50%.
    - from=start value
    - to=end value
    """
    def __init__(self, half_life=None, num_half_lives=10, scope="exponential-decay", **kwargs):
        """
        Args:
            half_life (Optional[int]): The half life period in number of timesteps. Use `num_half_lives` for a relative
                measure against `num_timesteps`.
            num_half_lives (Optional[int]): The number of sub-periods into which `num_timesteps` will be divided, each
                division being the length of time in which we decay 50%. This is an alternative to `half_life`.

        Keyword Args:
            see DecayComponent
        """
        assert isinstance(half_life, int) or isinstance(num_half_lives, int)

        super(ExponentialDecay, self).__init__(scope=scope, **kwargs)

        self.half_life_timesteps = half_life if half_life is not None else self.num_timesteps / num_half_lives

    def decay(self, time_steps_in_decay_window):
        if backend() == "tf":
            import tensorflow as tf
            return tf.train.exponential_decay(self.from_, time_steps_in_decay_window, self.half_life_timesteps, 0.5)
