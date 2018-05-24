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

from .decay import DecayComponent


class LinearDecay(DecayComponent):
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
    def __init__(self, from_=1.0, to_=0.1, start_timestep=0, num_timesteps=10000,
                 scope="linear-decay", **kwargs):
        """
        Args:
            from_ (float): The max value returned between 0 and `start_timestep`.
            to_ (float): The min value returned from [`start_timestep`+`num_timesteps`] onwards.
            start_timestep (int): The timestep at which to start the decay process.
            num_timesteps (int): The number of time steps over which to decay. Outputs will be stationary before and
                after this decaying period.
        """
        super(LinearDecay, self).__init__(from_=from_, to_=to_, start_timestep=start_timestep,
                                          num_timesteps=num_timesteps, scope=scope, **kwargs)

    def decay(self, time_step):
        progress = time_step / self.num_timesteps
        return progress * (self.from_ - self.to_) + self.to_

