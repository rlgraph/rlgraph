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

from yarl.utils import util
from yarl.components import Component


class DecayComponent(Component):
    """
    A base class Component that takes a time input and outputs some decaying-over-time value.
    """
    def __init__(self, scope="decay", **kwargs):
        """
        Keyword Args:
            from_ (float): The max value returned between 0 and `start_timestep`.
            to_ (float): The min value returned from [`start_timestep`+`num_timesteps`] onwards.
            start_timestep (int): The timestep at which to start the decay process.
            num_timesteps (int): The number of time steps over which to decay. Outputs will be stationary before and
                after this decaying period.
        """
        super(DecayComponent, self).__init__(scope=scope, **kwargs)

        self.from_ = kwargs.get("from", 1.0)
        self.to_ = kwargs.get("to", 0.1)
        self.start_timestep = kwargs.get("start_timestep", 0)
        self.num_timesteps = kwargs.get("num_timesteps", 10000)

        # Our interface.
        self.define_inputs("time_step")
        self.define_outputs("value")
        self.add_computation("time_step", "value", self._computation_value)

    def decay(self, time_steps_in_decay_window):
        """
        The function that returns the DataOp to actually compute the decay during the decay time period.

        Args:
            time_steps_in_decay_window (DataOp): The time-step value (already cast to float) based on
                `self.start_timestep` (not the global time-step value).
                E.g. time_step=10.0 if global-timestep=100 and `self.start_timestep`=90.

        Returns:
            DataOp: The decay'd value (may be based on time_steps_in_decay_window).
        """
        raise NotImplementedError

    def _computation_value(self, time_step):
        """
        Args:
            time_step (DataOp): The int-type DataOp that holds the current global time_step.

        Returns:
            DataOp: The decay'd value depending on the current time step.
        """
        return tf.cond(pred=(time_step <= self.start_timestep),
                       # We are still in pre-decay time.
                       true_fn=lambda: self.from_,
                       false_fn=lambda: tf.cond(pred=(time_step >= self.start_timestep + self.num_timesteps),
                                                # We are in post-decay time.
                                                true_fn=lambda: self.to_,
                                                # We are inside the decay time window.
                                                false_fn=lambda: self.decay(tf.cast(x=time_step - self.start_timestep,
                                                                                    dtype=util.dtype("float"))))
                       )

