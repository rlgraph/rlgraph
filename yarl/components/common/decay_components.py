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
from yarl.utils import util
from yarl.components import Component


class DecayComponent(Component):
    """
    A base class Component that takes a time input and outputs some decaying-over-time value.

    API:
    ins:
        time_step (int): The current time step.
    outs:
        value (float): The current decayed value based on the time step and c'tor settings.
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
        self.from_ = kwargs.pop("from_", kwargs.pop("from", 1.0))
        self.to_ = kwargs.pop("to_", kwargs.pop("to", 0.1))
        self.start_timestep = kwargs.pop("start_timestep", 0)
        self.num_timesteps = kwargs.pop("num_timesteps", 10000)

        # We only have time-step as input: Do not flatten.
        super(DecayComponent, self).__init__(scope=scope, flatten_ops=kwargs.pop("flatten_ops", False), **kwargs)

        # Our interface.
        self.define_inputs("time_step")
        self.define_outputs("value")
        self.add_graph_fn("time_step", "value", self._graph_fn_value)

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

    def _graph_fn_value(self, time_step):
        """
        Args:
            time_step (DataOp): The int-type DataOp that holds the current global time_step.

        Returns:
            DataOp: The decay'd value depending on the current time step.
        """
        if backend == "tf":
            import tensorflow as tf
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
        if backend == "tf":
            import tensorflow as tf
            return tf.train.polynomial_decay(self.from_, time_steps_in_decay_window, self.num_timesteps,
                                             self.to_, power=self.power)


# Create an alias for LinearDecay
LinearDecay = partial(PolynomialDecay, power=1.0)


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
        if backend == "tf":
            import tensorflow as tf
            return tf.train.exponential_decay(self.from_, time_steps_in_decay_window, self.half_life_timesteps, 0.5)
