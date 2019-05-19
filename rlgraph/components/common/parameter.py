# Copyright 2018/2019 The Rlgraph Authors, All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    pass


class Parameter(Component):
    """
    A time-dependent or constant parameter that can be used to implement learning rate or other decaying parameters.
    """
    def __init__(self, from_=None, to_=None, max_time_steps=None, resolution=1000, scope="parameter", **kwargs):
        """
        Args:
            from_ (float): The constant value or start value to use.
            to_ (Optional[float]): The value to move towards if this parameter is time-dependent.
            max_time_steps (Optional[int]): The maximum number of time-steps to use for percentage/decay calculations.
                IF not provided, the time-step percentage must be passed into API-calls to `get`.
            resolution (int): The resolution to use as "max time steps" value when calculating the current time step
                from the `time_percentage` parameter. The exact formula is: current_ts=time_percentage

        Keyword Args:
            from: Instead of arg `from_`.
            to: Instead of arg `to`.
        """
        kwargs_from = kwargs.pop("from", None)
        kwargs_to = kwargs.pop("to", None)

        super(Parameter, self).__init__(scope=scope, **kwargs)

        self.from_ = kwargs_from if kwargs_from is not None else from_ if from_ is not None else 1.0
        self.to_ = kwargs_to if kwargs_to is not None else to_ if to_ is not None else 0.0

        self.max_time_steps = max_time_steps
        self.resolution = resolution

    def check_input_completeness(self):
        # If max_time_steps is not given, we will rely on time_percentage input, therefore, it must be given.
        if self.max_time_steps is None and self.api_method_inputs["time_percentage"] is "flex":
            return False
        return super(Parameter, self).check_input_completeness()

    def check_input_spaces(self, input_spaces, action_space=None):
        time_pct_space = input_spaces["time_percentage"]

        # Time percentage is only not needed, iff backend=tf and we have a max_timesteps property with which we
        # can derive the percentage from the tf GLOBAL_TIMESTEP variable.
        if time_pct_space == "flex":
            assert get_backend() == "tf", "ERROR: `time_percentage` can only be left out if using tf as backend!"
            assert self.max_time_steps is not None, \
                "ERROR: `time_percentage` can only be left out if `self.max_time_steps` is not None!"

    @rlgraph_api
    def get(self, time_percentage=None):
        raise NotImplementedError

    def placeholder(self):
        """
        Creates a connection to a tf placeholder (completely outside the RLgraph meta-graph).
        Passes that placeholder through one run of our `get` API method and then returns the output op.
        That way, this parameter can be used inside a tf.optimizer object as the learning rate tensor.

        Returns:
            The tf op to calculate the learning rate from the `time_percentage` placeholder.
        """
        assert get_backend() == "tf"  # We must use tf for this to work.
        assert self.graph_builder is not None  # We must be in the build phase.
        # Get the placeholder (always the same!) for the `time_percentage` input.
        placeholder = self.graph_builder.get_placeholder("time_percentage", float, self)
        # Do the actual computation to get the current value for the parameter.
        op = self.api_methods["get"].func(self, placeholder)
        # Return the tf op.
        return op

    @classmethod
    def from_spec(cls, spec=None, **kwargs):
        map_ = {
            "lin": "linear-parameter",
            "linear": "linear-parameter",
            "polynomial": "polynomial-parameter",
            "poly": "polynomial-parameter",
            "exp": "exponential-parameter",
            "exponential": "exponential-parameter"
        }
        # Single float means constant parameter.
        if isinstance(spec, float):
            spec = dict(value=spec, type="constant-parameter")
        # List/tuple means simple (type)?/from/to setup.
        elif isinstance(spec, (tuple, list)):
            if len(spec) == 2:
                spec = dict(from_=spec[0], to_=spec[1], type="linear-parameter")
            elif len(spec) == 3:
                spec = dict(from_=spec[1], to_=spec[2], type=map_.get(spec[0], spec[0]))
            elif len(spec) == 4:
                type_ = map_.get(spec[0], spec[0])
                spec = dict(from_=spec[1], to_=spec[2], type=type_)
                if type_ == "polynomial-parameter":
                    spec["power"] = spec[3]
                elif type_ == "exponential-parameter":
                    spec["decay_rate"] = spec[3]
        return super(Parameter, cls).from_spec(spec, **kwargs)


class ConstantParameter(Parameter):
    """
    Always returns a constant value no matter what value `time_percentage` or GLOBAL_STEP have.
    """
    def __init__(self, value, scope="constant-parameter", **kwargs):
        super(ConstantParameter, self).__init__(from_=value, scope=scope, **kwargs)

    @rlgraph_api
    def _graph_fn_get(self, time_percentage=None):
        if get_backend() == "tf":
            if time_percentage is not None:
                return tf.fill(tf.shape(time_percentage), self.from_)
            else:
                return self.from_

    def placeholder(self):
        return self.from_


class PolynomialParameter(Parameter):
    """
    Returns the result of:
    to_ + (from_ - to_) * (1 - `time_percentage`) ** power
    """
    def __init__(self, power=1.0, scope="polynomial-parameter", **kwargs):
        super(PolynomialParameter, self).__init__(scope=scope, **kwargs)

        self.power = power

    @rlgraph_api
    def _graph_fn_get(self, time_percentage=None):
        if time_percentage is None:
            assert get_backend() == "tf"  # once more, just in case
            return tf.train.polynomial_decay(
                learning_rate=self.from_, global_step=tf.train.get_global_step(),
                decay_steps=self.max_time_steps,
                end_learning_rate=self.to_,
                power=self.power
            )
        else:
            # Get the fake current time-step from the percentage value.
            current_timestep = self.resolution * time_percentage
            if get_backend() == "tf":
                return tf.train.polynomial_decay(
                    learning_rate=self.from_, global_step=current_timestep,
                    decay_steps=self.resolution,
                    end_learning_rate=self.to_,
                    power=self.power
                )


class LinearParameter(PolynomialParameter):
    """
    Same as polynomial with power=1.0. Returns the result of:
    from_ - `time_percentage` * (from_ - to_)
    """
    def __init__(self, scope="linear-parameter", **kwargs):
        super(LinearParameter, self).__init__(power=1.0, scope=scope, **kwargs)


class ExponentialParameter(Parameter):
    """
    Returns the result of:
    to_ + (from_ - to_) * decay_rate ** `time_percentage`
    """
    def __init__(self, decay_rate=1.0, scope="exponential-parameter", **kwargs):
        super(ExponentialParameter, self).__init__(scope=scope, **kwargs)
        self.decay_rate = decay_rate

    @rlgraph_api
    def _graph_fn_get(self, time_percentage=None):
        if time_percentage is None:
            assert get_backend() == "tf"  # once more, just in case
            return tf.train.exponential_decay(
                learning_rate=self.from_ - self.to_, global_step=tf.train.get_global_step(),
                decay_steps=self.max_time_steps,
                decay_rate=self.decay_rate
            ) + self.to_
        else:
            # Get the fake current time-step from the percentage value.
            current_timestep = self.resolution * time_percentage
            if get_backend() == "tf":
                return tf.train.exponential_decay(
                    learning_rate=self.from_ - self.to_, global_step=current_timestep,
                    decay_steps=self.resolution,
                    decay_rate=self.decay_rate
                ) + self.to_
