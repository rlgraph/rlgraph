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

from rlgraph import get_backend
from rlgraph.components.common.time_dependent_parameters import Constant
from rlgraph.components.optimizers.optimizer import Optimizer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import DataOpTuple
from rlgraph.utils.util import force_list

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class LocalOptimizer(Optimizer):
    """
    A local optimizer performs optimization irrespective of any distributed semantics, i.e.
    it has no knowledge of other machines and does not implement any communications with them.
    """
    def __init__(self, learning_rate, clip_grad_norm=None, **kwargs):
        super(LocalOptimizer, self).__init__(
            learning_rate=learning_rate, scope=kwargs.pop("scope", "local-optimizer"), **kwargs
        )

        self.clip_grad_norm = clip_grad_norm
        if self.clip_grad_norm is not None:
            assert isinstance(self.clip_grad_norm, float) or isinstance(self.clip_grad_norm, int),\
                "ERROR: 'clip_grad_norm' must be of type float or int but is type {}".format(type(self.clip_grad_norm))

        self.input_complete = True

        # The wrapped, backend-specific optimizer object.
        self.optimizer = None

        # For define-by-run instances.
        self.optimizer_obj = None

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_step(self, variables, loss, loss_per_item, time_percentage, *inputs):
        # TODO n.b. PyTorch does not call api functions because other optimization semantics.
        if get_backend() == "tf":
            grads_and_vars = self._graph_fn_calculate_gradients(variables, loss, time_percentage)
            step_op = self._graph_fn_apply_gradients(grads_and_vars)
            return step_op
        elif get_backend() == "pytorch":
            # Instantiate optimizer with variables.
            if self.optimizer_obj is None:
                # self.optimizer is a lambda creating the respective optimizer
                # with params pre-filled.
                parameters = variables.values()
                self.optimizer_obj = self.optimizer(parameters)
            # Reset gradients.
            self.optimizer_obj.zero_grad()
            if not torch.isnan(loss):
                loss.backward()
            # Adjust learning rate via time-dependent parameter if not a constant.
            if not isinstance(self.learning_rate, Constant):
                lr = self.learning_rate.get(time_percentage)
                for param_group in self.optimizer_obj.param_groups:
                    param_group["lr"] = lr
            # Do the optimizer step.
            return self.optimizer_obj.step()

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_calculate_gradients(self, variables, loss, time_percentage):
        """
        Args:
            variables (DataOpTuple): A list of variables to calculate gradients for.
            loss (SingeDataOp): The total loss over a batch to be minimized.
        """
        if get_backend() == "tf":
            var_list = list(variables.values()) if isinstance(variables, dict) else force_list(variables)
            grads_and_vars = self.optimizer.compute_gradients(
                loss=loss,
                var_list=var_list
            )
            if self.clip_grad_norm is not None:
                for i, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grads_and_vars[i] = (tf.clip_by_norm(t=grad, clip_norm=self.clip_grad_norm), var)
            return DataOpTuple(grads_and_vars)

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_apply_gradients(self, grads_and_vars):
        if get_backend() == "tf":
            return self.optimizer.apply_gradients(
                grads_and_vars=grads_and_vars
            )

    def get_optimizer_variables(self):
        if get_backend() == "tf":
            return self.optimizer.variables()
        # Nothing to do for PyTorch -> no variable init issues.


class GradientDescentOptimizer(LocalOptimizer):
    """
    Classic gradient descent optimizer:
    "Stochastic Estimation of the Maximum of a Regression Function." - Kiefer and Wolfowitz, 1952
    """
    def __init__(self, learning_rate, **kwargs):
        super(GradientDescentOptimizer, self).__init__(
            learning_rate=learning_rate,
            scope=kwargs.pop("scope", "gradient-descent-optimizer"),
            **kwargs
        )

    def create_variables(self, input_spaces, action_space=None):
        if get_backend() == "tf":
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate.placeholder()
            )
        elif get_backend() == "pytorch":
            raise NotImplementedError


class AdamOptimizer(LocalOptimizer):
    """
    Adaptive momentum optimizer:
    https://arxiv.org/abs/1412.6980
    """
    def __init__(self, learning_rate, **kwargs):
        self.beta1 = kwargs.pop("beta_1", kwargs.pop("beta1", 0.9))
        self.beta2 = kwargs.pop("beta_2", kwargs.pop("beta2", 0.999))

        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate, scope=kwargs.pop("scope", "adam-optimizer"), **kwargs
        )

    def check_input_spaces(self, input_spaces, action_space=None):
        if get_backend() == "tf":
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate.placeholder(),
                beta1=self.beta1,
                beta2=self.beta2
            )
        elif get_backend() == "pytorch":
            # Cannot instantiate yet without weights.
            self.optimizer = lambda parameters: torch.optim.Adam(
                parameters,
                lr=self.learning_rate.from_,
                betas=(self.beta1, self.beta2)
            )


class NadamOptimizer(LocalOptimizer):
    """
    Nesterov-adaptive momentum optimizer which applies Nesterov's accelerated gradient to Adam:

    http://cs229.stanford.edu/proj2015/054_report.pdf
    """
    def __init__(self, learning_rate, **kwargs):
        self.beta1 = kwargs.pop("beta_1", kwargs.pop("beta1", 0.9))
        self.beta2 = kwargs.pop("beta_2", kwargs.pop("beta2", 0.999))
        self.schedule_decay = kwargs.pop("schedule_decay", 0.004)

        super(NadamOptimizer, self).__init__(
            learning_rate=learning_rate, scope=kwargs.pop("scope", "nadam-optimizer"), **kwargs
        )

    def check_input_spaces(self, input_spaces, action_space=None):
        if get_backend() == "tf":
            self.optimizer = tf.keras.optimizers.Nadam(
                lr=self.learning_rate.placeholder(),
                beta_1=self.beta1,
                beta_2=self.beta2,
                schedule_decay=self.schedule_decay
            )
        elif get_backend() == "pytorch":
            raise NotImplementedError


class AdagradOptimizer(LocalOptimizer):
    """
    Adaptive gradient optimizer which sets small learning rates for frequently appearing features
    and large learning rates for rare features:

    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, learning_rate, **kwargs):
        self.initial_accumulator_value = kwargs.pop("initial_accumulator_value", 0.1)

        super(AdagradOptimizer, self).__init__(
            learning_rate=learning_rate.placeholder(),
            scope=kwargs.pop("scope", "adagrad-optimizer"),
            **kwargs
        )

    def check_input_spaces(self, input_spaces, action_space=None):
        if get_backend() == "tf":
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate.placeholder(),
                initial_accumulator_value=self.initial_accumulator_value
            )
        elif get_backend() == "pytorch":
            # Cannot instantiate yet without weights.
            self.optimizer = lambda parameters: torch.optim.Adagrad(
                parameters,
                lr=self.learning_rate.from_,
                initial_accumulator_value=self.initial_accumulator_value
            )


class AdadeltaOptimizer(LocalOptimizer):
    """
    Adadelta optimizer which adapts learning rate over time:

    https://arxiv.org/abs/1212.5701
    """
    def __init__(self, learning_rate, **kwargs):
        self.rho = kwargs.pop("rho", 0.95)

        super(AdadeltaOptimizer, self).__init__(
            learning_rate=learning_rate, scope=kwargs.pop("scope", "adadelta-optimizer"), **kwargs
        )

    def check_input_spaces(self, input_spaces, action_space=None):
        if get_backend() == "tf":
            self.optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate.placeholder(),
                rho=self.rho
            )
        elif get_backend() == "pytorch":
            # Cannot instantiate yet without weights.
            self.optimizer = lambda parameters: torch.optim.Adadelta(
                parameters,
                lr=self.learning_rate.from_,
                rho=self.rho
            )


class SGDOptimizer(LocalOptimizer):
    """
    Stochastic gradient descent optimizer from tf.keras including support for momentum,
    learning-rate-decay and Nesterov momentum.
    """
    def __init__(self, learning_rate, **kwargs):
        self.momentum = kwargs.pop("momentum", 0.0)
        self.decay = kwargs.pop("decay", 0.0)
        self.nesterov = kwargs.pop("nesterov", False)

        super(SGDOptimizer, self).__init__(
            learning_rate=learning_rate, scope=kwargs.pop("scope", "sgd-optimizer"), **kwargs
        )

    def check_input_spaces(self, input_spaces, action_space=None):
        if get_backend() == "tf":
            self.optimizer = tf.keras.optimizers.SGD(
                lr=self.learning_rate.placeholder(),
                momentum=self.momentum,
                decay=self.decay,
                nesterov=self.nesterov
            )
        elif get_backend() == "pytorch":
            # Cannot instantiate yet without weights.
            self.optimizer = lambda parameters: torch.optim.SGD(
                parameters,
                lr=self.learning_rate.from_,
                momentum=self.momentum,
                weight_decay=self.decay,
                nesterov=self.nesterov
            )


class RMSPropOptimizer(LocalOptimizer):
    """
    RMSProp Optimizer as discussed by Hinton:

    https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, learning_rate, **kwargs):
        self.decay = kwargs.pop("decay", 0.99)
        self.momentum = kwargs.pop("momentum", 0.0)
        self.epsilon = kwargs.pop("epsilon", 0.1)

        super(RMSPropOptimizer, self).__init__(
            learning_rate=learning_rate, scope=kwargs.pop("scope", "rms-prop-optimizer"), **kwargs
        )

    def check_input_spaces(self, input_spaces, action_space=None):
        if get_backend() == "tf":
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate,
                decay=self.decay,
                momentum=self.momentum,
                epsilon=self.epsilon
            )
        elif get_backend() == "pytorch":
            # Cannot instantiate yet without weights.
            self.optimizer = lambda parameters: torch.optim.RMSprop(
                parameters,
                lr=self.learning_rate.from_,
                momentum=self.momentum,
                weight_decay=self.decay,
            )

