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
from yarl.components import Component

if backend == 'tf':
    import tensorflow as tf


class Optimizer(Component):
    """
    A component that takes a tuple of variables as in-Sockets and optimizes them according to some loss function
    or another criterion or method.

    API:
    ins:
        *inputs (any): Other necessary inputs for the specific type of optimizer (e.g. a time-step).
    outs:
        variables (tuple): The tuple of (trainable) variables to be optimized.
        deltas (tuple): The tuple of delta tensors to be added to each of the variables in the `variables` in-Socket.
        step (DataOp): Same as `deltas`, but also triggers actually applying the deltas to the `variables`.
    """
    def __init__(self, learning_rate, loss_function, *inputs, **kwargs):
        """
        Args:
            learning_rate (float): The learning rate to use.
            loss_function (Component): The LossFunction (Component) to minimize.
        """
        super(Optimizer, self).__init__(scope=kwargs.pop("scope", "optimizer"), **kwargs)

        self.learning_rate = learning_rate

        # TODO Note that we do not use this because we do not use minimize()
        self.loss_function = loss_function

        # Define our interface.
        self.define_inputs("variables", "loss", *inputs)
        self.define_outputs("gradients", "step")
        self.add_graph_fn(["variables", "loss"] + list(inputs), "gradients", self._graph_fn_calculate_gradients)
        self.add_graph_fn(["variables", "gradients"], "step", self._graph_fn_apply_gradients)

    def _graph_fn_calculate_gradients(self, variables, loss, *inputs):
        """
        Calculates the gradients for the given variables and the loss function (and maybe other child-class
            specific input parameters).

        Args:
            loss (SingeDataOp): The total loss over a batch to be minimized.
            variables (DataOpTuple): A list of variables to calculate gradients for.
            inputs (SingleDataOp): Custom SingleDataOp parameters, dependent on the optimizer type.

        Returns:
            DataOpTuple: The gradients per variable (same order as in input parameter `variables`).
        """
        raise NotImplementedError

    def _graph_fn_apply_gradients(self, grads_and_vars):
        """
        Changes the given variables based on the previously calculated gradients. `gradients` is the output of
            `self._graph_fn_calculate_gradients`.

        Args:
            grads_and_vars (DataOpTuple): The list of gradients and variables to be optimized.

        Returns:
            DataOp: The op to trigger the gradient-application step.
        """
        raise NotImplementedError


class LocalOptimizer(Optimizer):
    """
    A local optimizer performs optimization irrespective of any distributed semantics, i.e.
    it has no knowledge of other machines and does not implement any communications with them.
    """
    def __init__(self, learning_rate, loss_function, **kwargs):
        super(LocalOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        self.optimizer = None

    def _graph_fn_calculate_gradients(self, variables, loss, *inputs):
        if backend == 'tf':
            return self.optimizer.compute_gradients(
                loss=loss,
                var_list=variables
            )

    def _graph_fn_apply_gradients(self, grads_and_vars):
        if backend == 'tf':
            return self.optimizer.apply_gradients(
                grads_and_vars=grads_and_vars
            )


class GradientDescentOptimizer(LocalOptimizer):

    def __init__(self, learning_rate, loss_function, **kwargs):
        super(GradientDescentOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        if backend == 'tf':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)


class AdamOptimizer(LocalOptimizer):
    """
    Adaptive momentum optimizer:

    https://arxiv.org/abs/1412.6980
    """
    def __init__(self, learning_rate, loss_function, **kwargs):
        super(AdamOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        if backend == 'tf':
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=kwargs.pop('beta1', 0.9),
                beta2=kwargs.pop('beta2', 0.999)
            )


class NadamOptimizer(LocalOptimizer):
    """
    Nesterov-adaptive momentum optimizer which applies Nesterov's accelerated gradient to
    Adam:

    http://cs229.stanford.edu/proj2015/054_report.pdf
    """
    def __init__(self, learning_rate, loss_function, **kwargs):
        super(NadamOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        if backend == 'tf':
            self.optimizer = tf.keras.optimizers.Nadam(
                lr=self.learning_rate,
                beta_1=kwargs.pop('beta1', 0.9),
                beta_2=kwargs.pop('beta2', 0.999),
                schedule_decay=kwargs.pop('schedule_decay', 0.004),
            )


class AdagradOptimizer(LocalOptimizer):
    """
    Adaptive gradient optimizer which sets small learning rates for frequently appearing features
    and large learning rates for rare features:

    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, learning_rate, loss_function, **kwargs):
        super(AdagradOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        if backend == 'tf':
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate,
                initial_accumulator_value=kwargs.pop('initial_accumulator_value', 0.1)
            )


class AdadeltaOptimizer(LocalOptimizer):
    """
    Adadelta optimizer which adapts learning rate over time:

    https://arxiv.org/abs/1212.5701
    """
    def __init__(self, learning_rate, loss_function, **kwargs):
        super(AdadeltaOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        if backend == 'tf':
            self.optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate,
                rho=kwargs.pop('rho', 0.95)
            )


class SGDOptimizer(LocalOptimizer):
    """
    Stochastic gradient descent optimizer.
    """
    def __init__(self, learning_rate, loss_function, **kwargs):
        super(SGDOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        if backend == 'tf':
            self.optimizer = tf.keras.optimizers.SGD(
                lr=self.learning_rate,
                momentum=kwargs.pop('momentum', 0.0),
                decay=kwargs.pop('decay', 0.0),
            )


class RMSPropOptimizer(LocalOptimizer):
    """
    RMSPRop Optimizer as discussed by Hinton:

    https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, learning_rate, loss_function, **kwargs):
        super(RMSPropOptimizer, self).__init__(learning_rate, loss_function, **kwargs)
        if backend == 'tf':
            self.optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate,
                rho=kwargs.pop('rho', 0.95)
            )