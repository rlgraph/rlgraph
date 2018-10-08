# Copyright 2018 The RLgraph authors. All Rights Reserved.
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

from rlgraph import get_backend, get_distributed_backend
from rlgraph.components.optimizers.optimizer import Optimizer


if get_backend() == "tf" and get_distributed_backend() == "horovod":
    import horovod.tensorflow as hvd
elif get_backend() == "pytorch" and get_backend() == "horovod":
    import horovod.pytorch as hvd


class HorovodOptimizer(Optimizer):
    """
    This Optimizer provides a wrapper for the horovod optimizer package:

    https://github.com/uber/horovod

    Horovod is meant to be used as an alternative to distributed TensorFlow as it implements
    communication in a different way, as explained in the Horovod paper:

    arXiv:1802.05799

    This Horovod Optimizer expects a local LocalOptimizer spec (tensorflow) as input.
    """
    def __init__(self, local_optimizer=None, **kwargs):
        """
        Initializes a distributed horovod optimizer by wrapping a local optimizer.

        Args:
            local_optimizer (Optional[dict,LocalOptimizer]): The spec-dict for the wrapped LocalOptimizer object or
                a LocalOptimizer object itself.
        """
        super(HorovodOptimizer, self).__init__(**kwargs)

        # Create the horovod wrapper.
        wrapped_local_optimizer = Optimizer.from_spec(local_optimizer)
        self.local_optimizer = hvd.DistributedOptimizer(wrapped_local_optimizer)

        @api
        def step(self, variables, loss, *inputs):
            grads_and_vars = self._graph_fn_calculate_gradients(variables, loss, *inputs)
            return self._graph_fn_apply_gradients(grads_and_vars)

    def _graph_fn_calculate_gradients(self, variables, loss, *inputs):
        return self.local_optimizer._graph_fn_calculate_gradients(variables, loss, *inputs)

    def _graph_fn_apply_gradients(self, grads_and_vars):
        return self.local_optimizer._graph_fn_apply_gradients(grads_and_vars)

