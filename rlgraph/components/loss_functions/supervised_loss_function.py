# Copyright 2018/2019 ducandu GmbH, All Rights Reserved.
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

from rlgraph.components.loss_functions.loss_function import LossFunction
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api


class SupervisedLossFunction(LossFunction):
    """
    Calculates a loss based on predictions and labels.
    Should also optionally support time-ranked data.
    """
    def __init__(self, scope="supervised-loss-function", **kwargs):
        super(SupervisedLossFunction, self).__init__(scope=scope, **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        sanity_check_space(input_spaces["parameters"], must_have_batch_rank=True)
        sanity_check_space(input_spaces["labels"], must_have_batch_rank=True)

    @rlgraph_api
    def loss(self, parameters, labels, sequence_length=None, time_percentage=None):
        loss_per_item = self.loss_per_item(parameters, labels, sequence_length=sequence_length,
                                           time_percentage=time_percentage)
        total_loss = self.loss_average(loss_per_item)
        return total_loss, loss_per_item
