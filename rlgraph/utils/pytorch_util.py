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

from rlgraph import get_backend

if get_backend() == "pytorch":
    import torch


def pytorch_one_hot(tensor, depth=0):
    """
    One-hot utility function for PyTorch.

    Args:
        tensor (torch.Tensor): The input to be one-hot.
        depth (int): The max. number to be one-hot encoded (size of last rank).

    Returns:
        torch.Tensor: The one-hot encoded equivalent of the input array.
    """
    if get_backend() == "pytorch":
        tensor_one_hot = torch.FloatTensor(tensor.shape[0], depth)
        tensor_one_hot.zero_()
        tensor_one_hot.scatter_(1, tensor, 1)

        return tensor_one_hot
