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
import numpy as np

if get_backend() == "pytorch":
    import torch


class PyTorchVariable(object):
    """
    Wrapper to connect PyTorch parameters to names so they can be included
    in variable registries.
    """
    def __init__(self, name, parameters):
        """

        Args:
            name (str): Name of this variable.
            parameters (Union[generator, list]): List of parameters or generator yielding parameters.
        """
        self.name = name
        if isinstance(parameters, list):
            self.parameters = parameters
        else:
            self.parameters = list(parameters)


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


def pytorch_tile(tensor, n_tile, dim=0):
    """
    Tile utility as there is not `torch.tile`.
    Args:
        tensor (torch.Tensor): Tensor to tile.
        n_tile (int): Num tiles.
        dim (int): Dim to tile.

    Returns:
        torch.Tensor: Tiled tensor.
    """
    if isinstance(n_tile, torch.Size):
        n_tile = n_tile[0]
    init_dim = tensor.size(dim)
    repeat_idx = [1] * tensor.dim()
    repeat_idx[dim] = n_tile
    tensor = tensor.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(tensor, dim, order_index)
