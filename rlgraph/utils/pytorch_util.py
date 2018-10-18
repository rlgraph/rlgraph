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
    def __init__(self, name, ref):
        """

        Args:
            name (str): Name of this variable.
            ref (torch.nn.Module): Ref to the layer or network object.
        """
        self.name = name
        self.ref = ref

    def get_value(self):
        if get_backend() == "pytorch":
            if isinstance(self.ref, torch.nn.Module):
                return self.ref.weight

    def set_value(self, value):
        if get_backend() == "pytorch":
            if isinstance(self.ref, torch.nn.Module):
                if isinstance(value, torch.Tensor):
                    self.ref.weight = torch.nn.Parameter(value, requires_grad=True)
                elif isinstance(value, torch.nn.Parameter):
                    self.ref.weight = value
                else:
                    raise ValueError("Value assigned must be torch.Tensor or Parameter but is {}.".format(
                        type(value)
                    ))


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
        # Do converts.
        if isinstance(tensor, torch.FloatTensor):
            tensor = tensor.long()
        if isinstance(tensor, torch.IntTensor):
            tensor = tensor.long()
        # Un-squeeze 1d.
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        # print("one hot input: {}, dim: {}, shape {}, type {}".format(
        #     tensor, tensor.dim(), tensor.shape, tensor.dtype
        # ))
        # print("tensor.shape[0]", tensor.shape[0])
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


# TODO remove when we have handled pytorch placeholder inference better.
def get_input_channels(shape):
    """
    Helper for temporary issues with PyTorch shape inference.

    Args:
        shape (Tuple): Shape tuple.

    Returns:
        int: Num input channels.
    """
    # Have batch rank from placeholder space reconstruction:
    if len(shape) == 4:
        # Batch rank and channels first.
        return shape[1]
    elif len(shape) == 3:
        # No batch rank and channels first.
        return shape[0]


# TODO remove when new PyTorch versions support multi dim reductions/
def pytorch_reduce_mean(tensor, axes, keepdims):
    """
    Reduces a tensor along multiple axes as this is not natively supported.
    Args:
        tensor (torch.Tensor): Tensor to reduce.
        axes (list): Axes to reduce.
        keepdims (bool): If to keep dims.

    Returns:
        torch.Tensor: Result of reduction.
    """
    if keepdims:
        for axis in axes:
            tensor = tensor.mean(axis, keepdim=True)
    else:
        for axis in sorted(axes, reverse=True):
            tensor = tensor.mean(axis, keepdim=False)
    return tensor


if get_backend() == "pytorch":
    SMALL_NUMBER_TORCH = torch.tensor([1e-6])
    LOG_SMALL_NUMBER = torch.log(SMALL_NUMBER_TORCH)

    class SamePaddedConv2d(torch.nn.Module):
        """
        Implements a Conv2d layer with padding 'same' as PyTorch does not have
        padding options like TF.
        """
        def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1,
                     padding_layer=torch.nn.ReflectionPad2d):
            super().__init__()
            ka = kernel_size // 2
            kb = ka - 1 if kernel_size % 2 == 0 else ka

            self.layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride)
            self.net = torch.nn.Sequential(
                padding_layer((ka,kb,ka,kb)),
                self.layer
            )

            self.weight = self.layer.weight
            self.bias = self.layer.bias

        def forward(self, x):
            return self.net(x)

        def parameters(self):
            return self.layer.parameters()


