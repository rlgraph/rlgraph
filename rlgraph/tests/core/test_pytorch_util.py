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

import logging
import unittest

from rlgraph import get_backend
from rlgraph.tests import recursive_assert_almost_equal
from rlgraph.utils import root_logger, pytorch_one_hot

if get_backend() == "pytorch":
    import torch


class TestPyTorchUtil(unittest.TestCase):
    """
    Tests some torch utils.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_one_hot(self):
        """
        Tests a torch one hot function.
        """
        if get_backend() == "pytorch":
            # Flat action array.
            inputs = torch.tensor([0, 1], dtype=torch.int32)
            one_hot = pytorch_one_hot(inputs, depth=2)

            expected = torch.tensor([[1., 0.], [0., 1.]])
            recursive_assert_almost_equal(one_hot, expected)

            # Container space.
            inputs = torch.tensor([[0, 3, 2],[1, 2, 0]], dtype=torch.int32)
            one_hot = pytorch_one_hot(inputs, depth=4)

            expected = torch.tensor([[[1, 0, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]],[[0, 1, 0, 0],[0, 0, 1, 0],[1, 0, 0, 0,]]],
                                    dtype=torch.int32)
            recursive_assert_almost_equal(one_hot, expected)
