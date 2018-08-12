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

import numpy as np
from six.moves import xrange as range_
import unittest

from rlgraph.components.common.noise_components import *
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestNoiseComponents(unittest.TestCase):
    """
    Tests RLGraph's noise components.
    """

    # Decaying a value always without batch dimension (does not make sense for global time step).
    action_input_space = FloatBox(1000, add_batch_rank=False)

    def test_constant_noise(self):
        real_noise = 200.0

        noise_component = ConstantNoise(value=real_noise)
        test = ComponentTest(component=noise_component, action_space=self.action_input_space)

        for _ in range_(1000):
            test.test(("get_noise", None), expected_outputs=real_noise)

    def test_gaussian_noise(self):
        real_mean = 10.0
        real_sd = 2.0

        noise_component = GaussianNoise(mean=real_mean, stddev=real_sd)
        test = ComponentTest(component=noise_component, input_spaces=None, action_space=self.action_input_space)

        # Collect outputs in `collected` list to compare moments.
        collected = list()
        collect_outs = lambda component_test, outs: collected.append(outs)

        for _ in range_(1000):
            test.test(("get_noise", None), fn_test=collect_outs)

        test_mean = np.mean(collected)
        test_sd = np.std(collected)

        # Empiric mean should be within 2 sd of real mean
        self.assertGreater(real_mean, test_mean - test_sd * 2)
        self.assertLess(real_mean, test_mean + test_sd * 2)

        # Empiric sd should be within 80 % and 120 % interval
        self.assertGreater(real_sd, test_sd * 0.8)
        self.assertLess(real_sd, test_sd * 1.2)

    def test_ornstein_uhlenbeck_noise(self):
        ou_theta = 0.15
        ou_mu = 10.0
        ou_sigma = 2.0

        noise_component = OrnsteinUhlenbeckNoise(
            theta=ou_theta, mu=ou_mu, sigma=ou_sigma
        )
        test = ComponentTest(component=noise_component, action_space=self.action_input_space)

        # Collect outputs in `collected` list to compare moments.
        collected = list()
        collect_outs = lambda component_test, outs: collected.append(outs)

        for _ in range_(1000):
            test.test(("get_noise", None), fn_test=collect_outs)

        test_mean = np.mean(collected)
        test_sd = np.std(collected)

        print("Moments: {} / {}".format(test_mean, test_sd))

        # Empiric mean should be within 2 sd of real mean.
        self.assertGreater(ou_mu, test_mean - test_sd * 2)
        self.assertLess(ou_mu, test_mean + test_sd * 2)

        # Empiric sd should be within 45% and 200% interval.
        self.assertGreater(ou_sigma, test_sd * 0.45)
        self.assertLess(ou_sigma, test_sd * 2.0)

        # TODO: Maybe test time correlation?
