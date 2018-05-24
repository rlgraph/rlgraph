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

from yarl.components import Component


class Distribution(Component):
    """
    A distribution wrapper class that can incorporate a backend-specific distribution object that gets its parameters
    from an external source (e.g. a NN).
    """
    def __init__(self, scope="distribution", **kwargs):
        super(Distribution, self).__init__(scope=scope, **kwargs)

        # Define a generic Distribution interface.
        self.define_inputs("raw_input", "num_samples")
        self.define_outputs("sample")
        self.add_computation("raw_input", "distribution", self._computation_parameterize)
        self.add_computation(["distribution", "num_samples"], "sample", self._computation_sample)
        self.add_computation("distribution", "entropy", self._computation_entropy)

    def _computation_parameterize(self, raw_input):
        """
        Parameterizes this distribution (normally from an NN-output vector). Returns the backend-distribution object
        (a DataOp).

        Args:
            raw_input (DataOp): The input used to parameterize this distribution. This is normally a NN-output layer
                that, for example, can hold the two values for mean and variance for a univariate Gaussian
                distribution.

        Returns:
            DataOp: The parameterized backend-specific distribution object.
        """
        raise NotImplementedError

    def _computation_sample(self, distribution, num_samples):
        """
        Takes a sample of size n from the distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution DataOp to use for
                sampling.
            num_samples (int): The number of single samples to take.

        Returns:
            DataOp: The taken sample(s).
        """
        raise NotImplementedError

    def _computation_entropy(self, distribution):
        """
        Returns the DataOp holding the entropy value of the distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution whose entropy to
                calculate.

        Returns:
            DataOp: The distribution's entropy.
        """
        raise NotImplementedError

