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

from yarl import Specifiable, backend, YARLError
from yarl.components.layers import Initializer
# TODO: make this backend-dependent
import tensorflow as tf


class Space(Specifiable):
    """
    Space class (based and compatible with openAI).
    Provides a classification for state-, action- and reward spaces.
    """

    @property
    def shape(self):
        """
        The shape of this Space as a tuple.
        """
        raise NotImplementedError

    @property
    def rank(self):
        """
        The rank of the Space (e.g. 3 for a space with shape=(10, 7, 5)).
        """
        return len(self.shape)

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation.
        """
        raise NotImplementedError

    @property
    def dtype(self):
        """
        The dtype (as string) of this Space. Can be converted to tf/np/python dtypes via the misc.utils.dtype function.
        """
        raise NotImplementedError

    def get_tensor_variable(self, name, is_input_feed=False, **kwargs):
        """
        Returns a backend-specific variable/placeholder that matches the space's shape.

        Args:
            name (str): The name for the variable.
            is_input_feed (bool): Whether the returned object should be an input placeholder,
                instead of a full variable.

        Keyword Args:
            To be passed on to backend-specific methods.

        Returns:
            A Tensor Variable/Placeholder.
        """
        if backend() == "tf":
            if is_input_feed:
                return tf.placeholder(dtype=self.dtype, shape=self.shape, name=name)
            else:
                return tf.get_variable(name, shape=self.shape, dtype=self.dtype, **kwargs)
        else:
            raise YARLError("ERROR: Pytorch not supported yet!")

    #def get_initializer(self, specification):
    #    """
    #    Returns a backend-specific initializer object that matches the space's shape.
    #
    #    Args:
    #        specification (any): The specification to be passed into
    #
    #    Returns:
    #        An Initializer object.
    #    """
    #    return None  # optional

    def __repr__(self):
        return "Space(shape=" + str(self.shape) + ")"

    def __eq__(self, other):
        raise NotImplementedError

    def sample(self, seed=None):
        """
        Uniformly randomly samples an element from this space. This is more for testing purposes, e.g. to simulate
        a random environment.
        Args:
            seed (int): The random seed to use.
        Returns: The sampled element.
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Checks whether this space contains the given sample. This is more for testing purposes.
        Args:
            x: The element to check.
        Returns: Whether x is a valid member of this space.
        """
        raise NotImplementedError

