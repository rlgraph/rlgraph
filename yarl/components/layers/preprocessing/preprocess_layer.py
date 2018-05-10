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
from __future__ import print_function
from __future__ import division

from yarl.components.layers import LayerComponent
from yarl.spaces import Dict, Tuple


class PreprocessLayer(LayerComponent):
    """
    A LayerComponent that can serve as a preprocessing layer and also can act on complex container input
    spaces (Dict or Tuple).
    Do not override the `apply` computation method. Instead, override the `preprocess` method, which
    gets called automatically by `apply` after taking care of container inputs.
    It is not required to implement the `reset` logic (or store any state information at all).
    """
    def _computation_reset(self):
        """
        Returns:
            An op that resets this processor to some initial state.
            E.g. could be called whenever an episode ends.
            This could be useful if the preprocessor stores certain episode-sequence information
            to do the processing and this information has to be reset after the episode terminates.
        """
        pass  # Not mandatory.

    def _computation_apply(self, *inputs):
        """
        Args:
            inputs (any): The input to be "pre-processed".

        Returns:
            The op that pre-processes the input.
        """
        raise NotImplementedError

    def processed_shape(self, space):
        """
        TODO: make it such that children of this class don't have to re-implement this function. It should be automatic.

        Returns the processed shape for an input of space `space`.

        Args:
            space (Space): The Space of the input to _computation_apply.

        Returns:
            The shape of the return value of _computation_apply (if the input space is `space`).
        """
        # Use numpy semantics.
        sample = space.sample()
        out = self._computation_apply(sample)
        return out.shape

