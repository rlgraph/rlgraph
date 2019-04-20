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

import os
from copy import deepcopy

from rlgraph.components import PreprocessorStack


class RayActor(object):
    """
    Generic Ray actor. All classes implementing ray actors should inherit from this.
    """
    def get_host(self):
        """
        Returns host node identifier.

        Returns:
            str: Node name this agent is running on.
        """
        return os.uname()[1]

    def setup_preprocessor(self, preprocessing_spec, in_space):
        if preprocessing_spec is not None:
            preprocessing_spec = deepcopy(preprocessing_spec)
            in_space = deepcopy(in_space)
            # Set scopes.
            scopes = [preprocessor["scope"] for preprocessor in preprocessing_spec]
            # Set backend to python.
            for spec in preprocessing_spec:
                spec["backend"] = "python"
            processor_stack = PreprocessorStack(*preprocessing_spec, backend="python")
            build_space = in_space
            for sub_comp_scope in scopes:
                processor_stack.sub_components[sub_comp_scope].create_variables(input_spaces=dict(
                    preprocessing_inputs=build_space
                ), action_space=None)
                build_space = processor_stack.sub_components[sub_comp_scope].get_preprocessed_space(build_space)
            processor_stack.reset()
            return processor_stack
        else:
            return None
