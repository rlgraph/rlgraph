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

from yarl import backend
from yarl.components.layers import LayerComponent


class Distribution(LayerComponent):
    """
    A distribution wrapper class that can incorporate a backend-specific distribution object.
    """
    def __init__(self, *sub_components, class_=None, **kwargs):
        """
        Keyword Args:
            class (class): The wrapped tf.layers class to use.
            **kwargs (any): Kwargs to be passed to the native backend's layers's constructor.
        """
        assert class_, "ERROR: class_ parameter needs to be given as kwarg in c'tor of {}!".format(type(self).__name__)

        super(Distribution, self).__init__(*sub_components, **kwargs)
        self.class_ = class_(**kwargs)
        self.kwargs = kwargs

    def _computation_apply(self, input_):
        """
        Only can make_template from this function after(!) we know what the "output"?? socket's shape will be.
        """
        if backend() == "tf":
            return self.class_(input_, **self.kwargs)


