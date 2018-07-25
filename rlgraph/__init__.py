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

from yarl.version import __version__

import logging
# Convenience imports.
from yarl.utils.yarl_error import YARLError
from yarl.utils.specifiable import Specifiable
from yarl.utils.util import LARGE_INTEGER, SMALL_NUMBER
from yarl.backend_system import get_backend, get_distributed_backend, set_backend, set_distributed_backend, init_backend
# Libraries should add NullHandler() by default, as its the application code's
# responsibility to configure log handlers.
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())


__all__ = [
    "YARLError", "__version__", "Specifiable", "get_backend", "get_distributed_backend",
    "set_backend", "init_backend", "set_distributed_backend", "SMALL_NUMBER", "LARGE_INTEGER"
]
