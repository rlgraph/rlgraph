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

from rlgraph.utils.rlgraph_error import RLGraphError

# Default backend ('tf' for tensorflow or 'pt' for PyTorch)
BACKEND = "tf"

# Default distributed backend is distributed TensorFlow.
DISTRIBUTED_BACKEND = "ray"


distributed_compatible_backends = dict(
    tf=["distributed_tf", "ray", "horovod"]
)


def get_backend():
    return BACKEND


def get_distributed_backend():
    return DISTRIBUTED_BACKEND


def init_backend(_backend=None, _distributed_backend=None):
    """
    Initializes global backend variables.
    Args:
        _backend (Optional[str]): Execution backend for the computation graph. Defaults
            to TensorFlow ("tf").
        _distributed_backend (Optional[str]): Distributed backend. Must be compatible with _backend.
            Defaults to Ray ("ray).

    Returns:

    """
    if _backend is None:
        _backend = "tf"
    set_backend(_backend)
    if _distributed_backend is None:
        _distributed_backend = "ray"
    set_distributed_backend(_distributed_backend)


def set_distributed_backend(_distributed_backend):
    """
    Sets the distributed backend. Must be compatible with configured backend.

    Args:
        _distributed_backend (str): Specifier for distributed backend.
    """
    global distributed_backend

    if _distributed_backend is not None:
        distributed_backend = _distributed_backend
        # Distributed backend must be compatible with backend.
        if distributed_backend not in distributed_compatible_backends[backend]:
           raise RLGraphError("Distributed backend {} not compatible with backend {}. Compatible backends"
                           "are: {}".format(distributed_backend, backend, distributed_compatible_backends[backend]))

        if distributed_backend == 'distributed_tf':
            assert backend == "tf"
            try:
                import tensorflow
            except ModuleNotFoundError as e:
                raise RLGraphError("INIT ERROR: Cannot run distributed_tf without backend (tensorflow)! "
                                "Please install tensorflow first via `pip install tensorflow` or "
                                "`pip install tensorflow-gpu`.")
        elif distributed_backend == "horovod":
            try:
                import horovod
            except ModuleNotFoundError as e:
                raise RLGraphError("INIT ERROR: Cannot run RLGraph with distributed backend Horovod.")
        elif distributed_backend == "ray":
            try:
                import ray
            except ModuleNotFoundError as e:
                raise RLGraphError("INIT ERROR: Cannot run RLGraph with distributed backend Ray.")
        else:
            raise RLGraphError("Distributed backend {} not supported".format(distributed_backend))


def set_backend(backend_):
    """
    Gets or sets the computation backend for RLGraph.

    Args:
        backend_ (str): So far, only 'tf' supported.
    """
    global backend

    if backend_ is not None:
        backend = backend_
        # Try TensorFlow
        if get_backend() == "tf" or backend == "tf-eager":
            try:
                import tensorflow as tf
            except ModuleNotFoundError as e:
                raise RLGraphError("INIT ERROR: Cannot run RLGraph without backend (tensorflow)! "
                                "Please install tensorflow first via `pip install tensorflow` or "
                                "`pip install tensorflow-gpu`.")
        # TODO: remove once pytorch done.
        elif backend == "pt":
            raise RLGraphError("INIT ERROR: Backend 'PyTorch' not supported in RLGraph prototype. Use 'tf' instead.")
        else:
            raise RLGraphError("INIT ERROR: Backend '{}' not supported! Use 'tf' for tensorflow or 'pt' for PyTorch.")

        # Test for tf-eager support.
        if backend == "tf-eager":
            try:
                tf.enable_eager_execution()
            except AttributeError as e:
                raise RLGraphError("INIT ERROR: Cannot run RLGraph in backend 'tf-eager'! "
                                "Your version of tf ({}) does not support eager execution. Update with "
                                "`pip install --upgrade tensorflow`.".format(tf.__version__))