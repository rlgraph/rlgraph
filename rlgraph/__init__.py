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

from rlgraph.version import __version__

import json
import os
import logging

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

if "RLGRAPH_HOME" in os.environ:
    rl_graph_dir = os.environ.get("RLGRAPH_HOME")
else:
    rl_graph_dir = os.path.expanduser('~')
    rl_graph_dir = os.path.join(rl_graph_dir, ".rlgraph")


# TODO "tensorflow" for tensorflow?
# Default backend ('tf' for tensorflow or 'pytorch' for PyTorch)
BACKEND = "tf"

# Default distributed backend is distributed ray.
DISTRIBUTED_BACKEND = "distributed_tf"

distributed_compatible_backends = dict(
    tf=["distributed_tf", "ray", "horovod"],
    pytorch=["ray", "horovod"]
)


config_path = os.path.expanduser(os.path.join(rl_graph_dir, 'rlgraph.json'))
if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            config = json.load(f)
    except ValueError:
        config = dict()

    # Read from config or leave defaults.
    backend = config.get("BACKEND", None)
    if backend is not None:
        BACKEND = backend
    distributed_backend = config.get("DISTRIBUTED_BACKEND", None)
    if distributed_backend is not None:
        DISTRIBUTED_BACKEND = distributed_backend

# Create dir if necessary:
if not os.path.exists(rl_graph_dir):
    try:
        os.makedirs(rl_graph_dir)
    except OSError:
        pass


# Write to file if there was none:
if not os.path.exists(config_path):
    _config = {
        "BACKEND": BACKEND,
        "DISTRIBUTED_BACKEND": DISTRIBUTED_BACKEND,
    }
    try:
        with open(config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Overwrite backend if set in ENV.
if 'RLGRAPH_BACKEND' in os.environ:
    backend = os.environ.get('RLGRAPH_BACKEND', None)
    if backend is not None:
        logging.info("Setting BACKEND to '{}' per environment variable 'RLGRAPH_BACKEND'.".format(backend))
        BACKEND = backend

# Overwrite distributed-backend if set in ENV.
if 'RLGRAPH_DISTRIBUTED_BACKEND' in os.environ:
    distributed_backend = os.environ.get('RLGRAPH_DISTRIBUTED_BACKEND', None)
    if distributed_backend is not None:
        logging.info(
            "Setting DISTRIBUTED_BACKEND to '{}' per environment variable "
            "'RLGRAPH_DISTRIBUTED_BACKEND'.".format(distributed_backend)
        )
        DISTRIBUTED_BACKEND = distributed_backend


# Test compatible backend.
if DISTRIBUTED_BACKEND not in distributed_compatible_backends[BACKEND]:
    raise ValueError("Distributed backend {} not compatible with backend {}. Compatible backends"
                     "are: {}".format(DISTRIBUTED_BACKEND, BACKEND, distributed_compatible_backends[BACKEND]))


# Test imports.
if DISTRIBUTED_BACKEND == 'distributed_tf':
    assert BACKEND == "tf"
    try:
        import tensorflow
    except ImportError as e:
        raise ImportError(
            "INIT ERROR: Cannot run distributed_tf without backend (tensorflow)! Please install tensorflow first "
            "via `pip install tensorflow` or `pip install tensorflow-gpu`."
        )
elif DISTRIBUTED_BACKEND == "horovod":
    try:
        import horovod
    except ImportError as e:
        raise ValueError("INIT ERROR: Cannot run RLGraph with distributed backend Horovod.")
elif DISTRIBUTED_BACKEND == "ray":
    try:
        import ray
    except ImportError as e:
        raise ValueError("INIT ERROR: Cannot run RLGraph with distributed backend Ray.")
else:
    raise ValueError("Distributed backend {} not supported".format(DISTRIBUTED_BACKEND))


def get_backend():
    return BACKEND


def get_distributed_backend():
    return DISTRIBUTED_BACKEND


__all__ = [
    "__version__",  "get_backend", "get_distributed_backend"
]
