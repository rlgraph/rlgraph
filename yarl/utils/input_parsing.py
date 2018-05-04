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


def parse_execution_spec(execution_spec):
    """
    Parses execution parameters and inserts default values where necessary.

    Args:
        execution_spec: Execution spec dict. Must specify an execution mode
            "single" or "distributed". If mode "distributed", must specify a "distributed_spec"
            containing:
             - a key cluster_spec mapping to a ClusterSpec object,
             - a "job" for the job name,
             - an integer "task_index"
             - a boolean "global_shared_memory" to indicate if data collection is shared globally
               or workers collect and update locally. Defaults to true.

    Returns: The sanitized execution_spec dict.
    """
    # If no spec given,
    default_spec = dict(
        mode="single",
        distributed_spec=None,
        session_config=None
    )
    if not execution_spec:
        return default_spec

    execution_mode = execution_spec['mode']

    if execution_mode == "distributed":
        default_distributed = dict(
            job="ps",
            task_index=0,
            cluster_spec={
                "ps": ["localhost:22222"],
                "worker": ["localhost:22223"]
            },
            global_shared_memory=True
        )
        default_distributed.update(execution_spec.get("distributed_spec", {}))
        execution_spec["distributed_spec"] = default_distributed
        execution_spec["session_config"] = execution_spec.get("session_config")
        return execution_spec
    elif execution_mode == "multi-threaded":
        return execution_spec
    elif execution_mode == "single":
        return execution_spec
