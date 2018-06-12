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

from .util import default_dict


def parse_saver_spec(saver_spec):
    default_spec = dict(max_checkpoints=5)
    return default_dict(saver_spec, default_spec)


def parse_summary_spec(summary_spec):
    default_spec = dict()
    return default_dict(summary_spec, default_spec)


def parse_execution_spec(execution_spec):
    """
    Parses execution parameters and inserts default values where necessary.

    Args:
        execution_spec (Optional[dict]): Execution spec dict. Must specify an execution mode
            "single" or "distributed". If mode "distributed", must specify a "distributed_spec"
            containing:
             - a key cluster_spec mapping to a ClusterSpec object,
             - a "job" for the job name,
             - an integer "task_index"
             - a boolean "global_shared_memory" to indicate if data collection is shared globally
               or workers collect and update locally. Defaults to true.

    Returns:
        dict: The sanitized execution_spec dict.
    """
    # If no spec given.
    default_spec = dict(
        mode="single",
        distributed_spec=None,
        session_config=None,
        seed=None,  # random seed for the tf graph
        enable_profiler=False,  # enabling the tf profiler?
        profiler_frequency=1000  # with which frequency do we print out profiler information?
    )
    execution_spec = default_dict(execution_spec, default_spec)

    execution_mode = execution_spec.get("mode")
    if execution_mode == "distributed":
        default_distributed = dict(
            job="ps",
            task_index=0,
            cluster_spec=dict(
                ps=["localhost:22222"],
                worker=["localhost:22223"]
            ),
            global_shared_memory=True
        )
        default_dict(execution_spec.get("distributed_spec"), default_distributed)
        execution_spec["session_config"] = execution_spec.get("session_config")

    return execution_spec


def parse_observe_spec(observe_spec):
    """
    Parses parameters for `Agent.observe()` calls and inserts default values where necessary.

    Args:
        observe_spec (Optional[dict]): Observe spec dict.

    Returns:
        dict: The sanitized observe_spec dict.
    """
    # If no spec given.
    default_spec = dict(
        # Do we buffer observations in python before sending them through the graph?
        buffer_enabled=True,
        # Fill buffer with n records before sending them through the graph.
        buffer_size=100,  # only if buffer_enabled=True
        # The unit in which we measure frequency: one of "timesteps", "episodes", "sec".
    )
    observe_spec = default_dict(observe_spec, default_spec)

    return observe_spec


def parse_update_spec(update_spec):
    """
    Parses update/learning parameters and inserts default values where necessary.

    Args:
        update_spec (Optional[dict]): Update/Learning spec dict.

    Returns:
        dict: The sanitized update_spec dict.
    """
    # If no spec given.
    default_spec = dict(
        # Whether to perform calls to `Agent.update()` at all.
        do_updates=True,
        # The unit in which we measure frequency: one of "timesteps", "episodes", "sec".
        #unit="timesteps", # TODO: not supporting any other than timesteps
        # The number of 'units' to wait before we do any updating at all.
        steps_before_update=0,
        # The frequency with which we update (given in `unit`).
        update_interval=4,
        # The number of consecutive `Agent.update()` calls per update.
        update_steps=1,
        # The batch size with which to update (e.g. when pulling records from a memory).
        batch_size=64,
    )
    update_spec = default_dict(update_spec, default_spec)

    return update_spec
