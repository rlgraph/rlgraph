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
import logging


class RayExecutor(object):
    """
    Abstract distributed Ray executor.

    A Ray executor implements a specific distributed learning semantic by delegating
    distributed state management and execution to the Ray execution engine.

    """
    def __init__(self, distributed_spec):
        """

        Args:
            distributed_spec (dict): Contains all information necessary to set up and execute
                agents on a Ray cluster.
        """
        self.logger = logging.getLogger(__name__)
        self.distributed_spec = distributed_spec

    def init_agents(self):
        """
        Creates and initializes all remote agents on the Ray cluster.
        """
        raise NotImplementedError

    def execute_workload(self, workload):
        """
        Executes a given workload according to a specific distributed update semantic.

        Args:
            workload (dict): Dict specifying workload by describing environments, number of steps
                or episodes to execute, and termination conditions.

        Returns:
            dict: Summary statistics of distributed workload.
        """
        raise NotImplementedError
