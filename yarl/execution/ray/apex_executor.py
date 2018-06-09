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
from yarl.execution.ray.ray_executor import RayExecutor
from threading import Thread

import ray


class ApexExecutor(RayExecutor):
    """
    Implements the distributed update semantics of distributed prioritized experience replay (APE-X),
    as described in:

    https://arxiv.org/abs/1803.00933
    """

    def __init__(self, distributed_spec):
        super(ApexExecutor, self).__init__(distributed_spec)

    def init_agents(self):
        pass

    def execute_workload(self, workload):
        """
        Executes a workload via Ape-X semantics. The main loop performs the following
        steps until the specified number of steps or episodes is finished:

        - Retrieve sample batches via Ray from remote workers
        - Insert these into the local memory
        - Have a separate learn thread sample batches from the memory and compute updates
        - Sync weights to the shared model so remot eworkers can update their weights.
        """
        pass


class UpdateWorker(Thread):
    """
    Executes learning separate from the main event loop as described in the Ape-X paper.
    Communicates with the main thread via a queue.
    """

    def __init__(self, agent):
        super(UpdateWorker, self).__init__()

        # Agent to use for updating.
        self.agent = agent

        # Terminate when host process terminates.
        self.daemon = True

    def run(self):
        while True:
            # Fetch input for update, update
            pass

