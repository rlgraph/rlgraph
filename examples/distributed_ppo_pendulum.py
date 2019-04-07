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

"""
Example script for training a Proximal policy optimization agent on an OpenAI gym environment.

Usage:

python distributed_ppo_pendulum.py [--config configs/ppo_cartpole.json] [--env Pendulum-v0]

```
# Run script
python distributed_ppo_pendulum.py
```
"""

import json
import os
import sys

from absl import flags
from rlgraph.execution.ray import SyncBatchExecutor

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/distributed_ppo_pendulum.json', 'Agent config file.')
flags.DEFINE_string('env', 'Pendulum-v0', 'gym environment ID.')


def main(argv):
    try:
        FLAGS(argv)
    except flags.Error as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    agent_config_path = os.path.join(os.getcwd(), FLAGS.config)
    with open(agent_config_path, 'rt') as fp:
        agent_config = json.load(fp)

    env_spec = {
        "type": "openai",
        "gym_env": FLAGS.env
    }

    # Distributed synchronous optimisation on ray.
    executor = SyncBatchExecutor(
        environment_spec=env_spec,
        agent_config=agent_config,
    )
    results = executor.execute_workload(workload=dict(num_timesteps=500000, report_interval=50000,
                                                      report_interval_min_seconds=30))
    print(results)


if __name__ == '__main__':
    main(sys.argv)
