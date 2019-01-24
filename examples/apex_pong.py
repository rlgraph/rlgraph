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
Example script for training an APEX agent on the Arcade Learning Environment (ALE). This agent can be used both
for local and distributed training. You'll need to start Ray yourself using `ray start --head --redis-port 6379`.
If you want to use distributed training, just join the Ray cluster with `ray start --redis-address=master.host:6379`.

Usage:

python apex_ale.py [--config configs/apex_pong.json] [--gpu/--nogpu] [--env PongNoFrameSkip-v4] [--output results.csv]

Please make sure that Ray is configured as the distributed backend for RLgraph, e.g. by running this command:

```bash
echo '{"BACKEND":"tf","DISTRIBUTED_BACKEND":"ray"}' > $HOME/.rlgraph/rlgraph.json
```

Then you can start up the Ape-X agent:

```bash
# Start ray on the head machine
ray start --head --redis-port 6379
# Optionally join to this cluster from other machines with ray start --redis-address=...

# Run script
python apex_pong.py
```
"""

import csv
import json
import os
import sys

from absl import flags

from rlgraph.execution.ray import ApexExecutor


FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/apex_pong.json', 'Agent config file.')
flags.DEFINE_boolean('gpu', True, 'Use GPU for training.')
flags.DEFINE_string('env', 'PongNoFrameskip-v4', 'gym environment ID.')
flags.DEFINE_string('output', 'results.csv', 'Output rewards file.')


def main(argv):
    try:
        FLAGS(argv)
    except flags.Error as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    agent_config_path = os.path.join(os.getcwd(), FLAGS.config)
    with open(agent_config_path, 'rt') as fp:
        agent_config = json.load(fp)

    # GPU usage is enabled in the config per default, so we disable it if --nogpu was passed.
    if not FLAGS.gpu:
        agent_config['execution_spec']['gpu_spec']['gpus_enabled'] = False

    env_spec = {
        "type": "openai",
        "gym_env": FLAGS.env,
        "frameskip": 4,  # When using a NoFrameskip-environment, this setting will enable frameskipping within RLgraph
        "max_num_noops": 30,
        "episodic_life": False,
        "fire_reset": True
    }

    executor = ApexExecutor(
        environment_spec=env_spec,
        agent_config=agent_config,
    )

    print("Starting workload, this will take some time for the agents to build.")
    results = executor.execute_workload(workload=dict(num_timesteps=2000000, report_interval=50000,
                                                      report_interval_min_seconds=30))

    # Now we will save a CSV with the rewards timeseries for all workers and environments.
    print("Fetching worker learning results")
    agent_config_path = os.path.join(os.getcwd(), FLAGS.output)
    # First, we fetch all worker results as a large dict.
    all_results = executor.get_all_worker_results()
    with open(agent_config_path, 'wt') as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(["worker_num", "env_num", "total_times", "sample_times", "steps", "rewards"])
        # Then we iterate over all workers.
        for worker_num, result_dict in enumerate(all_results):
            # Workers step through multiple environments, so now we loop through those for each worker.
            for env_num, (step_list, reward_list, total_times_list, sample_times_list) in enumerate(zip(
                    result_dict['episode_timesteps'],
                    result_dict['episode_rewards'],
                    result_dict['episode_total_times'],
                    result_dict['episode_sample_times'])):

                # Lastly, we loop through the environment reward timeseries and save the results in our CSV.
                for steps, rewards, total_times, sample_times in zip(step_list, reward_list,
                                                                     total_times_list, sample_times_list):
                    row = [worker_num, env_num, total_times, sample_times, steps, rewards]
                    csvwriter.writerow(row)


if __name__ == '__main__':
    main(sys.argv)
