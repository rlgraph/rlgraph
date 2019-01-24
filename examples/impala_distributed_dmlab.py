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
Example script for training a distributed IMPALA [1] agent on an OpenAI gym environment.
A single-node agent uses multi-threading (via tf's queue runners) to collect experiences (using
the "mu"-policy) and a learner (main) thread to update the model (the "pi"-policy).

Usage:

python impala_distributed_dmlab.py [--config configs/impala_cartpole.json] [--env CartPole-v0]

```
# Run script
python impala_cartpole.py
```

[1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
    Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
"""

import json
import os
import sys

from absl import flags
import numpy as np
import time

from rlgraph.agents import IMPALAAgent
from rlgraph.environments import Environment


FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/impala_distributed_dmlab.json', 'Agent config file.')
flags.DEFINE_string('level', 'seekavoid_arena_01', 'Deepmind lab level name.')

flags.DEFINE_string('cluster_spec', './configs/impala_distributed_clusterspec.json', 'Cluster spec file.')

flags.DEFINE_boolean('learner', False, 'Start learner.')
flags.DEFINE_boolean('actor', False, 'Start actor.')
flags.DEFINE_integer('task', 0, 'Task index.')

flags.DEFINE_integer('updates', 50, 'Number of updates.')


def main(argv):
    try:
        FLAGS(argv)
    except flags.Error as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    agent_config_path = os.path.join(os.getcwd(), FLAGS.config)
    with open(agent_config_path, 'rt') as fp:
        agent_config = json.load(fp)

    cluster_spec_config_path = os.path.join(os.getcwd(), FLAGS.cluster_spec)
    with open(cluster_spec_config_path, 'rt') as fp:
        cluster_spec = json.load(fp)

    # Environment options
    env_spec = {
        "type": "deepmind-lab",
        "level_id": FLAGS.level,
        "frameskip": 4,
        "observations": ["RGB_INTERLEAVED", "INSTR"]
    }

    # Verbose usage errors
    if FLAGS.actor and FLAGS.learner:
        print("Please only use either --actor or --learner, not both.")
        sys.exit(1)

    # We dynamically update the distributed spec according to the job and task index
    if FLAGS.actor:
        agent_type = 'actor'
        distributed_spec = dict(
            job='actor',
            task_index=FLAGS.task,
            cluster_spec=cluster_spec
        )
        # Actors should only act on CPUs
        agent_config['execution_spec']['gpu_spec'].update({
            "gpus_enabled": False,
            "max_usable_gpus": 0,
            "num_gpus": 0
        })
    elif FLAGS.learner:
        agent_type = 'learner'
        distributed_spec = dict(
            job='learner',
            task_index=FLAGS.task,
            cluster_spec=cluster_spec
        )
    else:
        print("Please pass either --learner or --actor (or look at the CartPole example for single trainig mode).")
        sys.exit(1)

    # Set the sample size for the workers
    worker_sample_size = 100

    # Since we dynamically update the cluster spec according to the job and task index, we need to
    # manually update the execution spec as well.
    execution_spec = agent_config['execution_spec']
    execution_spec.update(dict(
        mode="distributed",
        distributed_spec=distributed_spec
    ))

    # Now, create the environment
    env = Environment.from_spec(env_spec)

    agent_spec = dict(
        type=agent_type,
        architecture="large",
        environment_spec=env_spec,
        worker_sample_size=worker_sample_size,
        state_space=env.state_space,
        action_space=env.action_space,
        # TODO: automate this (by lookup from NN).
        internal_states_space=IMPALAAgent.default_internal_states_space,
        execution_spec=execution_spec,
        # update_spec=dict(batch_size=2),
        # Summarize time-steps to have an overview of the env-stepping speed.
        summary_spec=dict(summary_regexp="time-step", directory="/tmp/impala_{}_{}/".format(agent_type, FLAGS.task))
    )

    agent_config.update(agent_spec)

    agent = IMPALAAgent(
        **agent_config
    )

    if FLAGS.learner:
        print("Starting learner for {} updates.".format(FLAGS.updates))
        for _ in range(FLAGS.updates):
            start_time = time.perf_counter()
            results = agent.update()
    else:
        # Actor just acts
        print("Starting actor. Terminate with SIGINT (Ctrl+C).")
        while True:
            agent.call_api_method("perform_n_steps_and_insert_into_fifo")  #.monitored_session.run([agent.enqueue_op])

    learn_updates = 100
    mean_returns = []
    for i in range(learn_updates):
        ret = agent.update()
        mean_return = _calc_mean_return(ret)
        mean_returns.append(mean_return)
        print("Iteration={} Loss={:.4f} Avg-reward={:.2f}".format(i, float(ret[1]), mean_return))

    print("Mean return: {:.2f} / over the last 10 episodes: {:.2f}".format(
        np.nanmean(mean_returns), np.nanmean(mean_returns[-10:])
    ))

    time.sleep(1)
    agent.terminate()
    time.sleep(1)


def _calc_mean_return(records):
    size = records[3]["rewards"].size
    rewards = records[3]["rewards"].reshape((size,))
    terminals = records[3]["terminals"].reshape((size,))
    returns = list()
    return_ = 0.0
    for r, t in zip(rewards, terminals):
        return_ += r
        if t:
            returns.append(return_)
            return_ = 0.0

    return np.nanmean(returns)


if __name__ == '__main__':
    main(sys.argv)
