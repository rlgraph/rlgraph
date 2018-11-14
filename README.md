[![PyPI version](https://badge.fury.io/py/rlgraph.svg)](https://badge.fury.io/py/rlgraph)
[![Python 3.5](https://img.shields.io/badge/python-3.5-orange.svg)](https://www.python.org/downloads/release/python-356/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/rlgraph/rlgraph/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/rlgraph/badge/?version=latest)](https://rlgraph.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/rlgraph/rlgraph.svg?branch=master)](https://travis-ci.org/rlgraph/rlgraph)

# RLgraph
Flexible computation graphs for deep reinforcement learning.

RLgraph is a framework to quickly prototype, define and execute reinforcement learning
algorithms both in research and practice. RLgraph is different from most other libraries as it can support
TensorFlow (or static graphs in general) or eager/define-by run execution (PyTorch) through
a single component interface.
 
RLgraph exposes a well defined API for using agents, and offers a novel component concept
for testing and assembly of machine learning models. By separating graph definition, compilation and execution,
multiple distributed backends and device execution strategies can be accessed without modifying
agent definitions. This means it is especially suited for a smooth transition from applied use case prototypes
to large scale distributed training.

The current state of RLgraph in version 0.1.3 is alpha. The core engine is substantially complete
and works for TensorFlow and PyTorch. Distributed execution on Ray is exemplified via Distributed
Prioritized Experience Replay (Ape-X), which also supports multi-gpu mode. Ape-X can be used both with PyTorch and
TensorFlow. Distributed TensorFlow can be tested via the IMPALA agent. We will add a number of key
algorithms and other distributed coordination mechanisms in the  coming weeks. Please create an issue
to discuss improvements or contributions.

For more detailed documentation on RLgraph and its API-reference, please visit
[our readthedocs page here](https://rlgraph.readthedocs.io).

## Install

The simplest way to install RLgraph is from pip:

```pip install rlgraph```

Note that some backends (e.g. ray) need additional dependencies (see setup.py).
For example, to install dependencies for the distributed backend ray, enter:

```pip install rlgraph[ray]```

To successfully run tests, please also install OpenAI gym, e.g.

```pip install gym[all]```

Upon calling RLgraph, a config JSON is created under ~.rlgraph/rlgraph.json
which can be used to change backend settings. The current default stable
backend is TensorFlow ("tf"). The PyTorch backend ("pytorch") does not support
all utilities available in TF yet. Namely, device handling for PyTorch is incomplete,
and we will likely wait until a stable PyTorch 1.0 release in the coming weeks.

### Quickstart / example usage

We provide an example script for training the Ape-X algorithm on ALE using Ray in the [examples](examples) folder.

First, you'll have to ensure, that Ray is used as the distributed backend. RLgraph checks the file
`~/.rlgraph/rlgraph.json` for this configuration. You can use this command to
configure RLgraph to use TensorFlow as the backend and Ray as the distributed backend:

```bash
echo '{"BACKEND":"tf","DISTRIBUTED_BACKEND":"ray"}' > $HOME/.rlgraph/rlgraph.json
```

Then you can run our Ape-X example:

```bash
# Start ray on the head machine
ray start --head --redis-port 6379
# Optionally join to this cluster from other machines with ray start --redis-address=...

# Run script
python apex_pong.py
```

You can also train a simple DQN agent locally on OpenAI gym environments such as CartPole (this doesn't require Ray):

```bash
python dqn_cartpole.py
```


## Import and use agents

Agents can be imported and used as follows:

```python
from rlgraph.agents import DQNAgent
environment = OpenAIGymEnv("Cartpole-v0")

# Create from .json file or dict, see agent API for all
# possible configuration parameters.
agent = DQNAgent(
  "configs/config.json",
  state_space=environment.state_space, 
  action_space=environment.action_space
)

# Get an action, take a step, observe reward.
state = environment.reset()
action, preprocessed state = agent.get_action(
  states=state,
  extra_returns="preprocessed_states"
)

# Execute step in environment.
next_state, reward, terminal, info =  environment.step(action)

# Observe result.
agent.observe(
  preprocessed_states=preprocessed_state,
  actions=action,
  internals=[],
  next_states=next_state,
  rewards=reward
)

# Call update when desired:
loss = agent.update()
```

## Distributed execution

RLgraph supports multiple distributed backends, as graph definition and execution are separate. For example, to use
a high performance version of distributed DQN (Ape-X), a corresponding ApexExecutor can distribute execution via Ray:

```python
from rlgraph.execution.ray import ApexExecutor

# See learning tests for example configurations e,g.
# rlgraph/tests/execution/test_apex_executor.py
env_spec = dict(type="openai", gym_env="CartPole-v0")

# Ray executor creating Ray actors.
exec = ApexExecutor(
  environment_spec=env_spec,
  agent_config=agent_config,
)

# Executes actual workload on distributed Ray cluster.
result = exec.execute_workload(workload=dict(num_timesteps=10000, report_interval=1000))

# Prints result metrics.
print(result)
```

More detailed examples coming soon.

## Cite

If you use RLgraph in your research, please cite the following paper: [link](https://arxiv.org/abs/1810.09028)


```
@ARTICLE{Schaarschmidt2018rlgraph,
    author = {{Schaarschmidt}, M. and {Mika}, S. and {Fricke}, K. and {Yoneki}, E.},
    title = "{RLgraph: Flexible Computation Graphs for Deep Reinforcement Learning}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1810.09028},
    year = 2018,
    month = oct
}
```