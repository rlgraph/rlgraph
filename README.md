# RLgraph
Flexible computation graphs for deep reinforcement learning.

RLgraph is a framework to quickly prototype, define and execute reinforcement learning
algorithms both in research and practice.
 
RLgraph exposes a well defined API for using agents, and offers a novel component concept
for testing and assembly of components. By separating graph definition, compilation and execution,
multiple distributed backends and device execution strategies can be accessed without modifying
agent definitions.

## Install

The simplest way to install RLgraph is from pip:

```pip install rlgraph```

Note that some backends (e.g. ray) need additional dependencies (see setup.py). For example, to install dependencies for the distributed backend ray, enter:

```pip install rlgraph[ray]```

To successfully run tests, please also install OpenAI gym, e.g.

```pip install gym[all]```

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

## More detailed examples and docs coming soon. 


