# RLgraph
A Framework for Flexible Deep Reinforcement Learning Graphs

RLGraph is a framework to quickly prototype, define and execute reinforcement learning
algorithms both in research and practice.
 
RLGraph exposes a well defined API for using agents, and offers a novel component concept for testing and assembly.
By separating graph definition, compilation and execution, multiple distributed backends
and device management can be accessed without modifying agent definitions.

## Install

The simplest way to install RLgraph is from pip:

```pip install rlgraph```

Note that some backends (e.g. ray) need additional dependencies (see setup.py). For example, to install dependencies for the distributed backend ray, enter:

```pip install rlgraph[ray]```

## Import and use agents

Agents can be imported and used as follows:

```python
from rlgraph.agents import DQNAgent

environment = OpenAIGymEnv("Cartpole-v0")

# Create from .json, file, see agent doc for all
# possible configuration parameters.
agent = DQNAgent(
	"config.json",
	 state_space=environment.state_space, 	 action_space=environment.action_space
)

# Get an action, take a step, observe reward
state = environment.reset()
action, preprocessed state = agent.get_action(
	states=state,
	extra_returns="preprocessed_states"
)

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

exec = ApexExecutor(
    environment_spec=env_spec,
    agent_config=agent_config,
)

# Executes actual workload.
result = exec(workload=dict(num_timesteps=10000, report_interval=1000))

# Prints result metrics
print(result)
```

## More detailed examples and docs coming soon. 


