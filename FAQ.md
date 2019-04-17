
# FAQ

Here we collect short answers to common questions and point to resources.


### How can I execute a gym environment?

The simplest way of executing an environment is the single-threaded executor. For an example, see
the ```examples/dqn_cartpole.py``` script. While single-threaded, the executor can still use
vectorization but executing on a list of environments and using batch actions. This is
controlled via the ```num_envs=1``` arg to the executor.

The Ray executor can execute a large number of environments transparently on a single node or 
a large cluster depending on the available resources. The ```apex_pong``` script trains a
distributed DQN variant (Ape-X) using Ray on a number of workers. Ray can also be used
with very few workers even on a laptop. For a number of examples, consider the 
(```TestApexExecutor``` under tests/execution) unit-test which executes e.g. CartPole with a hand-ful workers within
a few seconds.

There are currently two Ray executors (please create an issue if you think we should
add a new one): The Ape-X one for asynchronous distributed value-based learning, and a 
synchronous one for e.g. distributed PPO. It simply collects a number of batches in parallel
and merges them into one update. Examples are in the tests package (```TestSyncBatchExecutor``` in 
tests/execution).


## How can I use a custom environment? 

Both the single-threaded executor and all Ray executors accept:
- Environment spec dicts for pre-registered environments.
- Callables returning a new environment instance, e.g.:

``` 
env_spec = dict(
    type="openai",
    gym_env="CartPole-v0"
)
agent_config = config_from_path("configs/apex_agent_cartpole.json")

def create_env():
    return Environment.from_spec(env_spec)

executor = ApexExecutor(
    environment_spec=create_env,
    agent_config=agent_config,
)
```

## How can I use a more complex network structure?

The ``network_spec`` and ```value_function_spec``` configuration
parameters both take simple lists of layer configurations but also
accept instances of ```NeuralNetwork``` and ```ValueFuction``` respectively:

```agent = DQNAgent(network_spec=MyNetwork(), **kwargs)```

will use ```MyNetwork``` as the base of the policy object. 

In ```components/neural_networks/```, we show two examples of custom
value functions. The ```impala_networks``` module illustrates how to compose
an involved architecture combining LSTM and convolutional stacks from multiple
inputs for the IMPALA policy networks. In ```sac_networks```, we implement
a value function that concatenates states and actions which can require
splitting up a network into different stacks.