
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
()```TestApexExecutor```) unit-test which executes e.g. CartPole with a hand-ful workers within
a few seconds.

There are currently two Ray executors (please create an issue if you think we should
add a new one): The Ape-X one for asynchronous distributed value-based learning, and a 
synchronous one for e.g. distributed PPO. It simply collects a number of batches in parallel
and merges them into one update. Examples are in the tests package (```TestSyncBatchExecutor```).