## Release notes
Summarizes updates in recent releases.

## RLgraph 0.3.0 - 25.1.2019

- Added Ray executor for distributed policy optimization, e.g. distributed PPO on Ray.
- Allow use of api and graph functions from list comprehensions and lambdas
- Improved agent api to define graph functions 
- Fixed various build instabilities related to build order
- Fixed a bug for container actions where huber loss was applied to each action instead to the aggregate loss
- Fixed a number of bugs around space inference for PyTorch when using lists and numpy arrays to store internal state
- Simplified multi-gpu semantics for iterative in-graph multi gpu updates (e.g. on PPO).
- Allow for in-graph and external post-processing via extra flag
- Fixed bug in continuous action policies which made distribution parameters to be parsed incorrectly

## RLgraph 0.2.3 - 15.12.2018

- Improved LSTM-layer handling with keras-style api in network to manage sequences
- Added new LSTM example in examples folder
- Updated implementations to PyTorch 1.0
- Fixed various bugs around PyTorch type inference during build process 
- Improved memory usage of various Ray tasks by avoiding defensive copies,
  following improvements in Ray's memory management.
  
## RLgraph 0.2.2 - 3.12.2018
- Implemented support for advanced decorator options for PyTorch backend
- Various bugfixes in PyTorch utilities needed for PPO/Actor critic

## RLgraph 0.2.1 - 25.11.2018
- Updated actor-critic to support external value functions
- Fixed bugs related to hardcoded entropy for categoricals in loss functions

## RLgraph 0.2.0 - 23.11.2018
- Introduced support for container actions so actions can now be specified as dicts of
arbitrary numbers of sub-actions.
- Added agent a number of learning tests to CI

