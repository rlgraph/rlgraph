## Release notes
Summarizes updates in recent releases.

## RLgraph 0.3.4 - 29.3.2019

- Ray executors now allow passing in callables
  creating custom environments instead of environment specs.
- Further unified component state handling for define by run state.
  int types are not references and storing them in a internal registry 
  (like TF/torch parameter variables) means they will never be updated.
  Components can now self-describe via get_state() to inform about their 
  non-ref types (i.e. int variables). This for example now allows to run memory 
  tests with the same code requesting internal component variables for both backends.
- SAC agent now supports image inputs in the value function and container actions.
- Fixed a number of bugs related to action vectorization and preprocessing in SAC.

## RLgraph 0.3.3 - 25.2.2019

- Added soft actor critic implementation (contributed by @janislavjankov) 
- Separated out action adapters to be able to handle different bounded distributions
- Added a number of torch test cases to continuous integration
- Fixed a number of bugs in define-by-run mode related to arg splitting and merging.
- Fixed a number of shape bugs in various torch implementations
- Fixed a bug relating to assigning references instead of just copying weights when syncing torch Parmeter objects

## RLgraph 0.3.2 - 9.2.2019
- Fixed a number of bugs in internal state management for PyTorch which now
  allow to unify variable creation in most components
- Fixed bug in PyTorch GAE calculation
- Added PyTorch basic replay buffer implementation
- Renamed _variables() to variables() to obtain internal state of a component
- Changed some single node configurations in examples to use less memory and only
  one replay worker (Ape-X).

## RLgraph 0.3.1 - 27.1.2019

- Fixed count bug in synchronous Ray executor
- Fixed bugs related to episode-fetching in the ring-buffer 
  (only occurring when using episode update mode)
- Added reward-clipping option to GAE
- Added post-processing flag to DQN multi-gpu mode

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

