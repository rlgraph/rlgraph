## Release notes
Summarizes updates in release starting at 0.2.0

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

