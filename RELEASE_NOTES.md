## Release notes
Summarizes updates in recent releases.

## RLgraph 0.5.5 - 2019/07/02
- Added some new Components (MultiInputStreamNN, VariationalAutoEncoder, MultiLSTMLayer, JointCumulativeDistribution,
  different supervised LossFunctions, Models, etc..), incl. test cases.
- Added container Space support to all Agents (python buffers had to be expanded).
- Bug fix in BernoulliDistributionAdapter and Policy (with bool actions): Was returning wrong 
  parameters (must be probs, not raw NN output).
- tf backend: Allow returning None now in graph_fn as valid DataOps.
- Bug fix in LSTM tf backend: LSTMLayer would not compile on tf versions < 1.13.
- Added possibility to flatten/split a graph_fn call "alongside" some given "(self.)property". This makes it easier to
  split input args only to a certain extend (see e.g. distribution parameters in policy's API methods). 

## RLgraph 0.5.4 - 2019/06/07
- Added a prototype for a debug visualization util that automatically
  builds sub-sections of the (meta-)graph and visualizes this sub-graph
  in the browser (as pdf) using GraphViz.
  See FAQs for details on how to activate this feature.
  Installing the GraphViz engine (and pypi `graphviz`) is not a requirement.
  In the visualized sub-graph, where only the fault-relevant parts of
  the Agent are shown to reduce information overload, one can see
  immediately where Space (shape/type) problems occurred.
- Cleaner handling of exposing child API methods when calling:
  `Component.add_components()`. The auto-generated (exposed) parent API
  now has the same name and signature as the child's one (which makes it
  more consistent when tracking incoming Spaces into API-input-args
  (a Component's input-completeness is affected by this)).

## RLgraph 0.5.3 - 2019/06/03
- Fixed remaining issues with Keras-style NN assembly. This is now the
  recommended method for complex/multi-stream `NeuralNetworks`.

## RLgraph 0.5.2 - 2019/05/25
- Fixed bug in Worker's reporting: `mean_episode_reward` was calculated
  incorrectly as the return of the last episode (regardless of whether
  this episode was completed or not).
- Tuned learning tests for SAC and PPO.
- Added visualization tools for GridWorld envs. Rendering is now done in
  pygame (optional install and the env has additional heat-map and
  rewards/states-paths visualizing methods (png output).

## RLgraph 0.5.1 - 2019/05/24
- Fixed bug in PPOLossFunction affecting action spaces with shapes like
  (x, y, z, >1) and container action spaces.

## RLgraph 0.5.0 - 2019/05/22
- Fixed bug in PPOLossFunction in value-function target term. Here, the
  previous value-estimates need to be used (before the next update round)
  instead of the current estimates (from the ongoing (PPO-iterative)
  update round).
- Added new `TimeDependentParameter` classes for learning rate and other
  time-dependent parameters that may change over time. These replace
  the now obsoleted `DecayComponent`s.

  All json configs that use `exploration_spec` (Q-type Agents) must erase
  `start_timestep` and `num_timesteps` ..
  ```
  "exploration_spec": {
    "epsilon_spec": {
      "decay_spec": {
        ...
        "start_timestep": 0,   # <- erase this line
        "num_timesteps": 1000, # <- and this one
  }}}
  ```
  .. from the `decay_spec` within this `exploration_spec`. From now on, the Worker
  is responsible to pass into each `get_action()` and `update()` calls, a
  `time_percentage` value (between 0.0 and 1.0) that will make
  `start_timestep` and `num_timesteps` superfluous.
  To infer `time_percentage` automatically, the Worker needs some kind of maximum
  number of timestep value. There are different ways to pass in global max-timestep information:
  - Via the Worker's `max_timesteps` c'tor arg.
  - Via the Agent's `max_timesteps` c'tor arg.
  - Leave it to the Worker to figure out the max timesteps itself. A call to
  `Worker.execute_timesteps(timesteps=n)` will use n, a call to `Worker.execute_episodes(episodes=n,
  max_timesteps_per_episode=100)` will use 100 x n, etc.
  - If you are not using our Worker classes, make sure to pass in manually a `time_percentage`
  between 0.0 (start learning) and 1.0 (finished learning) into calls to
  `Agent.get_action()` and `Agent.update()`.
- Added `time_percentage` inputs to `Agent.update()` and `Agent.get_action()`
  calls. This enables all Components that own `TimeDependentParameter`
  sub-components to decay/change these values over time. This applies mostly to
  optimizers, loss-functions and (epsilon)-exploration components.
  See FAQs on how to configure decays for arbitrary hyper-parameters.
- Reduced number of `tf.placeholder`s to one per unique API input-arg name.
  Also, all placeholders have more descriptive names now (named after the API input-arg).
- The `Optimizer` Component's step API method will no longer return `loss` and `loss_per_item`
  as 2nd and 3rd return value. Instead only the `step_op` is returned.
  Make sure that all `Optimizer.step()` calls expect only one single return value (`step_op`).
- GridWorld: Bug fix in grid-maps for which start x/y-positions are
  different from (0, 0).
  Step reward was changed from -1.0 to -0.1 for better granularity/faster learning.

## RLgraph 0.4.1 - 2019/04/28
- Fixed bug in the SequencerHelper Component causing GAEs to be calculated incorrectly.
  This bug fix largely improved PPO learning performance (see MLAgents example script and
  config for "BananaCollector").

## RLgraph 0.4 - 2019/04/27

- Agents now support fully customized baselines where ```value_function_spec``` can now 
  be any instance of ValueFunction and does not need to be a list of layers.
  See FAQ for more detail.
- Added support for unity MLAgents environment.
- Added support for Keras-style functional neural network compositions.
  Details will be added to the FAQ.
- Added support for vectorised container actions in Ray executors.
- GAE standardization in PPO is now performed across the entire batch, not sampled sub-batches, 
  which may improve performance when the option is enabled.

## RLgraph 0.3.5/6 - 2019/04/02

- Fixed bug regarding build timing of graph functions calling other graph functions,
  where the call context now accounts for nested calls to be timed more accurately.
- Fixed a number of shape bugs related to container observations in the agent buffer.
- Fixed a bug in the PPO loss function related to updating prior log probs.

## RLgraph 0.3.4 - 2019/03/29

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

## RLgraph 0.3.3 - 2019/02/25

- Added soft actor critic implementation (contributed by @janislavjankov) 
- Separated out action adapters to be able to handle different bounded distributions
- Added a number of torch test cases to continuous integration
- Fixed a number of bugs in define-by-run mode related to arg splitting and merging.
- Fixed a number of shape bugs in various torch implementations
- Fixed a bug relating to assigning references instead of just copying weights when syncing torch Parmeter objects

## RLgraph 0.3.2 - 2019/02/09
- Fixed a number of bugs in internal state management for PyTorch which now
  allow to unify variable creation in most components
- Fixed bug in PyTorch GAE calculation
- Added PyTorch basic replay buffer implementation
- Renamed _variables() to variables() to obtain internal state of a component
- Changed some single node configurations in examples to use less memory and only
  one replay worker (Ape-X).

## RLgraph 0.3.1 - 2019/01/27

- Fixed count bug in synchronous Ray executor
- Fixed bugs related to episode-fetching in the ring-buffer 
  (only occurring when using episode update mode)
- Added reward-clipping option to GAE
- Added post-processing flag to DQN multi-gpu mode

## RLgraph 0.3.0 - 2019/01/25

- Added Ray executor for distributed policy optimization, e.g. distributed PPO on Ray.
- Allow use of api and graph functions from list comprehensions and lambdas
- Improved agent api to define graph functions 
- Fixed various build instabilities related to build order
- Fixed a bug for container actions where huber loss was applied to each action instead to the aggregate loss
- Fixed a number of bugs around space inference for PyTorch when using lists and numpy arrays to store internal state
- Simplified multi-gpu semantics for iterative in-graph multi gpu updates (e.g. on PPO).
- Allow for in-graph and external post-processing via extra flag
- Fixed bug in continuous action policies which made distribution parameters to be parsed incorrectly

## RLgraph 0.2.3 - 2018/12/15

- Improved LSTM-layer handling with keras-style api in network to manage sequences
- Added new LSTM example in examples folder
- Updated implementations to PyTorch 1.0
- Fixed various bugs around PyTorch type inference during build process 
- Improved memory usage of various Ray tasks by avoiding defensive copies,
  following improvements in Ray's memory management.
  
## RLgraph 0.2.2 - 2018/12/3
- Implemented support for advanced decorator options for PyTorch backend
- Various bugfixes in PyTorch utilities needed for PPO/Actor critic

## RLgraph 0.2.1 - 2018/11/25
- Updated actor-critic to support external value functions
- Fixed bugs related to hardcoded entropy for categoricals in loss functions

## RLgraph 0.2.0 - 2018/11/23
- Introduced support for container actions so actions can now be specified as dicts of
arbitrary numbers of sub-actions.
- Added agent a number of learning tests to CI

