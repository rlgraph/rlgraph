
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


### How can I use a custom environment?

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


### How can I train with decaying/changing parameters (like learning rates) over time?

This is easily done in json or other Agent config types since version 0.4.2 and it is applicable to
all hyper-parameters that are based on the `TimeDependentParameter` class, which should
be all learning rates, loss-function hyper-parameters (where this makes sense), and
exploration parameters (such as epsilon values in DQN exploration schemas).

In your json config, you can specify these parameters in one of the
following ways. Hereby, `from` is the start value, `to` is the final
value, `time_percentage` is the provided percentage (between 0.0 and 1.0)
of the time passed, and `power` and `decay_rate` are type-specific
parameters (see formulas below):

**For a constant/time-independent value:**
```
"optimizer_spec": {
    "type": "adam",
    "learning_rate": 0.001
}
```

**For a linearly decaying value** *(`from` - `time_percentage` * (`from` - `to`)):*
```
    "learning_rate": ["linear", 0.01, 0.0001]  # from=0.01, to=0.0001
    Or even simpler:
    "learning_rate": [0.01, 0.0001]  # <- if no type is given, assume "linear"
```

**For a polynomially decaying value** *(`to` + (`from` - `to`) * (1 - `time_percentage`) ** `power`):*
```
    "learning_rate": ["polynomial", 0.01, 0.0001]  # from=0.01, to=0.0001, power=2.0 (default)
    Or in case power is not 2.0:
    "learning_rate": ["polynomial", 0.01, 0.0001, 3.0]   <- power=3.0
```

**For an exponentially decaying value** *(`to` + (`from` - `to`) * `decay_rate` ** `time_percentage`):*
```
    "learning_rate": ["exponential", 0.01, 0.0001]  # from=0.01, to=0.0001, decay_rate=0.1 (default)
    Or in case decay_rate is not 0.1:
    "learning_rate": ["exponential", 0.01, 0.0001, 0.5]   <- decay_rate=0.5
```

The Component behind this is the `TimeDependentParameter` class. If you are implementing
your own Components that use time-dependent hyper-parameters (a new loss function, etc),
simply switch on time-decay support for the parameter as follows:
```
from rlgraph.components.common.time_dependent_parameters import TimeDependentParameter

class MyComponent(Component):
    def __init__(self, my_hyperparam=0.01):
        super(MyComponent, self).__init__()
        # Generate the sub-component and add it.
        self.my_hyperparam = TimeDependentParameter.from_spec(my_hyperparam)
        self.add_components(self.my_hyperparam)

    @rlgraph_api
    def get_decayed_value(self, time_percentage):  # <- this input is necessary for getting the value from any time-dependent parameter
        decayed_value = self.my_hyperparam.get(time_percentage)
        return decayed_value
```

The `time_percentage` value is passed by all Workers into
`Agent.get_action()` as well as `Agent.update()` and should thus arrive properly
inside all `update_from_external_batch/memory` and
`get_actions/actions_and_preprocessed_states` API-methods, where it can be further passed
into loss function and optimizer calls (and possibly other Components' APIs).

The Workers calculate this `time_percentage` value by keeping track of the time steps done
(each single acting is one time step) and dividing that number by some max timestep value,
which is either given to the Worker upon construction or later when calling `execute_timesteps`.
The Worker will try to infer a suitable max timestep value in any situation.
NOTE here that the Agent should not have to worry about time-step counting
(or time_percentage calculations). This is an execution detail that should be left outside an Agent.


### How can I use a more complex network structure?

There are many different ways to define your own neural networks in RLgraph
using our flexible NeuralNetwork Component class. It supports everything from simple
sequential setups, to sub-classing or custom `call` methods, and even a
full Keras-style assembly procedure (the new recommended way for
multi-stream and other complex NNs).
Here are the different ways allowing for arbitrary network complexity:

##### Sequential via lists of layer configs (recommended for simple, sequential NNs)
For simple, sequential networks, the method of choice is to provide a list of
layer configurations to the `NeuralNetwork.from_spec()` method or the
`NeuralNetwork` c'tor as in the following examples:

```
# Using the from_spec util:
my_sequential_network = NeuralNetwork.from_spec([
    {"type": "dense", "units": 10, "scope": "layer-1"},
    {"type": "dense", "units": 5, "scope": "layer-2"},
])

# This is the same as passing in all layer specs directly into the NeuralNetwork c'tor as *args:
my_sequential_network = NeuralNetwork(
    {"type": "dense", "units": 10, "scope": "layer-1"},
    {"type": "dense", "units": 5, "scope": "layer-2"},
)
```

##### Keras-Style functional API NeuralNetwork assembly (recommended for multi-stream/complex NNs).

From version 0.5.0 on, you can create complex NeuralNetworks by
using a Keras-style functional API, without sub-classing and without custom
`call` methods (see below). This way, you can write your networks Keras-like inside
your code (where you then also create and run your Agent).

For example, to generate an NN with 2 input streams, you can do:

```
# Define all dataflow on the fly using RLgraph Layer Components and
# calling them via `Layer([some input(s)])`:

# Define an input Space first (tuple of two input tensors).
input_space = Tuple([IntBox(3), FloatBox(shape=(4,))], add_batch_rank=True)

# One-hot flatten the int tensor (Tuple index 0).
flatten_layer_out = ReShape(flatten=True, flatten_categories=True)(input_space[0])

# Run the float tensor (Tuple index 1) through two dense layers.
dense_1_out = DenseLayer(units=3)(input_space[1])
dense_2_out = DenseLayer(units=5)(dense_1_out)

# Concat everything.
cat_out = ConcatLayer()(flatten_layer_out, dense_2_out)

# Use the `outputs` arg (like in Keras) to allow your network to trace back the
# data flow until the input space.
# You do not(!) need an `inputs` arg here as we only have one input (the Tuple).
my_keras_style_nn = NeuralNetwork(outputs=cat_out)

# Create an Agent and pass the NN into it as `network_spec`:
my_agent = DQNAgent( .. , network_spec=my_keras_style_nn, ..)
```

In case you don't like passing Tuples into multi-input NNs, you can also use
single Spaces like so:

```
# Simple list of two inputs.
input_spaces = [IntBox(3, add_batch_rank=True), FloatBox(shape=(4,), add_batch_rank=True)]

# ...
# build your network using input_spaces[0] and input_spaces[1] as inputs to some layers.
# ...

# Now we do have to specify an `inputs` c'tor arg so that the NN knows the order of the inputs.
my_keras_style_nn = NeuralNetwork(inputs=input_spaces, outputs=cat_out)
```


##### Using the base NeuralNetwork class plus a custom `call()` method.

For non-sequential networks (e.g. with many input streams that need
to be merged or for LSTM-containing networks with internal
states- or sequence-length inputs), you have the ability to use the base
NeuralNetwork class and pass a custom `call` method (taking a single
Tuple-space `inputs` arg) into the constructor like so:

```
def my_custom_call(self, inputs):
    # implicit split of the incoming Tuple space
    input1 = inputs[0]
    input2 = inputs[1]

    # Somewhat complex (non-sequential) dataflow.
    out1 = self.get_sub_component_by_name("d1").call(input1)
    out2 = self.get_sub_component_by_name("d2").call(input2)
    cat_output = self.get_sub_component_by_name("cat").call(out1, out2)

    return cat_output

my_multi_stream_network = NeuralNetwork(
    DenseLayer(units=3, scope="d1"),
    DenseLayer(units=3, scope="d2"),
    ConcatLayer(scope="cat"),
    # This makes sure `my_custom_call` is used and no automatic `call` is generated.
    api_methods={("call", my_custom_call)}
)
```

Similarly, you can construct an LSTM containing network as follows (note
again the single `inputs` arg containing all information the network
needs in a Tuple space):

```
def my_custom_call(self, inputs):
    # implicit split of the incoming Tuple space
    input_ = inputs[0]
    seq_lengths = inputs[1]

    # Special LSTM data flow using seq-lengths for dynamic LSTMs.
    dense_out = self.get_sub_component_by_name("dense").call(input_)
    lstm_out, last_internal_states = self.get_sub_component_by_name("lstm").call(dense_out, seq_lengths)
    return lstm_out, last_internal_states

my_lstm_network = NeuralNetwork(
    DenseLayer(units=3, scope="dense"),
    LSTMLayer(units=5, scope="lstm"),
    api_methods={("call", my_custom_call)}
)
```

##### Subclassing the base NeuralNetwork class.

To further customize a NN and its data flow, you can also subclass the
base NeuralNetwork class and create your own NN class. In your sub-class
you should override the `call` method and are allowed to add further
methods.

The ``network_spec`` and ```value_function_spec``` parameters in several
Components take simple lists of layer configurations (see examples above)
but also now accept instances of ```NeuralNetwork``` and ```ValueFuction``` respectively:

```agent = DQNAgent(network_spec=MyNetwork(), **kwargs)```
will use ```MyNetwork``` as the base of the policy object.

In ```components/neural_networks/```, we show two examples of custom
value functions. The ```impala_networks``` module illustrates how to compose
an involved architecture combining LSTM and convolutional stacks from multiple
inputs for the IMPALA policy networks. In ```sac_networks```, we implement
a value function that concatenates states and actions which can require
splitting up a network into different stacks.

Here is an example for NeuralNetwork subclassing to define custom
complex data flows:

```
class MyCustomNetwork(NeuralNetwork):
    def __init__(self, units_in_second_layer=5):
        super(MyCustomNetwork, self).__init__()

        self.layer1 = DenseLayer(units=10, scope="d1")
        self.layer2 = DenseLayer(units=units_in_second_layer, scope="d2")

        self.add_sub_component(self.layer1, self.layer2)

    # Override `call`.
    def call(self, inputs):
        out1 = self.get_sub_component_by_name("d1").call(inputs)
        out2 = self.get_sub_component_by_name("d2").call(out1)
        # "complex" dataflow.
        out_repeat_d1 = self.get_sub_component_by_name("d1").call(out2)
        return out_repeat_d1
```

