
# FAQ

Here we collect short answers to common questions and point to resources.


#### How can I execute a gym environment?

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


#### How can I use a custom environment?

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

#### How can I use a more complex network structure?

There are a couple of different ways to define your own NeuralNetworks.

###### Sequential via lists of layer configs
The simplest method is to provide a list of layer configurations to the
`from_spec` method as in
the following example:

```
my_sequential_network = NeuralNetwork.from_spec([
    {"type": "dense", "units": 10, "scope": "layer-1"},
    {"type": "dense", "units": 5, "scope": "layer-2"},
])

# This is the same as passing in all layer specs directly into the ctor as *args:
my_sequential_network = NeuralNetwork(
    {"type": "dense", "units": 10, "scope": "layer-1"},
    {"type": "dense", "units": 5, "scope": "layer-2"},
)
```

###### Using the base NeuralNetwork class plus a custom `call()` method.

For non-sequential networks (e.g. with many input streams that need
to be merged or for LSTM-containing networks with internal
states- or sequence-length inputs), you have the ability to use the plain
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

###### Subclassing the base NeuralNetwork class.

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


###### Keras-Style functional API NeuralNetwork assembly (since version 0.5.0).

Lastly, since version 0.5.0, we can create complex NeuralNetworks by
using a Keras-style functional API, without sub-classing and without custom
`call` methods. This way, you can write your networks Keras-like inside
your code (where you then also create and run your Agent).

For example, to mimic the above `MyCustomNetwork` class, you could
instead do:

```
# Define all dataflow on the fly using RLgraph Layer Components and
# calling them via `Layer([some input(s)])`:

# Define an input Space first (tuple of two input tensors).
input_space = Tuple([IntBox(3), FloatBox(shape=(4,))], add_batch_rank=True)

# One-hot flatten the int tensor.
flatten_layer_out = ReShape(flatten=True, flatten_categories=True)(input_space[0])
# Run the float tensor through two dense layers.
dense_1_out = DenseLayer(units=3)(input_space[1])
dense_2_out = DenseLayer(units=5)(dense_1_out)
# Concat everything.
cat_out = ConcatLayer()(flatten_layer_out, dense_2_out)

# Use the `outputs` arg to allow your network to trace back the data flow until the input space.
my_keras_style_nn = NeuralNetwork(outputs=cat_out)

# Create an Agent and pass the NN into it as `network_spec`:
my_agent - DQNAgent( .. , network_spec=my_keras_style_nn, ..)
```

