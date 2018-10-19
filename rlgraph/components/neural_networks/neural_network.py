# Copyright 2018 The RLgraph authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph import get_backend
from rlgraph.components import Component
from rlgraph.components.neural_networks.stack import Stack
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.utils import force_tuple, force_list
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "pytorch":
    import torch


class NeuralNetwork(Stack):
    """
    A NeuralNetwork is a Stack, in which the apply method is defined either by custom-API-method OR by connecting
    through all sub-Components' `apply` methods.
    In both cases, a dict should be returned with at least the `output` key set. Possible further keys could
    be `last_internal_states` for RNN-based NNs and other keys.
    """

    def __init__(self, *layers, **kwargs):
        """
        Args:
            *layers (Component): Same as `sub_components` argument of Stack. Can be used to add Layer Components
                (or any other Components) to this Network.

        Keyword Args:
            layers (Optional[list]): An optional list of Layer objects or spec-dicts to overwrite(!)
                *layers.
        """
        # Network object for fast-path execution where we do not repeatedely call `call` between layers.
        # self.network_obj = None

        # In case layers come in via a spec dict -> push it into *layers.
        layers_args = kwargs.pop("layers", layers)
        # Add a default scope (if not given) and pass on via kwargs.
        kwargs["scope"] = kwargs.get("scope", "neural-network")

        # Force the only API-method to be `apply`. No matter whether custom-API or auto-generated (via Stack).
        if "api_methods" not in kwargs:
            @rlgraph_api
            def apply(self, *inputs):
                out = self.apply_shadowed_(*inputs)
                if isinstance(out, dict):
                    assert "output" in out
                    return out
                else:
                    return dict(output=out)

            kwargs["api_methods"] = {("apply_shadowed_", "apply")}
        else:
            assert len(kwargs["api_methods"]) == 1, \
                "ERROR: Only 0 or 1 given API-methods are allowed in NeuralNetwork ctor! You provided " \
                "'{}'.".format(kwargs["api_methods"])
            # Make sure the only allowed api_method is `apply`.
            assert next(iter(kwargs["api_methods"]))[0] == "apply", \
                "ERROR: NeuralNetwork's custom API-method must be called `apply`! You named it '{}'.". \
                format(next(iter(kwargs["api_methods"]))[0])

        # Pytorch specific objects.
        self.network_obj = None
        self.non_layer_components = None

        super(NeuralNetwork, self).__init__(*layers_args, **kwargs)

    def build_auto_api_method(self, stack_api_method_name, component_api_method_name):
        if get_backend() == "pytorch" and self.execution_mode == "define_by_run":
            @rlgraph_api(name=stack_api_method_name, component=self)
            def method(self, *inputs, **kwargs):
                # Avoid jumping back between layers and calls at runtime.
                return self._pytorch_fast_path_exec(inputs, **kwargs)
        else:
            super(NeuralNetwork, self).build_auto_api_method(stack_api_method_name, component_api_method_name)

    def _pytorch_fast_path_exec(self, *inputs, **kwargs):
        """
        Builds a fast-path execution method for pytorch / eager.
        """
        inputs = inputs[0]
        forward_inputs = []
        for v in inputs:
            if v is not None:
                if isinstance(v, tuple):
                    # Unitary tuples
                    forward_inputs.append(v[0])
                else:
                    forward_inputs.append(v)
        result = self.network_obj.forward(*forward_inputs)
        # Problem: Not everything in the neural network stack is a true layer.
        for c in self.non_layer_components:
            result = getattr(c, "apply")(*force_list(result))
        return result

    def post_define_by_run_build(self):
        # Layer objects only exist after build - define torch neural network.
        layer_objects = []
        self.non_layer_components = []
        for component in self.sub_components.values():
            if hasattr(component, "layer"):
                # Store Layer object itself.
                layer_objects.append(component.layer)

                # Append activation fn if needed.
                # N.b. linear returns None here.
                if component.activation_fn is not None:
                    layer_objects.append(component.activation_fn)
            else:
                self.non_layer_components.append(component)
        self.network_obj = torch.nn.Sequential(*layer_objects)

    def has_rnn(self):
        # TODO: Maybe it would be better to create a child class (RecurrentNeuralNetwork with has_rrn=True and
        # TODO: other available information for its API-clients such as internal_states_space, etc..)
        # Set a convenience flag if one of our sub-Components is an LSTMLayer.
        return any(isinstance(sc, LSTMLayer) for sc in self.get_all_sub_components())
