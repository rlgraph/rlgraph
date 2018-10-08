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
from rlgraph.components.neural_networks.stack import Stack
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.utils import RLGraphError, force_tuple, force_list

if get_backend() == "pytorch":
    import torch


class NeuralNetwork(Stack):
    """
    Simple placeholder class that's a Stack.
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
        self.network_obj = None

        # In case layers come in via a spec dict -> push it into *layers.
        layers_args = kwargs.pop("layers", layers)
        # Add a default scope (if not given) and pass on via kwargs.
        kwargs["scope"] = kwargs.get("scope", "neural-network")
        super(NeuralNetwork, self).__init__(*layers_args, **kwargs)

        # Assert that the apply API-method has been defined and that it takes two input args:
        # `inputs` and `internal_states`.
        assert "apply" in self.api_methods

    def _build_stack(self, api_methods):
        """
        For each api-method in set `api_methods`, automatically create this Stack's own API-method by connecting
        through all sub-Component's API-methods. This is skipped if this Stack already has a custom API-method
        by that name.

        Args:
            api_methods (Set[Union[str,Tuple[str,str]]]): See ctor kwargs.
            #connection_rule (str): See ctor kwargs.
        """
        # Loop through the API-method set.
        for api_method_spec in api_methods:
            custom_api_method = None

            # API-method of sub-Components and this Stack should have different names.
            if isinstance(api_method_spec, tuple):
                # Custom method given, use that instead of creating one automatically.
                if callable(api_method_spec[1]):
                    stack_api_method_name = components_api_method_name = api_method_spec[0]
                    custom_api_method = api_method_spec[1]
                else:
                    stack_api_method_name, components_api_method_name = api_method_spec[0], api_method_spec[1]
            # API-method of sub-Components and this Stack should have the same name.
            else:
                stack_api_method_name = components_api_method_name = api_method_spec

            # Custom API-method was provided. Register it.
            if custom_api_method is not None:
                self.define_api_method(stack_api_method_name, custom_api_method)

            # API-method for this Stack does not exist yet -> Automatically create it.
            elif not hasattr(self, stack_api_method_name):
                @api(name=stack_api_method_name, component=self)
                def method(self_, *inputs):
                    if get_backend() == "pytorch" and self.execution_mode == "define_by_run":
                        # Avoid jumping back between layers and calls at runtime.
                        return self.fast_path_exec(inputs)
                    else:
                        result = inputs
                        for sub_component in self_.sub_components.values():  # type: Component
                            num_allowed_inputs = sub_component.get_number_of_allowed_inputs(components_api_method_name)
                            num_actual_inputs = len(result) if isinstance(result, (tuple, list)) else 1
                            # Check whether number of inputs to this sub-component's API-method is ok.
                            if num_allowed_inputs[0] > num_actual_inputs:
                                raise RLGraphError(
                                    "Number of given input args ({}) to Stack's API-method '{}' is too low! Needs to "
                                    "be at least {}.".format(num_actual_inputs, stack_api_method_name,
                                                             num_allowed_inputs[0])
                                )
                            elif num_allowed_inputs[1] is not None and num_allowed_inputs[1] < num_actual_inputs:
                                raise RLGraphError(
                                    "Number of given input args ({}) to Stack's API-method '{}' is too high! Needs to "
                                    "be at most {}.".format(num_actual_inputs, stack_api_method_name, num_allowed_inputs[1])
                                )

                            # TODO: python-Components: For now, we call each preprocessor's graph_fn
                            #  directly (assuming that inputs are not ContainerSpaces).
                            if self_.backend == "python" or get_backend() == "python":
                                graph_fn = getattr(sub_component, "_graph_fn_" + components_api_method_name)
                                if sub_component.api_methods[components_api_method_name].add_auto_key_as_first_param:
                                    result = graph_fn("", *force_tuple(result))
                                else:
                                    result = graph_fn(*force_tuple(result))
                            elif get_backend() == "pytorch":
                                # Do NOT convert to tuple, has to be in unpacked again immediately.n
                                result = getattr(sub_component, components_api_method_name)(*force_list(result))
                            elif get_backend() == "tf":
                                result = getattr(sub_component, components_api_method_name)(*force_tuple(result))
                        return result

        # Build fast-path execution method for pytorch / eager.
        if get_backend() == "pytorch":
            def fast_path_exec(*inputs):
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
            self.fast_path_exec = fast_path_exec

    def _post_define_by_run_build(self):
        # Layer objects only exist after build - define torch neural network.
        layer_objects = []
        self.non_layer_components = []
        for component in self.sub_components.values():
            if hasattr(component, "layer"):
                # Store Layer object itself.
                layer_objects.append(component.layer)

                # Append activation fn if needed.
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
