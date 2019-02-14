# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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
    A NeuralNetwork is a Stack, in which the `apply` method is defined either by custom-API-method OR by connecting
    through all sub-Components' `apply` methods.
    In both cases, a dict should be returned with at least the `output` key set. Possible further keys could
    be `last_internal_states` for RNN-based NNs and other keys.

    No other API methods other than `apply` should be defined/used.

    A NeuralNetwork Component correctly handles
    """

    def __init__(self, *layers, **kwargs):
        """
        Args:
            *layers (Component): Same as `sub_components` argument of Stack. Can be used to add Layer Components
                (or any other Components) to this Network.

        Keyword Args:
            layers (Optional[list]): An optional list of Layer objects or spec-dicts to overwrite(!)
                *layers.

            fold_time_rank (bool): Whether to overwrite the `fold_time_rank` option for the apply method.
                Only for auto-generated `apply` method. Default: None.

            unfold_time_rank (bool): Whether to overwrite the `unfold_time_rank` option for the apply method.
                Only for auto-generated `apply` method. Default: None.
        """
        # In case layers come in via a spec dict -> push it into *layers.
        layers_args = kwargs.pop("layers", layers)
        # Add a default scope (if not given) and pass on via kwargs.
        kwargs["scope"] = kwargs.get("scope", "neural-network")

        # Force the only API-method to be `apply`. No matter whether custom-API or auto-generated (via Stack).
        self.custom_api_given = True
        if not hasattr(self, "apply"):
            # Automatically create the `apply` stack.
            if "api_methods" not in kwargs:
                kwargs["api_methods"] = [dict(api="apply_shadowed_", component_api="apply")]
                self.custom_api_given = False
            # Sanity check `api_method` to contain only specifications on `apply`.
            else:
                assert len(kwargs["api_methods"]) == 1, \
                    "ERROR: Only 0 or 1 given API-methods are allowed in NeuralNetwork ctor! You provided " \
                    "'{}'.".format(kwargs["api_methods"])
                # Make sure the only allowed api_method is `apply`.
                assert next(iter(kwargs["api_methods"]))[0] == "apply", \
                    "ERROR: NeuralNetwork's custom API-method must be called `apply`! You named it '{}'.". \
                    format(next(iter(kwargs["api_methods"]))[0])

            # Follow given options.
            fold_time_rank = kwargs.pop("fold_time_rank", None)
            if fold_time_rank is not None:
                kwargs["api_methods"][0]["fold_time_rank"] = fold_time_rank
            unfold_time_rank = kwargs.pop("unfold_time_rank", None)
            if unfold_time_rank is not None:
                kwargs["api_methods"][0]["unfold_time_rank"] = unfold_time_rank

        # Pytorch specific objects.
        self.network_obj = None
        self.non_layer_components = None

        super(NeuralNetwork, self).__init__(*layers_args, **kwargs)

    def build_auto_api_method(self, stack_api_method_name, component_api_method_name, fold_time_rank=False,
                              unfold_time_rank=False, ok_to_overwrite=False):
        if get_backend() == "pytorch" and self.execution_mode == "define_by_run":
            @rlgraph_api(name=stack_api_method_name, component=self, ok_to_overwrite=ok_to_overwrite)
            def method(self, nn_input, *nn_inputs, **kwargs):
                # Avoid jumping back between layers and calls at runtime.
                return self._pytorch_fast_path_exec(*([nn_input] + list(nn_inputs)), **kwargs)

        # Auto apply-API -> Handle LSTMs correctly.
        elif self.custom_api_given is False:
            @rlgraph_api(component=self, ok_to_overwrite=ok_to_overwrite)
            def apply(self_, nn_input, *nn_inputs, **kwargs):
                inputs = [nn_input] + list(nn_inputs)
                # Keep track of the folding status.
                fold_status = "unfolded" if self.has_rnn() else None
                # Fold time rank? For now only support 1st arg folding/unfolding.
                original_input = inputs[0]
                if fold_time_rank is True:
                    args_ = tuple([self.folder.apply(original_input)] + list(inputs[1:]))
                    fold_status = "folded"
                else:
                    # TODO: If only unfolding: Assume for now that 2nd input is the original one (so we can infer
                    # TODO: batch/time dims).
                    if unfold_time_rank is True:
                        assert len(inputs) >= 2, \
                            "ERROR: In Stack: If unfolding w/o folding, second arg must be the original input!"
                        original_input = inputs[1]
                        args_ = tuple([inputs[0]] + list(inputs[2:]))
                    else:
                        args_ = inputs
                kwargs_ = kwargs

                # TODO: keep track of LSTMLayers that only return the last time-step (outputs after these Layers
                # TODO: can no longer be folded, their time-rank is gone for the rest of the NN.
                for i, sub_component in enumerate(self_.sub_components.values()):  # type: Component
                    if sub_component.scope in ["time-rank-folder_", "time-rank-unfolder_"]:
                        continue

                    # Unfold before an LSTM.
                    if isinstance(sub_component, LSTMLayer) and fold_status != "unfolded":
                        args_, kwargs_ = self._unfold(original_input, *args_, **kwargs_)
                        fold_status = "unfolded"
                    # Fold before a non-LSTM if not already done so.
                    elif not isinstance(sub_component, LSTMLayer) and fold_status == "unfolded":
                        args_, kwargs_ = self._fold(*args_, **kwargs_)
                        fold_status = "folded"

                    results = sub_component.apply(*args_, **kwargs_)

                    # Recycle args_, kwargs_ for reuse in next sub-Component's API-method call.
                    if isinstance(results, dict):
                        args_ = ()
                        kwargs_ = results
                    else:
                        args_ = force_tuple(results)
                        kwargs_ = {}

                if unfold_time_rank:
                    args_, kwargs_ = self._unfold(original_input, *args_, **kwargs_)
                if args_ == ():
                    return kwargs_
                elif len(args_) == 1:
                    return dict(output=args_[0])
                else:
                    return dict(output=args_)

        else:
            super(NeuralNetwork, self).build_auto_api_method(
                stack_api_method_name, component_api_method_name, fold_time_rank, unfold_time_rank, ok_to_overwrite
            )

    def _unfold(self, original_input, *args_, **kwargs_):
        if args_ == ():
            assert len(kwargs_) == 1, \
                "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
            key = next(iter(kwargs_))
            kwargs_ = {key: self.unfolder.apply(kwargs_[key], original_input)}
        else:
            assert len(args_) == 1, \
                "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
            args_ = (self.unfolder.apply(args_[0], original_input),)
        return args_, kwargs_

    def _fold(self, *args_, **kwargs_):
        if args_ == ():
            assert len(kwargs_) == 1, \
                "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
            key = next(iter(kwargs_))
            kwargs_ = {key: self.folder.apply(kwargs_[key])}
        else:
            args_ = (self.folder.apply(args_[0]),)
        return args_, kwargs_

    def add_layer(self, layer_component):
        """
        Adds an additional Layer Component (even after c'tor execution) to this NN.
        TODO: Currently, layers are always added to the end.

        Args:
            layer_component (Layer): The Layer object to be added to this NN.
        """
        assert self.custom_api_given is False,\
            "ERROR: Cannot add layer to neural network if `apply` API-method is a custom one!"
        assert hasattr(layer_component, self.map_api_to_sub_components_api["apply_shadowed_"]), \
            "ERROR: Layer to be added ({}) does not have an API-method called '{}'!".format(
                layer_component.scope, self.map_api_to_sub_components_api["apply_shadowed_"]
            )
        self.add_components(layer_component)
        self.build_auto_api_method("apply_shadowed_", self.map_api_to_sub_components_api["apply_shadowed_"],
                                   ok_to_overwrite=True)

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
        """
        Returns:
            True if one of our sub-Components is an LSTMLayer, False otherwise.
        """
        # TODO: Maybe it would be better to create a child class (RecurrentNeuralNetwork with has_rrn=True and
        # TODO: other available information for its API-clients such as internal_states_space, etc..)
        return any(isinstance(sc, LSTMLayer) for sc in self.get_all_sub_components())
