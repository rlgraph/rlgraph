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

from __future__ import absolute_import, division, print_function

import copy
import re

from rlgraph import get_backend
from rlgraph.components.layers.preprocessing.container_splitter import ContainerSplitter
from rlgraph.components.component import Component
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.components.neural_networks.stack import Stack
from rlgraph.spaces.containers import ContainerSpace
from rlgraph.utils import force_tuple, force_list
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.rlgraph_errors import RLGraphKerasStyleAssemblyError

if get_backend() == "pytorch":
    import torch


class NeuralNetwork(Stack):
    """
    A NeuralNetwork is a Stack, in which the `call` method is defined either by custom-API-method OR by connecting
    through all sub-Components' `call` methods. The signature of the `call` method is always (self, *inputs).
    In all cases, 1 or more values may be returned by `call`.
    No other API methods other than `call` should be defined/used.
    """
    def __init__(self, *layers, **kwargs):
        """
        Args:
            *layers (Component): Same as `sub_components` argument of Stack. Can be used to add Layer Components
                (or any other Components) to this Network.

        Keyword Args:
            layers (Optional[list]): An optional list of Layer objects or spec-dicts to overwrite(!)
                *layers.

            inputs (Optional[List[Space]]): A list of Spaces or a single Space object defining the input spaces for
                the `call` method of this network. Must be provided, if more than one input arg are needed by `call`
                to determine the order in which these inputs will come in.

            outputs (Optional[List[NNCallOutput]]): A list or single output NNCallOutput object,
                indicating that we have to infer the `call` method from the graph given by these outputs.
                This is used iff a NN is constructed by the Keras-style functional API.

            num_inputs (Optional[int]): An optional number of inputs the `call` method will take as `*inputs`.
                If not given, NN will try to infer this value automatically.

            fold_time_rank (bool): Whether to overwrite the `fold_time_rank` option for the apply method.
                Only for auto-generated `call` method. Default: None.

            unfold_time_rank (bool): Whether to overwrite the `unfold_time_rank` option for the `call` method.
                Only for auto-generated `call` method. Default: None.
        """
        # In case layers come in via a spec dict -> push it into *layers.
        layers_args = kwargs.pop("layers", layers)
        # Add a default scope (if not given) and pass on via kwargs.
        kwargs["scope"] = kwargs.get("scope", "neural-network")
        self.keras_style_api_outputs = force_list(kwargs.pop("outputs", None))
        self.keras_style_api_inputs = force_list(kwargs.pop("inputs", []))
        # If Keras-style inputs are given, just count those, otherwise allow for `num_inputs` hint (default: 1).
        self.num_inputs = len(self.keras_style_api_inputs)
        if self.num_inputs == 0:
            self.num_inputs = kwargs.pop("num_inputs", 1)
        self.num_outputs = min(len(self.keras_style_api_outputs), 1)

        # Force the only API-method to be `call`. No matter whether custom-API or auto-generated (via Stack).
        self.custom_call_given = True
        if not hasattr(self, "call"):
            # Automatically create the `call` stack.
            if "api_methods" not in kwargs:
                kwargs["api_methods"] = [dict(api="call_shadowed_", component_api="call")]
                self.custom_call_given = False
            # Sanity check `api_method` to contain only specifications on `call`.
            else:
                assert len(kwargs["api_methods"]) == 1, \
                    "ERROR: Only 0 or 1 given API-methods are allowed in NeuralNetwork ctor! You provided " \
                    "'{}'.".format(kwargs["api_methods"])
                # Make sure the only allowed api_method is `call`.
                assert next(iter(kwargs["api_methods"]))[0] == "call", \
                    "ERROR: NeuralNetwork's custom API-method must be called `call`! You named it '{}'.". \
                    format(next(iter(kwargs["api_methods"]))[0])

            # Follow given options.
            fold_time_rank = kwargs.pop("fold_time_rank", None)
            if fold_time_rank is not None:
                kwargs["api_methods"][0]["fold_time_rank"] = fold_time_rank
            unfold_time_rank = kwargs.pop("unfold_time_rank", None)
            if unfold_time_rank is not None:
                kwargs["api_methods"][0]["unfold_time_rank"] = unfold_time_rank

        assert len(self.keras_style_api_outputs) == 0 or self.custom_call_given is False, \
            "ERROR: If functional API is used to construct network, a custom `call` method must not be provided!"

        # Pytorch specific objects.
        self.network_obj = None
        self.non_layer_components = None

        super(NeuralNetwork, self).__init__(*layers_args, **kwargs)

        # In case we have more than one input (and not using Keras-style assembly),
        # add another input splitter here.
        self.inputs_splitter = None
        if self.num_inputs > 1:
            self.inputs_splitter = ContainerSplitter(tuple_length=self.num_inputs, scope=".helper-inputs-splitter")
            self.add_components(self.inputs_splitter)

    def build_auto_api_method(self, stack_api_method_name, component_api_method_name, fold_time_rank=False,
                              unfold_time_rank=False, ok_to_overwrite=False):

        if get_backend() == "pytorch" and self.execution_mode == "define_by_run":
            @rlgraph_api(name=stack_api_method_name, component=self, ok_to_overwrite=ok_to_overwrite)
            def method(self, nn_input, *nn_inputs, **kwargs):
                # Avoid jumping back between layers and calls at runtime.
                return self._pytorch_fast_path_exec(*([nn_input] + list(nn_inputs)), **kwargs)

        # Functional API (Keras Style assembly). TODO: Add support for pytorch.
        elif len(self.keras_style_api_outputs) > 0:
            self._build_call_via_keras_style_functional_api(*self.keras_style_api_outputs)

        # Auto call-API -> Handle LSTMs correctly.
        elif self.custom_call_given is False:
            self._build_auto_call_method(fold_time_rank, unfold_time_rank)

        # Have super class (Stack) handle registration of given custom `call` method.
        else:
            super(NeuralNetwork, self).build_auto_api_method(
                stack_api_method_name, component_api_method_name, fold_time_rank, unfold_time_rank, True
            )

    def _unfold(self, original_input, *args_, **kwargs_):
        if args_ == ():
            assert len(kwargs_) == 1, \
                "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
            key = next(iter(kwargs_))
            kwargs_ = {key: self.unfolder.call(kwargs_[key], original_input)}
        else:
            assert len(args_) == 1, \
                "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
        args_ = (self.unfolder.call(args_[0], original_input),)
        return args_, kwargs_

    def _fold(self, *args_, **kwargs_):
        if args_ == ():
            assert len(kwargs_) == 1, \
                "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
            key = next(iter(kwargs_))
            kwargs_ = {key: self.folder.call(kwargs_[key])}
        else:
            args_ = (self.folder.call(args_[0]),)
        return args_, kwargs_

    def add_layer(self, layer_component):
        """
        Adds an additional Layer Component (even after c'tor execution) to this NN.
        TODO: Currently, layers are always added to the end.

        Args:
            layer_component (Layer): The Layer object to be added to this NN.
        """
        assert self.custom_call_given is False,\
            "ERROR: Cannot add layer to neural network if `call` API-method is a custom one!"
        assert hasattr(layer_component, self.map_api_to_sub_components_api["call_shadowed_"]), \
            "ERROR: Layer to be added ({}) does not have an API-method called '{}'!".format(
                layer_component.scope, self.map_api_to_sub_components_api["call_shadowed_"]
            )
        self.add_components(layer_component)
        self.build_auto_api_method("call_shadowed_", self.map_api_to_sub_components_api["call_shadowed_"],
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
            result = getattr(c, "call")(*force_list(result))
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

    def _build_call_via_keras_style_functional_api(self, *layer_call_outputs):
        """
        Automatically builds our `call` method by traversing the given graph depth first via the following iterative
        procedure:

        Add given `layer_call_outputs` to a set.
        While still items in set that are not Spaces:
            For o in set:
                If o is lone output for its call OR all outputs are in set.
                    write call to code
                    erase outs from set
                    add ins to set
        Write `def call(self, ...)` from given Spaces.
        """
        output_set = set(layer_call_outputs)
        output_id = 0
        sub_components = set()

        def _all_siblings_in_set(output, set_):
            siblings = []
            need_to_find = output.num_outputs
            for o in set_:
                if o.component == output.component:
                    siblings.append(o)
            return len(siblings) == need_to_find, sorted(siblings, key=lambda s: s.output_slot)

        # Initialize var names for final outputs.
        for out in sorted(output_set):
            out.var_name = "out{}".format(output_id)
            output_id += 1

        # Write this NN's `call` API-method code dynamically, then execute it.
        call_code = "\treturn {}\n".format(", ".join([o.var_name for o in layer_call_outputs]))

        prev_output_set = None

        # Input Space-IDs that we know will be used.
        functional_api_input_ids = [space.id for space in self.keras_style_api_inputs]
        # If no inputs given -> Allow only a single-input arg setup (otherwise, there would be
        # ambiguity).
        auto_functional_api_single_input = None

        # Loop through all nodes.
        while len(output_set) > 0:
            output_list = list(output_set)

            output = next(iter(sorted(output_list)))

            # If only one output OR all outputs are in set -> Write the call.
            found_all, siblings = _all_siblings_in_set(output, output_set)
            if found_all is True:
                siblings_str = ", ".join([o.var_name for o in siblings])
            # Nothing has changed and it's the only output in list
            # Some output(s) may be dead ends (construct those as `_`).
            elif prev_output_set == output_set or (prev_output_set is None and len(output_set) == 1):
                indices = [s.output_slot for s in siblings]
                siblings_str = ""
                for i in range(output.num_outputs):
                    siblings_str += ", " + (siblings[indices.index(i)].var_name if i in indices else "_")
                siblings_str = siblings_str[2:]  # cut preceding ", "
            else:
                continue

            # Remove outs from set.
            for sibling in siblings:
                output_set.remove(sibling)
            # Add `ins` to set (or set to one of the `inputs[?]` for the `call` method.
            for pos, in_ in enumerate(output.inputs):
                # This input is a Space -> If we can find it in `self.keras_style_api_inputs`, use the correct
                # `inputs[?]` reference here, if not, may be a child of a container input, in which case:
                # tag it for now with `inputs[Space.id]`.
                if in_.space is not None:
                    # Given Space is in this NeuralNetwork's given inputs. Use its `*inputs`-index directly.
                    if in_.space.id in functional_api_input_ids:
                        in_.var_name = "inputs[{}]".format(functional_api_input_ids.index(in_.space.id))
                    # A child of an input container space. Add the necessary ContainerSplitter and `inputs`-index
                    # automatically.
                    else:
                        if len(functional_api_input_ids) == 0:
                            top_level_container_space = in_.space.get_top_level_container()
                            # Make sure it's always the same top-level container (only single input allowed in this
                            # case, due to arg-order ambiguity otherwise).
                            if auto_functional_api_single_input is not None:
                                if top_level_container_space != auto_functional_api_single_input:
                                    raise RLGraphKerasStyleAssemblyError(
                                        "When creating NeuralNetwork '{}' in Keras-style assembly and not providing "
                                        "the `inputs` arg, only one single input into the Network is allowed! You have "
                                        "{} and {}.".format(self.global_scope, auto_functional_api_single_input,
                                                            top_level_container_space)
                                    )
                            else:
                                auto_functional_api_single_input = top_level_container_space
                        # Look for this Space in `self.keras_style_api_inputs`.
                        index_chain = []
                        if self._get_container_space_index_chain(
                                self.keras_style_api_inputs if len(self.keras_style_api_inputs) > 0
                                else [auto_functional_api_single_input],
                                in_.space.id, index_chain
                        ) is False:
                            raise RLGraphKerasStyleAssemblyError(
                                "Input '{}' into NeuralNetwork '{}' was not found in any of the provided `inputs` "
                                "(or in the auto-derived input)!".format(in_.space, self.global_scope)
                            )
                        in_.var_name = "inputs[{}]".format("][".join(index_chain))
                elif in_.var_name is None:
                    in_.var_name = "out{}".format(output_id)
                    output_id += 1
                    output_set.add(in_)

            inputs_str = ", ".join([k + i.var_name for i, k in zip(output.inputs, output.kwarg_strings)])
            call_code = "\t{} = self.get_sub_component_by_name('{}').call({})\n".format(
                siblings_str, output.component.scope, inputs_str) + call_code
            sub_components.add(output.component)

            # Store previous state of our set.
            prev_output_set = output_set

        # Prepend inputs from left-over Space objects in set.
        call_code = \
            "@rlgraph_api(component=self, ok_to_overwrite=True)\n" + \
            "def call(self, *inputs):\n" + \
            call_code

        # Add all sub-components to this NN.
        self.add_components(*list(sub_components))

        # Execute the code and assign self.call to it.
        print("`call_code` for NN:")
        print(call_code)
        exec(call_code, globals(), locals())

    def _build_auto_call_method(self, fold_time_rank, unfold_time_rank):
        @rlgraph_api(component=self, ok_to_overwrite=True)
        def call(self_, *inputs):
            # Everything is lumped together in inputs[0] but is supposed to be split -> Do this here.
            if len(inputs) == 1 and self.num_inputs > 1:
                inputs = self.inputs_splitter.call(inputs[0])

            inputs = list(inputs)
            original_input = inputs[0]

            # Keep track of the folding status.
            fold_status = "unfolded" if self.has_rnn() else None
            # Fold time rank? For now only support 1st arg folding/unfolding.
            if fold_time_rank is True:
                args_ = tuple([self.folder.call(original_input)] + list(inputs[1:]))
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

            kwargs_ = {}

            # TODO: keep track of LSTMLayers that only return the last time-step (outputs after these Layers
            # TODO: can no longer be folded, their time-rank is gone for the rest of the NN.
            for i, sub_component in enumerate(self_.sub_components.values()):  # type: Component
                if re.search(r'^\.helper-', sub_component.scope):
                    continue

                # Unfold before an LSTM.
                if isinstance(sub_component, LSTMLayer) and fold_status != "unfolded":
                    args_, kwargs_ = self._unfold(original_input, *args_, **kwargs_)
                    fold_status = "unfolded"
                # Fold before a non-LSTM if not already done so.
                elif not isinstance(sub_component, LSTMLayer) and fold_status == "unfolded":
                    args_, kwargs_ = self._fold(*args_, **kwargs_)
                    fold_status = "folded"

                results = sub_component.call(*args_, **kwargs_)

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
                return args_[0]
            else:
                self.num_outputs = len(args_)
                return args_

    @staticmethod
    def _get_container_space_index_chain(spaces, space_id, _index_chain=None):
        """
        Finds `space_id` in `spaces` and returns the actual path from the top-level Space till the child-Space
        with id=space_id.

        Args:
            spaces (Union[List[Space],Tuple[Space],Dict[str,Space]]): The container Space or list of Spaces to look
                through.

            space_id (int): The ID of the Space, we are trying to find in `spaces`.

            _index_chain (List[str,int]): The indexing chain so far. Starts with the index of the matching parent Space
                in `spaces`. E.g. given:
                spaces=(Tuple([spaceA(id=0),Dict(a=SpaceB(id=2), b=SpaceC(id=5))]))
                space_id=5
                -> returns: [1, "b"] -> pick index 1 in Tuple, then key "b" in Dict.

        Returns:
            List[str]: A list of inputs indices, e.g. ["0", "'img'", "2"] to go from the top-level Space in `spaces`
                to the given Space's id.
        """
        assert isinstance(spaces, (tuple, list, dict)), \
            "ERROR: `spaces` must be tuple/list (Tuple Space) OR dict (Dict Space)!"

        for idx, in_space in (spaces.items() if isinstance(spaces, dict) else enumerate(spaces)):
            index_chain_copy = copy.deepcopy(_index_chain)
            # Found the ID.
            if in_space.id == space_id:
                _index_chain.append(str(idx) if isinstance(idx, int) else "\"" + idx + "\"")
                return True
            # Another container -> recurse.
            elif isinstance(in_space, ContainerSpace):
                index_chain_copy.append(str(idx) if isinstance(idx, int) else "\"" + idx + "\"")
                if NeuralNetwork._get_container_space_index_chain(in_space, space_id, index_chain_copy):
                    _index_chain[:] = index_chain_copy
                    return True

        # Not found -> Return False.
        return False
