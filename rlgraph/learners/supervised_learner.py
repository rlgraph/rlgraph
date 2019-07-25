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

from rlgraph.learners.learner import Learner
from rlgraph.components.algorithms.supervised_root_component import SupervisedRootComponent
from rlgraph.spaces import IntBox


class SupervisedLearner(Learner):
    """
    A SupervisedLearner is a graph-handling container for a supervised model Component.
    """
    def __init__(self, supervised_model_spec, add_time_rank=True, auto_build=True, **kwargs):
        """
        Creates a supervised learner for classification or regression.

        Args:
            supervised_model_spec (Union[dict,SupervisedModel ]): A specification dict to construct a SupervisedModel Component
                or a SupervisedModel Component directly.
        """
        self.sequence_length_in_states = kwargs.pop("sequence_length_in_state", None)
        super(SupervisedLearner, self).__init__(name=kwargs.pop("name", "supervised-learner"), **kwargs)

        labels_space = self.output_space.with_extra_ranks(add_batch_rank=True, add_time_rank=add_time_rank)
        self.input_spaces = dict(
            labels=labels_space,
            prediction_input=self.input_space.with_batch_rank()
        )
        if add_time_rank is True:
            self.input_spaces.update(dict(sequence_length=IntBox(shape=(), add_batch_rank=True)))

        self.root_component = SupervisedRootComponent(self, supervised_model_spec=supervised_model_spec)

        if auto_build is True:
            self.build()

    def predict(self, prediction_input):
        """
        Runs an input through the trained network and returns the output sampled
        from some distribution.

        Args:
            prediction_input (Union[np.ndarray, list]): Input.

        Returns:
            any: The prediction outputs.
        """
        # TODO: What if prediction_input is only a single sample (no batch). Generalize this method.
        # TODO: This will include redoing the contains() method of Spaces.
        #if not self.state_space.with_batch_rank().contains(prediction_input):
        #    batched_states = self.state_space.force_batch(prediction_input)
        #    ret = self.graph_executor.execute(("predict", prediction_input))
        #    return strip_list(ret)
        #else:
        return self.graph_executor.execute(("predict", prediction_input))

    def get_distribution_parameters(self, prediction_input):
        """
        Runs an input through the trained network and returns the pure distribution-parameter output,
        without the sampling step after that.

        Args:
            prediction_input (Union[np.ndarray, list]): Input.

        Returns:
            The distribution parameters returned by the NN.
        """
        return self.graph_executor.execute(("get_distribution_parameters", prediction_input))

    def get_loss(self, prediction_input, labels, sequence_length=None):
        return self.graph_executor.execute(("get_loss", [prediction_input, labels, sequence_length]))

    def update(self, batch=None, learning_rate=None):
        # Batch is not optional here.
        assert batch is not None
        # 0=step-op, 1=loss
        # Slice labels according to seq-lengths.
        # TODO: This is currently airtb-specific and needs to be much more generalized.
        _, loss = self.graph_executor.execute(("update", [batch["states"], batch["labels"], learning_rate], [0, 1]))
        return loss

    def __repr__(self):
        return "SupervisedLearner()"
