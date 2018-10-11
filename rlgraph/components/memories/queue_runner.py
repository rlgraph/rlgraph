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
from rlgraph.components.component import Component
from rlgraph.utils.ops import flatten_op
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class QueueRunner(Component):
    """
    A queue runner that contains n sub-components, of which an API-method is called. The return values are bundled
    into a FIFOQueue as inputs. Queue runner uses multi-threading and is started after session creation.

    API:
    enqueue() -> Returns a noop, but creates the enqueue ops for enqueuing data into the queue and hands these
        to the underlying queue-runner object.
    """
    def __init__(self, queue, api_method_name, return_slot,
                 # TODO: move these into data_producing_components-wrapper components
                 env_output_splitter,
                 fifo_input_merger,
                 next_states_slicer,
                 internal_states_slicer,
                 *data_producing_components, **kwargs):
        """
        Args:
            queue (Queue-like): The Queue (FIFOQueue), whose underlying `queue` object to use to enqueue item into.
            api_method_name (str): The name of the API method to call on all `sub_components` to get ops from
                which we will create enqueue ops for the queue.
            return_slot (int): The slot of the returned values to use as to-be-inserted record into the queue.
                Set to -1 if only one value is expected.
            #input_merger (Component): The record input-merger to use for merging things into a dict-record
            #    before inserting it into the queue.
            data_producing_components (Component): The components of this QueueRunner that produce the data to
                be enqueued.
        """
        super(QueueRunner, self).__init__(scope=kwargs.pop("scope", "queue-runner"), **kwargs)

        self.queue = queue
        self.api_method_name = api_method_name
        self.return_slot = return_slot

        self.env_output_splitter = env_output_splitter
        self.fifo_input_merger = fifo_input_merger
        self.next_states_slicer = next_states_slicer
        self.internal_states_slicer = internal_states_slicer

        # The actual backend-dependent queue object.
        self.queue_runner = None

        self.data_producing_components = data_producing_components

        # Add our sub-components (not the queue!).
        self.add_components(self.env_output_splitter, self.fifo_input_merger, self.next_states_slicer,
                            self.internal_states_slicer, *self.data_producing_components)

    @rlgraph_api
    def _graph_fn_setup(self):
        enqueue_ops = list()

        if get_backend() == "tf":
            for data_producing_component in self.data_producing_components:
                record = getattr(data_producing_component, self.api_method_name)()
                if self.return_slot != -1:
                    # Only care about one slot of the return values.
                    record = record[self.return_slot]
                # Create dict record from tuple return.
                #record = self.input_merger.merge(*record)

                # TODO: specific for IMPALA problem: needs to be generalized.
                preprocessed_s, actions, rewards, returns, terminals, next_states, action_log_probs, \
                    internal_states = self.env_output_splitter.split(record)

                last_next_state = self.next_states_slicer.slice(next_states, -1)
                initial_internal_states = self.internal_states_slicer.slice(internal_states, 0)
                #current_internal_states = self.internal_states_slicer.slice(internal_states, -1)

                record = self.fifo_input_merger.merge(
                    preprocessed_s, actions, rewards, terminals, last_next_state, action_log_probs,
                    initial_internal_states
                )

                # Insert results into the FIFOQueue.
                #insert_op = fifo_queue.insert_records(record)
                #return step_op, insert_op, current_internal_states, returns, terminals

                # Create enqueue_op from api_return.
                # TODO: This is kind of cheating, as we are producing an op from a component that's not ours.
                enqueue_op = self.queue.queue.enqueue(flatten_op(record))
                enqueue_ops.append(enqueue_op)

            self.queue_runner = tf.train.QueueRunner(self.queue.queue, enqueue_ops)
            # Add to standard collection, so all queue-runners will be started after session creation.
            tf.train.add_queue_runner(self.queue_runner)

            return tf.no_op()
