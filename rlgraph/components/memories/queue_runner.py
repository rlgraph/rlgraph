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
    def __init__(self, queue, api_method_name, *sub_components, **kwargs):
        """
        Args:
            queue (Queue-like): The Queue (FIFOQueue), whose underlying `queue` object to use to enqueue item into.
            api_method_name (str): The name of the API method to call on all `sub_components` to get ops from
                which we will create enqueue ops for the queue.
            sub_components (Component): The sub-components of this QueueRunner.
        """
        super(QueueRunner, self).__init__(scope=kwargs.pop("scope", "queue-runner"), **kwargs)

        self.queue = queue
        self.api_method_name = api_method_name

        self.queue_runner = None

        # Add our sub-components.
        self.add_components(*sub_components)

        self.define_api_method("setup", self._graph_fn_setup)

    def _graph_fn_setup(self):
        enqueue_ops = list()

        if get_backend() == "tf":
            for sub_component in self.sub_components.values():
                enqueue_op = self.call(getattr(sub_component, self.api_method_name))
                enqueue_ops.append(enqueue_op)

            self.queue_runner = tf.train.QueueRunner(self.queue.queue, enqueue_ops)
            # Add to standard collection, so all queue-runners will be started after session creation.
            tf.train.add_queue_runner(self.queue_runner)

            return tf.no_op()
