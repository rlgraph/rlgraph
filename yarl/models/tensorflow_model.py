# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.models import Model
import tensorflow as tf
import os


class TensorFlowModel(Model):

    def __init__(self, name="tf-model", saver_spec=None, summary_spec=None, execution_spec=None):
        """
        Args:
            name (str): The name of this model.
            summary_spec (dict): The specification dict for summary generation.
            saver_spec (dict): The saver specification for saving this graph to disk.
        """
        super(TensorFlowModel, self).__init__(
            name=name,
            saver_spec=saver_spec,
            summary_spec=summary_spec,
            execution_spec=execution_spec
        )
        # TODO should this be in the core component?
        self.global_training_timestep = None
        # Saver.
        self.saver = None
        self.saver_directory = None

        # tf.Scaffold.
        self.scaffold = None

        # The Server (for distributed mode).
        self.server = None  # The tf.Server object (if any).

        # Summary settings.
        self.summary_writer = None
        self.summary_configuration_op = None
        self.summaries = []  # List of summary objects of all our components.

        # The session for the computation graph.
        self.session = None
        self.monitored_session = None
        self.graph_default_context = None

    def finalize_backend(self):
        # After the graph is built -> Setup saver, summaries, etc..
        hooks = []  # Will be appended to in the following functions.
        self.setup_saver(hooks)
        self.setup_scaffold()
        self.setup_summaries(hooks)

        # Finalize our graph, create and enter the session.
        self.setup_session(hooks)

    def setup_graph(self):
        self.graph = tf.Graph()
        self.graph_default_context = self.graph.as_default()
        self.graph_default_context.__enter__()

    def call(self, sockets, input=None):
        fetch_list, feed_dict = self.get_execution_inputs(sockets=sockets, input_dict=input)
        ret = self.monitored_session.run(fetch_list, feed_dict=feed_dict)

        return ret

    def init_execution(self):
        """
        Creates and stores a tf server (and optionally joins it if we are a parameter-server).
        Only relevant, if we are running in distributed mode.
        """
        if self.execution_mode == "distributed":
            # Create the Server object.
            self.server = tf.train.Server(
                server_or_cluster_def=self.distributed_spec["cluster_spec"],
                job_name=self.distributed_spec["job"],
                task_index=self.distributed_spec["task_index"],
                protocol=self.distributed_spec.get("protocol"),
                config=self.distributed_spec.get("session_config"),
                start=True
            )
            if self.distributed_spec["job"] == "ps":
                # Just join and be done.
                self.server.join()
                quit()

    def store_model(self, path=None, add_timestep=True):
        if self.summary_writer is not None:
            self.summary_writer.flush()

        self.saver.save(
            sess=self.session,
            save_path=(path or self.saver_directory),
            # TODO: global_timestep
            global_step=(self.global_training_timestep if add_timestep is False else None),
            latest_filename=None,
            meta_graph_suffix="meta",
            write_meta_graph=True,
            write_state=True
        )

    def load_model(self, path=None):
        pass

    def setup_session(self, hooks):
        """
        Creates and then enters the session for this model. Also finalizes the graph.

        Args:
            hooks (list): A list of session hooks to use.
        """
        if self.execution_mode == "distributed":
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=self.scaffold,
                master=self.server.target,
                config=self.session_config,
                checkpoint_dir=None,
                checkpoint_filename_with_path=None
            )
            self.monitored_session = tf.train.MonitoredSession(
                session_creator=session_creator,
                hooks=hooks,
                stop_grace_period_secs=120  # Default value.
            )
        else:
            self.monitored_session = tf.train.SingularMonitoredSession(
                hooks=hooks,
                scaffold=self.scaffold,
                master='',
                config=self.session_config,
                checkpoint_dir=None
            )

        # Exit the graph-context and finalize the graph.
        if self.graph_default_context is not None:
            self.graph_default_context.__exit__(None, None, None)
        self.graph.finalize()

        # Enter the session to be ready for acting/learning.
        self.monitored_session.__enter__()
        self.session = self.monitored_session._tf_sess()

    def setup_saver(self, hooks):
        """
        Creates the tf.train.Saver object and stores it in self.saver.

        Args:
            hooks (list): List of hooks to use for Saver and Summarizer in Session. Should be appended to.
        """
        pass

    def setup_scaffold(self):
        pass

    def setup_summaries(self, hooks):
        """
        Args:
            hooks (list): List of hooks to use for Saver and Summarizer in Session. Should be appended to.
        """
        pass

    def assign_device(self, computation, socket, assigned_device):
        # TODO potentially validate device exists via fetching local devices.
        # Otherwise, what happens if device assigned not recognized?
        with tf.device(assigned_device):
            computation.update_from_input(socket, self.op_registry, self.socket_registry)