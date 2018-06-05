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
from tensorflow.python.client import device_lib


class TensorFlowModel(Model):
    """
    A TensorFlow-specific Model.

    Uses a finalized tf.Graph and runs it inside a tf.Session object,
    which can be used to fulfil `call` requests.
    """
    def __init__(self, name="tf-model", **kwargs):
        super(TensorFlowModel, self).__init__(name=name, **kwargs)
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
        self.summaries = list()  # List of summary objects of all our components.

        # The session for the computation graph.
        self.session = None
        self.monitored_session = None

        self.graph_default_context = None
        self.available_devices = device_lib.list_local_devices()

    def call(self, sockets, inputs=None):
        fetch_list, feed_dict = self.get_execution_inputs(output_socket_names=sockets, inputs=inputs)
        ret = self.monitored_session.run(fetch_list, feed_dict=feed_dict)
        if len(fetch_list) == 1:
            return ret[0]
        else:
            return ret

    def get_variable_values(self, variables):
        """
        Fetches the given variables from the graph and returns their current values.
        The returned structure corresponds to the data type and structure of `variables`
        (e.g. if a dict with variables as values comes in, a dict with the same keys and current values as values
        is returned).

        Args:
            variables (any): Any structure that contains variables.

        Returns:
            any: Values of the given variables in the exact same structure as `variables`.
        """
        self.logger.debug('Fetching values of variables {} from graph.'.format(variables))
        return self.monitored_session.run(variables, feed_dict=dict())

    def reset_backend(self):
        self.logger.debug("Resetting TensorFlow default graph.")
        tf.reset_default_graph()

    def init_execution(self):
        """
        Creates and stores a tf server (and optionally joins it if we are a parameter-server).
        Only relevant, if we are running in distributed mode.
        """
        if self.execution_mode == "distributed":
            self.logger.info("Setting up distributed TensorFlow execution mode.")
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
                self.logger.info("Job is parameter server, joining and waiting.")
                self.server.join()
                quit()

    def setup_graph(self):
        # Generate the tf-Graph object and enter its scope as default graph.
        self.graph = tf.Graph()
        self.graph_default_context = self.graph.as_default()
        self.graph_default_context.__enter__()
        # Set the random seed graph-wide.
        if self.seed is not None:
            self.logger.info("Initializing TensorFlow graph with seed {}".format(self.seed))
            tf.set_random_seed(self.seed)

    def complete_backend_setup(self):
        # After the graph is built -> Setup saver, summaries, etc..
        hooks = []  # Will be appended to in the following functions.
        self.setup_saver(hooks)
        self.setup_scaffold()
        self.setup_summaries(hooks)

        # Finalize our graph, create and enter the session.
        self.setup_session(hooks)

    def load_model(self, path=None):
        pass

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
        self.logger.info("Stored model to path: {}".format(path))

    def export_graph_definition(self, filename):
        """
        Exports TensorFlow meta graph to file.

        Args:
            filename (str): File to save meta graph. Should end in .meta
        """
        if not filename.endswith('.meta'):
            self.logger.warn('Filename for TensorFlow meta graph should end with .meta.')
        self.saver.export_meta_graph(filename=filename)

    def setup_saver(self, hooks):
        """
        Creates the tf.train.Saver object and stores it in self.saver.

        Args:
            hooks (list): List of hooks to use for Saver and Summarizer in Session. Should be appended to.
        """
        self.saver = tf.train.Saver(
            var_list=self.variables,
            reshape=False,
            sharded=False,
            max_to_keep=self.saver_spec.get('max_checkpoints', 5),
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=True,
            write_version=tf.train.SaverDef.V2,
            pad_step_number=False,
            save_relative_paths=True,
            filename=None
        )
        # TODO check if hooks required.

    def setup_scaffold(self):
        pass

    def setup_summaries(self, hooks):
        """
        Args:
            hooks (list): List of hooks to use for Saver and Summarizer in Session. Should be appended to.
        """
        pass

    def setup_session(self, hooks):
        """
        Creates and then enters the session for this model. Also finalizes the graph.

        Args:
            hooks (list): A list of session hooks to use.
        """
        if self.execution_mode == "distributed":
            self.logger.info("Setting up distributed TensorFlow session.")
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
            self.logger.info("Setting up singular monitored session for non-distributed mode.")
            self.monitored_session = tf.train.SingularMonitoredSession(
                hooks=hooks,
                scaffold=self.scaffold,
                master='',  # Default value.
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

    def assign_device(self, graph_fn, socket, assigned_device):
        if assigned_device not in self.available_devices:
            self.logger.error("Assigned device {} for graph_fn {} not in available devices:\n {}".
                format(assigned_device, graph_fn, self.available_devices))

        with tf.device(assigned_device):
            self.logger.debug("Assigning device {} to graph_fn {} via socket  {}".format(
                assigned_device, graph_fn, socket))
            graph_fn.update_from_input(socket, self.op_record_registry, self.in_socket_registry)

            # Store assigned names for debugging.
            if assigned_device not in self.device_component_assignments:
                self.device_component_assignments[assigned_device] = [str(graph_fn)]
            else:
                self.device_component_assignments[assigned_device].append(str(graph_fn))

    def get_available_devices(self):
        return self.available_devices

    def get_device_assignments(self, device_names=None):
        if device_names is None:
            return self.device_component_assignments
        else:
            assignments = dict()
            for device in self.device_component_assignments:
                if device in device_names:
                    assignments[device] = self.device_component_assignments[device]
            return assignments
