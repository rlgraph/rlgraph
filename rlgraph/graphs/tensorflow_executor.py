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

import os
import time

from rlgraph import get_backend, get_distributed_backend
import rlgraph.utils as util
from rlgraph.components.common.multi_gpu_synchronizer import MultiGpuSynchronizer
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.graphs.graph_executor import GraphExecutor

if get_backend() == "tf":
    import tensorflow as tf
    from tensorflow.python.client import device_lib
    from rlgraph.utils.specifiable_server import SpecifiableServer, SpecifiableServerHook
    from tensorflow.python.client import timeline


class TensorFlowExecutor(GraphExecutor):
    """
    A TensorFlow executioner manages execution via TensorFlow sessions.

    The following execution strategies are available:

    - 'default': Assign to CPU or provided default device if no assignment given,
        otherwise consider user device assignments.
    - 'custom': Completely user defined device strategy, graph executor just executes calls
    - 'multi_gpu_sync': Parallelizes updates across multiple GPUs by averaging gradients.
    """
    def __init__(self, **kwargs):
        super(TensorFlowExecutor, self).__init__(**kwargs)
        self.global_training_timestep = None
        self.session_config = self.execution_spec["session_config"]

        # The tf.Graph object to be run in a tf session.
        self.graph = None
        # Saver.
        self.saver = None
        self.saver_directory = None

        # tf.Scaffold.
        self.scaffold = None
        # Ops used by the scaffold to initialize variables and check variables for initialization.
        self.init_op = None
        self.local_init_op = None
        self.ready_op = None
        self.ready_for_local_init_op = None

        # # The tf.Server object (if any).
        self.server = None

        # Summary settings.
        self.summary_writer = None

        # The merged summary op to be used by the session to write the summaries.
        self.summary_op = None

        # The session for the computation graph.
        self.session = None
        self.monitored_session = None

        # The optimizer is a somewhat privileged graph component because it must manage
        # devices depending on the device strategy and we hence keep an instance here to be able
        # to request special device init ops.
        self.optimizer = None

        self.graph_default_context = None
        self.local_device_protos = device_lib.list_local_devices()

        # Just fetch CPUs. GPUs will be added when parsing the GPU configuration.
        self.available_devices = [x.name for x in self.local_device_protos if x.device_type == 'CPU']

        # Local session config which needs to be updated with device options during setup.
        self.tf_session_type = self.session_config.pop("type", "monitored-training-session")
        self.tf_session_config = tf.ConfigProto(**self.session_config)
        self.tf_session_options = None

        self.run_metadata = None

        # Tf Profiler config.
        self.profiling_enabled = self.execution_spec["enable_profiler"]
        if self.profiling_enabled is True:
            self.profiler = None
            self.profile_step = 0
            self.profiling_frequency = self.execution_spec["profiler_frequency"]
            self.run_metadata = tf.RunMetadata()
            if not self.disable_monitoring:
                self.tf_session_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        self.timeline_enabled = self.execution_spec["enable_timeline"]
        if self.timeline_enabled is True:
            if self.run_metadata is None:
                self.run_metadata = tf.RunMetadata()
            self.timeline_step = 0
            self.timeline_frequency = self.execution_spec["timeline_frequency"]
            if not self.disable_monitoring:
                self.tf_session_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        self.init_device_strategy()

        # # Initialize distributed backend.
        # distributed_backend_ = self.execution_spec.get("distributed_backend", "distributed_tf")
        #
        # self.logger.info("Updating global distributed backend setting with backend {}".format(distributed_backend_))
        # set_distributed_backend(distributed_backend_)

    def init_device_strategy(self):
        """
        Initializes default device and loads available devices.
        """
        self.device_strategy = self.execution_spec["device_strategy"]
        # Configures available GPUs.
        self.init_gpus()

        if self.device_strategy == "default":
            if self.execution_spec["device_map"] is not None:
                self.logger.warning(
                    "`device_map` given for device-strategy=`default`. Map will be ignored. Use "
                    "device-strategy=`custom` together with a `device_map`."
                )
            self.logger.info("Initializing graph executor with default device strategy. "
                             "Backend will assign all visible devices.")
            self.logger.info("GPUs enabled: {}. Usable GPUs: {}".format(self.gpus_enabled, self.gpu_names))
        elif self.device_strategy == 'multi_gpu_sync':
            assert self.gpus_enabled, "ERROR: device_strategy is 'multi_gpu_sync' but GPUs are not enabled. Please" \
                                      "check your gpu_spec and set gpus_enabled to True."
            self.default_device = self.execution_spec.get(
                "default_device", [x.name for x in self.local_device_protos if x.device_type == 'CPU'][0]
            )
            self.logger.info("Initializing graph executor with synchronized multi-gpu device strategy. "
                             "Default device: {}. Available gpus are: {}.".format(self.default_device, self.gpu_names))
        elif self.device_strategy == "custom":
            # Default device is user provided device or first CPU.
            default_device = self.execution_spec.get("default_device", None)
            if default_device is None:
                self.default_device = [x.name for x in self.local_device_protos if x.device_type == 'CPU'][0]
            else:
                self.default_device = default_device
                # Sanity check, whether given default device exists.
                # if self.default_device not in self.available_devices:
                #    raise RLGraphError("Provided `default_device` ('{}') is not in `available_devices` ({})".
                #                       format(self.default_device, self.available_devices))
            self.device_map = dict()
            # Clean up device map so it only contains devices that are actually available (otherwise,
            # use the default device).
            for component_name, device in self.execution_spec["device_map"].items():
                if device in self.available_devices:
                    self.device_map[component_name] = device
            self.logger.info("Initializing graph executor with custom device strategy (default device: {}).".
                             format(self.default_device))
        else:
            raise RLGraphError("Invalid device_strategy ('{}') for TensorFlowExecutor!".format(self.device_strategy))

    def build(self, root_components, input_spaces, optimizer=None, build_options=None, batch_size=32):
        # Use perf_counter for short tasks.
        start = time.perf_counter()
        # 0. Init phase: Component construction and nesting (child/parent Components).
        # Components can still be modified and re-arranged after this.
        self.init_execution()
        self.setup_graph()

        # 1. Build phase: Meta graph construction -> All of the root_component's API methods are being called once,
        # thereby calling other API-methods (of sub-Components). These API-method calls then build the meta-graph
        # (generating empty op-record columns around API methods and graph_fns).
        # TODO make compatible for multiple roots in graph builder.
        meta_build_times = []
        build_times = []

        for component in root_components:
            self._build_device_strategy(component, optimizer, batch_size=batch_size)
            start = time.perf_counter()
            meta_graph = self.meta_graph_builder.build(component, input_spaces)
            meta_build_times.append(time.perf_counter() - start)

            # 2. Build phase: Backend compilation, build actual TensorFlow graph from meta graph.
            # -> Inputs/Operations/variables
            build_time = self.graph_builder.build_graph_with_options(
                meta_graph=meta_graph, input_spaces=input_spaces, available_devices=self.available_devices,
                device_strategy=self.device_strategy, default_device=self.default_device, device_map=self.device_map,
                build_options=build_options
            )

            # Build time is a dict containing the cost of different parts of the build.
            build_times.append(build_time)

            # Check device assignments for inconsistencies or unused devices.
            self._sanity_check_devices()

            # Set up any remaining session or monitoring configurations.
            self.finish_graph_setup()

        return dict(
            total_build_time=time.perf_counter() - start,
            meta_graph_build_times=meta_build_times,
            build_times=build_times,
        )

    def execute(self, *api_method_calls):
        # Fetch inputs for the different API-methods.
        fetch_dict, feed_dict = self.graph_builder.get_execution_inputs(*api_method_calls)
        ret = self.monitored_session.run(
            fetch_dict, feed_dict=feed_dict, options=self.tf_session_options, run_metadata=self.run_metadata
        )

        if self.profiling_enabled:
            self.update_profiler_if_necessary()

        if self.timeline_enabled:
            self.update_timeline_if_necessary()

        # Return single values instead of lists of 1 item, but keep inner dicts as-are.
        ret = {key: (value[0] if len(ret[key]) == 1 and not isinstance(ret[key], dict) else tuple(value)
               if not isinstance(value, dict) else value) for key, value in ret.items()}

        # If only one key in ret, remove it.
        if len(api_method_calls) == 1:
            ret = ret[next(iter(ret))]

        return ret

    def update_profiler_if_necessary(self):
        """
        Updates profiler according to specification.
        """
        if self.profile_step % self.profiling_frequency == 0:
            self.profiler.add_step(self.profile_step, self.run_metadata)
            self.profiler.profile_operations(
                options=tf.profiler.ProfileOptionBuilder(
                    options=tf.profiler.ProfileOptionBuilder.time_and_memory()).with_node_names().build()
            )
        self.profile_step += 1

    def update_timeline_if_necessary(self):
        """
        Writes a timeline json file according to specification.
        """
        if self.timeline_step % self.timeline_frequency == 0:
            fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open("timeline_{:02d}.json".format(self.timeline_step), "w") as f:
                f.write(chrome_trace)
        self.timeline_step += 1

    def read_variable_values(self, variables):
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

    def init_execution(self):
        """
        Creates and stores a tf server (and optionally joins it if we are a parameter-server).
        Only relevant, if we are running in distributed mode.
        """
        if self.execution_mode == "distributed":
            if get_distributed_backend() == "distributed_tf":
                self.setup_distributed_tf()
            elif get_distributed_backend() == "horovod":
                self.setup_horovod_execution()

    def setup_distributed_tf(self):
        """
        Sets up distributed TensorFlow.
        """
        self.logger.info("Setting up distributed TensorFlow execution mode.")
        # Create a local server.
        if self.distributed_spec["cluster_spec"] is None:
            self.server = tf.train.Server.create_local_server()
        # Create an actual distributed Server.
        else:
            self.server = tf.train.Server(
                server_or_cluster_def=self.distributed_spec["cluster_spec"],
                job_name=self.distributed_spec["job"],
                task_index=self.distributed_spec["task_index"],
                protocol=self.distributed_spec["protocol"],
                start=True
            )

            if self.distributed_spec["job"] == "ps":
                # Just join and be done.
                self.logger.info("Job is parameter server, joining and waiting.")
                self.server.join()
                quit()

    def setup_horovod_execution(self):
        """
        Sets up Horovod.
        """
        # Check again to avoid import if unset which will crash if horovod is not installed.
        if get_distributed_backend() == "horovod":
            import horovod.tensorflow as hvd
            self.logger.info("Setting up Horovod execution.")
            hvd.init()
            config = tf.ConfigProto()
            config.gpu_options.visible_device_list = str(hvd.local_rank())
        
    def get_available_devices(self):
        return self.available_devices

    def get_device_assignments(self, device_names=None):
        if device_names is None:
            return self.graph_builder.device_component_assignments
        else:
            assignments = dict()
            for device in self.graph_builder.device_component_assignments:
                if device in device_names:
                    assignments[device] = self.graph_builder.device_component_assignments[device]
            return assignments

    def setup_graph(self):
        # Generate the tf-Graph object and enter its scope as default graph.
        self.graph = tf.Graph()
        self.graph_default_context = self.graph.as_default()
        self.graph_default_context.__enter__()
        # Set the random seed graph-wide.
        if self.seed is not None:
            self.logger.info("Initializing TensorFlow graph with seed {}".format(self.seed))
            tf.set_random_seed(self.seed)

    def finish_graph_setup(self):
        # After the graph is built -> Setup saver, summaries, etc..
        hooks = []  # Will be appended to in the following functions.
        self.setup_saver(hooks)
        self.setup_summaries(hooks)
        self.setup_scaffold()
        self.setup_specifiable_servers(hooks)

        # Finalize our graph, create and enter the session.
        self.setup_session(hooks)

        # NOT NECESSARY: SEEMS TO BE DONE AUTOMATICALLY BY SESSION
        # Start Queue Runners (if any).
        #self.start_queue_runners()

    def setup_saver(self, hooks):
        """
        Creates the tf.train.Saver object and stores it in self.saver.

        Args:
            hooks (list): List of hooks to use for Saver and Summarizer in Session. Should be appended to.
        """
        self.saver = tf.train.Saver(
           var_list=list(self.graph_builder.root_component.variables.values()),
           reshape=False,
           sharded=False,
           max_to_keep=self.saver_spec.get("max_checkpoints", 1) if self.saver_spec else None,
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

        # Add saver hook to session if saver spec was provided.
        if self.saver_spec is not None and (self.execution_mode == "single"
                                           or self.distributed_spec["task_index"] == 0):
           self.saver_directory = self.saver_spec["directory"]
           saver_hook = tf.train.CheckpointSaverHook(
               checkpoint_dir=self.saver_directory,
               # Either save_secs or save_steps must be set.
               save_secs=self.saver_spec["save_secs"],  # TODO: open question: how to handle settings?
               save_steps=self.saver_spec["save_steps"],
               saver=self.saver,
               checkpoint_basename=self.saver_spec["checkpoint_basename"],  # TODO: open question: how to handle settings?
               scaffold=None,  # None since not created yet.
               listeners=None
           )
           hooks.append(saver_hook)

    def setup_summaries(self, hooks):
        """
        Sets up tf.summary ops generated during the build of the graph inside the different Components.

        Args:
            hooks (list): List of hooks to use for Saver and Summarizer in Session. Should be appended to.
        """
        # Create our tf summary writer object.
        self.summary_writer = tf.summary.FileWriter(
            logdir=self.summary_spec["directory"],
            graph=self.graph,
            max_queue=10,
            flush_secs=120,
            filename_suffix=None
        )

        # Creates a single summary op to be used by the session to write the summary files.
        summary_list = list(self.graph_builder.root_component.summaries.values())
        if len(summary_list) > 0:
            self.summary_op = tf.summary.merge(inputs=summary_list)
            # Create an update saver hook for our summaries.
            summary_saver_hook = tf.train.SummarySaverHook(
                save_steps=self.summary_spec["save_steps"],  # Either one or the other has to be set.
                save_secs=self.summary_spec["save_secs"],
                output_dir=None,  # None since given via 'summary_writer' argument.
                summary_writer=self.summary_writer,
                scaffold=None,  # None since summary_op given directly here.
                summary_op=self.summary_op
            )
            # ... and append it to our list of hooks to use in the session.
            hooks.append(summary_saver_hook)

    def setup_scaffold(self):
        """
        Creates a tf.train.Scaffold object to be used by the session to initialize variables and to save models
        and summaries.
        Assigns the scaffold object to `self.scaffold`.
        """
        # Determine init_op and ready_op.
        var_list = list(self.graph_builder.root_component.variables.values())
        # We can not fetch optimizer vars.
        # self.logger.info("optimizer vars before init :")
        # self.logger.info(self.optimizer.optimizer.variables())
        # TODO let graph builder do this
        if self.optimizer is not None:
            var_list.extend(self.optimizer.get_optimizer_variables())

        if self.execution_mode == "single":
            self.init_op = tf.variables_initializer(var_list=var_list)
            self.ready_op = tf.report_uninitialized_variables(var_list=var_list)
        else:
            assert self.execution_mode == "distributed",\
                "ERROR: execution_mode can only be 'single' or 'distributed'! Is '{}'.".format(self.execution_mode)
            local_job_and_task = "/job:{}/task:{}/".format(self.execution_spec["distributed_spec"]["job"],
                                                          self.execution_spec["distributed_spec"]["task_index"])
            var_list_local = [var for var in var_list if not var.device or local_job_and_task in var.device]
            var_list_remote = [var for var in var_list if var.device and local_job_and_task not in var.device]
            self.init_op = tf.variables_initializer(var_list=var_list_remote)
            self.ready_for_local_init_op = tf.report_uninitialized_variables(var_list=var_list_remote)
            self.local_init_op = tf.variables_initializer(var_list=var_list_local)
            self.ready_op = tf.report_uninitialized_variables(var_list=var_list)

        def init_fn(scaffold, session):
            # NOTE: `self.load_from_file` is either True or a string value.
            # - No specific file given -> Use latest checkpoint.
            saver_dir = self.saver_spec.get("directory", "") if self.saver_spec else ""
            if self.load_from_file is True:
                assert self.saver_spec is not None,\
                    "ERROR: load_from_file is True but no saver_spec with 'directory' provided"
                file = tf.train.latest_checkpoint(
                    checkpoint_dir=saver_dir,
                    latest_filename=None
                )
            # - File given -> Look for it in cwd, then in our checkpoint directory.
            else:
                assert isinstance(self.load_from_file, str)
                file = self.load_from_file
                if not os.path.isfile(file):
                    file = os.path.join(saver_dir, self.load_from_file)

            if file is not None:
                scaffold.saver.restore(sess=session, save_path=file)

        # Create the tf.train.Scaffold object. Monitoring cannot be disabled for this.
        if not self.disable_monitoring:
            self.scaffold = tf.train.Scaffold(
                init_op=self.init_op,
                init_feed_dict=None,
                init_fn=init_fn if self.load_from_file else None,
                ready_op=self.ready_op,
                ready_for_local_init_op=self.ready_for_local_init_op,
                local_init_op=self.local_init_op,
                summary_op=self.summary_op,
                saver=self.saver,
                copy_from_scaffold=None
            )

    @staticmethod
    def setup_specifiable_servers(hooks):
        # Add the hook only if there have been SpecifiableServer objects created.
        # TODO: Change this registry to a tf collections based one. Problem: EnvStepper is created before the Graph,
        # TODO: So when the Graph gets entered, the registry (with the SpecifiableServer in it) is gone.
        if len(SpecifiableServer.INSTANCES) > 0:
            hooks.append(SpecifiableServerHook())

    def setup_session(self, hooks):
        """
        Creates and then enters the session for this model. Also finalizes the graph.

        Args:
            hooks (list): A list of session hooks to use.
        """
        if self.execution_mode == "distributed":
            self.logger.info("Setting up distributed TensorFlow session.")
            if self.server is None:
                raise RLGraphError(
                    "TensorflowGraphExecutor's Server is None! It could be that your DISTRIBUTED_BACKEND (currently "
                    "set to '{}') is not set to 'distributed_tf'. You can do so via the RLGraph config file in your "
                    "home directory or the ENV variable 'RLGRAPH_DISTRIBUTED_BACKEND=distributed_tf'.".
                    format(get_distributed_backend())
                )
            if self.tf_session_type == "monitored-session":
                session_creator = tf.train.ChiefSessionCreator(
                    scaffold=self.scaffold,
                    master=self.server.target,
                    config=self.tf_session_config,
                    checkpoint_dir=None,
                    checkpoint_filename_with_path=None
                )
                self.monitored_session = tf.train.MonitoredSession(
                    #is_chief=self.execution_spec["distributed_spec"]["task_index"] == 0,
                    session_creator=session_creator,
                    hooks=hooks,
                    stop_grace_period_secs=120  # Default value.
                )
            else:
                assert self.tf_session_type == "monitored-training-session",\
                    "ERROR: Invalid session type: {}!".format(self.tf_session_type)
                is_chief = self.execution_spec["distributed_spec"].get(
                    "is_chief", self.execution_spec["distributed_spec"]["task_index"] == 0
                )
                self.monitored_session = tf.train.MonitoredTrainingSession(
                    master=self.server.target,
                    is_chief=is_chief,
                    checkpoint_dir=None,  # TODO: specify?
                    save_checkpoint_secs=600,
                    save_summaries_secs=30,
                    log_step_count_steps=50000,
                    # scaffold=self.scaffold,
                    # Ignore other hooks
                    hooks=[hooks[-1]] if hooks else None,
                    config=self.tf_session_config,
                    stop_grace_period_secs=120  # Default value.
                )
        else:
            self.global_training_timestep = tf.get_variable(
                name="global-timestep", dtype=util.dtype("int"), trainable=False, initializer=0,
                collections=["global-timestep", tf.GraphKeys.GLOBAL_STEP])

            # If monitoring is disabled,
            if self.disable_monitoring:
                self.logger.info("Setting up default session for non-distributed mode.")
                self.monitored_session = tf.Session(config=self.tf_session_config)
            else:
                self.logger.info("Setting up singular monitored session for non-distributed mode.")
                self.monitored_session = tf.train.SingularMonitoredSession(
                    hooks=hooks,
                    scaffold=self.scaffold,
                    master='',  # Default value.
                    config=self.tf_session_config,
                    checkpoint_dir=None
                )

        # Exit the graph-context and finalize the graph.
        if self.graph_default_context is not None:
            self.graph_default_context.__exit__(None, None, None)

        # TODO back in
        # self.graph.finalize()

        if self.disable_monitoring:
            # If no monitoring, both just end up being simple sessions.
            self.session = self.monitored_session
            self.session.run(self.init_op)
        else:
            # Enter the session to be ready for acting/learning.
            self.monitored_session.__enter__()
            self.session = self.monitored_session._tf_sess()

        # Setup the tf Profiler.
        if self.profiling_enabled and not self.disable_monitoring:
            self.profiler = tf.profiler.Profiler(graph=self.session.graph)

    def load_model(self, path=None):
        self.logger.info("Attempting to restore model from path: {}.".format(path))
        self.saver.restore(self.session, path)

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

    def terminate(self):
        """
        Terminates the GraphExecutor, so it will no longer be usable.
        Things that need to be cleaned up should be placed into this function, e.g. closing sessions
        and other open connections.
        """
        # Close the tf.Session.
        self.monitored_session.close()

    def _build_device_strategy(self, root_component, root_optimizer, batch_size):
        """
        When using multiple GPUs or other special devices, additional graph components
        may be required to split up incoming data, load it to device memories, and aggregate
        results.

        In RLGraph, graph building and execution are separated so different device strategies can be
        plugged into single agent definitions. For example, one agent may use a single cpu or GPU,
        a local multi-gpu strategy and combine this with distributed sample collection via distributed
        TensorFlow or Ray.

        This method expands the meta graph according to the given device strategy if necessary.

        Args:
            root_component (Component): The root Component (will be used to create towers via `Component.copy()`).
            root_optimizer (Optimizer): The Optimizer object of the root Component.
            batch_size (int): The batch size that needs to be split between the different GPUs.
        """
        self.optimizer = root_optimizer

        if self.device_strategy == "multi_gpu_sync":
            assert self.num_gpus > 1, "ERROR: MultiGpuSync strategy needs more than one GPU available but" \
                                      "there are only {} GPUs visible.".format(self.num_gpus)
            self.logger.info("Building MultiGpuSync strategy with {} GPUs.".format(self.num_gpus))

            sub_graphs = []
            for i, device in enumerate(self.gpu_names):
                # Copy and assign GPU to copy.
                self.logger.info("Creating device sub-graph for device: {}.".format(device))
                # Only place the ops of the tower on the GPU (variables are shared with root).
                sub_graph = root_component.copy(device=device, scope="tower-{}".format(i))
                sub_graph.is_multi_gpu_tower = True

                sub_graphs.append(sub_graph)
                self.used_devices.append(device)

            # Setup and add MultiGpuSynchronizer to root.
            multi_gpu_optimizer = MultiGpuSynchronizer(batch_size=batch_size)
            root_component.add_components(multi_gpu_optimizer)
            multi_gpu_optimizer.setup_towers(sub_graphs, self.gpu_names)

    def _sanity_check_devices(self):
        """
        Checks device assignments to identify unused or conflicting assignments.
        """
        assignments = self.graph_builder.device_component_assignments
        # Devices can be used here or within graph build assignments.
        used_devices = list(assignments.keys()) + self.used_devices

        # Warn if some devices have not been assigned.
        self.logger.info("Checking if all visible devices are in use for strategy: {}. Available devices are: {}.".
                         format(self.device_strategy, self.available_devices))
        for device in self.available_devices:
            if device not in used_devices:
                self.logger.warning("Warning: Device {} is usable but has not been assigned.".format(
                    device
                ))

    def init_gpus(self):
        """
        Parses GPU specs and initializes GPU devices by adjusting visible CUDA devices to
        environment and setting memory allocation options.
        """
        gpu_spec = self.execution_spec.get("gpu_spec", None)

        if gpu_spec is not None:
            self.gpus_enabled = gpu_spec.get("gpus_enabled", False)
            self.max_usable_gpus = gpu_spec.get("max_usable_gpus", 0)

            if self.gpus_enabled:
                assert self.max_usable_gpus > 0, "ERROR: GPUs are enabled but max_usable_gpus are not >0 but {}".\
                    format(self.max_usable_gpus)
                gpu_names = sorted([x.name for x in self.local_device_protos if x.device_type == 'GPU'])
                cuda_visible_devices = gpu_spec.get("enable_cuda_devices", None)
                if len(gpu_names) < self.max_usable_gpus:
                    self.logger.warn("WARNING: max_usable_gpus is {} but only {} gpus are locally visible,"
                                     "using all available GPUs.".format(self.max_usable_gpus, len(gpu_names)))

                # Indicate specific CUDA devices to be used.
                if cuda_visible_devices is not None:
                    if isinstance(str, cuda_visible_devices):
                        # Assume "0, 3".
                        device_list = cuda_visible_devices.split(",")
                        num_provided_cuda_devices = len(device_list)
                        use_names = [gpu_names[int(device_id)] for device_id in device_list]
                    elif isinstance(cuda_visible_devices, list):
                        num_provided_cuda_devices = len(cuda_visible_devices)
                        use_names = [gpu_names[int(device_id)] for device_id in cuda_visible_devices]
                        cuda_visible_devices = ",".join(cuda_visible_devices)
                    else:
                        raise ValueError("ERROR: 'cuda_devices' must be string or list of device index "
                                         "values, e.g. [0, 1] or '0,1', but is: {}".format(type(cuda_visible_devices)))

                    # Must match number of allowed GPUs.
                    assert self.max_usable_gpus == num_provided_cuda_devices,\
                        "ERROR: Provided CUDA {} devices: {}, but max_usable_gpus is {}. Must match!"

                    # Expose these devices.
                    self.logger.info("GPU strategy: exposing CUDA devices with ids: {}".format(cuda_visible_devices))
                    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                    self.gpu_names = use_names

                else:
                    # Assign as many as specified.
                    visible_devices = []
                    use_names = []
                    for i, name in enumerate(gpu_names):
                        if len(use_names) < self.max_usable_gpus:
                            use_names.append(name)
                            visible_devices.append(str(i))
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
                    self.logger.info("GPU strategy initialized with GPUs enabled: {}".format(use_names))
                    self.gpu_names = use_names

                self.num_gpus = len(self.gpu_names)
                self.available_devices.extend(self.gpu_names)
                per_process_gpu_memory_fraction = gpu_spec.get("per_process_gpu_memory_fraction", None)
                if per_process_gpu_memory_fraction is not None:
                    self.tf_session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

                self.tf_session_config.gpu_options.allow_growth = gpu_spec.get("allow_memory_growth", False)
        else:
            # Do not allow any GPUs to be used.
            self.gpus_enabled = False
            self.logger.info("gpu_spec is None, disabling GPUs.")
