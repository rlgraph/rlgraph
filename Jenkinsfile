pipeline {
    agent {
        docker {
            image 'rlgraph/rlgraph:py3-full-torch-jq'
            args '-v ${WORKSPACE}:/rlgraph -u 0'
        }
    }
    stages {
        stage('setup') {
            steps {
                sh 'python --version'
            }
        }
        stage('learning-tests') {
            steps {
                // DDQN: CartPole
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_learning/short_tasks/test_dqn_agent_short_task_learning.py::TestDQNAgentShortTaskLearning::test_double_dueling_dqn_on_cart_pole';
                // IMPALA: Cartpole
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_learning/short_tasks/test_impala_agent_short_task_learning.py::TestIMPALAAgentShortTaskLearning::test_impala_on_cart_pole';
                // PPO: Cartpole.
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_learning/short_tasks/test_ppo_agent_short_task_learning.py::TestPPOShortTaskLearning::test_ppo_on_cart_pole';
                // SAC: GaussianDensityAsRewardEnv.
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_learning/short_tasks/test_sac_agent_short_task_learning.py::TestSACShortTaskLearning::test_sac_learning_on_gaussian_density_as_reward_env';

            }
        }
        stage('agent-functionality') {
            steps {
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_functionality/test_all_compile.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_functionality/test_base_agent_functionality.py';
                //sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_functionality/test_impala_agent_functionality.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_functionality/test_sac_agent_functionality.py';

            }
        }
        stage('execution-functionality') {
            steps {
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/execution/test_gpu_strategies.py::TestGpuStrategies::test_multi_gpu_dqn_agent_compilation';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/execution/test_gpu_strategies.py::TestGpuStrategies::test_multi_gpu_dqn_agent_learning_test_gridworld_2x2';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/execution/test_gpu_strategies.py::TestGpuStrategies::test_multi_gpu_ppo_agent_learning_test_gridworld_2x2';
            }
        }
        stage('core-lib') {
            steps {
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_api_methods.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_device_placements.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_graph_fns.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_input_incomplete_build.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_single_components.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_spaces.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_specifiable_server.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/core/test_specifiables.py';
            }
        }
        stage('components') {
            steps {
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_action_adapters.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_actor_components.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_batch_splitter.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_batch_apply.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_component_copy.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_decay_components.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_dict_preprocessor_stack.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_distributions.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_dqn_loss_functions.py';
                //sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_environment_stepper.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_epsilon_exploration.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_explorations.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_fifo_queue.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_local_optimizers.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_neural_networks.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_nn_layers.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_noise_components.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_policies.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_policies_on_container_actions.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_preprocess_layers.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_preprocessor_stacks.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_prioritized_replay.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_replay_memory.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_reshape_preprocessor.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_ring_buffer.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_sac_loss_function.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_sampler_component.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_sequence_preprocessor.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_slice.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_softmax.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_splitter_merger_components.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_stack.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_staging_area.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_string_layers.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_synchronizable.py';
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_v_trace_function.py';
                //sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components';
            }
        }
        stage('environments') {
            steps {
                sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/environments';

            }
        }
        stage('pytorch') {
            steps {
              sh "jq '.BACKEND = \"pytorch\"' ~/.rlgraph/rlgraph.json > tmp.json && mv tmp.json ~/.rlgraph/rlgraph.json";
              sh "jq '.DISTRIBUTED_BACKEND = \"ray\"' ~/.rlgraph/rlgraph.json > tmp.json && mv tmp.json ~/.rlgraph/rlgraph.json";
              // Test All compilations.
              sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/agent_functionality/test_all_compile.py';
              sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_dqn_loss_functions.py';
              sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_replay_memory.py';
              sh 'cd /rlgraph && python -m pytest -s rlgraph/tests/components/test_ring_buffer.py';
            }
        }
    }
}