# Copyright 2018/2019 RLcore.ai.
# ==============================================================================

from rlgraph.components.component import Component
from rlgraph.components.policies.policy import Policy
from rlgraph.utils.decorators import rlgraph_api


class SupervisedPredictor(Component):
    """
    A SupervisedPredictor is a wrapper Component that contains a Policy and makes it appear like a supervised
    predictor with a generic (not limited to actions) output space.
    """
    def __init__(self, network_spec, output_space=None, distribution_adapter_spec=None,
                 deterministic=False, scope="supervised-predictor",
                 **kwargs):
        super(SupervisedPredictor, self).__init__(scope=scope, **kwargs)

        self.deterministic = deterministic

        # We are going to incorporate a Policy so we can use all its functionality w/o exposing
        # its non matching API (RL vs SL) to the user.
        if isinstance(distribution_adapter_spec, dict) and "output_space" in distribution_adapter_spec:
            distribution_adapter_spec["action_space"] = distribution_adapter_spec["output_space"]
            del distribution_adapter_spec["output_space"]

        self.rlgraph_policy = Policy(network_spec, output_space, distribution_adapter_spec)

        self.add_components(self.rlgraph_policy)

    @rlgraph_api
    def predict(self, nn_inputs, deterministic=None):
        """
        Args:
            nn_inputs (any): The input(s) to our neural network.

            deterministic (Optional[bool]): Whether to draw the prediction sample deterministically
                (max likelihood) from the parameterized distribution or not.

        Returns:
            dict:
                - prediction: The final sample from the Distribution (including possibly the last internal states
                    of a RNN-based NN).
                - nn_outputs: The raw NeuralNetwork output (before distribution-parameters and sampling).
        """
        deterministic = self.deterministic if deterministic is None else deterministic
        out = self.rlgraph_policy.get_action(nn_inputs, deterministic)
        return dict(predictions=out["action"], parameters=out["parameters"], nn_outputs=out["nn_outputs"],
                    adapter_outputs=out["adapter_outputs"])

    @rlgraph_api
    def get_distribution_parameters(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The input(s) to our neural network.

        Returns:
            dict:
                - parameters: The raw (parameters) output of the DistributionAdapter layer.
                - nn_outputs: The raw NeuralNetwork output (before distribution-parameters and sampling).
        """
        out = self.rlgraph_policy.get_adapter_outputs_and_parameters(nn_inputs)
        # Add last internal states to return value.
        return dict(parameters=out["parameters"], nn_outputs=out["nn_outputs"],
                    adapter_outputs=out["adapter_outputs"])
