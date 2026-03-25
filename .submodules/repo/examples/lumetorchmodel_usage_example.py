"""
Example usage of LUMETorchModel with LCLS Cu Inj surrogate model.

This script demonstrates how to:
1. Load a TorchModel from a yaml configuration
2. Wrap it in LUMETorchModel for the LUME interface
3. Set inputs and get outputs using the LUME set/get pattern
"""

from lume_torch.base import LUMETorchModel
from lume_torch.models import TorchModel


def main():
    # Load the torch model from yaml configuration
    model = TorchModel("model_config.yaml")

    # Wrap it in LUMETorchModel
    ltmodel = LUMETorchModel(model)

    # Set input values
    print("\nSetting input: QUAD:IN20:121:BACT= -0.02")
    ltmodel.set({"QUAD:IN20:121:BACT": -0.02})

    # Get outputs
    outputs = ltmodel.get(ltmodel.torch_model.output_names)
    print(f"\nOutputs: {outputs}")

    # Get inputs (to verify what was set)
    inputs = ltmodel.get(ltmodel.torch_model.input_names)
    print(f"\nInputs: {inputs}")

    # Show all supported variables
    print(f"\nSupported variables: {ltmodel.supported_variables}")
    print(f"\nInput variables: {ltmodel.torch_model.input_names}")
    print(f"\nOutput variables: {ltmodel.torch_model.output_names}")

    # Test reset functionality
    ltmodel.reset()
    inputs_after_reset = ltmodel.get(ltmodel.torch_model.input_names)
    print(f"\nInputs after reset: {inputs_after_reset}")


if __name__ == "__main__":
    main()
