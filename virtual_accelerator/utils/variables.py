import os
from pathlib import Path
import yaml

from lume.variables import (
    Variable,
    ScalarVariable,
    NDVariable,
)


def get_pvs_by_element_name(model) -> dict[str, set[str]]:
    """Group supported PV names by normalized variable element name."""
    pvs_by_element: dict[str, set[str]] = {}

    for pv_name, variable in model.supported_variables.items():
        element_name = getattr(variable, "element_name", None)
        if not element_name:
            continue

        normalized_element_name = element_name.split("#")[0]
        pvs_by_element.setdefault(normalized_element_name, set()).add(pv_name)

    return pvs_by_element


def get_element_attr_mapping():
    """
    Get the mapping from element type to PV attributes and variable
    specifications.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Nested dictionary containing a mapping of cheetah element types to
    PV attributes and
        their variable specifications.
    """
    with open(
        os.path.join(Path(__file__).parent.resolve(), "slac_variable_config.yaml"), "r"
    ) as f:
        return yaml.safe_load(f)


def convert_to_torch_variables(
    variables: dict[str, Variable], dtype="float32"
) -> dict[str, Variable]:
    """
    Convert a dictionary of Variable instances into their corresponding Torch-based variants.

    For each Variable instance, this function:
    - Determines its concrete class (e.g., ScalarVariable, NDVariable)
    - Constructs the corresponding Torch class name by prepending "Torch"
      (e.g., ScalarVariable -> TorchScalarVariable)
    - Looks up the Torch class in `VARIABLE_CLASS_MAP`
    - Instantiates the Torch class using the same field values as the original
      Variable via Pydantic serialization

    Args:
        variables (dict[str, Variable]):
            Mapping of variable names to Variable instances (Pydantic models).

    Returns:
        dict[str, Variable]:
            Mapping of variable names to instantiated TorchVariable objects.

    Raises:
        KeyError:
            If the corresponding Torch class (e.g., "TorchScalarVariable") is not
            found in `VARIABLE_CLASS_MAP`.
        pydantic.ValidationError:
            If the Torch class cannot be instantiated with the provided data.

    Notes:
        - Assumes Torch class names follow the convention: "Torch" + <BaseClassName>.
        - Assumes field compatibility between base and Torch classes.
        - Uses `variable.dict()` for serialization (Pydantic v1). Replace with
          `model_dump()` if using Pydantic v2.
    """

    from lume_torch.variables import TorchScalarVariable, TorchNDVariable

    VARIABLE_CLASS_MAP = {
        "ScalarVariable": ScalarVariable,
        "NDVariable": NDVariable,
        "TorchScalarVariable": TorchScalarVariable,
        "TorchNDVariable": TorchNDVariable,
    }

    torch_variables: dict[str, Variable] = {}

    for name, variable in variables.items():
        cls = type(variable)
        kwargs = variable.model_dump()
        kwargs["dtype"] = dtype  # set the dtype for the Torch variable

        torch_class_name = "Torch" + cls.__name__

        # skip non Torch variables (e.g. StrVariable, EnumVariable) by checking if the corresponding Torch class exists in the mapping
        if torch_class_name not in VARIABLE_CLASS_MAP:
            continue

        torch_class = VARIABLE_CLASS_MAP[torch_class_name]
        torch_variables[name] = torch_class(**kwargs)

    return torch_variables
