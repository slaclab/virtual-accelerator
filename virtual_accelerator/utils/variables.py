import copy
import os
from pathlib import Path
import pandas as pd
from typing import Any
import warnings
import numpy as np

from lume.variables import Variable, ScalarVariable, NDVariable
from lume_torch.variables import TorchScalarVariable, TorchNDVariable
import yaml

VARIABLE_CLASS_MAP = {
    "ScalarVariable": ScalarVariable,
    "NDVariable": NDVariable,
    "TorchScalarVariable": TorchScalarVariable,
    "TorchNDVariable": TorchNDVariable,
}


def get_name_to_epics_mapping():
    """
    Get the mapping from element name to control system PV prefix for
    LCLS elements.

    Returns
    -------
    dict[str, str]
        Mapping of lattice element name -> control-system PV prefix.
    """
    fpath = os.path.join(
        Path(__file__).parent.resolve(),
        "lcls_elements.csv",
    )
    return (
        pd.read_csv(fpath, dtype=str)
        .set_index("Element")["Control System Name"]
        .dropna()
        .to_dict()
    )


def get_name_or_overlay_to_epics_mapping():
    """
    Get the mapping from element name or bmad overlay to control system
    PV prefix from file
    """
    fpath = os.path.join(
        Path(__file__).parent.resolve(),
        "cu_hxr_elements_to_device",
    )
    mapping = {}
    with Path(fpath).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            key, *rest = line.split()
            mapping[key] = rest[0] if rest else ""
    return mapping


def get_epics_to_name_or_overlay_mapping():
    return {v: k for k, v in get_name_or_overlay_to_epics_mapping().items()}


def get_epics_to_name_mapping():
    """
    Get the mapping from control system PV prefix to element name for
    LCLS elements.

    Returns
    -------
    dict[str, str]
        Mapping of control-system PV prefix -> lattice element name.
    """
    return {v: k for k, v in get_name_to_epics_mapping().items()}


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


def get_variables_from_element_name(
    element_class_name: str,
    control_name: str,
    element_attr_mapping: dict[str, dict[str, Any]],
) -> dict[str, Variable]:
    """
    Instantiate variables for a single lattice element using a SLAC variable
    configuration mapping.

    This function converts a device attribute configuration (typically loaded
    from the SLAC YAML variable config) into instantiated LUME `Variable`
    objects.

    For each configured attribute:
      - A full PV name is constructed as "{control_name}:{attr}".
      - The configured `variable_class` is instantiated.
      - Additional keyword arguments from the configuration are passed through.

    Special handling is necessary for NDVariables whose shape must be
    determined from the lattice element (e.g. screen image arrays).
    This should be handled by setting the shape of the variable
    AFTER calling this function.

    Parameters
    ----------
    element_class_name : str
        Name of the Cheetah lattice element class (e.g. "Screen", "BPM",
    "Quadrupole").

    control_name : str
        Control-system PV prefix for this device
        (e.g. "QUAD:IN20:425").

    element_attr_mapping : dict[str, dict[str, Any]]
        Nested dictionary containing a mapping of cheetah element types to PV
    attributes and their variable specifications.

    Returns
    -------
    dict[str, Variable]
        Mapping of full PV name -> instantiated LUME Variable
        (ScalarVariable or NDVariable).

    Notes
    -----
    Example structure for `element_attr_mapping`:
    {
        "Screen":
        {
            "Image:ArrayData": {
                "variable_class": NDVariable,
                "read_only": True,
                "unit": "pixel",
                "shape": (1,1)
            },
            "PNEUMATIC": {
                "variable_class": ScalarVariable,
                "read_only": False
            }
        },
        "Quadrupole":
        {
            "BCTRL": {
                "variable_class": ScalarVariable,
                "read_only": False,
                "unit": "kG-cm"
            },
            "BACT": {
                "variable_class": ScalarVariable,
                "read_only": True,
                "unit": "kG-cm"
            }
        }
    }
    """

    variables = {}

    # determine the cheetah element type as a string to look up in the config
    # mapping
    element_type = element_class_name
    element_attributes = element_attr_mapping.get(element_type)

    if element_attributes is None:
        warnings.warn(
            f"No variable configuration found for element type \
            {element_type!r}"
        )
        return variables

    # iterate over the configured attributes for this element type and
    # instantiate variables
    for attr, var_config in element_attributes.items():
        # create the variable name by combining the control name and attribute
        # (e.g. "QUAD:IN20:425:BCTRL")
        variable_name = f"{control_name}:{attr}"

        # copy the config so we do not mutate the original YAML mapping
        variable_info = copy.copy(var_config)

        # retrieve the variable class specified in the configuration
        variable_class = variable_info.pop("variable_class")

        # if the variable class was provided as a string (as it is in YAML),
        # convert it to the actual imported class
        if isinstance(variable_class, str):
            try:
                variable_class = VARIABLE_CLASS_MAP[variable_class]
            except KeyError as e:
                raise ValueError(
                    f"Unknown variable_class {variable_class!r} "
                    f"for {element_type}.{attr}"
                ) from e

        if variable_class == NDVariable:
            variable_info["shape"] = (
                1,
                1,
            )
        # default shape for NDVariables, should be updated after instantiation
        # create a variable instance using the specified variable class
        # and additional configuration parameters
        variables[variable_name] = variable_class(
            name=variable_name,
            **variable_info,
        )

    return variables


def split_control_and_observable(
    all_vars: dict[str, ScalarVariable],
) -> tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]:
    """
    Separate variables into control and observable groups.

    Parameters
    ----------
    all_vars : dict[str, ScalarVariable]
        Mapping of PV name to ScalarVariable.

    Returns
    -------
    tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]
        (control_vars, observable_vars)

    Notes
    -----
    Classification is based on the `read_only` attribute of
    each ScalarVariable:
        read_only == False → control variable
        read_only == True  → observable variable
    """
    control_vars = {k: v for k, v in all_vars.items() if not v.read_only}
    observable_vars = {k: v for k, v in all_vars.items() if v.read_only}
    return control_vars, observable_vars


def convert_to_torch_variables(
    variables: dict[str, Variable],
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
    torch_variables: dict[str, Variable] = {}

    for name, variable in variables.items():
        cls = type(variable)
        kwargs = variable.model_dump()

        torch_class_name = "Torch" + cls.__name__

        try:
            torch_class = VARIABLE_CLASS_MAP[torch_class_name]
        except KeyError as e:
            raise KeyError(
                f"No torch variable class registered for {cls.__name__!r} "
                f"(expected key {torch_class_name!r})"
            ) from e

        torch_variables[name] = torch_class(**kwargs)

    return torch_variables

