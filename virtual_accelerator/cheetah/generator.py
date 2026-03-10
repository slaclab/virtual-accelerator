from __future__ import annotations
import importlib
from typing import Any
import os
import yaml
from pathlib import Path
from cheetah.accelerator import Segment, Element
import warnings
from lume.variables import Variable, ScalarVariable, NDVariable
from copy import copy


SLAC_VARIABLE_CONFIG_FILE = os.path.join(
    Path(__file__).parent.resolve(),
    "slac_variable_config.yaml",
)

VARIABLE_CLASS_MAP = {
    "ScalarVariable": ScalarVariable,
    "NDVariable": NDVariable,
}


def get_variables_from_element(
    element: Element, control_name: str, element_attr_mapping: dict[str, dict[str, Any]]
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

    Special handling is applied for NDVariables whose shape must be determined
    from the lattice element (e.g. screen image arrays).

    Parameters
    ----------
    element : Element
        Cheetah lattice element instance (e.g. Screen, BPM, Quadrupole).

    control_name : str
        Control-system PV prefix for this device
        (e.g. "QUAD:IN20:425").

    element_attr_mapping : dict[str, dict[str, Any]]
        Nested dictionary containing a mapping of cheetah element types to PV attributes and
        their variable specifications.

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
                "shape": None
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

    # determine the cheetah element type as a string to look up in the config mapping
    element_type = type(element).__name__
    element_attributes = element_attr_mapping.get(element_type)

    if element_attributes is None:
        warnings.warn(
            f"No variable configuration found for element type {element_type!r}"
        )
        return variables

    # iterate over the configured attributes for this element type and instantiate variables
    for attr, var_config in element_attributes.items():

        # create the variable name by combining the control name and attribute
        # (e.g. "QUAD:IN20:425:BCTRL")
        variable_name = f"{control_name}:{attr}"

        # copy the config so we do not mutate the original YAML mapping
        variable_info = copy(var_config)

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

        # if this is an NDVariable and the attribute is "Image:ArrayData",
        # we need to determine the shape from the lattice element's resolution
        if variable_class is NDVariable and attr == "Image:ArrayData":
            variable_info["shape"] = tuple(element.resolution)

        # create a variable instance using the specified variable class
        # and additional configuration parameters
        variables[variable_name] = variable_class(
            name=variable_name,
            **variable_info,
        )

    return variables


def get_variables_from_segment(
    segment: Segment,
    device_mapping: dict[str, str],
    element_attr_mapping: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Variable]:
    """
    Get variables for all devices in a lattice segment.

    This function iterates over all devices in a Cheetah lattice segment and
    uses the provided device mapping and variable configuration to instantiate
    LUME Variables for each device.

    Parameters
    ----------
    segment : Segment
        Cheetah accelerator segment containing the lattice elements.

    device_mapping : dict[str, str]
        Mapping of lattice element name -> control-system PV prefix.

    element_attr_mapping : dict[str, dict[str, dict[str, Any]]]
        Device-type -> PV attribute -> variable specification mapping
        loaded from the SLAC variable YAML configuration.

    Returns
    -------
    dict[str, Variable]
        Mapping of full PV name -> instantiated LUME Variable
        (ScalarVariable or NDVariable).

    Notes
    -----
    An example `device_mapping` might look like:
    device_mapping = {"QE03": "QUAD:IN20:511", "KLYS01": "KLYS:IN20:1001"}

    See `get_variables_from_element` for details on the specification of `element_attr_mapping`.

    """
    # 
    all_variables = {}

    for element in segment.elements:
        if (type(element).__name__ == 'Drift'): 
            continue
        elif element.name.upper() in device_mapping: 
            control_name = device_mapping[element.name.upper()]
        else:
            warnings.warn(f"Element {element.name} not found in device mapping")
            continue

        element_variables = get_variables_from_element(
            element, control_name, element_attr_mapping
        )
        all_variables.update(element_variables)

    return all_variables


def generate_variables(
    segment: Segment,
    device_mapping: dict[str,str],
    variable_config_file: str | None = None,
) -> dict[str, ScalarVariable]:
    """
    Generate LUME variables for a SLAC-style control system from a Cheetah lattice.

    This is the primary high-level entry point for building variables used by the
    virtual accelerator. The function performs the following steps:

      1. Resolves the variable configuration (either directly provided or
         loaded from a YAML configuration file).
      2. Determines which devices from the SLAC elements table exist in the
         provided lattice.
      3. Instantiates the appropriate Variable objects for each device.
      4. Returns both the variables and a reverse mapping to lattice elements.

    Parameters
    ----------
    lattice : Segment
        Cheetah accelerator segment containing the lattice elements.

    device_mapping: dict
        Mapping from Madname to Control Name

    variable_config_file : str, optional
        Path to a YAML variable configuration file

    Returns
    -------
    dict[str, Variable]

        all_vars
            Mapping of full PV name -> instantiated Variable
            (ScalarVariable or NDVariable).
    """
    
    if variable_config_file is None:
        variable_config_file = str(SLAC_VARIABLE_CONFIG_FILE)

    with open(variable_config_file, "r") as f:
        element_attr_mapping = yaml.safe_load(f)

    all_vars = get_variables_from_segment(
        segment, device_mapping, element_attr_mapping
    )

    return all_vars


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
