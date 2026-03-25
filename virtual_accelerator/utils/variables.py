import copy
import os
from pathlib import Path
import pandas as pd
from typing import Any
import warnings
import numpy as np

from lume.variables import Variable, ScalarVariable, NDVariable
import yaml

VARIABLE_CLASS_MAP = {
    "ScalarVariable": ScalarVariable,
    "NDVariable": NDVariable,
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


def get_epics_to_name_mapping():
    """
    Get the mapping from control system PV prefix to element name for
    LCLS elements.

    Returns
    -------
    dict[str, str]
        Mapping of control-system PV prefix -> lattice element name.
    """
    return {v: k for k, v in get_name_or_overlay_to_epics_mapping().items()}


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


def get_cu_hxr_screen_variables(control_variables, element_list):
    """
    Get screen attributes for cu_hxr from yaml file

    Parameters
    ----------
    control_varialbes dictionary
    screen_list one or more of: ['OTRH1', 'OTRH2', 'OTR2', 'OTR3', 'OTR4',
                                 'OTR11', 'OTR12', 'OTR21', 'OTRDMP']

    Returns
    -------
    dict[element: attributes ]
    - number of pixels in x and y
    - resolution um/pixel
    - bit_depth
    - orientation
    """

    with open(
        os.path.join(Path(__file__).parent.resolve(), "cu_hxr_profmon_info.yaml"), "r"
    ) as f:
        screen_data = yaml.safe_load(f)

    screen_attributes = {}
    for element in screen_data.keys():
        if element not in element_list:
            continue
        image_name = screen_data[element]["name"] + ":Image:ArrayData"
        nCol = screen_data[element]["nCol"]
        nRow = screen_data[element]["nRow"]
        control_variables[image_name] = NDVariable(
            name=image_name, unit="", read_only=True, shape=(nCol, nRow)
        )
        screen_attributes[element] = {
            "bins": np.array([nCol, nRow]),
            "resolution": screen_data[element]["res"],
            "bit_depth": screen_data[element]["bitdepth"],
            "orient": np.array(
                [screen_data[element]["orientX"], screen_data[element]["orientY"]],
            ),
        }
    return control_variables, screen_attributes
