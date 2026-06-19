from typing import Any
from pytao import Tao
import re
from lume.variables import Variable

from lume_bmad.actions import (
    ScreenSpec,
    ScreenImageVariable,
    ScreenResolutionVariable,
    ScreenImageShapeVariable,
)

import virtual_accelerator.bmad.actions as _bmad_actions

import logging

logger = logging.getLogger(__name__)

# Pre-compile regex patterns for performance
KLYSTRON_SEGMENT_PATTERN = re.compile(r"^(K\d+_\d+)[A-Z]$")
KLYSTRON_PATTERN = re.compile(r"^(K\d+_\d+)[A-Z]?$")

# Mapping of element types to canonical types for variable configuration
ELEMENT_TYPE_MAPPING = {
    "VKicker": "VerticalCorrector",
    "HKicker": "HorizontalCorrector",
}


def get_normalized_element_names(tao: Tao):
    elements = tao.lat_list("*", "ele.name")

    # Remove sentinel elements and preserve Tao's unique element IDs to avoid
    # ambiguous lookups when multiple elements share the same base name.
    return list(
        dict.fromkeys(elem for elem in elements if elem not in ("BEGINNING", "END"))
    )


def get_element_type(tao: Tao, element_name: str) -> str:
    """Get the element type for an element, applying any necessary mappings or normalizations."""
    try:
        element_type = tao.ele_head(element_name)["key"]
    except Exception as exc:
        logger.warning(
            "Error getting element type for %s: %s. Defaulting to 'Unknown'.",
            element_name,
            exc,
        )
        return "Unknown"

    # handle BPMs
    if element_type == "Monitor" and element_name.startswith("BPM"):
        element_type = "BPM"

    # Apply element type mappings
    element_type = ELEMENT_TYPE_MAPPING.get(element_type, element_type)
    return element_type


def get_variables(
    tao: Tao,
    element_attr_mapping: dict[str, dict[str, dict[str, Any]]] = None,
):
    """
    Get variables for all controllable devices.

    This function iterates over devices and
    uses the provided device mapping and variable configuration to instantiate
    LUME Variables for each device.

    Parameters
    ----------
    tao : Tao
        Tao object containing the lattice elements.
    element_attr_mapping : dict[str, dict[str, dict[str, Any]]], optional
        Device-type -> PV attribute -> variable specification mapping
        loaded from the SLAC variable YAML configuration.

    Returns
    -------
    list[Variable]
        List of instantiated LUME Variables for all controllable devices.

    Notes
    -----
    See `get_variables_from_element_name` for details on the specification of `element_attr_mapping`.

    Warnings for unknown element types are deduplicated within each invocation;
    calling this function multiple times may repeat warnings for the same types.

    """
    all_variables = []
    warned_types: set[str] = set()

    normalized_elements = get_normalized_element_names(tao)

    # iterate over the normalized element names and create variables for those that are in the device mapping
    for element_name in normalized_elements:
        element_type = get_element_type(tao, element_name)

        # check if element type is in the variable configuration mapping, if not skip it with a warning
        if element_type not in element_attr_mapping:
            if element_type not in warned_types:
                logger.warning(
                    "Element type %s not found in variable configuration mapping. Skipping all elements of this type.",
                    element_type,
                )
                warned_types.add(element_type)
            continue

        # get the element pv suffix mapping for this element type from the variable configuration
        element_pv_suffix_mapping = element_attr_mapping[element_type]

        all_variables.extend(
            create_variables_from_element(
                tao=tao,
                element_name=element_name,
                class_mapping=element_pv_suffix_mapping,
            )
        )

    return all_variables


def create_variables_from_element(
    tao: Tao,
    element_name: str,
    class_mapping: dict[str, Any],
) -> list[Variable]:
    """
    Create variables for an element

    Parameters
    ----------
    tao : Tao
        Tao object containing the lattice elements.
    element_name : str
        Name of the element to create variables for.
    class_mapping : dict[str, Any]
        Mapping of PV attribute suffix -> variable specification.
        Each value may be a variable class name string or a dict
        containing a `variable_class` field.

    Returns
    -------
    list[Variable]
        List of instantiated LUME Variables for the given element based on the provided class mapping.

    """

    variables = []

    base_pv = tao.ele(element_name).head.alias
    for attr, var_spec in class_mapping.items():
        pv_name = f"{base_pv}:{attr}"

        if isinstance(var_spec, dict):
            var_class_name = var_spec["variable_class"]
        else:
            var_class_name = var_spec

        # use the configured class name to get the class from actions.py
        var_class = getattr(_bmad_actions, var_class_name, None)
        if var_class is None:
            raise KeyError(
                f"Unknown variable class '{var_class_name}'. "
                "Ensure it is defined in virtual_accelerator.bmad.actions."
            )
        variable = var_class(name=pv_name, element_name=element_name)
        variables.append(variable)

    return variables


def get_screen_variables(
    tao: Tao,
    screen_name: str,
    config_dict: dict[str, dict[str, Any]],
):
    """
    Get variables for a screen element based on the provided configuration.

    Parameters
    ----------
    tao : Tao
        Tao object containing the lattice elements.
    screen_name : str
        Name of the screen element to get variables for.
    config_dict : dict[str, dict[str, Any]]
        Mapping of screen name -> variable configuration dict. The variable configuration dict should specify the PV attribute suffixes and corresponding variable class names to instantiate for that screen.

    Returns
    -------
    list[Variable]
        List of instantiated LUME Variables for the given screen based on the provided configuration.

    """

    base_pv = tao.ele(screen_name).head.alias
    if screen_name not in config_dict:
        raise ValueError(f"Screen {screen_name} not found in configuration dictionary.")

    screen_config = config_dict[screen_name]

    shape = screen_config["shape"]
    pixel_size = screen_config["pixel_size"]

    screen_spec = ScreenSpec(
        element_name=screen_name,
        shape=tuple(shape),
        pixel_size=float(pixel_size),
    )

    # create screen variables based on the configuration for this screen
    variables = [
        ScreenImageVariable.from_screen_spec(
            name=f"{base_pv}:Image:ArrayData",
            screen_spec=screen_spec,
        ),
        ScreenResolutionVariable.from_screen_spec(
            name=f"{base_pv}:RESOLUTION",
            screen_spec=screen_spec,
        ),
        ScreenImageShapeVariable.from_screen_spec(
            name=f"{base_pv}:Image:ArraySize0_RBV",
            screen_spec=screen_spec,
            index=0,
        ),
        ScreenImageShapeVariable.from_screen_spec(
            name=f"{base_pv}:Image:ArraySize1_RBV",
            screen_spec=screen_spec,
            index=1,
        ),
    ]

    return variables
