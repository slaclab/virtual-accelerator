from typing import Any
from pytao import Tao
import re
from lume.variables import Variable
from virtual_accelerator.bmad import actions as bmad_actions

from lume_bmad.actions import (
    ScreenSpec,
    ScreenImageVariable,
    ScreenResolutionVariable,
    ScreenImageShapeVariable,
)

import logging

logger = logging.getLogger(__name__)

# Pre-compile regex patterns for performance
KLYSTRON_SEGMENT_PATTERN = re.compile(r"^(K\d+_\d+)[A-Z]$")
KLYSTRON_PATTERN = re.compile(r"^K\d{2}_\d[A-Z]#?$")

# Mapping of element types to canonical types for variable configuration
ELEMENT_TYPE_MAPPING = {
    "VKicker": "VerticalCorrector",
    "HKicker": "HorizontalCorrector",
}


def set_overlay_aliases(tao: Tao):
    elements = tao.lat_list("*", "ele.name")

    elements = list(
        dict.fromkeys(elem for elem in elements if elem not in ("BEGINNING", "END"))
    )

    # if an element matches the klystron segment pattern similar to K21_1D#1, normalize it to the base klystron name without the segment suffix K21_1
    for elem in elements:
        match = KLYSTRON_PATTERN.match(
            elem.split("#")[0]
        )  # ignore any alias suffixes for matching
        if match:
            overlay_element = elem[:-3]

            # set the klystron overlay element alias to match the element alias
            alias = tao.ele(elem).head.alias
            logging.debug(
                f"Setting alias for klystron overlay element {overlay_element} to {alias}"
            )
            tao.ele(overlay_element).head.alias = alias


def get_overlay_alias(tao: Tao, element_name: str) -> str:
    elements = tao.lat_list("*", "ele.name")

    # find an element that contains the element_name as a substring
    for elem in elements:
        if element_name in elem:
            return tao.ele(elem).head.alias


def get_normalized_element_names(tao: Tao):
    elements = tao.lat_list("*", "ele.name")

    # Remove sentinel elements and preserve Tao's unique element IDs to avoid
    # ambiguous lookups when multiple elements share the same base name.
    elements = list(
        dict.fromkeys(elem for elem in elements if elem not in ("BEGINNING", "END"))
    )

    # if an element matches the klystron segment pattern similar to K21_1D#1, normalize it to the base klystron name without the segment suffix K21_1
    normalized_elements = []
    for elem in elements:
        match = KLYSTRON_PATTERN.match(
            elem.split("#")[0]
        )  # ignore any alias suffixes for matching
        if match:
            normalized_elements.append(elem[:-3])

        else:
            normalized_elements.append(elem)

    # remove duplicates while preserving order
    normalized_elements = list(dict.fromkeys(normalized_elements))

    return normalized_elements


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

    # handle klystrons
    if element_type == "Overlay" and element_name.startswith("K"):
        element_type = "Klystron"

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
        List of instantiated LUME Variables

    Notes
    -----
    See `get_variables_from_element_name` for details on the specification of `element_attr_mapping`.

    """
    all_variables = []

    normalized_elements = get_normalized_element_names(tao)

    # iterate over the normalized element names and create variables for those that are in the device mapping
    for element_name in normalized_elements:
        element_type = get_element_type(tao, element_name)

        # check if element type is in the variable configuration mapping, if not skip it with a warning
        if element_type not in element_attr_mapping:
            # raise warning and skip if element type is not in the variable configuration mapping
            logger.warning(
                f"Element type {element_type} for element {element_name} not found in variable configuration mapping. Skipping."
            )
            continue

        # get alias
        if element_type == "Klystron":
            alias = get_overlay_alias(tao, element_name)
        else:
            alias = tao.ele(element_name).head.alias

        # get the element pv suffix mapping for this element type from the variable configuration
        element_pv_suffix_mapping = element_attr_mapping[element_type]

        all_variables.extend(
            create_variables_from_element(
                element_name=element_name,
                base_pv=alias,
                class_mapping=element_pv_suffix_mapping,
            )
        )

    return all_variables


def create_variables_from_element(
    element_name: str,
    base_pv: str,
    class_mapping: dict[str, Any],
) -> list[Variable]:
    """
    Create variables for an element

    Parameters
    ----------
    element_name : str
        Name of the element to create variables for.
    base_pv : str
        Base PV name to use for the variables.
    class_mapping : dict[str, Any]
        Mapping of PV attribute suffix -> variable specification.

    Returns
    -------
    list[Variable]
        List of instantiated LUME Variables for the given element based on the provided class mapping.

    """

    variables = []

    for attr, var_spec in class_mapping.items():
        pv_name = f"{base_pv}:{attr}"

        if isinstance(var_spec, dict):
            var_class_name = var_spec["variable_class"]
        else:
            var_class_name = var_spec

        # Resolve variable class names from the actions module explicitly.
        var_class = getattr(bmad_actions, var_class_name, None)
        if var_class is None:
            raise ValueError(
                f"Unknown Bmad variable class {var_class_name!r} for {element_name}.{attr}"
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
