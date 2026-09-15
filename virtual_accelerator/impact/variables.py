from typing import Any

from impact import Impact
from lume.variables import Variable
from virtual_accelerator.impact import actions as impact_actions
from virtual_accelerator.impact.screen_actions import (
    ScreenSpec,
    ScreenImageVariable,
    ScreenResolutionVariable,
    ScreenImageShapeVariable,
)

import logging


logger = logging.getLogger(__name__)

SUPPORTED_ELEMENT_TYPES = {"quadrupole", "write_beam", "solrf"}


def get_normalized_element_type(impact: Impact, element_name):
    element = impact.ele[element_name]
    element_type = element["type"]
    if element_type == "quadrupole":
        element_type = "Quadrupole"
    if element_type == "write_beam":
        element_type = "Screen"

    return element_type


def get_all_element_types(impact: Impact):
    return {
        name: get_normalized_element_type(impact, name)
        for name, element in impact.ele.items()
        if element["type"] in SUPPORTED_ELEMENT_TYPES
    }


def get_variables(
    impact: Impact,
    element_attr_mapping: dict[str, dict[str, dict[str, Any]]],
    screen_config_dict: dict[str, dict[str, Any]],
    alias_dict: dict[str, str],
):
    all_variables = []

    all_element_types = get_all_element_types(impact)
    for element_name, element_type in all_element_types.items():
        # if the element is a screen, add screen variables based on the screen configuration
        if element_type == "Screen":
            if element_name not in screen_config_dict:
                logger.warning(
                    f"Screen {element_name} found in lattice but missing from screen configuration. Skipping screen variables for this element."
                )
                continue

            screen_variables = get_screen_variables(
                impact, alias_dict[element_name], element_name, screen_config_dict
            )
            all_variables.extend(screen_variables)
            continue

        # check if element type is in the variable configuration mapping, if not skip it with a warning
        if element_type not in element_attr_mapping:
            # raise warning and skip if element type is not in the variable configuration mapping
            logger.warning(
                f"Element type {element_type} for element {element_name} not found in variable configuration mapping. Skipping."
            )
            continue

        # get the element pv suffix mapping for this element type from the variable configuration
        element_pv_suffix_mapping = element_attr_mapping[element_type]

        all_variables.extend(
            create_variables_from_element(
                element_name=element_name,
                base_pv=alias_dict[element_name],
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
    Instantiate variables for one element from a PV-class mapping.

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
        Instantiated variables for the given element.

    Raises
    ------
    ValueError
        If a configured variable class name cannot be resolved.

    """

    variables = []

    for attr, var_spec in class_mapping.items():
        pv_name = f"{base_pv}:{attr}"

        if isinstance(var_spec, dict):
            var_class_name = var_spec["variable_class"]
        else:
            var_class_name = var_spec

        # Resolve variable class names from the actions module explicitly.
        var_class = getattr(impact_actions, var_class_name, None)
        if var_class is None:
            raise ValueError(
                f"Unknown Impact variable class {var_class_name!r} for {element_name}.{attr}"
            )
        variable = var_class(name=pv_name, element_name=element_name)
        variables.append(variable)

    return variables


def get_screen_variables(
    impact: Impact,
    base_pv: str,
    screen_name: str,
    config_dict: dict[str, dict[str, Any]],
):
    """
    Build screen image-related variables from screen configuration.

    Parameters
    ----------
    impact : Impact
        Active Impact instance containing the currently loaded lattice.
    screen_name : str
        Screen element name to build variables for.
    base_pv : str
        Base PV name to use for the screen variables.
    config_dict : dict[str, dict[str, Any]]
        Mapping of screen name -> configuration with ``shape`` and
        ``pixel_size``.

    Returns
    -------
    list[Variable]
        Screen variables including image array data, resolution, and array
        dimensions.

    Raises
    ------
    ValueError
        If ``screen_name`` is missing from ``config_dict``.

    """

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
    image_screen_spec = ScreenSpec(
        element_name=screen_name,
        shape=tuple(shape),
        pixel_size=float(pixel_size) * 1e-6,  # convert from microns to meters
    )
    variables = [
        ScreenImageVariable.from_screen_spec(
            name=f"{base_pv}:Image:ArrayData",
            screen_spec=image_screen_spec,
        ),
        ScreenResolutionVariable.from_screen_spec(
            name=f"{base_pv}:RESOLUTION",
            screen_spec=screen_spec,
        ),
        ScreenImageShapeVariable.from_screen_spec(
            name=f"{base_pv}:Image:ArraySize0_RBV",
            screen_spec=screen_spec,
            index=1,  # need to reverse the order of the shape for the ArraySize0_RBV and ArraySize1_RBV variables since they are in row-major order
        ),
        ScreenImageShapeVariable.from_screen_spec(
            name=f"{base_pv}:Image:ArraySize1_RBV",
            screen_spec=screen_spec,
            index=0,  # need to reverse the order of the shape for the ArraySize0_RBV and ArraySize1_RBV variables since they are in row-major order
        ),
    ]

    return variables
