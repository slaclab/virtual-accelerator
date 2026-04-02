from typing import Any
from pytao import Tao
import re
import yaml
from lume.variables import NDVariable
from pathlib import Path
import numpy as np
from virtual_accelerator.utils.variables import (
    get_element_attr_mapping,
    get_name_or_overlay_to_epics_mapping,
    get_variables_from_element_name,
)

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

    # Remove sentinel elements and deduplicate by stripping Bmad instance suffixes
    elements_set = {
        elem.split("#")[0] for elem in elements if elem not in ("BEGINNING", "END")
    }

    return sorted(elements_set)


def get_variables(
    tao: Tao,
    device_mapping: dict[str, str] = None,
    element_attr_mapping: dict[str, dict[str, dict[str, Any]]] = None,
):
    """
    Get variables for all controlable devices.

    This function iterates over devices and
    uses the provided device mapping and variable configuration to instantiate
    LUME Variables for each device.

    Parameters
    ----------
    tao : Tao
        Tao object containing the lattice elements.
    device_mapping : dict[str, str], optional
        Mapping of lattice element name -> control-system PV prefix.
    element_attr_mapping : dict[str, dict[str, dict[str, Any]]], optional
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

    See `get_variables_from_element_name` for details on the specification of `element_attr_mapping`.

    """
    all_variables = {}
    device_mapping = device_mapping or get_name_or_overlay_to_epics_mapping()
    element_attr_mapping = element_attr_mapping or get_element_attr_mapping()

    normalized_elements = get_normalized_element_names(tao)

    # Track which (base_element, device_name) pairs we've already created variables for
    # This handles cavity segments (e.g., K21_3A, K21_3B, K21_3C) that map to the same device
    processed_devices = set()

    # iterate over the normalized element names and create variables for those that are in the device mapping
    for element_name in normalized_elements:
        # Try to find the element in device_mapping
        # First try the element name directly
        if element_name in device_mapping:
            device_name = device_mapping[element_name]
            base_element = element_name
        else:
            # If not found, check if this is a klystron element with segment suffix (e.g., K21_3B)
            # and try matching the base element (e.g., K21_3)
            klystron_match = KLYSTRON_SEGMENT_PATTERN.match(element_name)
            if not klystron_match:
                # Not in mapping and not a klystron segment, skip it
                continue

            base_element = klystron_match.group(1)
            if base_element not in device_mapping:
                # Element not in mapping, skip it
                continue

            device_name = device_mapping[base_element]

        # Create a unique key for this device by base element and control name
        device_key = (base_element, device_name)

        # Skip if we've already processed this device (handles multi-segment cavities)
        if device_key in processed_devices:
            continue

        processed_devices.add(device_key)

        # Only query TAO for elements we actually need
        element_type = tao.ele_head(element_name)["key"]

        # Apply element type mappings
        element_type = ELEMENT_TYPE_MAPPING.get(element_type, element_type)

        # Handle Lcavity and Lcavity_Overlay types
        if element_type == "Lcavity":
            element_type = "Lcavity_Overlay"
        elif element_type == "Overlay" and device_name.split(":")[0] == "KLYS":
            element_type = "Lcavity_Overlay"

        element_variables = get_variables_from_element_name(
            element_type, device_name, element_attr_mapping
        )

        all_variables.update(element_variables)

    return all_variables


def get_cu_hxr_screen_variables(
    tao: Tao,
    control_variables: dict[str, NDVariable],
    screen_list: list[str],
) -> tuple[dict[str, NDVariable], dict[str, dict[str, Any]], list[str]]:
    """
    Get screen attributes for cu_hxr from yaml file

    Parameters
    ----------
    tao : Tao
        The TAO instance.
    control_variables : dict[str, NDVariable]
        Dictionary of control variables.
    screen_list : list[str]
        List of screen elements to include. One or more of: ['OTRH1', 'OTRH2', 'OTR2', 'OTR3', 'OTR4',
                                                             'OTR11', 'OTR12', 'OTR21', 'OTRDMP']

    Returns
    -------
    tuple[dict[str, NDVariable], dict[str, dict[str, Any]], list[str]]
        - control_variables: Updated dictionary of control variables including screen variables.
        - screen_attributes: Dictionary of screen attributes for each element.
            - number of pixels in x and y
            - resolution um/pixel
            - bit_depth
            - orientation
        - used_screens: List of screens that were found in the lattice and included in the control variables.
    """

    config_path = Path(__file__).parent / ".." / "utils" / "cu_hxr_profmon_info.yaml"
    with open(config_path) as f:
        screen_data = yaml.safe_load(f)

    normalized_elements = get_normalized_element_names(tao)

    screen_attributes = {}
    used_screens = []
    for element in normalized_elements:
        if element not in screen_list:
            continue

        if element not in screen_data:
            continue

        elem_config = screen_data[element]
        image_name = elem_config["name"] + ":Image:ArrayData"
        nCol = elem_config["nCol"]
        nRow = elem_config["nRow"]

        control_variables[image_name] = NDVariable(
            name=image_name, unit="", read_only=True, shape=(nCol, nRow)
        )
        screen_attributes[element] = {
            "bins": np.array([nCol, nRow]),
            "resolution": elem_config["res"],
            "bit_depth": elem_config["bitdepth"],
            "orient": np.array([elem_config["orientX"], elem_config["orientY"]]),
        }

        used_screens.append(element)

    return control_variables, screen_attributes, used_screens
