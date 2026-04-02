from typing import Any
from pytao import Tao
import os
import yaml
from lume.variables import NDVariable
from pathlib import Path
import numpy as np
from virtual_accelerator.utils.variables import (
    get_element_attr_mapping,
    get_name_or_overlay_to_epics_mapping,
    get_variables_from_element_name,
)


def get_normalized_element_names(tao: Tao):
    elements = tao.lat_list("*", "ele.name")

    # if "BEGINNING" or "END" elements are present remove them from the list
    if "BEGINNING" in elements:
        elements.remove("BEGINNING")
    if "END" in elements:
        elements.remove("END")

    # Strip Bmad instance suffixes (e.g. QE01#1, QE01#2 -> QE01) and deduplicate.
    normalized_elements = []
    seen_elements = set()
    for element_name in elements:
        base_element_name = element_name.split("#", 1)[0]
        if base_element_name not in seen_elements:
            seen_elements.add(base_element_name)
            normalized_elements.append(base_element_name)

    return normalized_elements


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

    # iterate over the normalized element names and create variables for those that are in the device mapping
    for element_name in normalized_elements:
        if element_name in device_mapping:
            element_type = tao.ele_head(element_name)["key"]
            control_name = device_mapping[element_name]
            if element_type == "VKicker":
                element_type = "VerticalCorrector"
            if element_type == "HKicker":
                element_type = "HorizontalCorrector"
            if element_type == "Overlay" and control_name.split(":")[0] == "KLYS":
                element_type = "Lcavity_Overlay"
            element_variables = get_variables_from_element_name(
                element_type, control_name, element_attr_mapping
            )

            all_variables.update(element_variables)

    return all_variables


def get_cu_hxr_screen_variables(tao, control_variables, screen_list):
    """
    Get screen attributes for cu_hxr from yaml file

    Parameters
    ----------
    tao : object
        The TAO instance.
    control_variables : dict
        Dictionary of control variables.
    screen_list : list
        List of screen elements to include. One or more of: ['OTRH1', 'OTRH2', 'OTR2', 'OTR3', 'OTR4',
                                                             'OTR11', 'OTR12', 'OTR21', 'OTRDMP']

    Returns
    -------
    tuple[dict[str, NDVariable], dict[str, dict[str, any]], list[str]]
        - control_variables: Updated dictionary of control variables including screen variables.
        - screen_attributes: Dictionary of screen attributes for each element.
            - number of pixels in x and y
            - resolution um/pixel
            - bit_depth
            - orientation
        - used_screens: List of screens that were found in the lattice and included in the control variables.
    """

    with open(
        os.path.join(
            Path(__file__).parent.resolve(), "../utils/cu_hxr_profmon_info.yaml"
        ),
        "r",
    ) as f:
        screen_data = yaml.safe_load(f)

    normalized_elements = get_normalized_element_names(tao)

    screen_attributes = {}
    used_screens = []
    for element in normalized_elements:
        if element not in screen_list:
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

        used_screens.append(element)

    return control_variables, screen_attributes, used_screens
