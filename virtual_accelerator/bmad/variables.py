from typing import Any
import warnings
from pytao import Tao
from virtual_accelerator.utils.variables import (
    get_element_attr_mapping,
    get_name_to_epics_mapping,
    get_variables_from_element_name,
)


def get_variables_from_tao(
    tao: Tao,
    device_mapping: dict[str, str] = None,
    element_attr_mapping: dict[str, dict[str, dict[str, Any]]] = None,
):
    """
    Get variables for all devices in a lattice segment.

    This function iterates over all devices in a Tao lattice and
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
    device_mapping = device_mapping or get_name_to_epics_mapping()
    element_attr_mapping = element_attr_mapping or get_element_attr_mapping()

    # get element names
    element_names = tao.lat_list("*", "ele.name")
    element_types = tao.lat_list("*", "ele.key")

    # map element types to standards
    element_types = [
        "HorizontalCorrector" if etype == "HKicker" else etype
        for etype in element_types
    ]
    element_types = [
        "VerticalCorrector" if etype == "VKicker" else etype for etype in element_types
    ]

    for element_name, element_type in zip(element_names, element_types):
        # handle slave elements
        if "#" in element_name:
            lord_info = tao.lord_control(element_name)[0]
            element_name = lord_info["name"]
            element_type = lord_info["key"]

        if element_type in ["Drift", "Marker", "Cavity", "BPM"]:
            continue
        elif element_name in device_mapping:
            control_name = device_mapping[element_name.upper()]
        else:
            warnings.warn(f"Element {element_name} not found in device mapping")
            continue

        element_variables = get_variables_from_element_name(
            element_type, control_name, element_attr_mapping
        )

        all_variables.update(element_variables)

    return all_variables
