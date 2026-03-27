from typing import Any
from pytao import Tao
from virtual_accelerator.utils.variables import (
    get_element_attr_mapping,
    get_name_or_overlay_to_epics_mapping,
    get_variables_from_element_name,
)


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

    for element_name in device_mapping.keys():
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
