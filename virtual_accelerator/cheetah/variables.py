from typing import Any
from cheetah.accelerator import Screen, Segment, SuperimposedElement
import warnings
from lume.variables import Variable
from virtual_accelerator.utils.variables import (
    get_variables_from_element_name,
    get_name_to_epics_mapping,
    get_element_attr_mapping,
)


def get_variables_from_segment(
    segment: Segment,
    device_mapping: dict[str, str] = None,
    element_attr_mapping: dict[str, dict[str, dict[str, Any]]] = None,
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
    #
    all_variables = {}
    device_mapping = device_mapping or get_name_to_epics_mapping()
    element_attr_mapping = element_attr_mapping or get_element_attr_mapping()

    for element in segment.elements:
        if type(element).__name__ in ["Drift", "Marker", "Cavity"]:
            continue
        elif element.name.upper() in device_mapping:
            control_name = device_mapping[element.name.upper()]
        else:
            warnings.warn(f"Element {element.name} not found in device mapping")
            continue

        if isinstance(element, SuperimposedElement):
            element_variables = get_variables_from_element_name(
                type(element.base_element).__name__,
                control_name,
                element_attr_mapping,
            )

            # iterate through the superimposed elements and get variables for each
            for sub_element in element.superimposed_element.elements:
                if type(sub_element).__name__ in ["Drift", "Marker", "Cavity"]:
                    continue
                elif sub_element.name.upper() in device_mapping:
                    sub_control_name = device_mapping[sub_element.name.upper()]
                else:
                    warnings.warn(
                        f"Element {sub_element.name} not found in device mapping"
                    )
                    continue

                sub_element_variables = get_variables_from_element_name(
                    type(sub_element).__name__,
                    sub_control_name,
                    element_attr_mapping,
                )
                element_variables.update(sub_element_variables)

        else:
            element_variables = get_variables_from_element_name(
                type(element).__name__, control_name, element_attr_mapping
            )

        # if element type is a screen then modify the output variable
        if isinstance(element, Screen):
            element_variables[
                f"{control_name}:Image:ArrayData"
            ].shape = element.resolution

        all_variables.update(element_variables)

    return all_variables
