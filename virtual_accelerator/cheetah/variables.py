"""Variable factory utilities for action-based Cheetah model integration.

This module builds action variables from a Cheetah lattice segment and SLAC PV
configuration mappings by resolving per-element variable classes from
``virtual_accelerator.cheetah.actions``.
"""

from typing import Any
import warnings
from lume.variables import Variable
from virtual_accelerator.utils.variables import (
    get_element_attr_mapping,
)
from virtual_accelerator.cheetah import actions as cheetah_actions


SKIPPED_ELEMENT_TYPES = {
    "Drift",
    "Marker",
    "Cavity",
    "Undulator",
    "Dipole",
    "Aperture",
}

# Element-type aliases bridge Cheetah runtime names to SLAC config keys.
ELEMENT_TYPE_ALIASES = {
    "TransverseDeflectingCavity": "Crab_Cavity",
}

# Screen variables require an explicit mapping because the YAML configuration
# currently uses BMAD screen class names that do not apply to this action layer.
SCREEN_VARIABLE_CLASS_MAPPING = {
    "Image:ArrayData": "ScreenImageVariable",
    "PNEUMATIC": "ScreenPneumaticVariable",
    "Image:ArraySize1_RBV": "ScreenImageArraySizeVariable",
    "Image:ArraySize0_RBV": "ScreenImageArraySizeVariable",
    "RESOLUTION": "ScreenResolutionVariable",
    "IMAGE": "ScreenImageVariable",
    "X": "ScreenXVariable",
    "Y": "ScreenYVariable",
}

SCREEN_ARRAY_SIZE_INDEX_BY_SUFFIX = {
    "Image:ArraySize1_RBV": 0,
    "Image:ArraySize0_RBV": 1,
}


def _resolve_variable_class_name(var_spec: Any) -> str:
    """Extract the variable class name from config entries.

    The SLAC mapping can provide either a string class name or a dictionary
    containing ``variable_class`` metadata.
    """
    if isinstance(var_spec, dict):
        return var_spec["variable_class"]
    return var_spec


def _resolve_element_variable_mapping(
    element_type: str,
    element_attr_mapping: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve PV attribute-to-class mapping for a Cheetah element type.

    Screen mappings are handled separately to force action-layer classes.
    All other element types are resolved through alias normalization and YAML
    lookup.
    """
    if element_type == "Screen":
        return SCREEN_VARIABLE_CLASS_MAPPING

    mapping_key = ELEMENT_TYPE_ALIASES.get(element_type, element_type)
    return element_attr_mapping.get(mapping_key)


def _resolve_control_name(
    element_name: str, device_mapping: dict[str, str]
) -> str | None:
    """Resolve control-system base PV from an element name.

    Supports direct element-name matches and split-element fallback by removing
    ``#<index>`` suffixes.
    """
    normalized_name = element_name.upper()
    if normalized_name in device_mapping:
        return device_mapping[normalized_name]

    split_name = normalized_name.split("#", 1)[0]
    return device_mapping.get(split_name)


def _instantiate_element_variables(
    element_name: str,
    control_name: str,
    class_mapping: dict[str, Any],
    image_shape: tuple[int, ...] | None = None,
) -> dict[str, Variable]:
    """Instantiate mapped action variables for a single logical element.

    Parameters
    ----------
    element_name : str
        Logical element identifier used by action variables.
    control_name : str
        Base PV prefix used to build full variable names.
    class_mapping : dict[str, Any]
        Mapping of PV suffix to variable class descriptor.
    image_shape : tuple[int, ...] | None, optional
        Explicit NDVariable shape for screen image variables.

    Returns
    -------
    dict[str, Variable]
        Mapping of full PV names to instantiated action variables.
    """
    element_variables: dict[str, Variable] = {}

    for attr, var_spec in class_mapping.items():
        var_class_name = _resolve_variable_class_name(var_spec)
        var_class = getattr(cheetah_actions, var_class_name, None)
        if var_class is None:
            raise ValueError(
                f"Unknown Cheetah variable class {var_class_name!r} for {element_name}.{attr}"
            )

        variable_name = f"{control_name}:{attr}"
        init_kwargs = {
            "name": variable_name,
            "element_name": element_name,
        }

        if issubclass(var_class, cheetah_actions.CheetahReadOnlyNDVariable):
            init_kwargs["shape"] = image_shape or (1,)

        if var_class_name == "ScreenImageArraySizeVariable":
            size_index = SCREEN_ARRAY_SIZE_INDEX_BY_SUFFIX.get(attr)
            if size_index is None:
                raise ValueError(
                    f"No screen array-size index configured for suffix {attr!r}"
                )
            init_kwargs["index"] = size_index

        element_variables[variable_name] = var_class(**init_kwargs)

    return element_variables


def get_variables_from_segment(
    segment,
    device_mapping: dict[str, str],
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

    Variable classes are resolved from `virtual_accelerator.cheetah.actions`
    using the configured mapping for each element type.

    """
    all_variables = {}
    processed_control_names: set[str] = set()
    element_attr_mapping = element_attr_mapping or get_element_attr_mapping()

    for element in segment.elements:
        element_type = type(element).__name__
        if element_type in SKIPPED_ELEMENT_TYPES:
            continue

        control_name = _resolve_control_name(element.name, device_mapping)
        if control_name is None:
            warnings.warn(f"Element {element.name} not found in device mapping")
            continue

        if control_name in processed_control_names:
            # Skip duplicate split-elements that map to one control device.
            continue
        processed_control_names.add(control_name)

        class_mapping = _resolve_element_variable_mapping(
            element_type, element_attr_mapping
        )
        if class_mapping is None:
            warnings.warn(
                f"Element type {element_type} for {element.name} not found in variable mapping"
            )
            continue

        base_element_name = element.name.split("#", 1)[0]
        image_shape = None
        if element_type == "Screen":
            image_shape = tuple(element.resolution)

        element_variables = _instantiate_element_variables(
            element_name=base_element_name,
            control_name=control_name,
            class_mapping=class_mapping,
            image_shape=image_shape,
        )

        all_variables.update(element_variables)

    return all_variables
