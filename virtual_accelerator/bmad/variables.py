from typing import Any
from pytao import Tao
from pytao.model import ElementNotFoundError
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

SKIPPED_TYPES = ["Drift", "Marker", "Instrument", "Fixer"]


def set_overlay_aliases(tao: Tao):
    """Propagate aliases from segmented klystron elements to overlay elements.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.

    Notes
    -----
    Elements matching the klystron segment pattern (for example ``K21_1C#1``)
    are normalized to their overlay root (for example ``K21_1``), and the
    overlay alias is set to match the segment alias.
    """
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
    """Return the alias for the first lattice element matching a substring.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.
    element_name : str
        Substring used to locate a matching element name.

    Returns
    -------
    str
        Alias of the first matching element, if found.
    """
    elements = tao.lat_list("*", "ele.name")

    # find an element that contains the element_name as a substring
    for elem in elements:
        if element_name in elem:
            return tao.ele(elem).head.alias


def get_normalized_element_names(tao: Tao):
    """Return lattice element names normalized for variable generation.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.

    Returns
    -------
    list[str]
        Ordered element names with sentinels removed, klystron segments
        normalized to overlay roots, and duplicates removed.
    """
    elements = tao.lat_list("*", "ele.name")

    # Remove sentinel elements and preserve Tao's unique element IDs to avoid
    # ambiguous lookups when multiple elements share the same base name.
    elements = list(
        dict.fromkeys(elem for elem in elements if elem not in ("BEGINNING", "END"))
    )

    normalized_elements = []

    # if an element has "#<number>" suffix then it is a split element, remove the suffix -- duplicates will be removed later while preserving order
    elements = [elem.split("#")[0] for elem in elements]

    # if an element matches the klystron segment pattern similar to K21_1D#1, normalize it to the base klystron name without the segment suffix K21_1
    for elem in elements:
        match = KLYSTRON_PATTERN.match(elem)
        if match:
            normalized_elements.append(
                elem[:-1]
            )  # remove the segment suffix to get the overlay element name

        else:
            normalized_elements.append(elem)

    # remove duplicates while preserving order
    normalized_elements = list(dict.fromkeys(normalized_elements))

    return normalized_elements


def get_element_type(tao: Tao, element_name: str) -> str:
    """Get normalized device type for a lattice element.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.
    element_name : str
        Lattice element name to classify.

    Returns
    -------
    str
        Normalized element type key used for variable mapping.

    Notes
    -----
    This helper applies BPM and klystron-specific remapping and then applies
    canonical type aliases from ``ELEMENT_TYPE_MAPPING``.
    """
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

    # handle screens
    if element_type == "Monitor" and (
        element_name.startswith("OTR")
        or element_name.startswith("PR")
        or element_name.startswith("YAG")
    ):
        element_type = "Screen"

    # handle transverse deflecting cavities
    if element_type == "Lcavity" and element_name.startswith("TC"):
        element_type = "TransverseDeflectingCavity"

    # handle klystrons
    if element_type == "Overlay" and element_name.startswith("K"):
        element_type = "Klystron"

    # Apply element type mappings
    element_type = ELEMENT_TYPE_MAPPING.get(element_type, element_type)
    return element_type


def get_all_element_types(tao: Tao) -> dict[str, str]:
    """Get a mapping of all lattice element names to their normalized types.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.

    Returns
    -------
    dict[str, str]
        Mapping of element name -> normalized element type.
    """
    elements = get_normalized_element_names(tao)
    return {elem: get_element_type(tao, elem) for elem in elements}


def get_variables(
    tao: Tao,
    element_attr_mapping: dict[str, dict[str, dict[str, Any]]],
    screen_config_dict: dict[str, dict[str, Any]],
):
    """
    Build variables for supported lattice elements.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.
    element_attr_mapping : dict[str, dict[str, dict[str, Any]]], optional
        Mapping of element type -> PV suffix -> variable specification.
        If omitted, callers are expected to pass a mapping upstream.
    screen_config_dict : dict[str, dict[str, Any]], optional
        Mapping of screen element name -> screen configuration with ``shape``
        and ``pixel_size``.

    Returns
    -------
    list[Variable]
        Instantiated variables for all supported elements.

    Notes
    -----
    Elements in ``SKIPPED_TYPES`` are ignored. Unknown element types are logged
    and skipped.

    """
    all_variables = []

    normalized_elements = get_normalized_element_names(tao)

    # iterate over the normalized element names and create variables for those that are in the device mapping
    for element_name in normalized_elements:
        element_type = get_element_type(tao, element_name)

        # get alias
        if element_type == "Klystron":
            alias = get_overlay_alias(tao, element_name)
        else:
            try:
                alias = tao.ele(element_name).head.alias
            except ElementNotFoundError:
                logger.warning(
                    f"Element {element_name} not found in Tao lattice. Skipping variable generation for this element."
                )
                continue

        # skip element types that are in the SKIPPED_TYPES list
        if element_type in SKIPPED_TYPES:
            continue

        # if the element is a screen, add screen variables based on the screen configuration
        if element_type == "Screen":
            if element_name not in screen_config_dict:
                logger.warning(
                    f"Screen {element_name} found in lattice but missing from screen configuration. Skipping screen variables for this element."
                )
                continue

            screen_variables = get_screen_variables(
                tao, element_name, screen_config_dict
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
    Build screen image-related variables from screen configuration.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.
    screen_name : str
        Screen element name to build variables for.
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
