import importlib.util
import os
from pathlib import Path


TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "../bmad", "test_beam")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_BMAD_DEPS = has_module("pytao") and has_module("lume_bmad")


def assert_bmad_model_initialization(
    get_model,
    required_control_variable: str | None = None,
) -> None:
    model = get_model(custom_beam_path=TEST_BEAM_PATH)

    assert len(model.control_variables) > 0
    if required_control_variable is not None:
        assert required_control_variable in model.control_variables


def assert_bmad_model_twiss_outputs(get_model) -> None:
    model = get_model(custom_beam_path=TEST_BEAM_PATH)
    outputs = model.get(["a.beta", "b.beta", "name"])

    assert len(outputs["a.beta"]) == len(model.tao.lat_list("*", "ele.name"))
    assert len(outputs["b.beta"]) == len(model.tao.lat_list("*", "ele.name"))
    assert outputs["name"][0] == "BEGINNING"
    assert outputs["name"][-1] == "END"


def assert_bmad_model_track_beam_custom_path(get_model) -> None:
    # This test ensures shared track_beam setup works when custom_beam_path is given.
    model = get_model(track_beam=True, custom_beam_path=TEST_BEAM_PATH)
    assert model is not None


def assert_element_pvs_match_tao_lattice(
    model,
    element_key: str,
    element_attrs: tuple[str, ...] = ("BCTRL", "BACT", "BDES", "BMIN", "BMAX"),
) -> None:
    """
    Verify that all lattice elements of a given type have corresponding PV mappings.

    This utility checks:
    1. All elements of the specified type have PV prefix mappings.
    2. All expected PV attributes for those elements exist in model.supported_variables.

    Parameters
    ----------
    model : LUMEModel
        The LUME BMAD model instance to check.
    element_key : str
        The element type key to match (e.g., "Quadrupole", "HKicker", "VKicker").
    element_attrs : tuple[str, ...]
        The PV attributes to check for each element (e.g., "BCTRL", "BACT", etc.).

    Raises
    ------
    AssertionError
        If any elements lack mappings or expected PVs are missing.
    """
    element_names = model.tao.lat_list("*", "ele.name")
    element_keys = model.tao.lat_list("*", "ele.key")

    # Collect all unique element names of the specified type
    matching_elements = {
        element_name.split("#")[0]
        for element_name, key in zip(element_names, element_keys)
        if element_name not in ("BEGINNING", "END") and key == element_key
    }

    if not matching_elements:
        # No elements of this type, test passes vacuously
        return

    # Build mapping from element name to PV prefix
    pv_prefix_by_element = {
        element_name: pv_prefix
        for pv_prefix, element_name in model.transformer.control_name_to_bmad.items()
    }

    # Check that all elements have PV mappings
    missing_mapping = sorted(
        element_name
        for element_name in matching_elements
        if element_name not in pv_prefix_by_element
    )
    assert not missing_mapping, (
        f"{element_key} elements missing PV prefix mapping: "
        + ", ".join(missing_mapping)
    )

    # Check that all expected PVs exist in supported_variables
    supported_variable_names = set(model.supported_variables)
    missing_pvs = {}

    for element_name in sorted(matching_elements):
        pv_prefix = pv_prefix_by_element[element_name]
        expected_pvs = {f"{pv_prefix}:{attr}" for attr in element_attrs}
        absent_pvs = sorted(expected_pvs - supported_variable_names)
        if absent_pvs:
            missing_pvs[element_name] = absent_pvs

    assert not missing_pvs, (
        f"{element_key} PVs missing from model.supported_variables: "
        + "; ".join(
            f"{element}: {', '.join(pvs)}" for element, pvs in missing_pvs.items()
        )
    )


def assert_screen_image_pvs_in_supported_variables(
    model,
    screen_elements: tuple[str, ...] | list[str] | None = None,
    screen_attrs: tuple[str, ...] = (
        "Image:ArrayData",
        "Image:ArraySize1_RBV",
        "Image:ArraySize0_RBV",
        "RESOLUTION",
    ),
) -> None:
    """
    Verify image-related PVs for screen elements are present in supported variables.

    Parameters
    ----------
    model : LUMEModel
        The LUME BMAD model instance to check.
    screen_elements : tuple[str, ...] | list[str] | None
        Optional explicit list of lattice screen element names.
        If omitted, uses ``model.dump_locations``.
    screen_attrs : tuple[str, ...]
        Screen PV attributes to check for each screen element.

    Raises
    ------
    AssertionError
        If screen elements are missing PV prefix mappings or if expected
        expected screen PVs are absent from ``model.supported_variables``.
    """
    if screen_elements is None:
        screen_elements = tuple(getattr(model, "dump_locations", ()) or ())

    if not screen_elements:
        return

    pv_prefix_by_element = {
        element_name: pv_prefix
        for pv_prefix, element_name in model.transformer.control_name_to_bmad.items()
    }

    missing_mapping = sorted(
        element_name
        for element_name in screen_elements
        if element_name not in pv_prefix_by_element
    )
    assert not missing_mapping, (
        "Screen elements missing PV prefix mapping: " + ", ".join(missing_mapping)
    )

    supported_variable_names = set(model.supported_variables)
    expected_image_pvs = {
        f"{pv_prefix_by_element[element_name]}:{attr}"
        for element_name in screen_elements
        for attr in screen_attrs
    }
    missing_image_pvs = sorted(expected_image_pvs - supported_variable_names)

    assert not missing_image_pvs, (
        "Screen image PVs missing from model.supported_variables: "
        + ", ".join(missing_image_pvs)
    )
