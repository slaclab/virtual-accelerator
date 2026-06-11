import importlib.util
import math
import os
from numbers import Real
from pathlib import Path
from lume.variables import ScalarVariable

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


def assert_magnet_pvs_match_tao_lattice(
    model,
    element_key: str,
    element_attrs: tuple[str, ...] = (
        "BCTRL",
        "BACT",
        "BDES",
        "BMIN",
        "BMAX",
        "STATCTRLSUB.T",
        "CTRL",
    ),
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


def assert_bpm_pvs_match_tao_lattice(
    model,
    bpm_attrs: tuple[str, ...] = ("X", "Y"),
) -> None:
    """
    Verify that mapped BPM elements expose expected BPM PVs.

    Parameters
    ----------
    model : LUMEModel
        The LUME BMAD model instance to check.
    bpm_attrs : tuple[str, ...]
        BPM PV attributes expected for each BPM device.

    Raises
    ------
    AssertionError
        If BPM lattice elements are missing PV prefix mappings, or if expected
        BPM PVs are missing from ``model.supported_variables``.
    """
    element_names = model.tao.lat_list("*", "ele.name")
    bpm_elements = {
        element_name.split("#")[0]
        for element_name in element_names
        if element_name not in ("BEGINNING", "END")
        and "BPM" in element_name.split("#")[0]
    }

    if not bpm_elements:
        return

    pv_prefix_by_element = {
        element_name
        for _, element_name in model.transformer.control_name_to_bmad.items()
    }

    missing_mapping = sorted(
        element_name
        for element_name in bpm_elements
        if element_name not in pv_prefix_by_element
    )
    assert not missing_mapping, "BPM elements missing PV prefix mapping: " + ", ".join(
        missing_mapping
    )

    pv_prefix_by_element = {
        element_name: pv_prefix
        for pv_prefix, element_name in model.transformer.control_name_to_bmad.items()
    }

    supported_variable_names = set(model.supported_variables)
    missing_bpm_pvs = {}

    for element_name in sorted(bpm_elements):
        pv_prefix = pv_prefix_by_element[element_name]
        expected_pvs = {f"{pv_prefix}:{attr}" for attr in bpm_attrs}
        absent_pvs = sorted(expected_pvs - supported_variable_names)
        if absent_pvs:
            missing_bpm_pvs[element_name] = absent_pvs

    assert not missing_bpm_pvs, (
        "BPM PVs missing from model.supported_variables: "
        + "; ".join(
            f"{element}: {', '.join(pvs)}" for element, pvs in missing_bpm_pvs.items()
        )
    )


def assert_roundtrip_pv_get_set(
    model,
) -> None:
    """
    Verify that all writable PVs roundtrip through set/get unchanged.

    Parameters
    ----------
    model : LUMEModel
        The model instance under test.
    """

    def assert_value_equal(pv_name, expected_value, actual_value) -> None:
        if (
            isinstance(expected_value, Real)
            and isinstance(actual_value, Real)
            and not isinstance(expected_value, bool)
            and not isinstance(actual_value, bool)
        ):
            assert math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-9,
            ), (
                f"Expected {pv_name!r} readback to be {expected_value!r}, "
                f"got {actual_value!r}"
            )
            return

        equality_result = actual_value == expected_value
        if hasattr(equality_result, "all"):
            assert bool(equality_result.all()), (
                f"Expected {pv_name!r} readback to be {expected_value!r}, "
                f"got {actual_value!r}"
            )
            return

        assert actual_value == expected_value, (
            f"Expected {pv_name!r} readback to be {expected_value!r}, "
            f"got {actual_value!r}"
        )

    supported_variables = model.supported_variables
    assert supported_variables, "model.supported_variables is empty"

    writable_supported_variables = [
        name for name, val in supported_variables.items() if val.read_only is False
    ]
    if not writable_supported_variables:
        writable_supported_variables = [
            name
            for name, variable in supported_variables.items()
            if not getattr(variable, "read_only", True)
        ]

    assert writable_supported_variables, (
        "No writable variables found in model.supported_variables"
    )

    for pv_name in writable_supported_variables:
        if pv_name is not "track_type":
            if isinstance(supported_variables[pv_name], ScalarVariable):
                original_value = model.get(pv_name) + 0.001  # add small offset to ensure set does something
                assert original_value != 0, f"Original value for {pv_name} is zero, cannot perform roundtrip test with offset"
                model.set({pv_name: original_value})
                roundtrip_value = model.get(pv_name)
        assert_value_equal(pv_name, original_value, roundtrip_value)
