import math
import os
from collections.abc import Iterable, Sequence
from numbers import Real
from lume_bmad.model import LUMEBmadModel
from lume_cheetah import LUMECheetahModel
from pathlib import Path
from lume_cheetah import LUMECheetahModel

from virtual_accelerator.utils.variables import get_pvs_by_element_name

TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "../bmad", "test_beam")

DEFAULT_MAGNET_PV_ATTRS = (
    "BCTRL",
    "BACT",
    "BDES",
    "BMIN",
    "BMAX",
    "STATCTRLSUB.T",
    "CTRL",
)
DEFAULT_SCREEN_PV_ATTRS = (
    "Image:ArrayData",
    "Image:ArraySize1_RBV",
    "Image:ArraySize0_RBV",
    "RESOLUTION",
    "X",
    "Y",
)
DEFAULT_BPM_PV_ATTRS = ("X", "Y", "TMIT")

def _normalize_element_name(element_name: str) -> str:
    """Return an element name without any split-index suffix.

    Parameters
    ----------
    element_name : str
        Raw element name from lattice metadata (for example ``Q1#2``).

    Returns
    -------
    str
        Base element name with any ``#...`` suffix removed.
    """
    return element_name.split("#", 1)[0]


def _collect_elements_by_type(
    element_names: Sequence[str],
    element_types: Sequence[str],
    requested_type: str,
) -> set[str]:
    """Collect normalized lattice element names for a specific element type.

    Parameters
    ----------
    element_names : Sequence[str]
        Element names from a lattice/segment description.
    element_types : Sequence[str]
        Element type identifiers aligned positionally with ``element_names``.
    requested_type : str
        Element type to select.

    Returns
    -------
    set[str]
        Unique normalized element names for entries whose type matches
        ``requested_type``.

    Raises
    ------
    AssertionError
        If ``element_names`` and ``element_types`` are not the same length.
    """
    assert len(element_names) == len(element_types), (
        "Element-name and element-type sequences must have the same length."
    )

    return {
        _normalize_element_name(element_name)
        for element_name, element_type in zip(element_names, element_types)
        if element_name not in ("BEGINNING", "END") and element_type == requested_type
    }


def _assert_elements_have_pv_mapping_and_attrs(
    model,
    element_names: Iterable[str],
    required_attrs: tuple[str, ...],
    element_group_label: str,
) -> None:
    """Assert PV mapping completeness and required suffixes for elements.

    Parameters
    ----------
    model : LUMEModel
        Model exposing ``supported_variables``.
    element_names : Iterable[str]
        Element names to validate. Split-element names are normalized.
    required_attrs : tuple[str, ...]
        PV suffixes that must exist for each element.
    element_group_label : str
        Label used in assertion messages to identify the validated element group.

    Raises
    ------
    AssertionError
        If any element is missing from element-name mapping metadata or if one
        or more required PV suffixes are missing for any element.
    """
    normalized_elements = sorted(
        {
            _normalize_element_name(element_name)
            for element_name in element_names
            if element_name not in ("BEGINNING", "END")
        }
    )

    if not normalized_elements:
        return

    pvs_by_element = get_pvs_by_element_name(model)

    missing_mapping = sorted(
        element_name
        for element_name in normalized_elements
        if element_name not in pvs_by_element
    )
    assert not missing_mapping, (
        f"{element_group_label} elements missing variables with element_name mapping: "
        + ", ".join(missing_mapping)
    )

    missing_pvs = {}
    for element_name in normalized_elements:
        element_pvs = pvs_by_element[element_name]
        absent_attrs = sorted(
            attr
            for attr in required_attrs
            if not any(pv_name.endswith(f":{attr}") for pv_name in element_pvs)
        )
        if absent_attrs:
            missing_pvs[element_name] = absent_attrs

    assert not missing_pvs, (
        f"{element_group_label} PV attrs missing from model.supported_variables: "
        + "; ".join(
            f"{element}: {', '.join(attrs)}" for element, attrs in missing_pvs.items()
        )
    )


def _get_tao_lattice_element_metadata(model: LUMEBmadModel) -> tuple[list[str], list[str]]:
    """Get element names and keys from a Tao lattice.

    Parameters
    ----------
    model : LUMEBmadModel
        Model exposing a Tao interface at ``model.tao``.

    Returns
    -------
    tuple[list[str], list[str]]
        Pair of lists ``(element_names, element_keys)`` from ``tao.lat_list``.
    """
    return model.tao.lat_list("*", "ele.name"), model.tao.lat_list("*", "ele.key")


def _get_cheetah_segment_element_metadata(model: LUMECheetahModel) -> tuple[list[str], list[str]]:
    """Get element names and runtime class names from a Cheetah segment.

    Parameters
    ----------
    model : LUMECheetahModel
        Model exposing ``model.simulator.segment.elements``.

    Returns
    -------
    tuple[list[str], list[str]]
        Pair ``(element_names, element_types)`` where ``element_types`` are
        class names such as ``Quadrupole`` or ``Screen``.

    Raises
    ------
    AssertionError
        If required simulator/segment metadata is not available or an element
        has an invalid ``name`` attribute.
    """

    elements = model.simulator.segment.elements

    element_names: list[str] = []
    element_types: list[str] = []

    for element in elements:
        element_name = getattr(element, "name", None)
        assert isinstance(element_name, str) and element_name, (
            f"Cheetah segment element {element!r} does not expose a valid name."
        )
        element_names.append(element_name)
        element_types.append(type(element).__name__)

    return element_names, element_types


def assert_bmad_model_initialization(
    get_model,
    required_control_variable: str | None = None,
) -> None:
    """Assert that a BMAD model initializes and exposes writable controls.

    Parameters
    ----------
    get_model : Callable[..., LUMEBmadModel]
        Factory used to build the model under test. The helper passes
        ``custom_beam_path=TEST_BEAM_PATH``.
    required_control_variable : str | None, optional
        If provided, this PV must be writable in ``model.supported_variables``.

    Raises
    ------
    AssertionError
        If no writable controls are found or if ``required_control_variable`` is
        not writable.
    """
    model = get_model(custom_beam_path=TEST_BEAM_PATH)

    writable_control_variables = {
        name
        for name, variable in model.supported_variables.items()
        if not getattr(variable, "read_only", True)
    }

    assert len(writable_control_variables) > 0
    if required_control_variable is not None:
        assert required_control_variable in writable_control_variables


def assert_bmad_model_twiss_outputs(get_model) -> None:
    """Assert basic Twiss/readback output integrity for a BMAD model.

    Parameters
    ----------
    get_model : Callable[..., LUMEBmadModel]
        Factory used to build the model under test.

    Raises
    ------
    AssertionError
        If Twiss output lengths do not match lattice length or if expected
        endpoint names are not present.
    """
    model = get_model(custom_beam_path=TEST_BEAM_PATH)
    outputs = model.get(["a.beta", "b.beta", "name"])

    assert len(outputs["a.beta"]) == len(model.tao.lat_list("*", "ele.name"))
    assert len(outputs["b.beta"]) == len(model.tao.lat_list("*", "ele.name"))
    assert outputs["name"][0] == "BEGINNING"
    assert outputs["name"][-1] == "END"


def assert_bmad_model_track_beam_custom_path(get_model) -> None:
    """Assert that a BMAD model can initialize with ``track_beam=True``.

    Parameters
    ----------
    get_model : Callable[..., LUMEBmadModel]
        Factory used to build the model under test.

    Raises
    ------
    AssertionError
        If the returned model instance is ``None``.
    """
    # This test ensures shared track_beam setup works when custom_beam_path is given.
    model = get_model(track_beam=True, custom_beam_path=TEST_BEAM_PATH)
    assert model is not None


def assert_magnet_pvs_match_tao_lattice(
    model,
    element_key: str,
    excluded_elements: Iterable[str] = (),
    element_attrs: tuple[str, ...] = DEFAULT_MAGNET_PV_ATTRS,
) -> None:
    """Assert magnet PV coverage for a Tao-backed model.

    Parameters
    ----------
    model : LUMEBmadModel
        BMAD model exposing Tao lattice metadata.
    element_key : str
        Tao element key to validate (for example ``Quadrupole``).
    excluded_elements : Iterable[str], optional
        Element names to skip from validation.
    element_attrs : tuple[str, ...], optional
        PV suffixes that must exist for each validated element.
    """
    element_names, element_keys = _get_tao_lattice_element_metadata(model)
    assert_magnet_pvs_match_lattice_elements(
        model=model,
        element_key=element_key,
        element_names=element_names,
        element_keys=element_keys,
        excluded_elements=excluded_elements,
        element_attrs=element_attrs,
    )


def assert_magnet_pvs_match_cheetah_segment(
    model,
    element_key: str,
    excluded_elements: Iterable[str] = (),
    element_attrs: tuple[str, ...] = DEFAULT_MAGNET_PV_ATTRS,
) -> None:
    """Assert magnet PV coverage for a Cheetah-backed model.

    Parameters
    ----------
    model : LUMECheetahModel
        Cheetah model exposing segment metadata.
    element_key : str
        Cheetah element class name to validate (for example ``Quadrupole``).
    excluded_elements : Iterable[str], optional
        Element names to skip from validation.
    element_attrs : tuple[str, ...], optional
        PV suffixes that must exist for each validated element.
    """
    element_names, element_keys = _get_cheetah_segment_element_metadata(model)
    assert_magnet_pvs_match_lattice_elements(
        model=model,
        element_key=element_key,
        element_names=element_names,
        element_keys=element_keys,
        excluded_elements=excluded_elements,
        element_attrs=element_attrs,
    )


def assert_magnet_pvs_match_lattice_elements(
    model,
    element_key: str,
    element_names: Sequence[str],
    element_keys: Sequence[str],
    excluded_elements: Iterable[str] = (),
    element_attrs: tuple[str, ...] = DEFAULT_MAGNET_PV_ATTRS,
) -> None:
    """Assert magnet PV coverage from explicit lattice metadata sequences.

    Parameters
    ----------
    model : LUMEModel
        Model exposing ``supported_variables``.
    element_key : str
        Element type/class identifier to match.
    element_names : Sequence[str]
        Element names aligned positionally with ``element_keys``.
    element_keys : Sequence[str]
        Element type identifiers aligned positionally with ``element_names``.
    excluded_elements : Iterable[str], optional
        Element names to skip from validation after normalization.
    element_attrs : tuple[str, ...], optional
        PV suffixes required for each validated element.

    Raises
    ------
    AssertionError
        If metadata lengths do not match, if required element mappings are
        missing, or if required PV suffixes are missing.
    """
    matching_elements = _collect_elements_by_type(
        element_names=element_names,
        element_types=element_keys,
        requested_type=element_key,
    )
    excluded_elements_normalized = {
        _normalize_element_name(element_name) for element_name in excluded_elements
    }
    matching_elements = {
        element_name
        for element_name in matching_elements
        if element_name not in excluded_elements_normalized
    }

    _assert_elements_have_pv_mapping_and_attrs(
        model=model,
        element_names=matching_elements,
        required_attrs=element_attrs,
        element_group_label=element_key,
    )


def assert_screen_image_pvs_in_supported_variables(
    model,
    screen_elements: tuple[str, ...] | list[str] | None = None,
    screen_attrs: tuple[str, ...] = DEFAULT_SCREEN_PV_ATTRS,
) -> None:
    """
    Verify image-related PVs for screen elements are present in supported variables.

    Parameters
    ----------
    model : LUMEModel
        Model exposing ``supported_variables``.
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

    _assert_elements_have_pv_mapping_and_attrs(
        model=model,
        element_names=screen_elements,
        required_attrs=screen_attrs,
        element_group_label="Screen",
    )


def assert_screen_image_pvs_match_tao_lattice(
    model,
    screen_attrs: tuple[str, ...] = DEFAULT_SCREEN_PV_ATTRS,
) -> None:
    """Assert screen image PV coverage using Tao ``dump_locations``.

    Parameters
    ----------
    model : LUMEBmadModel
        BMAD model that provides ``dump_locations``.
    screen_attrs : tuple[str, ...], optional
        PV suffixes required for each screen element.
    """
    assert_screen_image_pvs_in_supported_variables(
        model=model,
        screen_elements=tuple(getattr(model, "dump_locations", ()) or ()),
        screen_attrs=screen_attrs,
    )


def assert_screen_image_pvs_match_cheetah_segment(
    model,
    screen_attrs: tuple[str, ...] = DEFAULT_SCREEN_PV_ATTRS,
) -> None:
    """Assert screen image PV coverage for screen elements in a Cheetah segment.

    Parameters
    ----------
    model : LUMECheetahModel
        Cheetah model exposing segment elements.
    screen_attrs : tuple[str, ...], optional
        PV suffixes required for each screen element.
    """
    element_names, element_types = _get_cheetah_segment_element_metadata(model)
    screen_elements = _collect_elements_by_type(
        element_names=element_names,
        element_types=element_types,
        requested_type="Screen",
    )
    assert_screen_image_pvs_in_supported_variables(
        model=model,
        screen_elements=tuple(sorted(screen_elements)),
        screen_attrs=screen_attrs,
    )


def assert_bpm_pvs_match_tao_lattice(
    model,
    bpm_attrs: tuple[str, ...] = DEFAULT_BPM_PV_ATTRS,
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
    element_names, _ = _get_tao_lattice_element_metadata(model)
    bpm_elements = {
        _normalize_element_name(element_name)
        for element_name in element_names
        if element_name not in ("BEGINNING", "END")
        and "BPM" in _normalize_element_name(element_name)
    }

    assert_bpm_pvs_match_elements(
        model=model,
        bpm_elements=bpm_elements,
        bpm_attrs=bpm_attrs,
    )


def assert_bpm_pvs_match_cheetah_segment(
    model,
    bpm_attrs: tuple[str, ...] = DEFAULT_BPM_PV_ATTRS,
) -> None:
    """Assert BPM PV coverage for BPM elements in a Cheetah segment.

    Parameters
    ----------
    model : LUMECheetahModel
        Cheetah model exposing segment elements.
    bpm_attrs : tuple[str, ...], optional
        PV suffixes required for each BPM element.
    """
    element_names, element_types = _get_cheetah_segment_element_metadata(model)
    bpm_elements = _collect_elements_by_type(
        element_names=element_names,
        element_types=element_types,
        requested_type="BPM",
    )

    assert_bpm_pvs_match_elements(
        model=model,
        bpm_elements=bpm_elements,
        bpm_attrs=bpm_attrs,
    )


def assert_bpm_pvs_match_elements(
    model,
    bpm_elements: Iterable[str],
    bpm_attrs: tuple[str, ...] = DEFAULT_BPM_PV_ATTRS,
) -> None:
    """Assert BPM PV coverage for an explicit BPM element name collection.

    Parameters
    ----------
    model : LUMEModel
        Model exposing ``supported_variables``.
    bpm_elements : Iterable[str]
        BPM element names to validate.
    bpm_attrs : tuple[str, ...], optional
        PV suffixes required for each BPM element.
    """

    _assert_elements_have_pv_mapping_and_attrs(
        model=model,
        element_names=bpm_elements,
        required_attrs=bpm_attrs,
        element_group_label="BPM",
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

    Notes
    -----
    - Only writable scalar-like controls are exercised.
    - ``track_type`` is excluded because it has model-level semantics that do
      not fit this generic scalar roundtrip check.
    """

    def assert_value_equal(pv_name, expected_value, actual_value) -> None:
        """Assert equality with support for numeric tolerance and array values."""
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
        if pv_name == "track_type":
            continue

        current_value = model.get(pv_name)

        if isinstance(current_value, bool):
            target_value = not current_value
        elif isinstance(current_value, Real):
            if pv_name.endswith(":PNEUMATIC"):
                target_value = 0.0 if float(current_value) >= 0.5 else 1.0
            else:
                target_value = float(current_value) + 0.001
        else:
            # Roundtrip helper currently targets scalar-like writable controls.
            continue

        model.set({pv_name: target_value})
        roundtrip_value = model.get(pv_name)
        assert_value_equal(pv_name, target_value, roundtrip_value)
