import numpy as np
from pathlib import Path

import pytest
import yaml

from lume.exceptions import ReadOnlyError
from virtual_accelerator.tests.dependency_profiles import (
    HAS_BMAD_DEPS,
    HAS_CHEETAH_DEPS,
    HAS_IMPACT_DEPS,
    HAS_LCLS_LATTICE,
)
from virtual_accelerator.models.cu_hxr import (
    IMPACT_GROUP_PV_MAPPING,
    get_cu_inj_impact_model,
    get_cu_hxr_bmad_model,
    get_cu_hxr_cheetah_model,
)
from virtual_accelerator.tests._bmad_model_test_utils import (
    assert_bpm_pvs_match_cheetah_segment,
    assert_screen_image_pvs_match_cheetah_segment,
    TEST_BEAM_PATH,
    assert_bpm_pvs_match_tao_lattice,
    assert_bmad_model_initialization,
    assert_bmad_model_twiss_outputs,
    assert_bmad_model_track_beam_custom_path,
    assert_magnet_pvs_match_lattice_elements,
    assert_magnet_pvs_match_cheetah_segment,
    assert_magnet_pvs_match_tao_lattice,
    assert_roundtrip_pv_get_set,
    assert_screen_image_pvs_in_supported_variables,
    assert_screen_image_pvs_match_tao_lattice,
)

CU_HXR_PROFMON_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "cu_hxr_profmon_info.yaml"
)


def _load_cu_hxr_screen_config():
    with CU_HXR_PROFMON_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _has_impact_executable() -> bool:
    if not HAS_IMPACT_DEPS:
        return False

    from impact import tools

    try:
        tools.find_executable(exename="ImpactTexe", envname="IMPACTT_BIN")
    except Exception:
        return False

    return True


HAS_IMPACT_EXECUTABLE = _has_impact_executable()
IMPACT_SKIP_REASON = (
    "requires impact optional dependencies, LCLS_LATTICE, "
    "and ImpactTexe executable (IMPACTT_BIN)"
)

SCREEN_PV_ATTRS = (
    "Image:ArrayData",
    "Image:ArraySize1_RBV",
    "Image:ArraySize0_RBV",
    "RESOLUTION",
)


def _get_impact_lattice_element_metadata(model):
    from virtual_accelerator.impact.variables import get_normalized_element_type

    impact = model.impact_model.simulator
    element_names = []
    element_types = []

    for element_name in impact.ele:
        element_names.append(element_name)
        element_types.append(get_normalized_element_type(impact, element_name))

    return element_names, element_types


@pytest.mark.requires_bmad
@pytest.mark.requires_lcls_lattice
@pytest.mark.skipif(
    not HAS_BMAD_DEPS or not HAS_LCLS_LATTICE,
    reason="requires bmad optional dependencies and LCLS_LATTICE",
)
class TestCUHXRBmad:
    def test_initialization(self):
        assert_bmad_model_initialization(
            lambda **kwargs: get_cu_hxr_bmad_model(end_element="OTR4", **kwargs),
            required_control_variable="QUAD:IN20:631:BCTRL",
        )

        assert_bmad_model_track_beam_custom_path(
            lambda **kwargs: get_cu_hxr_bmad_model(end_element="OTR4", **kwargs)
        )

        model = get_cu_hxr_bmad_model(
            end_element="OTR4", track_beam=True, custom_beam_path=TEST_BEAM_PATH
        )
        assert_screen_image_pvs_match_tao_lattice(model, screen_attrs=SCREEN_PV_ATTRS)

        # test getting all of the supported variables to ensure no errors with screen variable setup
        _ = model.get(list(model.supported_variables))

        # test beam up to end of TD11
        model = get_cu_hxr_bmad_model(
            end_element="TD11", track_beam=True, custom_beam_path=TEST_BEAM_PATH
        )
        _ = model.get(list(model.supported_variables))

    def test_bact_readback_is_not_writable(self):
        model = get_cu_hxr_bmad_model(
            end_element="OTR4", track_beam=True, custom_beam_path=TEST_BEAM_PATH
        )

        with pytest.raises(ReadOnlyError, match="is read-only"):
            model.set({"QUAD:IN20:631:BACT": 0.0})

    def test_cu_hxr_twiss(self):
        assert_bmad_model_twiss_outputs(get_cu_hxr_bmad_model)

    def test_sub_lattice(self):
        model = get_cu_hxr_bmad_model("QE04#1", "OTR2")
        assert len(model.supported_variables) < 50

        # test getting partial lattice with beam tracking
        model = get_cu_hxr_bmad_model(
            end_element="OTR4", track_beam=True, custom_beam_path=TEST_BEAM_PATH
        )

    def test_cu_hxr_screen(self):
        model = get_cu_hxr_bmad_model(
            end_element="OTR4", track_beam=True, custom_beam_path=TEST_BEAM_PATH
        )

        # get OTR2 image
        image = model.get("OTRS:IN20:571:Image:ArrayData")
        assert image.shape == (1040, 1392)

        # get initial OTR4 image
        image = model.get("OTRS:IN20:711:Image:ArrayData")
        assert image.shape == (1040, 1392)

        # set some control variables
        model.set({"QUAD:IN20:631:BCTRL": 0.0})

        # get updated OTR4 image
        updated_image = model.get("OTRS:IN20:711:Image:ArrayData")
        assert updated_image.shape == (1040, 1392)

        # make sure it changed
        assert not (image == updated_image).all()

    def test_cu_hxr_screen_resolution_matches_yaml_and_expected_range(self):
        screen_config = _load_cu_hxr_screen_config()

        otr4_config = screen_config["OTR4"]
        resolution_pv = f"{otr4_config['name']}:RESOLUTION"
        expected_resolution = float(otr4_config["pixel_size"])

        model = get_cu_hxr_bmad_model(
            end_element="OTR4", track_beam=True, custom_beam_path=TEST_BEAM_PATH
        )
        resolution = float(model.get(resolution_pv))

        assert np.isclose(resolution, expected_resolution)
        assert 10.0 < resolution < 20.0

    def test_cu_hxr_lcavity(self):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)

        enld = model.get("KLYS:LI21:31:ENLD")
        model.set({"KLYS:LI21:31:ENLD": enld + 5.0})
        ampl = model.get("KLYS:LI21:31:ENLD")
        assert np.isclose(ampl, enld + 5.0)

    @pytest.mark.parametrize("element_type", ["Quadrupole", "HKicker", "VKicker"])
    def test_magnet_pvs_match_tao_lattice(self, element_type):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)
        assert_magnet_pvs_match_tao_lattice(model, element_type)

    def test_bpm_pvs_match_tao_lattice(self):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)
        assert_bpm_pvs_match_tao_lattice(model)

    def test_roundtrip_pv_get_set(self):
        model = get_cu_hxr_bmad_model(
            custom_beam_path=TEST_BEAM_PATH, end_element="OTR4"
        )
        assert_roundtrip_pv_get_set(model)

    def test_screen_pvs_match_tao_lattice(self):
        model = get_cu_hxr_bmad_model(
            custom_beam_path=TEST_BEAM_PATH, end_element="OTR4", track_beam=True
        )
        assert_screen_image_pvs_match_tao_lattice(model, screen_attrs=SCREEN_PV_ATTRS)


class TestCUHXRCheetah:
    pytestmark = [
        pytest.mark.requires_cheetah,
        pytest.mark.requires_lcls_lattice,
        pytest.mark.skipif(
            not HAS_CHEETAH_DEPS or not HAS_LCLS_LATTICE,
            reason="requires cheetah optional dependencies and LCLS_LATTICE",
        ),
    ]

    def test_initialization(self):
        model = get_cu_hxr_cheetah_model()
        writable_control_variables = {
            name
            for name, variable in model.supported_variables.items()
            if not getattr(variable, "read_only", True)
        }

        assert len(model.supported_variables) > 0
        assert len(writable_control_variables) > 0

        # Smoke test that reading all variables works after initialization.
        _ = model.get(list(model.supported_variables))

    def test_bact_readback_is_not_writable(self):
        model = get_cu_hxr_cheetah_model()

        bact_pv = next(
            name for name in model.supported_variables if name.endswith(":BACT")
        )

        with pytest.raises(ReadOnlyError, match="is read-only"):
            model.set({bact_pv: 0.0})

    def test_cu_hxr_screen(self):
        model = get_cu_hxr_cheetah_model()

        image_pv = next(
            name
            for name in model.supported_variables
            if name.endswith(":Image:ArrayData")
        )
        image = np.asarray(model.get(image_pv))
        assert image.ndim == 2
        assert image.size > 0

        control_pv = next(
            name
            for name, variable in model.supported_variables.items()
            if name.endswith(":BCTRL") and not getattr(variable, "read_only", True)
        )
        current_value = float(model.get(control_pv))
        target_value = current_value + 0.001
        model.set({control_pv: target_value})
        assert np.isclose(float(model.get(control_pv)), target_value)

        updated_image = np.asarray(model.get(image_pv))
        assert updated_image.shape == image.shape
        assert np.isfinite(updated_image).all()

    def test_cu_hxr_screen_resolution_matches_yaml_and_expected_range(self):
        model = get_cu_hxr_cheetah_model()

        resolution_pv = "OTRS:IN20:541:RESOLUTION"
        assert resolution_pv in model.supported_variables

        otr1_element = next(
            element
            for element in model.simulator.segment.elements
            if element.name.split("#", 1)[0].lower() == "otr1"
        )
        expected_resolution = float(otr1_element.pixel_size[0]) * 1e6

        resolution = float(model.get(resolution_pv))
        assert np.isclose(resolution, expected_resolution)
        assert 5.0 < resolution < 30.0

    def test_screen_pvs_match_cheetah_segment(self):
        model = get_cu_hxr_cheetah_model()
        assert_screen_image_pvs_match_cheetah_segment(
            model,
            screen_attrs=SCREEN_PV_ATTRS,
        )

    def test_bpm_pvs_have_expected_suffixes(self):
        model = get_cu_hxr_cheetah_model()
        assert_bpm_pvs_match_cheetah_segment(model)

    def test_roundtrip_pv_get_set(self):
        model = get_cu_hxr_cheetah_model()
        assert_roundtrip_pv_get_set(model)

    @pytest.mark.parametrize(
        ("element_type", "excluded_elements"),
        [
            ("Quadrupole", ()),
            ("HorizontalCorrector", ("xcapm2",)),
            ("VerticalCorrector", ("ycapm2",)),
        ],
    )
    def test_magnet_pvs_match_cheetah_segment(self, element_type, excluded_elements):
        model = get_cu_hxr_cheetah_model()
        assert_magnet_pvs_match_cheetah_segment(
            model,
            element_type,
            excluded_elements=excluded_elements,
        )


class TestCUInjImpact:
    pytestmark = [
        pytest.mark.requires_impact,
        pytest.mark.requires_lcls_lattice,
        pytest.mark.skipif(
            not HAS_IMPACT_DEPS or not HAS_LCLS_LATTICE or not HAS_IMPACT_EXECUTABLE,
            reason=IMPACT_SKIP_REASON,
        ),
    ]

    @pytest.fixture
    def model(self):
        return get_cu_inj_impact_model(n_particles=100)

    def test_initialization(self, model):
        writable_control_variables = {
            name
            for name, variable in model.supported_variables.items()
            if not getattr(variable, "read_only", True)
        }

        assert len(model.supported_variables) > 0
        assert len(writable_control_variables) > 0

        # Smoke test that reading all variables works after initialization.
        _ = model.get(list(model.supported_variables))

    def test_group_actions_are_registered_and_writable(self, model):
        from virtual_accelerator.impact.actions import ImpactGroupVariable

        expected_group_pvs = {
            group_config["pv"] for group_config in IMPACT_GROUP_PV_MAPPING.values()
        }
        missing_group_pvs = sorted(expected_group_pvs - set(model.supported_variables))
        assert not missing_group_pvs

        for group_name, group_config in IMPACT_GROUP_PV_MAPPING.items():
            group_pv = group_config["pv"]
            group_variable = model.supported_variables[group_pv]

            assert isinstance(group_variable, ImpactGroupVariable)
            assert not getattr(group_variable, "read_only", True)
            assert np.isclose(group_variable.scale, group_config.get("scale", 1.0))

        # Use one representative mapped PV for roundtrip set/get behavior.
        _, test_group_config = next(iter(IMPACT_GROUP_PV_MAPPING.items()))
        test_group_pv = test_group_config["pv"]
        original_value = float(model.get(test_group_pv))
        updated_value = original_value + 1e-4
        model.set({test_group_pv: updated_value})
        assert np.isclose(float(model.get(test_group_pv)), updated_value)

        # Reset to original value to avoid side effects across tests.
        model.set({test_group_pv: original_value})

    def test_bact_readback_is_not_writable(self, model):
        bact_pv = next(
            name for name in model.supported_variables if name.endswith(":BACT")
        )

        with pytest.raises(ReadOnlyError, match="is read-only"):
            model.set({bact_pv: 0.0})

    def test_screen_image_outputs(self, model):
        image_pv = next(
            name
            for name in model.supported_variables
            if name.endswith(":Image:ArrayData")
        )
        base_pv = image_pv.rsplit(":", 2)[0]

        image = np.asarray(model.get(image_pv))
        assert image.ndim == 2
        assert image.size > 0
        assert np.isfinite(image).all()
        assert image.min() >= 0.0
        assert image.max() <= 1.0

        resolution = float(model.get(f"{base_pv}:RESOLUTION"))
        size0 = int(model.get(f"{base_pv}:Image:ArraySize0_RBV"))
        size1 = int(model.get(f"{base_pv}:Image:ArraySize1_RBV"))

        assert image.shape == (size1, size0)
        assert resolution > 0.0

    def test_quadrupole_pvs_match_impact_lattice(self, model):
        element_names, element_types = _get_impact_lattice_element_metadata(model)

        assert_magnet_pvs_match_lattice_elements(
            model=model,
            element_key="Quadrupole",
            element_names=element_names,
            element_keys=element_types,
        )

    def test_screen_pvs_match_impact_lattice(self, model):
        element_names, element_types = _get_impact_lattice_element_metadata(model)
        screen_elements = [
            element_name
            for element_name, element_type in zip(element_names, element_types)
            if element_type == "Screen"
        ]

        assert screen_elements
        assert_screen_image_pvs_in_supported_variables(
            model=model,
            screen_elements=screen_elements,
            screen_attrs=SCREEN_PV_ATTRS,
        )

    def test_bctrl_roundtrip_get_set(self, model):
        bctrl_pv = next(
            name
            for name, variable in model.supported_variables.items()
            if name.endswith(":BCTRL") and not getattr(variable, "read_only", True)
        )

        current_value = float(model.get(bctrl_pv))
        target_value = current_value + 0.001
        model.set({bctrl_pv: target_value})
        assert np.isclose(float(model.get(bctrl_pv)), target_value)

        # Reset to original value to avoid side effects across tests.
        model.set({bctrl_pv: current_value})

    def test_end_element(self):
        model = get_cu_inj_impact_model(n_particles=2, end_element="YAG03")

        # assert certain elements are not in the simulation
        assert "OTR2" not in model.impact_model.simulator.ele.keys()
        assert "YAG03" in model.impact_model.simulator.ele.keys()

        # assert that the supported variables do not include the grouped removed element
        assert "group:L0B_scale" not in model.supported_variables
        assert "group:L0B_phase" not in model.supported_variables

        # assert PVs for the removed element are not in the supported variables
        assert "OTRS:Image:571:ArraySize0_RBV" not in model.supported_variables
        assert "OTRS:Image:571:ArraySize1_RBV" not in model.supported_variables
        assert "OTRS:Image:571:RESOLUTION" not in model.supported_variables
        assert "ACCL:IN20:400:L0B_ADES" not in model.supported_variables
