import numpy as np
from pathlib import Path

import pytest
import yaml

from lume.exceptions import ReadOnlyError
from virtual_accelerator.tests.dependency_profiles import (
    HAS_BMAD_DEPS,
    HAS_CHEETAH_DEPS,
    HAS_LCLS_LATTICE,
)
from virtual_accelerator.tests._bmad_model_test_utils import (
    TEST_BEAM_PATH,
    assert_bpm_pvs_match_tao_lattice,
    assert_bmad_model_initialization,
    assert_bmad_model_twiss_outputs,
    assert_bmad_model_track_beam_custom_path,
    assert_magnet_pvs_match_tao_lattice,
    assert_roundtrip_pv_get_set,
    assert_screen_image_pvs_in_supported_variables,
)

CU_HXR_PROFMON_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "cu_hxr_profmon_info.yaml"
)


def _resolve_runtime_getter(has_optional_deps: bool, getter_name: str):
    if not (has_optional_deps and HAS_LCLS_LATTICE):
        return False, None

    try:
        from virtual_accelerator.models import cu_hxr as cu_hxr_models

        return True, getattr(cu_hxr_models, getter_name)
    except Exception:
        return False, None


def _load_cu_hxr_screen_config():
    with CU_HXR_PROFMON_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


HAS_BMAD_RUNTIME, get_cu_hxr_bmad_model = _resolve_runtime_getter(
    HAS_BMAD_DEPS,
    "get_cu_hxr_bmad_model",
)

HAS_CHEETAH_RUNTIME, get_cu_hxr_cheetah_model = _resolve_runtime_getter(
    HAS_CHEETAH_DEPS,
    "get_cu_hxr_cheetah_model",
)


@pytest.mark.requires_bmad
@pytest.mark.requires_lcls_lattice
@pytest.mark.skipif(
    not HAS_BMAD_RUNTIME,
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
        assert_screen_image_pvs_in_supported_variables(model)

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
        assert len(model.supported_variables) < 40

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
        model = get_cu_hxr_bmad_model()
        assert_magnet_pvs_match_tao_lattice(model, element_type)

    def test_bpm_pvs_match_tao_lattice(self):
        model = get_cu_hxr_bmad_model()
        assert_bpm_pvs_match_tao_lattice(model)

    def test_roundtrip_pv_get_set(self):
        model = get_cu_hxr_bmad_model(
            custom_beam_path=TEST_BEAM_PATH, end_element="OTR4"
        )
        assert_roundtrip_pv_get_set(model)


class TestCUHXRCheetah:
    pytestmark = [
        pytest.mark.requires_cheetah,
        pytest.mark.requires_lcls_lattice,
        pytest.mark.skipif(
            not HAS_CHEETAH_RUNTIME,
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
        screen_config = _load_cu_hxr_screen_config()

        model = get_cu_hxr_cheetah_model()

        resolution_pv, expected_resolution = next(
            (
                f"{config_entry['name']}:RESOLUTION",
                float(config_entry["pixel_size"]),
            )
            for config_entry in screen_config.values()
            if f"{config_entry['name']}:RESOLUTION" in model.supported_variables
        )

        resolution = float(model.get(resolution_pv))
        assert np.isclose(resolution, expected_resolution)
        assert 5.0 < resolution < 30.0

    def test_quadrupole_pvs_have_expected_suffixes(self):
        model = get_cu_hxr_cheetah_model()

        expected_suffixes = {
            "BCTRL",
            "BACT",
            "BDES",
            "BMIN",
            "BMAX",
            "STATCTRLSUB.T",
            "CTRL",
        }

        quadrupole_control_pvs = [
            name
            for name in model.supported_variables
            if name.startswith("QUAD:") and name.endswith(":BCTRL")
        ]
        assert quadrupole_control_pvs

        for control_pv in quadrupole_control_pvs:
            base_pv = control_pv.rsplit(":", 1)[0]
            for suffix in expected_suffixes:
                assert f"{base_pv}:{suffix}" in model.supported_variables

    def test_bpm_pvs_have_expected_suffixes(self):
        model = get_cu_hxr_cheetah_model()

        bpm_x_pvs = [
            name
            for name in model.supported_variables
            if name.startswith("BPMS:") and name.endswith(":X")
        ]
        assert bpm_x_pvs

        for x_pv in bpm_x_pvs:
            base_pv = x_pv.rsplit(":", 1)[0]
            assert f"{base_pv}:Y" in model.supported_variables
            assert f"{base_pv}:TMIT" in model.supported_variables

    def test_roundtrip_pv_get_set(self):
        model = get_cu_hxr_cheetah_model()
        assert_roundtrip_pv_get_set(model)