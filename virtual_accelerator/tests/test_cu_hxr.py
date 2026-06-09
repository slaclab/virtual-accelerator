import os
import importlib.util

import pytest
from virtual_accelerator.tests._bmad_model_test_utils import (
    HAS_BMAD_DEPS,
    TEST_BEAM_PATH,
    assert_bpm_pvs_match_tao_lattice,
    assert_bmad_model_initialization,
    assert_bmad_model_twiss_outputs,
    assert_bmad_model_track_beam_custom_path,
    assert_magnet_pvs_match_tao_lattice,
    assert_screen_image_pvs_in_supported_variables,
)
from virtual_accelerator.models.cu_hxr import (
    get_cu_hxr_bmad_model,
    get_cu_hxr_cheetah_model,
)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_CHEETAH_DEPS = _has_module("cheetah") and _has_module("lume_cheetah")
HAS_LCLS_LATTICE = bool(os.environ.get("LCLS_LATTICE"))


@pytest.mark.skipif(not HAS_BMAD_DEPS, reason="requires bmad optional dependencies")
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

        # get initial OTR4 image
        image = model.get("OTRS:IN20:711:Image:ArrayData")
        assert image.shape == (1392, 1040)

        # set some control variables
        model.set({"QUAD:IN20:631:BCTRL": 0.0})

        # get updated OTR4 image
        updated_image = model.get("OTRS:IN20:711:Image:ArrayData")
        assert updated_image.shape == (1392, 1040)

        # make sure it changed
        assert not (image == updated_image).all()

    @pytest.mark.xfail(reason="need to update klystron implementation")
    def test_cu_hxr_lcavity(self):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)

        enld = model.get("KLYS:LI21:31:ENLD")
        enld = enld + 5
        model.set({"KLYS:LI21:31:ENLD": enld})
        ampl = model.get("KLYS:LI21:31:ENLD")
        assert ampl == enld

    def test_quadrupole_pvs_match_tao_lattice(self):
        model = get_cu_hxr_bmad_model()
        assert_magnet_pvs_match_tao_lattice(model, "Quadrupole")

    def test_hkicker_pvs_match_tao_lattice(self):
        model = get_cu_hxr_bmad_model()
        assert_magnet_pvs_match_tao_lattice(model, "HKicker")

    def test_vkicker_pvs_match_tao_lattice(self):
        model = get_cu_hxr_bmad_model()
        assert_magnet_pvs_match_tao_lattice(model, "VKicker")

    def test_bpm_pvs_match_tao_lattice(self):
        model = get_cu_hxr_bmad_model()
        assert_bpm_pvs_match_tao_lattice(model)


@pytest.mark.skipif(
    (not HAS_CHEETAH_DEPS) or (not HAS_LCLS_LATTICE),
    reason="requires cheetah optional dependencies and LCLS_LATTICE",
)
class TestCUHXRCheetah:
    def test_initialization(self):
        model = get_cu_hxr_cheetah_model()

        assert model.get(["OTRS:IN20:541:Image:ArrayData"])[
            "OTRS:IN20:541:Image:ArrayData"
        ].shape == (1392, 1040)
