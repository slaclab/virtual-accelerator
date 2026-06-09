import os

import pytest
from virtual_accelerator.tests._bmad_model_test_utils import (
    HAS_BMAD_DEPS,
    TEST_BEAM_PATH,
    assert_bpm_pvs_match_tao_lattice,
    assert_bmad_model_initialization,
    assert_bmad_model_track_beam_custom_path,
    assert_bmad_model_twiss_outputs,
    assert_magnet_pvs_match_tao_lattice,
    assert_screen_image_pvs_in_supported_variables,
)
from virtual_accelerator.models.facet2 import (
    get_facet_bmad_model,
    get_facet_staged_model,
)


HAS_FACET_LATTICE = bool(os.environ.get("FACET2_LATTICE"))


@pytest.mark.skipif(
    (not HAS_BMAD_DEPS) or (not HAS_FACET_LATTICE),
    reason="requires bmad optional dependencies and FACET2_LATTICE",
)
class TestFACET2Bmad:
    def test_initialization(self):
        assert_bmad_model_initialization(get_facet_bmad_model)

    def test_twiss(self):
        assert_bmad_model_twiss_outputs(get_facet_bmad_model)

    def test_track_beam_custom_path(self):
        assert_bmad_model_track_beam_custom_path(get_facet_bmad_model)

    def test_screen_image_pvs_in_supported_variables(self):
        model = get_facet_bmad_model(track_beam=True, custom_beam_path=TEST_BEAM_PATH)
        assert_screen_image_pvs_in_supported_variables(model)

        # test getting all of the supported variables to ensure no errors with screen variable setup
        _ = model.get(list(model.supported_variables))

    def test_screen_variables(self):
        model = get_facet_bmad_model(
            track_beam=True, custom_beam_path=TEST_BEAM_PATH, end_element="PR10711"
        )
        # Check that screen image variables are included in supported variables.
        assert_screen_image_pvs_in_supported_variables(model)

        pv_prefix_by_element = {
            element_name: pv_prefix
            for pv_prefix, element_name in model.transformer.control_name_to_bmad.items()
        }
        screen_element = "PR10571"
        screen_pv = f"{pv_prefix_by_element[screen_element]}:Image:ArrayData"

        # test specific output from one of the screens to ensure it's properly set up
        output = model.get(screen_pv)
        assert output.shape == (1392, 1040)

        # test to make sure that changing an upstream variable that should affect the screen output
        current_value = model.get("QUAD:IN10:371:BCTRL")
        model.set({"QUAD:IN10:371:BCTRL": current_value + 0.1})
        new_output = model.get(screen_pv)
        assert not (new_output == output).all()  # Check that the screen output changed

    def test_staged_model(self):
        staged_model = get_facet_staged_model(
            surrogate_inputs="machine", n_particles=1000, end_element="PR10711"
        )

        mapping = staged_model.lume_model_instances[1].transformer.control_name_to_bmad
        pv_prefix_by_element = {
            element_name: pv_prefix for pv_prefix, element_name in mapping.items()
        }
        for screen_element in ["PR10571", "PR10711"]:
            screen_pv = f"{pv_prefix_by_element[screen_element]}:Image:ArrayData"
            assert screen_pv in staged_model.supported_variables

    @pytest.mark.xfail(reason="known FACET2 quadrupoles are missing EPICS mappings")
    def test_quadrupole_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model()
        assert_magnet_pvs_match_tao_lattice(model, "Quadrupole")

    @pytest.mark.xfail(reason="known FACET2 HKickers are missing EPICS mappings")
    def test_hkicker_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model()
        assert_magnet_pvs_match_tao_lattice(model, "HKicker")

    @pytest.mark.xfail(reason="known FACET2 VKickers are missing EPICS mappings")
    def test_vkicker_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model()
        assert_magnet_pvs_match_tao_lattice(model, "VKicker")

    def test_bpm_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model()
        assert_bpm_pvs_match_tao_lattice(model)
