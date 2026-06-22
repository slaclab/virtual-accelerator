import pytest
from virtual_accelerator.tests._bmad_model_test_utils import (
    HAS_BMAD_DEPS,
    HAS_FACET2_LATTICE,
    HAS_FACET_SURROGATE_DEPS,
    TEST_BEAM_PATH,
    assert_bpm_pvs_match_tao_lattice,
    assert_bmad_model_initialization,
    assert_bmad_model_track_beam_custom_path,
    assert_bmad_model_twiss_outputs,
    assert_magnet_pvs_match_tao_lattice,
    assert_screen_image_pvs_in_supported_variables,
)

pytestmark = [
    pytest.mark.requires_bmad,
    pytest.mark.requires_facet2_lattice,
]

if HAS_BMAD_DEPS and HAS_FACET2_LATTICE:
    from virtual_accelerator.models.facet2 import (
        get_facet_bmad_model,
        get_facet_staged_model,
    )
    from virtual_accelerator.utils.variables import get_pvs_by_element_name
else:
    pytest.skip(
        "requires bmad optional dependencies and FACET2_LATTICE",
        allow_module_level=True,
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

        screen_pvs = get_pvs_by_element_name(model)["PR10571"]
        # get the PV name that contains "Image:ArrayData" which is the expected output PV for the screen image
        screen_pv = next(pv for pv in screen_pvs if "Image:ArrayData" in pv)

        # test specific output from one of the screens to ensure it's properly set up
        output = model.get(screen_pv)
        assert output.shape == (1392, 1040)

        # test to make sure that changing an upstream variable that should affect the screen output
        current_value = model.get("QUAD:IN10:371:BCTRL")
        model.set({"QUAD:IN10:371:BCTRL": current_value + 0.1})
        new_output = model.get(screen_pv)
        assert not (new_output == output).all()  # Check that the screen output changed

    @pytest.mark.requires_surrogate
    @pytest.mark.skipif(
        not HAS_FACET_SURROGATE_DEPS,
        reason="requires staged-model optional dependencies",
    )
    def test_staged_model(self):
        staged_model = get_facet_staged_model(
            surrogate_inputs="machine", n_particles=1000, end_element="PR10711"
        )

        pvs_by_element = get_pvs_by_element_name(staged_model.lume_model_instances[1])
        for screen_element in ["PR10571", "PR10711"]:
            screen_pv = next(
                pv for pv in pvs_by_element[screen_element] if "Image:ArrayData" in pv
            )
            assert screen_pv in staged_model.supported_variables

    # @pytest.mark.xfail(reason="known FACET2 quadrupoles are missing EPICS mappings")
    def test_quadrupole_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model(end_element="PR10711")
        assert_magnet_pvs_match_tao_lattice(model, "Quadrupole")

    def test_hkicker_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model(end_element="PR10711")
        assert_magnet_pvs_match_tao_lattice(model, "HKicker")

    def test_vkicker_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model(end_element="PR10711")
        assert_magnet_pvs_match_tao_lattice(model, "VKicker")

    def test_bpm_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model(end_element="PR10711")
        assert_bpm_pvs_match_tao_lattice(model)

    def test_facet_custom_variables(self):
        model = get_facet_bmad_model(end_element="PR10711")
        # test that the L0B phase feedback variable is included since L0B is in the lattice
        assert "KLYS:LI10:41:SFB_PDES" in model.supported_variables

        # test that the L0B phase feedback variable is properly set up as a writable action variable
        model.get("KLYS:LI10:41:SFB_PDES")

        # test that the variable is writable
        model.set({"KLYS:LI10:41:SFB_PDES": 10.0})
        assert model.get("KLYS:LI10:41:SFB_PDES") == 10.0
