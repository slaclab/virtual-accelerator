import pytest
import numpy as np

from lume.exceptions import ReadOnlyError

from virtual_accelerator.models.facet2 import (
    IMPACT_GROUP_PV_MAPPING,
    get_facet_bmad_model,
    get_facet_impact_model,
    get_facet_staged_model,
)

from virtual_accelerator.tests.dependency_profiles import (
    HAS_BMAD_DEPS,
    HAS_FACET2_LATTICE,
    HAS_FACET_SURROGATE_DEPS,
    HAS_IMPACT_DEPS,
    HAS_LCLS_LATTICE,
)

from virtual_accelerator.tests._bmad_model_test_utils import (
    TEST_BEAM_PATH,
    assert_bpm_pvs_match_tao_lattice,
    assert_bmad_model_initialization,
    assert_bmad_model_track_beam_custom_path,
    assert_bmad_model_twiss_outputs,
    assert_magnet_pvs_match_lattice_elements,
    assert_magnet_pvs_match_tao_lattice,
    assert_screen_image_pvs_in_supported_variables,
)
from virtual_accelerator.utils.variables import get_element_attr_mapping, get_pvs_by_element_name
from virtual_accelerator.tests.test_cu_hxr import (
    HAS_IMPACT_EXECUTABLE,
    IMPACT_SKIP_REASON,
    SCREEN_PV_ATTRS,
    _get_impact_lattice_element_metadata,
)


@pytest.mark.requires_bmad
@pytest.mark.requires_facet2_lattice
@pytest.mark.skipif(
    not HAS_BMAD_DEPS or not HAS_FACET2_LATTICE,
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
        model = get_facet_bmad_model(
            track_beam=True, custom_beam_path=TEST_BEAM_PATH, end_element="PR10711"
        )
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

    def test_sbend(self):
        model = get_facet_bmad_model(
            track_beam=True,
            start_element="L0AFEND",
            end_element="BPM10781",
            custom_beam_path=TEST_BEAM_PATH,
        )

        nominal_value = model.get("BEND:IN10:661:BCTRL")
        assert np.isclose(
            nominal_value, 0.125, rtol=1e-2
        )  # Check that the nominal value is correct

        # change the setpoint by a fixed percentage
        scale_factor = 0.01
        new_value = nominal_value * (1 + scale_factor)
        model.set({"BEND:IN10:661:BCTRL": new_value})
        updated_value = model.get("BEND:IN10:661:BCTRL")
        assert np.isclose(
            updated_value, new_value, rtol=1e-2
        )  # Check that the updated value is correct
        ele_attrs = model.tao.ele_gen_attribs("BX10661")

        # Check that the DG/G ratio is correct
        assert np.isclose(ele_attrs["DG"] / ele_attrs["G"], scale_factor, rtol=1e-4)

        # This change should also affect the downstream BPM reading
        bpm_reading = model.get("BPMS:IN10:781:X")
        assert (
            bpm_reading < -1.0
        )  # check that there is a significant deflection in the negative X direction

    def test_tcav(self):
        # test that the TCAV works as expected
        model = get_facet_bmad_model(
            track_beam=True,
            start_element="L0AFEND",
            end_element="PR10711",
            custom_beam_path=TEST_BEAM_PATH,
        )

        # set the TCAV voltage
        model.set(
            {
                "KLYS:LI10:51:REFPOC": 10.0,
                "KLYS:LI10:51:ADES": 0.3,
                "KLYS:LI10:51:MODECFG": "ACCEL_STDBY",
            }
        )  # Set TCAV voltage to 0.3  MV, phase to 10 degrees, and enable the TCAV
        assert (
            model.tao.ele_gen_attribs("TCY10490")["VOLTAGE"] == 0.3e6
        )  # Check that the TCAV voltage was set correctly
        assert np.isclose(
            model.tao.ele_gen_attribs("TCY10490")["PHI0"], 10.0 / 360.0
        )  # Check that the TCAV phase is 10 degrees
        assert (
            model.tao.ele("TCY10490").key == "Crab_Cavity"
        )  # Check that the TCAV is a crab cavity
        assert model.tao.ele("TCY10490").head.is_on  # Check that the TCAV is enabled

        # measure the deflection at the downstream bpm
        assert np.isclose(
            model.get("BPMS:IN10:651:X"), 0.0, atol=1e-4
        )  # Check that the beam is not deflected in X
        assert np.isclose(
            model.get("BPMS:IN10:651:Y"), 1.939, rtol=1e-2
        )  # Check that the beam is deflected in Y by 2 mm
        # NOTE: this value requires the bmad fixer elements to be disabled

        # disable the TCAV
        model.set({"KLYS:LI10:51:MODECFG": "STDBY"})  # Set TCAV to standby mode

        # measure the deflection at the downstream bpm again
        assert np.isclose(
            model.get("BPMS:IN10:651:Y"), 0.0, atol=1e-4
        )  # Check that the beam is no longer deflected

        # re-enable the TCAV
        model.set(
            {"KLYS:LI10:51:MODECFG": "ACCEL_STDBY"}
        )  # Set TCAV back to ACCEL_STDBY mode
        assert np.isclose(
            model.get("BPMS:IN10:651:Y"), 1.939, rtol=1e-2
        )  # Check that the TCAV deflected the beam again

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

    @pytest.mark.parametrize(
        "element_type", ["Quadrupole", "HKicker", "VKicker", "SBend"]
    )
    def test_magnet_pvs_match_tao_lattice(self, element_type):
        model = get_facet_bmad_model(end_element="PR10711")
        assert_magnet_pvs_match_tao_lattice(model, element_type)

    def test_bpm_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model(end_element="PR10711")
        assert_bpm_pvs_match_tao_lattice(model)

    def test_facet_custom_variables(self):
        model = get_facet_bmad_model(end_element="PR10711")
        # test that the L0B phase feedback variable is included since L0B is in the lattice
        for var in ["KLYS:LI10:41:SFB_PDES"]:
            assert var in model.supported_variables.keys()
            value = model.get(var)
            # test that the variable is writable
            model.set({var: value * 1.1})
            assert np.isclose(model.get(var), value * 1.1)


class TestFACETImpact:
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
        return get_facet_impact_model(n_particles=100)

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

        # remove L0AFEND screen
        screen_elements = [e for e in screen_elements if e != "L0AFEND"]

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
        model = get_facet_impact_model(n_particles=2, end_element="PR10241")

        # assert certain elements are not in the simulation
        assert "PR10571" not in model.impact_model.simulator.ele.keys()
        assert "PR10241" in model.impact_model.simulator.ele.keys()

        # assert that the supported variables do not include the grouped removed element
        assert "group:L0BF_scale" not in model.supported_variables
        assert "group:L0BF_phase" not in model.supported_variables

        # assert PVs for the removed element are not in the supported variables
        assert "PR10571:Image:ArraySize0_RBV" not in model.supported_variables
        assert "PR10571:Image:ArraySize1_RBV" not in model.supported_variables
        assert "PR10571:RESOLUTION" not in model.supported_variables
        assert "KLYS:IN10:41:ADES" not in model.supported_variables

    def test_custom_pvs(self):
        # test custom solenoid PVs
        model = get_facet_impact_model(n_particles=2, end_element="PR10241")
        suffixes = get_element_attr_mapping()["Solenoid"].keys()
        for suffix in suffixes:
            pv = f"SOLN:IN10:121:{suffix}"
            assert pv in model.supported_variables


