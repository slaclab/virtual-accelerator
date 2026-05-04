import os
import importlib.util

import pytest
from virtual_accelerator.tests._bmad_model_test_utils import (
    HAS_BMAD_DEPS,
    TEST_BEAM_PATH,
    assert_bmad_model_initialization,
    assert_bmad_model_track_beam_custom_path,
    assert_bmad_model_twiss_outputs,
)
from virtual_accelerator.models.facet2 import get_facet_bmad_model


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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

    def test_screen_variables(self):
        model = get_facet_bmad_model(
            track_beam=True, custom_beam_path=TEST_BEAM_PATH, end_element="PR10711"
        )
        # Check that screen variables are included in supported variables when tracking is enabled
        assert "OTRS:IN10:571:Image:ArrayData" in model.supported_variables
        assert "OTRS:IN10:711:Image:ArrayData" in model.supported_variables

        # test specific output from one of the screens to ensure it's properly set up
        output = model.get("OTRS:IN10:571:Image:ArrayData")
        assert output.shape == (1392, 1040)

        # test to make sure that changing an upstream variable that should affect the screen output
        current_value = model.get("QUAD:IN10:371:BCTRL")
        model.set({"QUAD:IN10:371:BCTRL": current_value + 0.1})
        new_output = model.get("OTRS:IN10:571:Image:ArrayData")
        assert not (new_output == output).all()  # Check that the screen output changed

    @pytest.mark.xfail(reason="known FACET2 quadrupoles are missing EPICS mappings")
    def test_quadrupole_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model()

        element_names = model.tao.lat_list("*", "ele.name")
        element_keys = model.tao.lat_list("*", "ele.key")
        quadrupole_elements = {
            element_name.split("#")[0]
            for element_name, element_key in zip(element_names, element_keys)
            if element_name not in ("BEGINNING", "END") and element_key == "Quadrupole"
        }

        pv_prefix_by_element = {
            element_name: pv_prefix
            for pv_prefix, element_name in model.transformer.control_name_to_bmad.items()
        }

        missing_mapping = sorted(
            element_name
            for element_name in quadrupole_elements
            if element_name not in pv_prefix_by_element
        )
        assert not missing_mapping, (
            "Quadrupole elements missing PV prefix mapping: "
            + ", ".join(missing_mapping)
        )

        quadrupole_attrs = ("BCTRL", "BACT", "BDES", "BMIN", "BMAX")
        supported_variable_names = set(model.supported_variables)

        missing_pvs = {}
        for element_name in sorted(quadrupole_elements):
            pv_prefix = pv_prefix_by_element[element_name]
            expected_pvs = {f"{pv_prefix}:{attr}" for attr in quadrupole_attrs}
            absent_pvs = sorted(expected_pvs - supported_variable_names)
            if absent_pvs:
                missing_pvs[element_name] = absent_pvs

        assert not missing_pvs, (
            "Quadrupole PVs missing from model.supported_variables: "
            + "; ".join(
                f"{element}: {', '.join(pvs)}" for element, pvs in missing_pvs.items()
            )
        )

    @pytest.mark.xfail(reason="known FACET2 HKickers are missing EPICS mappings")
    def test_hkicker_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model()

        element_names = model.tao.lat_list("*", "ele.name")
        element_keys = model.tao.lat_list("*", "ele.key")
        hkicker_elements = {
            element_name.split("#")[0]
            for element_name, element_key in zip(element_names, element_keys)
            if element_name not in ("BEGINNING", "END") and element_key == "HKicker"
        }

        pv_prefix_by_element = {
            element_name: pv_prefix
            for pv_prefix, element_name in model.transformer.control_name_to_bmad.items()
        }

        missing_mapping = sorted(
            element_name
            for element_name in hkicker_elements
            if element_name not in pv_prefix_by_element
        )
        assert not missing_mapping, (
            "HKicker elements missing PV prefix mapping: " + ", ".join(missing_mapping)
        )

        hkicker_attrs = ("BCTRL", "BACT", "BDES", "BMIN", "BMAX")
        supported_variable_names = set(model.supported_variables)

        missing_pvs = {}
        for element_name in sorted(hkicker_elements):
            pv_prefix = pv_prefix_by_element[element_name]
            expected_pvs = {f"{pv_prefix}:{attr}" for attr in hkicker_attrs}
            absent_pvs = sorted(expected_pvs - supported_variable_names)
            if absent_pvs:
                missing_pvs[element_name] = absent_pvs

        assert not missing_pvs, (
            "HKicker PVs missing from model.supported_variables: "
            + "; ".join(
                f"{element}: {', '.join(pvs)}" for element, pvs in missing_pvs.items()
            )
        )

    @pytest.mark.xfail(reason="known FACET2 VKickers are missing EPICS mappings")
    def test_vkicker_pvs_match_tao_lattice(self):
        model = get_facet_bmad_model()

        element_names = model.tao.lat_list("*", "ele.name")
        element_keys = model.tao.lat_list("*", "ele.key")
        vkicker_elements = {
            element_name.split("#")[0]
            for element_name, element_key in zip(element_names, element_keys)
            if element_name not in ("BEGINNING", "END") and element_key == "VKicker"
        }

        pv_prefix_by_element = {
            element_name: pv_prefix
            for pv_prefix, element_name in model.transformer.control_name_to_bmad.items()
        }

        missing_mapping = sorted(
            element_name
            for element_name in vkicker_elements
            if element_name not in pv_prefix_by_element
        )
        assert not missing_mapping, (
            "VKicker elements missing PV prefix mapping: " + ", ".join(missing_mapping)
        )

        vkicker_attrs = ("BCTRL", "BACT", "BDES", "BMIN", "BMAX")
        supported_variable_names = set(model.supported_variables)

        missing_pvs = {}
        for element_name in sorted(vkicker_elements):
            pv_prefix = pv_prefix_by_element[element_name]
            expected_pvs = {f"{pv_prefix}:{attr}" for attr in vkicker_attrs}
            absent_pvs = sorted(expected_pvs - supported_variable_names)
            if absent_pvs:
                missing_pvs[element_name] = absent_pvs

        assert not missing_pvs, (
            "VKicker PVs missing from model.supported_variables: "
            + "; ".join(
                f"{element}: {', '.join(pvs)}" for element, pvs in missing_pvs.items()
            )
        )
