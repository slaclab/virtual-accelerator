import os
import importlib.util

from pathlib import Path
import pytest
from virtual_accelerator.models.cu_hxr import (
    get_cu_hxr_bmad_model,
    get_cu_hxr_cheetah_model,
)

TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "../bmad", "test_beam")


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_BMAD_DEPS = _has_module("pytao") and _has_module("lume_bmad")
HAS_CHEETAH_DEPS = _has_module("cheetah") and _has_module("lume_cheetah")
HAS_LCLS_LATTICE = bool(os.environ.get("LCLS_LATTICE"))


@pytest.mark.skipif(not HAS_BMAD_DEPS, reason="requires bmad optional dependencies")
class TestCUHXRBmad:
    def test_initialization(self):
        model = get_cu_hxr_bmad_model(
            end_element="OTR4", custom_beam_path=TEST_BEAM_PATH
        )

        assert "QUAD:IN20:631:BCTRL" in model.control_variables

        model = get_cu_hxr_bmad_model(
            end_element="OTR4", track_beam=True, custom_beam_path=TEST_BEAM_PATH
        )
        assert "OTRS:IN20:711:Image:ArrayData" in model.supported_variables

    def test_cu_hxr_twiss(self):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)

        outputs = model.get(["a.beta", "b.beta", "name"])

        assert len(outputs["a.beta"]) == len(model.tao.lat_list("*", "ele.name"))
        assert len(outputs["b.beta"]) == len(model.tao.lat_list("*", "ele.name"))
        assert outputs["name"][0] == "BEGINNING"
        assert outputs["name"][-1] == "END"

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
        image = model.get(["OTRS:IN20:711:Image:ArrayData"])[
            "OTRS:IN20:711:Image:ArrayData"
        ]
        assert image.shape == (1392, 1040)

        # set some control variables
        model.set({"QUAD:IN20:631:BCTRL": 0.0})

        # get updated OTR4 image
        updated_image = model.get(["OTRS:IN20:711:Image:ArrayData"])[
            "OTRS:IN20:711:Image:ArrayData"
        ]
        assert updated_image.shape == (1392, 1040)

        # make sure it changed
        assert not (image == updated_image).all()

    def test_cu_hxr_lcavity(self):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)

        enld = model.get(["KLYS:LI21:31:ENLD"])["KLYS:LI21:31:ENLD"]
        enld = enld + 5
        model.set({"KLYS:LI21:31:ENLD": enld})
        ampl = model.get(["KLYS:LI21:31:ENLD"])
        assert ampl["KLYS:LI21:31:ENLD"] == enld


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
