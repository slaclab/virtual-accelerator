import os

from pathlib import Path
from virtual_accelerator.models.cu_hxr import (
    get_cu_hxr_bmad_model,
    get_cu_hxr_cheetah_model,
)

TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "../bmad", "test_beam")

class TestCUHXRBmad:
    def test_initialization(self):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)

        assert "QUAD:IN20:631:BCTRL" in model.control_variables

        model = get_cu_hxr_bmad_model(track_beam=True, custom_beam_path=TEST_BEAM_PATH)
        assert "OTRS:IN20:711:Image:ArrayData" in model.supported_variables

    def test_cu_hxr_twiss(self):
        model = get_cu_hxr_bmad_model(custom_beam_path=TEST_BEAM_PATH)

        outputs = model.get(["a.beta", "b.beta", "name"])

        assert len(outputs["a.beta"]) == len(model.tao.lat_list("*", "ele.name"))
        assert len(outputs["b.beta"]) == len(model.tao.lat_list("*", "ele.name"))
        assert outputs["name"][0] == "BEGINNING"
        assert outputs["name"][-1] == "END"

    def test_sub_lattice(self):
        model = get_cu_hxr_bmad_model("QE04#1","OTR2")
        assert len(model.supported_variables) < 40

        # test getting partial lattice with beam tracking
        model = get_cu_hxr_bmad_model(end_element="OTR4", track_beam=True)


    def test_cu_hxr_screen(self):
        model = get_cu_hxr_bmad_model(track_beam=True, custom_beam_path=TEST_BEAM_PATH)

        # set tracking
        model.set({"track_type": 1})

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


class TestCUHXRCheetah:
    def test_initialization(self):
        model = get_cu_hxr_cheetah_model()

        assert model.get(["OTRS:IN20:541:Image:ArrayData"])[
            "OTRS:IN20:541:Image:ArrayData"
        ].shape == (1392, 1040)
