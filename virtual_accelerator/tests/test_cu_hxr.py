import os

from pathlib import Path
from virtual_accelerator.models.cu_hxr import (
    get_cu_hxr_bmad_model,
    get_cu_hxr_cheetah_model,
)


class TestCUHXRBmad:
    def test_cu_hxr_twiss(self):
        model = get_cu_hxr_bmad_model()

        beam_path = os.path.join(Path(__file__).parent, "../bmad", "test_beam")
        model.tao.cmd(f"set beam_init position_file = {beam_path}")

        outputs = model.get(["a.beta", "b.beta", "name"])

        assert len(outputs["a.beta"]) == len(model.tao.lat_list("*", "ele.name"))
        assert len(outputs["b.beta"]) == len(model.tao.lat_list("*", "ele.name"))
        assert outputs["name"][0] == "BEGINNING"
        assert outputs["name"][-1] == "END"

    def test_cu_hxr_screen(self):
        model = get_cu_hxr_bmad_model()

        beam_path = os.path.join(Path(__file__).parent, "../bmad", "test_beam")
        model.tao.cmd(f"set beam_init position_file = {beam_path}")

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


class TestCUHXRCheetah:
    def test_initialization(self):
        model = get_cu_hxr_cheetah_model()

        assert model.get(["OTRS:IN20:541:Image:ArrayData"])[
            "OTRS:IN20:541:Image:ArrayData"
        ].shape == (1392, 1040)
