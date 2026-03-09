import os
from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model

class TestCUHXR:
    def test_cu_hxr_twiss(self):
        model = get_cu_hxr_bmad_model()

        outputs = model.get(["a.beta","b.beta", "name"])

        assert len(outputs["a.beta"]) == 3219
        assert len(outputs["b.beta"]) == 3219
        assert outputs["name"][0] == "BEGINNING"
        assert outputs["name"][-1] == "END"

    def test_cu_hxr_screen(self):
        # may be needed to set LCLS_LATTICE env variable to run this test, but should be set in CI already
        model = get_cu_hxr_bmad_model()

        # set tracking
        model.set({"track_type": 1})

        # get initial OTR4 image
        image = model.get(["OTRS:IN20:711:Image:ArrayData"])["OTRS:IN20:711:Image:ArrayData"]
        assert image.shape == (1024, 1024)

        # set some control variables
        model.set({"QUAD:IN20:631:BDES": 0.0})

        # get updated OTR4 image
        updated_image = model.get(["OTRS:IN20:711:Image:ArrayData"])["OTRS:IN20:711:Image:ArrayData"]
        assert updated_image.shape == (1024, 1024)

        # make sure it changed
        assert not (image == updated_image).all()
