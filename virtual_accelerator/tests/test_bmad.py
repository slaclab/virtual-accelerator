import os

import pytest
from virtual_accelerator.tests._bmad_model_test_utils import (
    HAS_BMAD_DEPS,
    HAS_LCLS_LATTICE,
)

pytestmark = [
    pytest.mark.requires_bmad,
    pytest.mark.requires_lcls_lattice,
]

if HAS_BMAD_DEPS and HAS_LCLS_LATTICE:
    from pytao import Tao

    from virtual_accelerator.bmad.actions import (
        BminVariable,
        BmaxVariable,
        ControlStateVariable,
        StatusVariable,
        QuadrupoleBACTVariable,
        QuadrupoleBCTRLVariable,
    )
    from virtual_accelerator.bmad.variables import create_variables_from_element
else:
    pytest.skip(
        "requires bmad optional dependencies and LCLS_LATTICE",
        allow_module_level=True,
    )


class TestBmadVariables:
    @pytest.fixture
    def tao(self):
        lattice_root = os.environ["LCLS_LATTICE"]
        init_file = os.path.join(lattice_root, "bmad/models/cu_hxr/tao.init")
        return Tao(f"-init {init_file} -noplot -slice_lattice YAG03:OTR4")

    def test_create_variable_classes(self, tao):
        # test quadrupole variables
        quadrupole_elements = ["QE01", "QE02", "QE03", "QE04"]
        mapping = {
            "BCTRL": "QuadrupoleBCTRLVariable",
            "BCTRL.DRVL": "BminVariable",
            "BCTRL.DRVH": "BmaxVariable",
            "BACT": "QuadrupoleBACTVariable",
            "BDES": "QuadrupoleBCTRLVariable",
            "BMAX": "BmaxVariable",
            "BMIN": "BminVariable",
            "STATCTRLSUB.T": "StatusVariable",
            "CTRL": "ControlStateVariable",
        }

        for element in quadrupole_elements:
            variables = create_variables_from_element(
                element, base_pv=tao.ele(element).head.alias, class_mapping=mapping
            )

            # element pv
            base_pv = tao.ele(element).head.alias

            # Check that the expected variable classes are present for each quadrupole element
            variable_classes = {var.name: var for var in variables}
            assert isinstance(
                variable_classes[f"{base_pv}:BCTRL"], QuadrupoleBCTRLVariable
            )
            assert isinstance(
                variable_classes[f"{base_pv}:BACT"], QuadrupoleBACTVariable
            )
            assert isinstance(variable_classes[f"{base_pv}:BMIN"], BminVariable)
            assert isinstance(variable_classes[f"{base_pv}:BMAX"], BmaxVariable)
            assert isinstance(variable_classes[f"{base_pv}:CTRL"], ControlStateVariable)
            assert isinstance(
                variable_classes[f"{base_pv}:STATCTRLSUB.T"], StatusVariable
            )
            assert isinstance(
                variable_classes[f"{base_pv}:BDES"], QuadrupoleBCTRLVariable
            )
            assert isinstance(variable_classes[f"{base_pv}:BCTRL.DRVL"], BminVariable)
            assert isinstance(variable_classes[f"{base_pv}:BCTRL.DRVH"], BmaxVariable)
