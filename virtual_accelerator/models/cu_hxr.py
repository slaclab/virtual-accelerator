import os
import numpy as np

from lume_bmad.model import LUMEBmadModel
from lume.variables import NDVariable

from virtual_accelerator.bmad.cu_transformer import (
    CUBmadTransformer,
)
from virtual_accelerator.bmad.utils import (
    import_control_variables,
    import_output_variables,
)


def get_cu_hxr_bmad_model():
    """
    Get the LUMEBmadModel for the CU_HXR lattice.

    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the CU_HXR lattice.
    """

    LCLS_LATTICE = os.environ["LCLS_LATTICE"]
    print(f"LCLS_LATTICE: {LCLS_LATTICE}")

    init_file = os.path.join(LCLS_LATTICE, "bmad/models/cu_hxr/tao.init")

    input_fname = os.path.join(os.path.dirname(__file__), "../bmad/hxr_input.yaml")

    control_vars, control_name_to_bmad = import_control_variables(input_fname)
    output_vars = {} # import_output_variables("hxr_output.yaml")

    # handle OTR2
    control_name_to_bmad["OTRS:IN20:711"] = "OTR4"
    output_vars["OTRS:IN20:711:Image:ArrayData"] = NDVariable(
        name="OTRS:IN20:711:Image:ArrayData",
        unit="",
        read_only=True,
        shape=(1024, 1024),
    )
    screen_attributes = {
        "OTR4": {
            "bins": np.array([1024, 1024]),  # number of pixels in x and y
            "resolution": 0.01,  # mm/pixel
        }
    }  ## TODO replace with correct values

    transformer = CUBmadTransformer(
        control_name_to_bmad=control_name_to_bmad, screen_attributes=screen_attributes
    )

    model = LUMEBmadModel(
        init_file=init_file,
        control_variables=control_vars,
        output_variables=output_vars,
        transformer=transformer,
        dump_locations=["OTR4"],
    )

    model.tao.cmd("set beam_init position_file = $LCLS_LATTICE/bmad/beams/bmad_set_beam2000_pg")

    return model
