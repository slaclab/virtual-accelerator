import os
from lume_bmad.model import LUMEBmadModel

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

    init_file = os.path.join(LCLS_LATTICE, "bmad/models/cu_hxr/tao_beam.init")

    control_vars, control_name_to_bmad = import_control_variables("hxr_input.yaml")
    #output_vars = import_output_variables("hxr_output.yaml")
    output_vars = {}

    transformer = CUBmadTransformer(control_name_to_bmad=control_name_to_bmad)

    model = LUMEBmadModel(
        init_file=init_file,
        control_variables=control_vars,
        output_variables=output_vars,
        transformer=transformer,
    )

    return model
