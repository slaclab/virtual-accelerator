import os

from virtual_accelerator.utils.variables import (
    get_epics_to_name_mapping,
    split_control_and_observable,
    convert_to_torch_variables,
)
import torch

from ..utils.optional_dependencies import import_optional_symbol


def get_sc_diag0_cheetah_model():
    """
    Get the LUMECheetahModel for the SC_DIAG0 lattice from BEAM0 to OTRDG02.

    Returns
    -------
    LUMECheetahModel
        Instance of the LUMECheetahModel for the SC_DIAG0 lattice.
    """

    LUMECheetahModel = import_optional_symbol(
        "lume_cheetah",
        "LUMECheetahModel",
        feature="SC DIAG0 Cheetah model",
        extra="cheetah",
    )
    CheetahSimulator = import_optional_symbol(
        "lume_cheetah",
        "CheetahSimulator",
        feature="SC DIAG0 Cheetah model",
        extra="cheetah",
    )
    Segment = import_optional_symbol(
        "cheetah.accelerator",
        "Segment",
        feature="SC DIAG0 Cheetah model",
        extra="cheetah",
    )
    ParticleBeam = import_optional_symbol(
        "cheetah.particles",
        "ParticleBeam",
        feature="SC DIAG0 Cheetah model",
        extra="cheetah",
    )
    from virtual_accelerator.cheetah.transformer import SLACCheetahTransformer
    from virtual_accelerator.cheetah.variables import get_variables_from_segment

    incoming_beam = ParticleBeam.from_twiss(
        beta_x=torch.tensor(9.34),
        alpha_x=torch.tensor(-1.6946),
        emittance_x=torch.tensor(1e-7),
        beta_y=torch.tensor(9.34),
        alpha_y=torch.tensor(-1.6946),
        emittance_y=torch.tensor(1e-7),
        energy=torch.tensor(90e6),
    )
    incoming_beam.particle_charges = torch.tensor(1.0)

    # Get path to lattice files
    lcls_lattice = os.environ.get("LCLS_LATTICE")

    # Create lattice from file
    segment = Segment.from_lattice_json(
        os.path.join(lcls_lattice, "cheetah/sc_diag0.json")
    )

    # Ensure screen elements can support vectorization
    for element in segment.elements:
        element_type = type(element).__name__
        if element_type == "Screen":
            element.method = "kde"

    # Define the simulator using lattice and particle beam
    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=incoming_beam,
    )

    # get control system device to cheetah mapping
    control_name_to_element_name = {
        k: v.lower() for k, v in get_epics_to_name_mapping().items()
    }

    # Create transformer that handles maps get/set calls
    transformer = SLACCheetahTransformer(
        control_name_to_cheetah=control_name_to_element_name
    )

    # Get supported control system variables
    # for the model
    variables = get_variables_from_segment(segment)
    torch_variables = convert_to_torch_variables(variables)
    # Define the controllable and observable variables
    control_variables, observable_variables = split_control_and_observable(
        torch_variables
    )

    # Create model
    model = LUMECheetahModel(
        simulator=simulator,
        transformer=transformer,
        control_variables=control_variables,
        observable_variables=observable_variables,
    )

    return model
