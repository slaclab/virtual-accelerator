import os
from pathlib import Path
from lume_cheetah import LUMECheetahModel, CheetahSimulator
from virtual_accelerator.cheetah.transformer import SLACCheetahTransformer
from virtual_accelerator.cheetah.variables import get_variables_from_segment
from virtual_accelerator.utils.variables import (
    get_epics_to_name_mapping,
    split_control_and_observable
)
from cheetah.accelerator import Segment
from cheetah.particles import ParticleBeam
import torch


def get_sc_diag0_cheetah_model():
    """
    Get the LUMECheetahModel for the SC_DIAG0 lattice from BEAM0 to OTRDG02.

    Returns
    -------
    LUMECheetahModel
        Instance of the LUMECheetahModel for the SC_DIAG0 lattice.
    """

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

    # Define the controllable and observable variables
    control_variables, observable_variables = split_control_and_observable(variables)

    # Create model
    model = LUMECheetahModel(
        simulator=simulator,
        transformer=transformer,
        control_variables=control_variables,
        observable_variables=observable_variables,
    )

    return model