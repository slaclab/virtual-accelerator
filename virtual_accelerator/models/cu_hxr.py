import os
import numpy as np
from pytao import Tao

from lume_bmad.model import LUMEBmadModel
from lume.variables import NDVariable
from lume_cheetah import LUMECheetahModel,CheetahSimulator
from virtual_accelerator.cheetah.transformer import SLACCheetahTransformer
from virtual_accelerator.cheetah.variables import get_variables_from_segment
from virtual_accelerator.bmad.variables import get_variables_from_tao
from virtual_accelerator.utils.variables import get_epics_to_name_mapping, split_control_and_observable
from cheetah.accelerator import Segment
from cheetah.particles import ParticleBeam
import torch

from virtual_accelerator.bmad.cu_transformer import (
    CUBmadTransformer,
)


def get_cu_hxr_bmad_model():
    """
    Get the LUMEBmadModel for the CU_HXR lattice from OTR2 to ENDDMPH_2.

    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the CU_HXR lattice.
    """

    LCLS_LATTICE = os.environ["LCLS_LATTICE"]
    init_file = os.path.join(LCLS_LATTICE, "bmad/models/cu_hxr/tao.init")
    tao = Tao(f"-init {init_file} -noplot")

    control_name_to_element_name = get_epics_to_name_mapping()
    variables = get_variables_from_tao(tao)

    # Define the controllable and observable variables
    control_variables, observable_variables = split_control_and_observable(variables)

    # handle OTR4
    control_variables["OTRS:IN20:711:Image:ArrayData"] = NDVariable(
        name="OTRS:IN20:711:Image:ArrayData",
        unit="",
        read_only=True,
        shape=(1024, 1024),
    )
    screen_attributes = {
        "OTR4": {
            "bins": np.array([1024, 1024]),  # number of pixels in x and y
            "resolution": 10,  # um/pixel
        }
    }  ## TODO replace with correct values

    transformer = CUBmadTransformer(
        control_name_to_bmad=control_name_to_element_name, screen_attributes=screen_attributes
    )

    model = LUMEBmadModel(
        tao=tao,
        control_variables=control_variables,
        output_variables=observable_variables,
        transformer=transformer,
        dump_locations=["OTR4"],
    )

    model.tao.cmd("set beam_init position_file = $LCLS_LATTICE/bmad/beams/bmad_set_beam2000_pg")

    return model

def get_cu_hxr_cheetah_model():
    """
    Get the LUMECheetahModel for the CU_HXR lattice from GUN to OTR2.

    Returns
    -------
    LUMECheetahModel
        Instance of the LUMECheetahModel for the CU_HXR lattice.
    """
    # Get path to beam distributions
    # beam_dist = os.environ.get(
    #    'BEAM_DISTRIBUTION',
    #    '/sdf/group/ad/sw/machine-learning/Linac-Simulation-Server/simulation_server/beams'
    # )
    # Create Cheetah particle Beam from file
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
    lcls_lattice = os.environ.get('LCLS_LATTICE')

    # Create lattice from file
    segment = Segment.from_lattice_json(
        os.path.join(lcls_lattice, "cheetah/nc_hxr.json")
    )

    #Set end destination from full lattice
    segment = segment.subcell(end='otr2')

    # Define the simulator using lattice and particle beam
    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=incoming_beam,
    )

    # get control system device to cheetah mapping
    control_name_to_element_name = {k: v.lower() for k,v in get_epics_to_name_mapping().items()}

    # Create transformer that handles maps get/set calls
    transformer = SLACCheetahTransformer(control_name_to_cheetah=control_name_to_element_name)

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