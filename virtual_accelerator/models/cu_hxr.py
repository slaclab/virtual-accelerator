import os
from pathlib import Path
from pytao import Tao

from lume_bmad.model import LUMEBmadModel
from lume_cheetah import LUMECheetahModel, CheetahSimulator
from virtual_accelerator.cheetah.transformer import SLACCheetahTransformer
from virtual_accelerator.cheetah.variables import get_variables_from_segment
from virtual_accelerator.bmad.variables import (
    get_variables,
    get_cu_hxr_screen_variables,
)
from virtual_accelerator.utils.variables import (
    get_epics_to_name_or_overlay_mapping,
    get_epics_to_name_mapping,
    split_control_and_observable,
)
from cheetah.accelerator import Segment
from cheetah.particles import ParticleBeam
import torch

from virtual_accelerator.bmad.cu_transformer import (
    CUBmadTransformer,
)


def get_cu_hxr_bmad_model(
    start_element="OTR2", end_element="END", track_beam=False, custom_beam_path=None
):
    """
    Get the LUMEBmadModel for the CU_HXR lattice from OTR2 to END.

    Parameters
    -------------
    start_element: str, optional
        The starting element for the model. Default is "OTR2".
    end_element: str, optional
        The ending element for the model. Default is "END".
    track_beam: bool, optional
        Whether to enable beam tracking in the model. Default is False.
    custom_beam_path: str, optional
        Path to custom beam file for tracking. If None, will use default design beam. Default is None.


    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the CU_HXR lattice.
    """

    LCLS_LATTICE = os.environ["LCLS_LATTICE"]
    init_file = os.path.join(LCLS_LATTICE, "bmad/models/cu_hxr/tao.init")
    tao = Tao(f"-init {init_file} -noplot -slice_lattice {start_element}:{end_element}")

    # get supported variables from tao lattice and get mapping from control
    # system device names to bmad element names
    control_name_to_element_name = get_epics_to_name_or_overlay_mapping()
    variables = get_variables(tao)

    # Define the controllable and observable variables
    control_variables, observable_variables = split_control_and_observable(variables)
    # handle Profile Monitors
    screens = ["OTR3", "OTR4", "OTR11", "OTR12", "OTR21"]
    control_variables, screen_attributes, used_screens = get_cu_hxr_screen_variables(
        tao, control_variables, screens
    )

    transformer = CUBmadTransformer(
        control_name_to_bmad=control_name_to_element_name,
        screen_attributes=screen_attributes,
    )

    model = LUMEBmadModel(
        tao=tao,
        control_variables=control_variables,
        output_variables=observable_variables,
        transformer=transformer,
        dump_locations=used_screens,
    )

    if track_beam:
        if start_element == "OTR2" and custom_beam_path is None:
            beam_path = os.path.join(
                Path(__file__).parent, "../bmad", "bmad_set_beam2000_pg"
            )
        elif custom_beam_path is not None:
            beam_path = custom_beam_path
        else:
            raise ValueError(
                "Cannot have track_beam=True for start_element != OTR2 without providing custom_beam_path"
            )

        model.tao.cmd(f"set beam_init position_file = {beam_path}")
        model.set({"track_type": 1})

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
    #    '/sdf/group/ad/sw/machine-learning/
    # Linac-Simulation-Server/simulation_server/beams'
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
    lcls_lattice = os.environ.get("LCLS_LATTICE")

    # Create lattice from file
    segment = Segment.from_lattice_json(
        os.path.join(lcls_lattice, "cheetah/nc_hxr.json")
    )

    # Set end destination from full lattice
    segment = segment.subcell(end="otr2")

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
