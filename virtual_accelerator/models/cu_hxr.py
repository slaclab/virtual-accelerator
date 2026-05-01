import os

from virtual_accelerator.bmad.factory import BmadModelSpec, build_bmad_model
from virtual_accelerator.utils.optional_dependencies import import_optional
from virtual_accelerator.utils.variables import (
    get_epics_to_name_or_overlay_mapping,
    split_control_and_observable,
)


def _check_optional_modules(module_names: list[str], feature: str, extra: str) -> None:
    """Validate all optional modules for a feature in a single gate check."""
    for module_name in module_names:
        import_optional(module_name, feature=feature, extra=extra)


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

    spec = BmadModelSpec(
        feature="CU HXR Bmad model",
        lattice_env_var="LCLS_LATTICE",
        tao_init_relpath="bmad/models/cu_hxr/tao.init",
        screens=("OTR3", "OTR4", "OTR11", "OTR12", "OTR21"),
        profmon_config_filename="cu_hxr_profmon_info.yaml",
        default_beam_relpath="bmad/bmad_set_beam2000_pg",
        default_track_start="OTR2",
    )
    return build_bmad_model(
        spec=spec,
        start_element=start_element,
        end_element=end_element,
        track_beam=track_beam,
        custom_beam_path=custom_beam_path,
    )


def get_cu_hxr_cheetah_model():
    """
    Get the LUMECheetahModel for the CU_HXR lattice from GUN to OTR2.

    Returns
    -------
    LUMECheetahModel
        Instance of the LUMECheetahModel for the CU_HXR lattice.
    """
    _check_optional_modules(
        [
            "lume_cheetah",
            "cheetah.accelerator",
            "cheetah.particles",
            "virtual_accelerator.cheetah.transformer",
            "virtual_accelerator.cheetah.variables",
        ],
        feature="CU HXR Cheetah model",
        extra="cheetah",
    )

    from lume_cheetah import LUMECheetahModel, CheetahSimulator
    from cheetah.accelerator import Segment
    from cheetah.particles import ParticleBeam
    from virtual_accelerator.cheetah.transformer import SLACCheetahTransformer
    from virtual_accelerator.cheetah.variables import get_variables_from_segment
    import torch

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
    database_path = os.path.join(
        lcls_lattice, "bmad/conversion/from_oracle/lcls_elements.csv"
    )
    control_name_to_element_name = get_epics_to_name_or_overlay_mapping(database_path)
    element_name_to_control_name = {
        v: k for k, v in control_name_to_element_name.items()
    }

    # Create transformer that handles maps get/set calls
    transformer = SLACCheetahTransformer(
        control_name_to_cheetah=control_name_to_element_name
    )

    # Get supported control system variables
    # for the model
    variables = get_variables_from_segment(segment, element_name_to_control_name)

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
