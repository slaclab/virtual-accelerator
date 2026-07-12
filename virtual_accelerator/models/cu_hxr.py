import os

from lume.staged_model import StagedModel
from virtual_accelerator.utils.optional_dependencies import (
    import_optional,
    import_optional_symbol,
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

    from virtual_accelerator.bmad.factory import BmadModelSpec, build_bmad_model

    spec = BmadModelSpec(
        feature="CU HXR Bmad model",
        lattice_env_var="LCLS_LATTICE",
        tao_init_relpath="bmad/models/cu_hxr/tao.init",
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


def get_cu_hxr_injector_surrogate_model(
    n_particles: int = 1000,
):
    """
    Get the surrogate model for the CU_HXR injector to OTR2.
    Parameters
    ----------
    n_particles: int
        Number of particles to simulate.
    """

    from virtual_accelerator.surrogates.injector_surrogate import InjectorSurrogate

    injector_surrogate = InjectorSurrogate(n_particles=n_particles)
    return injector_surrogate


# get lume model instances for each stage of the accelerator
def get_cu_hxr_staged_model(n_particles: int = 1000, **kwargs) -> StagedModel:
    """

    Parameters
    ----------
    n_particles: int
        Number of particles to simulate.
    **kwargs:
        Keyword arguments to be passed to the bmad LUMEModel instances as needed.

    Returns
    -------
    StagedModel
        Instance of the StagedModel for the CU_HXR lattice.
    """

    injector_surrogate = get_cu_hxr_injector_surrogate_model(n_particles=n_particles)
    cu_hxr_bmad_model = get_cu_hxr_bmad_model(
        track_beam=True, start_element="OTR2", **kwargs
    )

    staged_model = StagedModel([injector_surrogate, cu_hxr_bmad_model])

    return staged_model


def get_cu_hxr_cheetah_model():
    """
    Get the LUMECheetahModel for the CU_HXR lattice from GUN to OTR2.

    Returns
    -------
    LUMECheetahModel
        Instance of the LUMECheetahModel for the CU_HXR lattice.
    """
    LUMECheetahModel = import_optional_symbol(
        "lume_cheetah",
        "LUMECheetahModel",
        feature="CU HXR Cheetah model",
        extra="cheetah",
    )
    CheetahSimulator = import_optional_symbol(
        "lume_cheetah",
        "CheetahSimulator",
        feature="CU HXR Cheetah model",
        extra="cheetah",
    )
    Segment = import_optional_symbol(
        "cheetah.accelerator",
        "Segment",
        feature="CU HXR Cheetah model",
        extra="cheetah",
    )
    ParticleBeam = import_optional_symbol(
        "cheetah.particles",
        "ParticleBeam",
        feature="CU HXR Cheetah model",
        extra="cheetah",
    )
    get_variables_from_segment = import_optional_symbol(
        "virtual_accelerator.cheetah.variables",
        "get_variables_from_segment",
        feature="CU HXR Cheetah model",
        extra="cheetah",
    )
    get_mad_control_mapping = import_optional_symbol(
        "virtual_accelerator.cheetah.utils",
        "get_mad_control_mapping",
        feature="CU HXR Cheetah model",
        extra="cheetah",
    )
    torch = import_optional("torch", feature="CU HXR Cheetah model", extra="cheetah")

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
        num_particles=1000,
        energy=torch.tensor(90e6),
    )
    incoming_beam.particle_charges = torch.tensor(1.0)

    # Get path to lattice files
    lcls_lattice = os.environ.get("LCLS_LATTICE")
    if lcls_lattice is None:
        raise ValueError("LCLS_LATTICE environment variable must be set")

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
    element_name_to_control_name = get_mad_control_mapping(database_path)

    # Get supported control system variables
    # for the model
    variables = get_variables_from_segment(segment, element_name_to_control_name)

    # Create model using action-based variable integration.
    model = LUMECheetahModel(simulator=simulator, action_variables=list(variables.values()))

    return model