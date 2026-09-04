from copy import copy
import os

from lume.staged_model import StagedModel

IMPACT_GROUP_PV_MAPPING = {
    "group:L0A_phase": {"pv": "ACCL:IN20:300:L0A_PDES", "element": "L0A_entrance"},
    "group:L0B_phase": {"pv": "ACCL:IN20:400:L0B_PDES", "element": "L0B_entrance"},
    "group:L0A_scale": {
        "pv": "ACCL:IN20:300:L0A_ADES",
        "scale": 1e6,
        "element": "L0A_entrance",
    },
    "group:L0B_scale": {
        "pv": "ACCL:IN20:400:L0B_ADES",
        "scale": 1e6,
        "element": "L0B_entrance",
    },
    "group:GUN_phase": {"pv": "GUN:IN20:1:GN1_PDES", "element": "GUN"},
    "group:GUN_scale": {"pv": "GUN:IN20:1:GN1_ADES", "scale": 1e6, "element": "GUN"},
}


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
        default_beam_relpath="bmad_set_beam2000_pg",
        default_track_start="OTR2",
    )
    return build_bmad_model(
        spec=spec,
        start_element=start_element,
        end_element=end_element,
        track_beam=track_beam,
        custom_beam_path=custom_beam_path,
        custom_tao_commands=[
            "set bmad_com lr_wakes_on=false",
            "set bmad_com sr_wakes_on=false",
        ],
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


def get_cu_hxr_cheetah_model(n_particles: int = 1000):
    """
    Get the LUMECheetahModel for the CU_HXR lattice.

    Returns
    -------
    LUMECheetahModel
        Instance of the LUMECheetahModel for the CU_HXR lattice.
    """
    import torch
    from cheetah.accelerator import Segment
    from cheetah.particles import ParticleBeam
    from lume_cheetah import LUMECheetahModel, CheetahSimulator
    from virtual_accelerator.cheetah.variables import get_variables_from_segment

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
        num_particles=n_particles,
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

    # Define the simulator using lattice and particle beam
    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=incoming_beam,
    )

    # Get supported control system variables
    # for the model
    variables = get_variables_from_segment(segment)

    # Create model using action-based variable integration.
    model = LUMECheetahModel(
        simulator=simulator,
        action_variables=list(variables.values()),
    )

    return model


def get_cu_inj_impact_model(n_particles: int = 100, end_element="OTR2"):
    from virtual_accelerator.impact.factory import (
        ImpactModelSpec,
        build_impact_model,
        get_actions_from_groups,
    )

    spec = ImpactModelSpec(
        lattice_env_var="LCLS_LATTICE",
        distgen_file="distgen/models/cu_inj/v0/distgen.yaml",
        impact_yaml_file="impact/models/cu_inj/v0/ImpactT.yaml",
        profmon_config_filename="cu_hxr_profmon_info.yaml",
        n_particles=n_particles,
        numprocs=1,
        space_charge=False,
        stop_location=end_element,
    )
    model = build_impact_model(spec)

    # register custom actions for linac L0A and L0B sections
    group_actions = get_actions_from_groups(model.impact_model.simulator, spec)

    for action in group_actions:
        old_name = copy(action.name)
        action.name = IMPACT_GROUP_PV_MAPPING[old_name]["pv"]
        action.scale = IMPACT_GROUP_PV_MAPPING[old_name].get("scale", 1.0)

        if (
            IMPACT_GROUP_PV_MAPPING[old_name]["element"]
            in model.impact_model.simulator.ele
        ):
            model.register_impact_action_variable(action)

    return model
