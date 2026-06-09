import os

from virtual_accelerator.bmad.factory import BmadModelSpec, build_bmad_model


def get_facet_bmad_model(
    start_element="L0AFEND", end_element="END", track_beam=False, custom_beam_path=None
):
    """
    Get the LUMEBmadModel for the FACET-II lattice from L0AFEND to END.

    Parameters
    -------------
    start_element: str, optional
        The starting element for the model. Default is "L0AFEND".
    end_element: str, optional
        The ending element for the model. Default is "END".
    track_beam: bool, optional
        Whether to enable beam tracking in the model. Default is False.
    custom_beam_path: str, optional
        Path to custom beam file for tracking. If None, will use default design beam. Default is None.


    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the FACET-II lattice.
    """

    spec = BmadModelSpec(
        feature="FACET-II Bmad model",
        lattice_env_var="FACET2_LATTICE",
        tao_init_relpath="bmad/models/f2_elec/tao.init",
        mapping_beampath=None,
        screens=("PR10571", "PR10711"),
        profmon_config_filename="facet2_profmon_info.yaml",
        default_beam_relpath="beams/2024-10-22_oneBunch.h5",
        default_track_start="L0AFEND",
    )
    return build_bmad_model(
        spec=spec,
        start_element=start_element,
        end_element=end_element,
        track_beam=track_beam,
        custom_beam_path=custom_beam_path,
        custom_tao_commands=["set bmad_com absolute_time_tracking=true"],
    )


def get_facet_staged_model(n_particles=10000, surrogate_inputs="machine", **kwargs):
    """
    Get the StagedModel for the FACET-II lattice from PRR10241 to END, with an injector surrogate model.

    Parameters
    -------------
    n_particles: int, optional
        Number of particles to simulate in the surrogate model. Default is 1000.
    surrogate_inputs: str, optional
        Input for the surrogate model either "machine" or "sim". Default is "machine".
    **kwargs:
        Keyword arguments to be passed to the bmad LUMEModel instance as needed.

    Returns
    -------
    StagedModel
        Instance of the StagedModel for the FACET-II lattice.
    """
    from facet2_inj_ml_model import load_model
    from virtual_accelerator.surrogates.beam_output import BeamOutputModel
    from virtual_accelerator.models.facet2 import get_facet_bmad_model
    from lume.staged_model import StagedModel

    injector_surrogate = BeamOutputModel(
        load_model(surrogate_inputs),
        n_particles=n_particles,
        t0=3.15391398e-09,
        p0c=6.3e06,
        z0=0.9420843,
    )

    # need to provide a beam distribution file to initialize the bmad model
    fname = os.getcwd() + "/input_beam.h5"
    injector_surrogate.final_particles.write(fname)

    facet_bmad_model = get_facet_bmad_model(
        start_element="PR10241", track_beam=True, custom_beam_path=fname, **kwargs
    )

    staged_model = StagedModel([injector_surrogate, facet_bmad_model])

    return staged_model
