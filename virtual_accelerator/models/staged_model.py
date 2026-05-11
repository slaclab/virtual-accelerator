from lume.staged_model import StagedModel


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

    from virtual_accelerator.surrogates.injector_surrogate import InjectorSurrogate
    from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model

    injector_surrogate = InjectorSurrogate(n_particles=n_particles)
    cu_hxr_bmad_model = get_cu_hxr_bmad_model(
        track_beam=True, start_element="OTR2", **kwargs
    )

    staged_model = StagedModel([injector_surrogate, cu_hxr_bmad_model])

    return staged_model
