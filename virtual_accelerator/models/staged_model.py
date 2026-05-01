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


def get_facet_staged_model(n_particles: int = 1000, **kwargs) -> StagedModel:
    """

    Parameters
    ----------
    n_particles: int
        Number of particles to simulate in the FACET injector surrogate beam output.
    **kwargs:
        Keyword arguments to be passed to the FACET Bmad LUMEModel instance.

    Returns
    -------
    StagedModel
        Instance of the staged FACET injector surrogate followed by the FACET Bmad model.
    """

    from virtual_accelerator.surrogates.facet_injector_surrogate import (
        FacetInjectorSurrogate,
    )
    from virtual_accelerator.models.facet import get_facet_bmad_model

    facet_injector_surrogate = FacetInjectorSurrogate(n_particles=n_particles)
    facet_bmad_model = get_facet_bmad_model(**kwargs)

    staged_model = StagedModel([facet_injector_surrogate, facet_bmad_model])

    return staged_model
