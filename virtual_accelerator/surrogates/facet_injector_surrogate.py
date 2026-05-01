from virtual_accelerator.surrogates.injector_surrogate import BeamOutputWrapper
from virtual_accelerator.utils.optional_dependencies import import_optional_symbol

LUMETorchModel = import_optional_symbol(
    "lume_torch.base",
    "LUMETorchModel",
    feature="FACET-II injector surrogate",
    extra="surrogate",
)
load_model = import_optional_symbol(
    "facet2_inj_ml_model",
    "load_model",
    feature="FACET-II injector surrogate model package",
    extra="surrogate",
)


class FacetInjectorSurrogate(BeamOutputWrapper):
    """LUME wrapper around the FACET-II injector covariance surrogate."""

    def __init__(self, n_particles: int = 10000, p0c: float = 1e8) -> None:
        tm = load_model(use_cpu=True)
        model = LUMETorchModel(tm)
        super().__init__(model, n_particles=n_particles, p0c=p0c)
