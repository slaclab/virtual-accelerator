import sys
import types

from lume.model import LUMEModel
from lume.variables.particle_group import ParticleGroupVariable

from virtual_accelerator.models.staged_model import (
    StagedModel,
    get_facet_staged_model,
)


class FakeFacetInjectorSurrogate(LUMEModel):
    def __init__(self, n_particles: int = 1000):
        super().__init__()
        self.n_particles = n_particles
        self._cache = {}

    def _get(self, names):
        return {name: self._cache.get(name) for name in names}

    def _set(self, values):
        self._cache.update(values)

    def reset(self):
        self._cache = {}

    @property
    def supported_variables(self):
        return {
            "output_beam": ParticleGroupVariable(name="output_beam", read_only=True)
        }


class FakeFacetBmadModel(LUMEModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._cache = {}

    def _get(self, names):
        return {name: self._cache.get(name) for name in names}

    def _set(self, values):
        self._cache.update(values)

    def reset(self):
        self._cache = {}

    @property
    def supported_variables(self):
        return {
            "input_beam": ParticleGroupVariable(name="input_beam"),
        }


def test_get_facet_staged_model_factory(monkeypatch):
    surrogate_module = types.ModuleType(
        "virtual_accelerator.surrogates.facet_injector_surrogate"
    )
    surrogate_module.FacetInjectorSurrogate = FakeFacetInjectorSurrogate
    monkeypatch.setitem(
        sys.modules,
        "virtual_accelerator.surrogates.facet_injector_surrogate",
        surrogate_module,
    )

    facet_module = types.ModuleType("virtual_accelerator.models.facet")

    def get_facet_bmad_model(**kwargs):
        return FakeFacetBmadModel(**kwargs)

    facet_module.get_facet_bmad_model = get_facet_bmad_model
    monkeypatch.setitem(sys.modules, "virtual_accelerator.models.facet", facet_module)

    model = get_facet_staged_model(n_particles=256, end_element="MFFF")

    assert isinstance(model, StagedModel)
    assert len(model.lume_model_instances) == 2
    assert isinstance(model.lume_model_instances[0], FakeFacetInjectorSurrogate)
    assert model.lume_model_instances[0].n_particles == 256
    assert isinstance(model.lume_model_instances[1], FakeFacetBmadModel)
    assert model.lume_model_instances[1].kwargs["end_element"] == "MFFF"