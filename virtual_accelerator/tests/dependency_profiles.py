import importlib.util
import os


def has_module(name: str) -> bool:
    """Return True when an importable module spec is available."""
    return importlib.util.find_spec(name) is not None


HAS_BMAD_DEPS = has_module("pytao") and has_module("lume_bmad")
HAS_CHEETAH_DEPS = has_module("cheetah") and has_module("lume_cheetah")
HAS_SURROGATE_RUNTIME_DEPS = (
    has_module("lume_torch")
    and has_module("torch")
    and has_module("beamphysics")
    and has_module("distgen")
)
HAS_FACET_SURROGATE_DEPS = HAS_SURROGATE_RUNTIME_DEPS and has_module(
    "facet2_inj_ml_model"
)
HAS_INJECTOR_SURROGATE_DEPS = (
    HAS_SURROGATE_RUNTIME_DEPS and HAS_CHEETAH_DEPS and has_module("lcls_cu_inj_model")
)
HAS_SURROGATE_DEPS = HAS_INJECTOR_SURROGATE_DEPS

HAS_STAGED_MODEL_DEPS = (
    HAS_BMAD_DEPS
    and HAS_CHEETAH_DEPS
    and HAS_INJECTOR_SURROGATE_DEPS
    and HAS_FACET_SURROGATE_DEPS
)

HAS_LCLS_LATTICE = bool(os.environ.get("LCLS_LATTICE"))
HAS_FACET2_LATTICE = bool(os.environ.get("FACET2_LATTICE"))
