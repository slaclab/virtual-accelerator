import importlib
import importlib.util
import os
import sys


def has_module(name: str) -> bool:
    """Return True when a module can be found AND successfully imported.

    Uses :func:`importlib.import_module` to verify the module is actually
    importable, not just that its spec exists on disk.  This handles cases
    where a package is installed but has unresolvable dependencies at import
    time — for example, a third-party package that relies on a standard-library
    module removed in Python 3.13 (``imghdr``, ``cgi``, ``telnetlib``, …).

    Returns False for any module that raises an exception on import so that
    callers can treat it as "not available" and skip the relevant tests rather
    than letting the collection error propagate.
    """
    # Fast path: already imported and cached in sys.modules
    if name in sys.modules:
        return True
    # Fast path: module file not present at all
    if importlib.util.find_spec(name) is None:
        return False
    # Verify the module can actually be imported without errors
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


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
