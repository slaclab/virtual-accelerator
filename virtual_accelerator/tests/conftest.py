import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

try:
    from virtual_accelerator.models.cu_hxr import get_cu_hxr_cheetah_model
except ModuleNotFoundError:
    get_cu_hxr_cheetah_model = None

try:
    from virtual_accelerator.models.sc_diag0 import get_sc_diag0_cheetah_model
except ModuleNotFoundError:
    get_sc_diag0_cheetah_model = None

CHEETAH_SUBMODULES = ["diag0", "transformer", "utils", "variables"]
CHEETAH_MODEL_FACTORIES = []
if get_cu_hxr_cheetah_model is not None:
    CHEETAH_MODEL_FACTORIES.append(("cu_hxr", get_cu_hxr_cheetah_model))
if get_sc_diag0_cheetah_model is not None:
    CHEETAH_MODEL_FACTORIES.append(("sc_diag0", get_sc_diag0_cheetah_model))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "for_every_cheetah_module: run the test for every virtual_accelerator.cheetah submodule",
    )
    config.addinivalue_line(
        "markers",
        "for_every_cheetah_model: run the test for every Cheetah-based virtual accelerator model",
    )


def pytest_generate_tests(metafunc):
    if metafunc.definition.get_closest_marker("for_every_cheetah_module"):
        metafunc.parametrize("module_name", CHEETAH_SUBMODULES)
    if metafunc.definition.get_closest_marker("for_every_cheetah_model"):
        if not CHEETAH_MODEL_FACTORIES:
            pytest.skip("No Cheetah model factories available in this environment")
        metafunc.parametrize(
            "model_name,model_factory",
            CHEETAH_MODEL_FACTORIES,
            ids=[name for name, _ in CHEETAH_MODEL_FACTORIES],
        )
