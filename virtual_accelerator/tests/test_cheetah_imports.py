import importlib
import pkgutil
from pathlib import Path

import pytest


CHEETAH_SUBMODULES = ["diag0", "transformer", "utils", "variables"]


def test_virtual_accelerator_package_import():
    package = importlib.import_module("virtual_accelerator")
    assert package is not None
    assert hasattr(package, "__file__")
    assert Path(package.__file__).exists()


def test_virtual_accelerator_cheetah_package_import():
    cheetah_pkg = importlib.import_module("virtual_accelerator.cheetah")
    assert cheetah_pkg is not None
    assert hasattr(cheetah_pkg, "__path__")
    assert any(Path(path).exists() for path in cheetah_pkg.__path__)


def test_cheetah_submodules_present():
    cheetah_pkg = importlib.import_module("virtual_accelerator.cheetah")
    available = {module.name for module in pkgutil.iter_modules(cheetah_pkg.__path__)}
    assert set(CHEETAH_SUBMODULES).issubset(available)


@pytest.mark.for_every_cheetah_module
def test_cheetah_submodule_import(module_name):
    module = importlib.import_module(f"virtual_accelerator.cheetah.{module_name}")
    assert module.__name__.endswith(f".{module_name}")
    assert hasattr(module, "__file__")
    assert Path(module.__file__).exists()
