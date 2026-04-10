CHEETAH_SUBMODULES = ["diag0", "transformer", "utils", "variables"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "for_every_cheetah_module: run the test for every virtual_accelerator.cheetah submodule",
    )


def pytest_generate_tests(metafunc):
    if metafunc.definition.get_closest_marker("for_every_cheetah_module"):
        metafunc.parametrize("module_name", CHEETAH_SUBMODULES)
