import importlib.util
import pytest

# Skip entire module at collection time when lume_torch is absent — avoids
# an ImportError inside injector_surrogate.py before any skip logic fires.
pytest.importorskip(
    "lume_torch",
    reason="requires lume-torch: pip install virtual-accelerator[surrogate]",
)

from virtual_accelerator.surrogates.injector_surrogate import InjectorSurrogate  # noqa: E402


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


pytestmark = pytest.mark.skipif(
    not _has_module("cheetah"),
    reason="requires surrogate optional dependencies: pip install virtual-accelerator[surrogate,cheetah]",
)


def test_injector_surrogate():
    # test to make sure that the surrogate can be
    # initialized and returns an output beam distribution
    surrogate = InjectorSurrogate(n_particles=1000)
    output = surrogate.get(["output_beam"])
    assert "output_beam" in output
    beam = output["output_beam"]
    assert beam.x.shape[0] == 1000

    # check to make sure that changing a control variable changes
    # the output beam distribution
    initial_beam = surrogate.get(["output_beam"])["output_beam"]
    surrogate.set({"QUAD:IN20:525:BCTRL": -5.0})
    updated_beam = surrogate.get(["output_beam"])["output_beam"]
    assert not (initial_beam.x == updated_beam.x).all()
    assert surrogate.get(["QUAD:IN20:525:BCTRL"])["QUAD:IN20:525:BCTRL"] == -5.0


def test_injector_surrogate_outputs_are_physical():
    "Avoids bugs due to YAML/loading errors that can be silent"
    surrogate = InjectorSurrogate(n_particles=1000)

    outputs = surrogate.get(
        [
            "OTRS:IN20:571:XRMS",
            "OTRS:IN20:571:YRMS",
            "sigma_z",
            "norm_emit_x",
            "norm_emit_y",
        ]
    )

    xrms = outputs["OTRS:IN20:571:XRMS"]
    yrms = outputs["OTRS:IN20:571:YRMS"]
    sigma_z = outputs["sigma_z"]
    norm_emit_x = outputs["norm_emit_x"]
    norm_emit_y = outputs["norm_emit_y"]

    assert 0.0 < xrms < 1.0e4
    assert 0.0 < yrms < 1.0e4
    assert 0.0 < sigma_z < 1.0e2
    assert 0.0 < norm_emit_x < 1.0e-3
    assert 0.0 < norm_emit_y < 1.0e-3
