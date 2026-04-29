import pytest
from lume.model import LUMEModel

# Skip entire module at collection time when lume-torch is absent — avoids
# an ImportError inside injector_surrogate.py before any skip logic fires.
pytest.importorskip(
    "lume_torch",
    reason="requires lume-torch: pip install virtual-accelerator[surrogate]",
)
pytest.importorskip(
    "cheetah",
    reason="requires surrogate optional dependencies: pip install virtual-accelerator[surrogate]",
)
from lume_torch.variables import TorchNDVariable
import torch

from virtual_accelerator.surrogates.injector_surrogate import (
    InjectorSurrogate,
    BeamOutputWrapper,
)  # noqa: E402


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


TEST_COVARIANCE_MATRIX = torch.diag(
    torch.tensor([1.0e-3, 1.0e5, 1.0e-3, 1.0e5, 1.0e-3, 1.0e5], dtype=torch.float32)
)


class TestLumeModel(LUMEModel):
    """Minimal test model exposing a 6x6 covariance matrix variable."""

    def __init__(self):
        super().__init__()

        self._cache = {"covariance_matrix": TEST_COVARIANCE_MATRIX.clone()}

    def _get(self, names):
        return {name: self._cache[name] for name in names}

    def _set(self, values):
        self._cache.update(values)

    def reset(self):
        self._cache = {"covariance_matrix": TEST_COVARIANCE_MATRIX.clone()}

    @property
    def supported_variables(self):
        return {
            "covariance_matrix": TorchNDVariable(
                name="covariance_matrix", unit="", shape=(6, 6)
            )
        }


def test_beam_output_wrapper():
    surrogate = TestLumeModel()
    wrapped = BeamOutputWrapper(surrogate, n_particles=1000000, p0c=1e8)

    output = wrapped.get(["output_beam", "covariance_matrix"])
    assert "output_beam" in output
    assert output["covariance_matrix"].shape == (6, 6)
    assert output["output_beam"].x.shape[0] == 1000000

    # check that the covariance matrix is being converted to cheetah units correctly
    cov = torch.tensor(
        output["output_beam"].cov("x", "px", "y", "py", "z", "pz")
    ).float()
    assert torch.allclose(cov, TEST_COVARIANCE_MATRIX, atol=1e3, rtol=1e-2)
