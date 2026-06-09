import pytest
from unittest.mock import Mock
from scipy import constants

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
pytest.importorskip(
    "lcls_cu_inj_model",
    reason="requires packaged Cu injector model: pip install virtual-accelerator[surrogate]",
)
from lume_torch.variables import TorchNDVariable
from lume_torch.models.torch_model import TorchModel
import torch

from virtual_accelerator.surrogates.injector_surrogate import (
    InjectorSurrogate,
)  # noqa: E402
from virtual_accelerator.surrogates.beam_output import BeamOutputModel


def test_injector_surrogate():
    # test to make sure that the surrogate can be
    # initialized and returns an output beam distribution
    surrogate = InjectorSurrogate(n_particles=1000)
    beam = surrogate.final_particles
    assert beam.x.shape[0] == 1000

    # check to make sure that changing a control variable changes
    # the output beam distribution
    initial_beam = surrogate.final_particles
    surrogate.set({"QUAD:IN20:525:BCTRL": -5.0})
    updated_beam = surrogate.final_particles
    assert not (initial_beam.x == updated_beam.x).all()
    assert surrogate.get("QUAD:IN20:525:BCTRL") == -5.0


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


def make_dummy_torch_model() -> TorchModel:
    """Return a minimal TorchModel-like object for BeamOutputWrapper tests."""
    model = Mock(spec=TorchModel)
    model.input_variables = []
    model.output_variables = [
        TorchNDVariable(name="covariance_matrix", unit="", shape=(6, 6))
    ]
    model.input_names = []
    model.evaluate.return_value = {"covariance_matrix": TEST_COVARIANCE_MATRIX.clone()}
    return model


def test_beam_output_model():
    surrogate = make_dummy_torch_model()
    wrapped = BeamOutputModel(surrogate, n_particles=1000000, p0c=1e8)

    output = wrapped.get(["covariance_matrix"])
    beam = wrapped.final_particles
    assert output["covariance_matrix"].shape == (6, 6)
    assert beam.x.shape[0] == 1000000

    # convert the covariance matrix time to z using speed of light
    test_matrix = TEST_COVARIANCE_MATRIX.clone()
    test_matrix[4, :] *= -constants.speed_of_light
    test_matrix[:, 4] *= -constants.speed_of_light

    # check that the covariance matrix is being converted to cheetah units correctly
    cov = torch.tensor(beam.cov("x", "px", "y", "py", "z", "pz")).float()
    assert torch.allclose(cov, test_matrix, atol=1e-3, rtol=1e-3)
