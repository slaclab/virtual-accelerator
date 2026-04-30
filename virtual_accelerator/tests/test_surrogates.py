import pytest
from lume.model import LUMEModel
import os

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
from lume_torch.variables import TorchNDVariable, TorchScalarVariable
import torch

from virtual_accelerator.surrogates.injector_surrogate import (
    InjectorSurrogate,
    BeamOutputWrapper,
)  # noqa: E402


class FakeInjectorBackend(LUMEModel):
    """Minimal injector backend for deterministic InjectorSurrogate tests."""

    _CONTROL_DEFAULTS = {
        "CAMR:IN20:186:R_DIST": 0.4,
        "Pulse_length": 3.0,
        "FBCK:BCI0:1:CHRG_S": 0.25,
        "SOLN:IN20:121:BCTRL": 0.45,
        "QUAD:IN20:121:BCTRL": -0.01,
        "QUAD:IN20:122:BCTRL": -0.01,
        "ACCL:IN20:300:L0A_ADES": -9.5,
        "ACCL:IN20:300:L0A_PDES": -9.5,
        "ACCL:IN20:400:L0B_ADES": 9.8,
        "ACCL:IN20:400:L0B_PDES": 9.8,
        "QUAD:IN20:361:BCTRL": -2.0,
        "QUAD:IN20:371:BCTRL": 2.0,
        "QUAD:IN20:425:BCTRL": -1.05,
        "QUAD:IN20:441:BCTRL": -0.18,
        "QUAD:IN20:511:BCTRL": 2.8,
        "QUAD:IN20:525:BCTRL": -3.2,
    }

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self._controls = dict(self._CONTROL_DEFAULTS)
        self._refresh_outputs()

    def _refresh_outputs(self):
        quad = float(self._controls["QUAD:IN20:525:BCTRL"])
        solenoid = float(self._controls["SOLN:IN20:121:BCTRL"])
        self._cache = dict(self._controls)
        self._cache.update(
            {
                "OTRS:IN20:571:XRMS": torch.tensor(120.0 + abs(quad) * 12.0),
                "OTRS:IN20:571:YRMS": torch.tensor(115.0 + abs(solenoid) * 20.0),
                "sigma_z": torch.tensor(3.2 + abs(quad) * 0.2),
                "norm_emit_x": torch.tensor(4.5e-7 + abs(quad) * 1.0e-8),
                "norm_emit_y": torch.tensor(5.0e-7 + abs(solenoid) * 5.0e-8),
            }
        )

    def _get(self, names):
        return {name: self._cache[name] for name in names}

    def _set(self, values):
        self._controls.update(values)
        self._refresh_outputs()

    @property
    def supported_variables(self):
        return {
            name: TorchScalarVariable(name=name, unit="", read_only=False)
            for name in self._cache
        }


@pytest.fixture
def fake_injector_backend(monkeypatch):
    backend = FakeInjectorBackend()
    monkeypatch.setattr(InjectorSurrogate, "_load_model", lambda self: backend)
    return backend


def test_injector_surrogate(fake_injector_backend):
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


def test_injector_surrogate_outputs_are_physical(fake_injector_backend):
    "Avoids bugs due to loader/config errors that can be silent."
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


def test_injector_surrogate_loads_latest_registered_model(monkeypatch):
    tracking_uris = []
    model_uris = []
    patched_configs = []
    backend = FakeInjectorBackend()

    monkeypatch.setenv("api_key", "test-api-key")
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.setattr(
        "virtual_accelerator.surrogates.injector_surrogate.mlflow.set_tracking_uri",
        lambda uri: tracking_uris.append(uri),
    )
    monkeypatch.setattr(
        "virtual_accelerator.surrogates.injector_surrogate.mlflow.pyfunc.load_model",
        lambda uri: model_uris.append(uri) or object(),
    )
    monkeypatch.setattr(
        InjectorSurrogate,
        "enable_amsc_x_api_key",
        staticmethod(lambda config: patched_configs.append(config)),
    )
    monkeypatch.setattr(
        InjectorSurrogate,
        "_coerce_model",
        classmethod(lambda cls, _loaded_model: backend),
    )

    surrogate = InjectorSurrogate(
        n_particles=10,
        tracking_uri="https://mlflow.example.org/",
        registered_model_name="custom-injector-model",
        model_version=None,
        mlflow_config={
            "mlflow": {
                "username": "user@example.org",
                "api_key_env": "api_key",
            }
        },
    )

    assert surrogate.model is backend
    assert tracking_uris == ["https://mlflow.example.org/"]
    assert model_uris == ["models:/custom-injector-model/latest"]
    assert patched_configs == [
        {
            "mlflow": {
                "username": "user@example.org",
                "api_key_env": "api_key",
            }
        }
    ]
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "user@example.org"


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
