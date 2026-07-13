import pytest
from unittest.mock import Mock
from scipy import constants
import yaml
from virtual_accelerator.tests.dependency_profiles import HAS_INJECTOR_SURROGATE_DEPS

pytestmark = [
    pytest.mark.requires_surrogate,
]

if HAS_INJECTOR_SURROGATE_DEPS:
    from lume_torch.variables import TorchNDVariable, TorchScalarVariable
    from lume_torch.models.torch_model import TorchModel
    import torch

    from virtual_accelerator.surrogates.augmentations import (
        BCTRLFamilyAugmentation,
        InjectorCovarianceAugmentation,
        load_augmentations_from_yaml,
    )
    from virtual_accelerator.surrogates.injector_surrogate import (
        InjectorSurrogate,
        compute_covariance_matrix,
    )
    from virtual_accelerator.surrogates.beam_output import BeamOutputModel

    TEST_COVARIANCE_MATRIX = torch.diag(
        torch.tensor([1.0e-3, 1.0e5, 1.0e-3, 1.0e5, 1.0e-3, 1.0e5], dtype=torch.float32)
    )
else:
    pytest.skip(
        "requires surrogate optional dependencies: pip install virtual-accelerator[surrogate]",
        allow_module_level=True,
    )


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

    # try getting all of the other PVs
    all_pvs = surrogate.get(surrogate.supported_variables.keys())
    for suffix in [
        "BCTRL",
        "BACT",
        "BMIN",
        "BMAX",
        "CTRL",
        "STATCTRLSUB.T",
        "BCTRL.DRVL",
        "BCTRL.DRVH",
    ]:
        assert all_pvs[f"QUAD:IN20:525:{suffix}"] is not None


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


def make_dummy_torch_model(include_bctrl: bool = False):
    """Return a minimal TorchModel-like object for BeamOutputWrapper tests."""
    model = Mock(spec=TorchModel)
    model.input_variables = []
    model.input_names = []

    if include_bctrl:
        model.input_variables.append(
            TorchScalarVariable(
                name="QUAD:IN20:525:BCTRL", unit="kG", default_value=0.0
            )
        )
        model.input_names.append("QUAD:IN20:525:BCTRL")

    model.output_variables = [
        TorchNDVariable(name="covariance_matrix", unit="", shape=(6, 6))
    ]
    model.evaluate.return_value = {"covariance_matrix": TEST_COVARIANCE_MATRIX.clone()}
    return model


def make_injector_scalar_dummy_torch_model():
    model = Mock(spec=TorchModel)
    model.input_variables = []
    model.input_names = []
    model.output_variables = [
        TorchScalarVariable(name="OTRS:IN20:571:XRMS", unit="um"),
        TorchScalarVariable(name="OTRS:IN20:571:YRMS", unit="um"),
        TorchScalarVariable(name="sigma_z", unit="um"),
        TorchScalarVariable(name="norm_emit_x", unit="m"),
        TorchScalarVariable(name="norm_emit_y", unit="m"),
    ]
    model.evaluate.return_value = {
        "OTRS:IN20:571:XRMS": torch.tensor(50.0),
        "OTRS:IN20:571:YRMS": torch.tensor(75.0),
        "sigma_z": torch.tensor(30.0),
        "norm_emit_x": torch.tensor(2.0e-6),
        "norm_emit_y": torch.tensor(3.0e-6),
    }
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


def test_beam_output_model_accepts_singleton_batched_covariance():
    surrogate = make_dummy_torch_model()
    surrogate.evaluate.return_value = {
        "covariance_matrix": TEST_COVARIANCE_MATRIX.clone().unsqueeze(0)
    }

    wrapped = BeamOutputModel(surrogate, n_particles=100000, p0c=1e8)
    beam = wrapped.final_particles

    test_matrix = TEST_COVARIANCE_MATRIX.clone()
    test_matrix[4, :] *= -constants.speed_of_light
    test_matrix[:, 4] *= -constants.speed_of_light

    cov = torch.tensor(beam.cov("x", "px", "y", "py", "z", "pz")).float()
    assert torch.allclose(cov, test_matrix, atol=1e-3, rtol=1e-3)


def test_beam_output_model_bctrl_family_augmentation_round_trip():
    surrogate = make_dummy_torch_model(include_bctrl=True)
    wrapped = BeamOutputModel(
        surrogate,
        n_particles=10000,
        p0c=1e8,
        augmentations=[BCTRLFamilyAugmentation()],
    )

    wrapped.set({"QUAD:IN20:525:BDES": -5.0})

    assert wrapped.get("QUAD:IN20:525:BCTRL") == -5.0
    assert wrapped.get("QUAD:IN20:525:BACT") == -5.0
    assert wrapped.get("QUAD:IN20:525:BMIN") == -100.0
    assert wrapped.get("QUAD:IN20:525:BMAX") == 100.0
    assert wrapped.get("QUAD:IN20:525:CTRL") == "Ready"
    assert wrapped.get("QUAD:IN20:525:STATCTRLSUB.T") == "Ready"


def test_injector_covariance_augmentation_derives_covariance_matrix():
    surrogate = make_injector_scalar_dummy_torch_model()
    augmentation = InjectorCovarianceAugmentation(energy_eV=135.0e6)
    wrapped = BeamOutputModel(
        surrogate,
        n_particles=1000,
        p0c=135.0e6,
        augmentations=[augmentation],
    )

    covariance_matrix = torch.as_tensor(wrapped.get("covariance_matrix"))
    expected_covariance_matrix = torch.from_numpy(
        compute_covariance_matrix(wrapped._cache, energy=135.0e6)
    )

    assert covariance_matrix.shape == (6, 6)
    assert torch.allclose(covariance_matrix, expected_covariance_matrix)


def test_load_surrogate_augmentations_from_yaml(tmp_path):
    config = {
        "schema_version": 1,
        "augmentations": [
            {"type": "bctrl_family"},
            {"type": "injector_covariance", "energy_eV": 135.0e6},
        ],
    }
    config_path = tmp_path / "augmentations.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    augmentations = load_augmentations_from_yaml(config_path)

    assert len(augmentations) == 2
    assert isinstance(augmentations[0], BCTRLFamilyAugmentation)
    assert isinstance(augmentations[1], InjectorCovarianceAugmentation)
