import numpy as np
import pytest


pytest.importorskip("zfel")

from virtual_accelerator.models.cu_hxr_zfel import (
    build_cu_hxr_zfel_runner_config,
    get_cu_hxr_zfel_model,
    get_cu_hxr_zfel_runner,
)
from virtual_accelerator.zfel.undulator_mapping import HXR_CELLS


class DummyRunner:
    """
    Minimal stand-in for lume-pva Runner.

    This allows runner configuration to be tested without
    starting an EPICS server.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.config = config

    @staticmethod
    def generate_config(model, prefix=""):
        return {
            "prefix": prefix,
            "protocol": ["pva"],
            "update_rate": 0.1,
            "variables": {
                name: {
                    "pv": f"{prefix}{name}",
                }
                for name in model.supported_variables
            },
        }


def test_cu_hxr_zfel_model_variables():
    model = get_cu_hxr_zfel_model()

    supported = model.supported_variables

    for cell in HXR_CELLS:
        assert f"KAct_{cell}" in supported
        assert f"DSKAct_{cell}" in supported

    expected_diagnostics = {
        "power_max",
        "exit_power",
        "pulse_energy",
        "pulse_intensity_mean",
        "pulse_intensity_p80",
        "pulse_intensity_std_relative",
    }

    assert expected_diagnostics.issubset(supported)

    state = model.get(
        [
            "KAct_14",
            "DSKAct_14",
            "pulse_energy",
            "pulse_intensity_mean",
            "pulse_intensity_p80",
            "pulse_intensity_std_relative",
        ]
    )

    assert np.isclose(state["KAct_14"], 3.5)
    assert np.isclose(state["DSKAct_14"], 3.5)

    assert state["pulse_energy"] > 0.0

    assert np.isclose(
        state["pulse_intensity_mean"],
        state["pulse_energy"],
    )

    assert np.isclose(
        state["pulse_intensity_p80"],
        state["pulse_energy"],
    )

    assert state["pulse_intensity_std_relative"] == 0.0


def test_scalar_kact_write_updates_zfel_backend():
    model = get_cu_hxr_zfel_model()

    baseline = model.get(
        [
            "KAct_47",
            "pulse_energy",
        ]
    )

    target_k = baseline["KAct_47"] - 0.02

    model.set(
        {
            "KAct_47": target_k,
        }
    )

    changed = model.get(
        [
            "KAct_47",
            "pulse_energy",
        ]
    )

    assert np.isclose(
        changed["KAct_47"],
        target_k,
    )

    assert not np.isclose(
        changed["pulse_energy"],
        baseline["pulse_energy"],
        rtol=1e-8,
        atol=0.0,
    )


def test_runner_config_uses_machine_style_pvs():
    model = get_cu_hxr_zfel_model()

    config = build_cu_hxr_zfel_runner_config(
        DummyRunner,
        model,
        prefix="VA:",
        protocols=("ca",),
        update_rate=0.5,
    )

    assert config["prefix"] == ""
    assert config["protocol"] == ["ca"]
    assert config["update_rate"] == 0.5

    assert config["variables"]["KAct_14"]["pv"] == "VA:USEG:UNDH:1450:KAct"

    assert config["variables"]["DSKAct_14"]["pv"] == "VA:USEG:UNDH:1450:DSKAct"

    assert config["variables"]["pulse_energy"]["pv"] == "VA:ZFEL:PULSE_ENERGY"

    assert config["variables"]["pulse_intensity_mean"]["pv"] == "VA:GDET:FEE1:361:ENRC"

    assert (
        config["variables"]["pulse_intensity_p80"]["pv"]
        == "VA:GDET:FEE1:361:ENRCHSTCUHBR"
    )


def test_zfel_runner_factory_returns_configured_runner():
    runner = get_cu_hxr_zfel_runner(
        DummyRunner,
        protocols=("ca",),
    )

    assert isinstance(runner, DummyRunner)

    assert runner.config["variables"]["KAct_14"]["pv"] == "VA:USEG:UNDH:1450:KAct"

    assert runner.config["variables"]["pulse_energy"]["pv"] == "VA:ZFEL:PULSE_ENERGY"
