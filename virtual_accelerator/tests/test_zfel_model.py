import numpy as np
import pytest


pytest.importorskip("zfel")

from virtual_accelerator.zfel.model import ZFELModel


def test_zfel_model_initialization():
    model = ZFELModel()

    state = model.get(
        [
            "Kact",
            "DSKact",
            "unduK",
            "power_max",
            "exit_power",
            "pulse_energy",
        ]
    )

    assert np.asarray(state["Kact"]).shape == (32,)
    assert np.asarray(state["DSKact"]).shape == (32,)
    assert np.asarray(state["unduK"]).shape == (320,)

    assert np.isfinite(state["power_max"])
    assert np.isfinite(state["exit_power"])
    assert np.isfinite(state["pulse_energy"])

    assert state["pulse_energy"] > 0.0


def test_setting_k_changes_zfel_result():
    model = ZFELModel()

    baseline = model.get(
        [
            "Kact",
            "DSKact",
            "unduK",
            "pulse_energy",
        ]
    )

    changed_kact = np.asarray(
        baseline["Kact"],
        dtype=float,
    ).copy()

    changed_kact[-1] -= 0.02

    model.set(
        {
            "Kact": changed_kact,
            "DSKact": baseline["DSKact"],
        }
    )

    changed = model.get(
        [
            "Kact",
            "unduK",
            "pulse_energy",
        ]
    )

    assert np.isclose(
        changed["Kact"][-1],
        changed_kact[-1],
    )

    assert not np.allclose(
        changed["unduK"],
        baseline["unduK"],
    )

    assert not np.isclose(
        changed["pulse_energy"],
        baseline["pulse_energy"],
        rtol=1e-8,
        atol=0.0,
    )


def test_reset_restores_initial_state():
    model = ZFELModel()

    baseline = model.get(
        [
            "Kact",
            "DSKact",
            "unduK",
            "pulse_energy",
        ]
    )

    changed_kact = np.asarray(
        baseline["Kact"],
        dtype=float,
    ).copy()

    changed_kact[-1] -= 0.02

    model.set(
        {
            "Kact": changed_kact,
        }
    )

    model.reset()

    reset_state = model.get(
        [
            "Kact",
            "DSKact",
            "unduK",
            "pulse_energy",
        ]
    )

    assert np.allclose(
        reset_state["Kact"],
        baseline["Kact"],
    )

    assert np.allclose(
        reset_state["DSKact"],
        baseline["DSKact"],
    )

    assert np.allclose(
        reset_state["unduK"],
        baseline["unduK"],
    )

    assert np.isclose(
        reset_state["pulse_energy"],
        baseline["pulse_energy"],
        rtol=1e-10,
        atol=0.0,
    )
