import numpy as np
import pytest


pytest.importorskip("zfel")

from virtual_accelerator.zfel.model import ZFELBackend


def test_zfel_model_initialization():
    model = ZFELBackend()

    state = model.get(
        [
            "KAct",
            "DSKAct",
            "unduK",
            "power_max",
            "exit_power",
            "pulse_energy",
        ]
    )

    assert np.asarray(state["KAct"]).shape == (32,)
    assert np.asarray(state["DSKAct"]).shape == (32,)
    assert np.asarray(state["unduK"]).shape == (320,)

    assert np.isfinite(state["power_max"])
    assert np.isfinite(state["exit_power"])
    assert np.isfinite(state["pulse_energy"])

    assert state["pulse_energy"] > 0.0


def test_setting_k_changes_zfel_result():
    model = ZFELBackend()

    baseline = model.get(
        [
            "KAct",
            "DSKAct",
            "unduK",
            "pulse_energy",
        ]
    )

    changed_kact = np.asarray(
        baseline["KAct"],
        dtype=float,
    ).copy()

    changed_kact[-1] -= 0.02

    model.set(
        {
            "KAct": changed_kact,
            "DSKAct": baseline["DSKAct"],
        }
    )

    changed = model.get(
        [
            "KAct",
            "unduK",
            "pulse_energy",
        ]
    )

    assert np.isclose(
        changed["KAct"][-1],
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
    model = ZFELBackend()

    baseline = model.get(
        [
            "KAct",
            "DSKAct",
            "unduK",
            "pulse_energy",
        ]
    )

    changed_kact = np.asarray(
        baseline["KAct"],
        dtype=float,
    ).copy()

    changed_kact[-1] -= 0.02

    model.set(
        {
            "KAct": changed_kact,
        }
    )

    model.reset()

    reset_state = model.get(
        [
            "KAct",
            "DSKAct",
            "unduK",
            "pulse_energy",
        ]
    )

    assert np.allclose(
        reset_state["KAct"],
        baseline["KAct"],
    )

    assert np.allclose(
        reset_state["DSKAct"],
        baseline["DSKAct"],
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
