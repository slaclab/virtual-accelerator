import numpy as np
import pytest


pytest.importorskip("zfel")

from virtual_accelerator.models.cu_hxr_zfel import (
    get_cu_hxr_zfel_model,
)
from virtual_accelerator.zfel.undulator_mapping import HXR_CELLS


def test_cu_hxr_zfel_model_variables():
    model = get_cu_hxr_zfel_model()

    supported = model.supported_variables

    for cell in HXR_CELLS:
        assert f"KAct_{cell}" in supported
        assert f"DSKAct_{cell}" in supported
        assert f"KAct_{cell}" in supported
        assert f"DSKAct_{cell}" in supported

        expected_pvs = {
            "ZFEL:POWER_MAX",
            "ZFEL:EXIT_POWER",
            "ZFEL:PULSE_ENERGY",
            "GDET:FEE1:361:ENRC",
            "GDET:FEE1:361:ENRCHSTCUHBR",
            "ZFEL:PULSE_INTENSITY_STD_REL",
            "ZFEL:MODEL_EVAL_ID",
        }

    assert expected_pvs.issubset(supported)

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


def test_machine_style_pv_aliases():
    model = get_cu_hxr_zfel_model()

    state = model.get(
        [
            "KAct_14",
            "USEG:UNDH:1450:KAct",
            "pulse_intensity_p80",
            "GDET:FEE1:361:ENRCHSTCUHBR",
        ]
    )

    assert np.isclose(
        state["KAct_14"],
        state["USEG:UNDH:1450:KAct"],
    )

    assert np.isclose(
        state["pulse_intensity_p80"],
        state["GDET:FEE1:361:ENRCHSTCUHBR"],
    )


def test_machine_style_pv_write():
    model = get_cu_hxr_zfel_model()

    target_k = 3.48

    model.set(
        {
            "USEG:UNDH:4750:KAct": target_k,
        }
    )

    state = model.get(
        [
            "KAct_47",
            "USEG:UNDH:4750:KAct",
        ]
    )

    assert np.isclose(state["KAct_47"], target_k)
    assert np.isclose(
        state["USEG:UNDH:4750:KAct"],
        target_k,
    )
