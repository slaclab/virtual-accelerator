import numpy as np

from virtual_accelerator.zfel.undulator_mapping import (
    HXR_CELLS,
    build_hxr_mapping,
)


def test_hxr_cells_have_expected_layout():
    expected_cells = tuple(
        list(range(14, 21))
        + list(range(22, 28))
        + list(range(29, 48))
    )

    assert HXR_CELLS == expected_cells
    assert len(HXR_CELLS) == 32

    # Bypass-chicane locations are not active undulator segments.
    assert 21 not in HXR_CELLS
    assert 28 not in HXR_CELLS


def test_constant_k_maps_to_expected_zfel_profile():
    kact = np.full(len(HXR_CELLS), 3.5)
    dskact = np.full(len(HXR_CELLS), 3.5)

    mapping = build_hxr_mapping(
        kact,
        dskact,
        z_steps=320,
    )

    k_zfel = np.asarray(mapping["k_zfel"])

    assert k_zfel.shape == (320,)
    assert np.isfinite(k_zfel).all()
    assert np.allclose(k_zfel, 3.5)


def test_changing_segment_changes_zfel_profile():
    kact = np.full(len(HXR_CELLS), 3.5)
    dskact = np.full(len(HXR_CELLS), 3.5)

    baseline = build_hxr_mapping(
        kact,
        dskact,
        z_steps=320,
    )

    changed_kact = kact.copy()
    changed_dskact = dskact.copy()

    changed_kact[-1] -= 0.02
    changed_dskact[-1] -= 0.02

    changed = build_hxr_mapping(
        changed_kact,
        changed_dskact,
        z_steps=320,
    )

    baseline_k = np.asarray(baseline["k_zfel"])
    changed_k = np.asarray(changed["k_zfel"])

    assert baseline_k.shape == changed_k.shape
    assert not np.allclose(changed_k, baseline_k)
    assert np.min(changed_k) < np.min(baseline_k)