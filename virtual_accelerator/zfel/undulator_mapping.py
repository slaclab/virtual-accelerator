from typing import Sequence

import numpy as np


# ------------------------------------------------------------------
# LCLS HXR machine geometry
# ------------------------------------------------------------------

# Physical HXR cell slots covered by the present taper environment.
HXR_SLOTS: tuple[int, ...] = tuple(range(14, 48))

# Active HXR undulator cells.
# This matches the current real-machine Badger environment.
HXR_CELLS: tuple[int, ...] = (
    tuple(range(14, 21))
    + tuple(range(22, 28))
    + tuple(range(29, 48))
)

# Cells omitted from the active-undulator list.
INACTIVE_HXR_CELLS: tuple[int, ...] = (21, 28)

FIRST_HXR_CELL = HXR_SLOTS[0]
LAST_HXR_CELL = HXR_SLOTS[-1]

CELL_LENGTH_M = 4.4
UNDULATOR_LENGTH_M = 3.4
INTERSPACE_LENGTH_M = 1.0

N_ACTIVE_SEGMENTS = len(HXR_CELLS)

PHYSICAL_SPAN_M = len(HXR_SLOTS) * CELL_LENGTH_M
MAGNETIC_LENGTH_M = N_ACTIVE_SEGMENTS * UNDULATOR_LENGTH_M


def _validate_endpoints(
    kact: Sequence[float],
    dskact: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and return the per-segment K endpoints.
    """

    kact_array = np.asarray(kact, dtype=float).reshape(-1)
    dskact_array = np.asarray(dskact, dtype=float).reshape(-1)

    if kact_array.size != N_ACTIVE_SEGMENTS:
        raise ValueError(
            f"kact must contain {N_ACTIVE_SEGMENTS} values, "
            f"got {kact_array.size}."
        )

    if dskact_array.size != N_ACTIVE_SEGMENTS:
        raise ValueError(
            f"dskact must contain {N_ACTIVE_SEGMENTS} values, "
            f"got {dskact_array.size}."
        )

    if not np.all(np.isfinite(kact_array)):
        raise ValueError("kact contains non-finite values.")

    if not np.all(np.isfinite(dskact_array)):
        raise ValueError("dskact contains non-finite values.")

    return kact_array, dskact_array


def build_hxr_mapping(
    kact: Sequence[float],
    dskact: Sequence[float],
    *,
    z_steps: int = 320,
    physical_points_per_cell: int = 100,
) -> dict[str, np.ndarray | float]:
    """
    Build both the physical HXR map and the effective zfel K profile.

    Parameters
    ----------
    kact
        K at the upstream end of each active HXR segment.

    dskact
        K at the downstream end of each active HXR segment.

    z_steps
        Number of longitudinal integration steps for zfel.

        The default is 320:
            32 segments x 10 zfel steps per segment.

    physical_points_per_cell
        Plotting resolution for the real 149.6 m machine layout.

    Returns
    -------
    dict
        Physical machine representation:
            z_physical_m
            k_physical
            physical_cell

        zfel representation:
            z_zfel_m
            k_zfel
            zfel_cell

        Geometry:
            physical_span_m
            magnetic_length_m
    """

    kact_array, dskact_array = _validate_endpoints(
        kact,
        dskact,
    )

    z_steps = int(z_steps)
    physical_points_per_cell = int(physical_points_per_cell)

    if z_steps < N_ACTIVE_SEGMENTS:
        raise ValueError(
            "z_steps must be at least the number of active segments."
        )

    if physical_points_per_cell < 2:
        raise ValueError(
            "physical_points_per_cell must be at least 2."
        )

    # ==============================================================
    # 1. PHYSICAL MACHINE COORDINATE
    #
    # Includes:
    #   - 3.4 m active undulators
    #   - 1.0 m interspaces
    #   - inactive cells 21 and 28
    #
    # NaN is used outside active magnetic sections so plotting does
    # not falsely imply that the drift/interspace has an undulator K.
    # ==============================================================

    n_physical_points = (
        len(HXR_SLOTS) * physical_points_per_cell
    )

    dz_physical = PHYSICAL_SPAN_M / n_physical_points

    z_physical_m = (
        np.arange(n_physical_points, dtype=float) + 0.5
    ) * dz_physical

    physical_cell = (
        FIRST_HXR_CELL
        + np.floor(
            z_physical_m / CELL_LENGTH_M
        ).astype(int)
    )

    cell_start_m = (
        physical_cell - FIRST_HXR_CELL
    ) * CELL_LENGTH_M

    local_z_m = z_physical_m - cell_start_m

    # NaN means: no active undulator field represented here.
    k_physical = np.full(
        n_physical_points,
        np.nan,
        dtype=float,
    )

    for segment_index, cell in enumerate(HXR_CELLS):

        mask = (
            (physical_cell == cell)
            & (local_z_m < UNDULATOR_LENGTH_M)
        )

        fraction = (
            local_z_m[mask] / UNDULATOR_LENGTH_M
        )

        k_physical[mask] = (
            kact_array[segment_index]
            + fraction
            * (
                dskact_array[segment_index]
                - kact_array[segment_index]
            )
        )

    # ==============================================================
    # 2. ZFEL MAGNETIC COORDINATE
    #
    # Current zfel does not explicitly model the machine drifts,
    # interspaces, or chicanes.
    #
    # We therefore concatenate only the 32 active 3.4 m magnetic
    # segments:
    #
    #   total magnetic length = 32 x 3.4 m = 108.8 m
    #
    # Each zfel K value represents the center of one integration step.
    # ==============================================================

    z_edges_m = np.linspace(
        0.0,
        MAGNETIC_LENGTH_M,
        z_steps + 1,
        dtype=float,
    )

    z_zfel_m = 0.5 * (
        z_edges_m[:-1] + z_edges_m[1:]
    )

    segment_index = np.floor(
        z_zfel_m / UNDULATOR_LENGTH_M
    ).astype(int)

    segment_index = np.clip(
        segment_index,
        0,
        N_ACTIVE_SEGMENTS - 1,
    )

    segment_start_m = (
        segment_index * UNDULATOR_LENGTH_M
    )

    local_fraction = (
        z_zfel_m - segment_start_m
    ) / UNDULATOR_LENGTH_M

    k_zfel = (
        kact_array[segment_index]
        + local_fraction
        * (
            dskact_array[segment_index]
            - kact_array[segment_index]
        )
    )

    hxr_cells_array = np.asarray(
        HXR_CELLS,
        dtype=int,
    )

    zfel_cell = hxr_cells_array[segment_index]

    return {
        "z_physical_m": z_physical_m,
        "k_physical": k_physical,
        "physical_cell": physical_cell,
        "z_zfel_m": z_zfel_m,
        "k_zfel": k_zfel,
        "zfel_cell": zfel_cell,
        "physical_span_m": PHYSICAL_SPAN_M,
        "magnetic_length_m": MAGNETIC_LENGTH_M,
    }