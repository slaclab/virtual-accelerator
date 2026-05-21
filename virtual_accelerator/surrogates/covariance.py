from typing import Any, Mapping

import numpy as np
from scipy import constants


def compute_covariance_matrix(state: Mapping[str, Any], energy: float) -> np.ndarray:
    """Compute a diagonal 6x6 covariance matrix from scalar beam parameters.

    The matrix is in trace space (x, x', y, y', z, z') with no off-diagonal terms.

    Parameters
    ----------
    state : dict
        Model output state containing XRMS, YRMS, sigma_z, norm_emit_x, norm_emit_y.
    energy : float
        Beam energy in eV, used to convert normalized emittance to geometric.

    Returns
    -------
    np.ndarray
        6x6 diagonal covariance matrix.
    """
    sigma_x = state["OTRS:IN20:571:XRMS"] * 1e-6  # microns -> meters
    sigma_y = state["OTRS:IN20:571:YRMS"] * 1e-6
    sigma_z = state["sigma_z"] * 1e-6

    relativistic_gamma = energy / (
        constants.value("electron mass energy equivalent in MeV") * 1e6
    )
    emit_x = state["norm_emit_x"] / relativistic_gamma  # geometric emittance
    emit_y = state["norm_emit_y"] / relativistic_gamma

    cov = np.zeros((6, 6))
    cov[0, 0] = sigma_x**2
    cov[2, 2] = sigma_y**2
    cov[1, 1] = emit_x / cov[0, 0]
    cov[3, 3] = emit_y / cov[2, 2]
    cov[4, 4] = sigma_z**2
    # cov[5, 5] is left as zero — energy spread not available from model
    return cov
