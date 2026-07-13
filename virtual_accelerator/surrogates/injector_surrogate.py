from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from lcls_cu_inj_model import load_model

from virtual_accelerator.surrogates.augmentations import (
    BCTRLFamilyAugmentation,
    InjectorCovarianceAugmentation,
    compute_injector_covariance_matrix,
    load_augmentations_from_config,
    load_augmentations_from_yaml,
)
from virtual_accelerator.surrogates.beam_output import (
    BeamOutputAugmentation,
    BeamOutputModel,
)


def compute_covariance_matrix(state: Mapping[str, Any], energy: float) -> np.ndarray:
    """Compute a diagonal 6x6 covariance matrix from scalar beam
    parameters for a specific (lcls_cu_inj_model) surrogate model.

    The matrix is in OpenPMDBeamphysics units with no off-diagonal terms.
    variable order: [x, px, y, py, z, pz]
    units: [m, eV/c, m, eV/c, s, eV/c]

    Parameters
    ----------
    state : dict
        Model output state containing XRMS, YRMS, sigma_z, norm_emit_x, norm_emit_y.
    energy : float
        Beam energy in eV.

    Returns
    -------
    np.ndarray
        6x6 diagonal covariance matrix.
    """
    return compute_injector_covariance_matrix(state, energy_eV=energy)


class InjectorSurrogate(BeamOutputModel):
    """
    Custom wrapper class for the LCLS injector surrogate model, which
    computes the covariance matrix from the surrogate's scalar beam parameters.

    Hardcoded to use the LCLS injector surrogate model which dumps beam at OTR2
    """

    def __init__(
        self,
        p0c: float = 135.0e6,
        augmentation_config: Mapping[str, Any] | None = None,
        augmentation_config_path: str | None = None,
        augmentations: Sequence[BeamOutputAugmentation] | None = None,
        **kwargs,
    ):
        active_augmentations: list[BeamOutputAugmentation] = [
            BCTRLFamilyAugmentation(),
            InjectorCovarianceAugmentation(energy_eV=p0c),
        ]

        if augmentation_config is not None:
            active_augmentations.extend(
                load_augmentations_from_config(augmentation_config)
            )

        if augmentation_config_path is not None:
            active_augmentations.extend(
                load_augmentations_from_yaml(augmentation_config_path)
            )

        if augmentations is not None:
            active_augmentations.extend(augmentations)

        super().__init__(
            load_model(),
            p0c=p0c,
            augmentations=active_augmentations,
            **kwargs,
        )
