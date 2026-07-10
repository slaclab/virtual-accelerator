import torch

from lume.variables import EnumVariable, ScalarVariable
from virtual_accelerator.surrogates.beam_output import BeamOutputModel

from lcls_cu_inj_model import load_model

from typing import Any, Mapping

import numpy as np
from scipy import constants


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
    cov[1, 1] = emit_x**2 * energy**2 / cov[0, 0]
    cov[3, 3] = emit_y**2 * energy**2 / cov[2, 2]
    cov[4, 4] = sigma_z**2
    # cov[5, 5] is left as zero — energy spread not available from model
    return cov


class InjectorSurrogate(BeamOutputModel):
    """
    Custom wrapper class for the LCLS injector surrogate model, which
    computes the covariance matrix from the surrogate's scalar beam parameters.

    Hardcoded to use the LCLS injector surrogate model which dumps beam at OTR2
    """

    def __init__(self, **kwargs):
        super().__init__(load_model(), **kwargs)
        self.p0c = 135.0e6  # eV/c

    def _get(self, variable_names: list[str]) -> dict[str, Any]:
        """Wrap the generic `get` method to handle BACT and BDES variables which map to BCTRL variables"""

        results = {}
        for name in variable_names:
            if name.endswith(":BACT") or name.endswith(":BDES"):
                full_name = ":".join(name.split(":")[:-1]) + ":BCTRL"
                results[name] = super()._get([full_name])[full_name]
            elif name.endswith(":BMIN") or name.endswith(":BCTRL.DRVL"):
                results[name] = -100.0  # default value for BMIN
            elif name.endswith(":BMAX") or name.endswith(":BCTRL.DRVH"):
                results[name] = 100.0  # default value for BMAX
            elif name.endswith(":STATCTRLSUB.T"):
                results[name] = "Ready"  # default value for STATCTRLSUB.T
            elif name.endswith(":CTRL"):
                results[name] = "Ready"  # default value for CTRL
            else:
                results[name] = super()._get([name])[name]

        return results

    def _set(self, values: Mapping[str, Any]):
        """Wrap the generic `set` method to handle BDES variables which map to BCTRL variables"""
        new_values = {}
        for name, value in values.items():
            if name.endswith(":BDES"):
                new_values["".join(name.split(":")[:-1]) + ":BCTRL"] = value
            else:
                new_values[name] = value

        super()._set(new_values)

    @property
    def supported_variables(self) -> Mapping[str, Any]:
        """Wrap the generic `supported_variables` method to add BDES and BACT variables for quadrupoles"""
        vars = super().supported_variables
        new_vars = {}
        for name in vars.keys():
            if name.endswith(":BCTRL"):
                base_name = ":".join(name.split(":")[:-1])
                unit = vars[name].unit
                for suffix in [":BACT", ":BMIN", ":BMAX", ":BCTRL.DRVL", ":BCTRL.DRVH"]:
                    new_vars[base_name + suffix] = ScalarVariable(
                        unit=unit, name=base_name + suffix, read_only=True
                    )

                new_vars[base_name + ":BDES"] = ScalarVariable(
                    unit=unit, name=base_name + ":BDES", read_only=False
                )

                for suffix in [":STATCTRLSUB.T", ":CTRL"]:
                    new_vars[base_name + suffix] = EnumVariable(
                        name=base_name + suffix, options=["Ready"], read_only=True
                    )

        return {**vars, **new_vars}

    def update_state(self):
        """Wrap the generic `update_state` method to calculate the custom covariance matrix"""
        # Update cache with surrogate outputs
        self._cache.update(
            self.surrogate.get(list(self.surrogate.supported_variables.keys()))
        )

        # Compute the covariance matrix from surrogate outputs and store in cache
        cov = compute_covariance_matrix(self._cache, self.p0c)
        self._cache["covariance_matrix"] = torch.from_numpy(cov)

        # Generate the output beam ParticleGroup
        self._generate_output_beam()
