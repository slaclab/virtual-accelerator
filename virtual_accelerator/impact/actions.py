from typing import Any


from impact import Impact
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable, EnumVariable
import numpy as np

import logging

logger = logging.getLogger(__name__)


class ImpactGroupVariable(ScalarVariable, WritableActionMixin):
    """Base class for group variables in the Impact simulator."""

    group_name: str
    group_key: str
    scale: float = 1.0
    offset: float = 0.0

    def _get(self, simulator: Impact) -> Any:
        return (simulator[self.group_name][self.group_key] - self.offset) / self.scale

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator[self.group_name][self.group_key] = value * self.scale + self.offset


class ImpactScalarVariable(ScalarVariable):
    """Base class for scalar variables in the Impact simulator."""

    element_name: str

    def _get_ele_attr(self, simulator: Impact) -> Any:
        return simulator.ele[self.element_name]

    def _set_ele_attr(self, simulator: Impact, attribute_name: str, value: Any) -> None:
        simulator.ele[self.element_name][attribute_name] = value


class ImpactEnumVariable(EnumVariable):
    """Base class for enum variables in the Impact simulator."""

    element_name: str


class _ReadbackFromControlMixin(ReadOnlyActionMixin):
    """Common readback behavior for variables that share control get logic."""

    read_only: bool = True

    def _get(self, simulator: Impact) -> Any:
        # Skip ReadOnlyActionMixin's abstract _get and delegate to the next class.
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator: Impact, value: Any) -> None:
        raise RuntimeError(f"{self.name} is read-only")


def reconstruct_fields(field: dict[str, Any], z: np.ndarray) -> np.ndarray:
    """
    Reconstruct the normalized on-axis field from Impact-T Fourier coefficients.

    Parameters
    ----------
    field : dict[str, Any]
        A solrf fieldmap component (e.g. ``fieldmap["field"]["Bz"]``) holding the
        Fourier coefficients (``"fourier_coefficients"``), leading edge (``"z0"``),
        and periodic length (``"L"``).
    z : np.ndarray
        Longitudinal positions (m) at which to evaluate the field.

    Returns
    -------
    np.ndarray
        The reconstructed on-axis field at each ``z``, normalized to unit peak.
    """
    z = np.asarray(z, dtype=float)
    coefs = field[
        "fourier_coefficients"
    ]  # key used by lume-impact's in-memory solrf fieldmap
    n_cos = (len(coefs) - 1) // 2
    c0, c_cos, c_sin = coefs[0], coefs[1::2], coefs[2::2]
    n = np.arange(1, n_cos + 1)
    arg = np.outer(2 * np.pi * (z - field["L"] / 2 - field["z0"]) / field["L"], n)
    return c0 / 2 + np.cos(arg) @ c_cos + np.sin(arg) @ c_sin


def calc_effective_length(sol: dict[str, Any]) -> float:
    """
    Compute the field-squared effective length of a solenoid Bz fieldmap.

    Uses ``integral(Bz**2 dz) / max(Bz)**2`` over the physical hump (``z > 0``),
    the effective length that characterizes the solenoid focusing strength.

    Parameters
    ----------
    sol : dict[str, Any]
        Solenoid Bz fieldmap component with keys ``"z0"``, ``"z1"``, ``"L"``, and
        ``"fourier_coefficients"``.

    Returns
    -------
    float
        The effective length in meters.
    """
    z = np.linspace(sol["z0"], sol["z1"], 1001)
    Bz = reconstruct_fields(sol, z)  # normalized on-axis field (peak ~ 1)
    # Effective length using only z > 0 (single physical solenoid hump)
    mask = z > 0
    z_pos, Bz_pos = z[mask], Bz[mask]
    Bpeak_pos = np.max(np.abs(Bz_pos))

    trapz = (
        np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    )  # np.trapz removed in NumPy 2.0
    int_Bz2_pos = trapz(Bz_pos**2, z_pos)
    L_eff_focus_pos = int_Bz2_pos / Bpeak_pos**2
    return L_eff_focus_pos


class SolenoidBCTRLVariable(ImpactScalarVariable, WritableActionMixin):
    read_only: bool = False
    unit: str = "kG-m"

    def _load_solrf(self, simulator: Impact) -> Any:
        # Use the fieldmap already parsed into the simulator
        ele_attr = self._get_ele_attr(simulator)
        fieldmap = simulator.fieldmaps[ele_attr["filename"]]
        return fieldmap["field"]["Bz"]

    def _calculate_effective_length(self, simulator: Impact) -> Any:
        sol = self._load_solrf(simulator)
        return calc_effective_length(sol)

    def _get_bctrl_value(self, simulator: Impact) -> Any:
        ele_attr = self._get_ele_attr(simulator)
        # kG-m: field scale * effective length, with a T->kG factor of 10.
        return (
            ele_attr["solenoid_field_scale"]
            * self._calculate_effective_length(simulator)
            * 10
        )

    def _set_bctrl_value(self, simulator: Impact, value: Any) -> None:
        self._set_ele_attr(
            simulator,
            "solenoid_field_scale",
            value / (self._calculate_effective_length(simulator) * 10),
        )

    def _get(self, simulator):
        return self._get_bctrl_value(simulator)

    def _set(self, simulator, value):
        self._set_bctrl_value(simulator, value)


class SolenoidBACTVariable(_ReadbackFromControlMixin, SolenoidBCTRLVariable):
    """BACT readback of the solenoid."""


class QuadrupoleBCTRLVariable(ImpactScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Quadrupoles"""

    read_only: bool = False
    unit: str = "kG"

    def _get_bctrl_value(self, simulator: Impact) -> Any:
        ele_attr = self._get_ele_attr(simulator)
        return -ele_attr["b1_gradient"] * ele_attr["L_effective"] * 10

    def _set_bctrl_value(self, simulator: Impact, value: Any) -> None:
        ele_attr = self._get_ele_attr(simulator)
        self._set_ele_attr(
            simulator, "b1_gradient", -value / (ele_attr["L_effective"] * 10)
        )

    def _get(self, simulator: Impact) -> Any:
        return self._get_bctrl_value(simulator)

    def _set(self, simulator: Impact, value: Any) -> None:
        self._set_bctrl_value(simulator, value)


class QuadrupoleBACTVariable(_ReadbackFromControlMixin, QuadrupoleBCTRLVariable):
    """Action that operates on the BACT property of Quadrupoles"""


class StatusVariable(ImpactScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the status of a device (e.g. STATCTRLSUB.T)"""

    read_only: bool = True

    def _get(self, simulator: Impact) -> Any:
        return 0  # TODO: add logic for status of device


class BminVariable(ImpactScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMIN/DRVL property of a device"""

    read_only: bool = True

    def _get(self, simulator: Impact) -> Any:
        return -100  # TODO: add logic for these limits


class BmaxVariable(ImpactScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMAX/DRVH property of a device"""

    read_only: bool = True

    def _get(self, simulator: Impact) -> Any:
        return 100  # TODO: add logic for these limits


class ControlStateVariable(ImpactEnumVariable, ReadOnlyActionMixin):
    """Action that operates on the control state (e.g. CTRL) of a device"""

    read_only: bool = True
    options: list[str] = ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"]
    default_value: str = "Ready"

    def _get(self, simulator: Impact) -> Any:
        return "Ready"
