"""Action-variable implementations for Cheetah-backed virtual accelerator PVs.

This module provides the Cheetah action layer used by the virtual accelerator
package. Conversion and composite-device behavior live directly on action
classes.
"""

from cheetah.accelerator import Screen
from lume_cheetah.actions import (
    CheetahReadOnlyEnumVariable,
    CheetahReadOnlyNDVariable,
    CheetahReadOnlyScalarVariable,
    CheetahWritableScalarVariable,
)
from lume_torch.variables import TorchScalarVariable
from lume.actions import ReadOnlyActionMixin
from lume.variables import EnumVariable


BCTRL_LIMIT = 100.0


def get_magnetic_rigidity(energy):
    """Calculate magnetic rigidity ($B\rho$) in kG-m for energy in eV."""
    return 33.356 * energy / 1e9


class _ReadOnlyActionMixin(ReadOnlyActionMixin):
    read_only: bool = True


class _ReadbackFromControlMixin(ReadOnlyActionMixin):
    """Read-only mixin for readbacks that reuse control `_get` logic."""

    read_only: bool = True

    def _get(self, simulator):
        # Skip ReadOnlyActionMixin's abstract _get and delegate to the next base.
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator, value) -> None:
        raise RuntimeError(f"{self.name} is read-only")


class QuadrupoleBCTRLVariable(CheetahWritableScalarVariable):
    """Quadrupole control/desired magnetic strength (BCTRL/BDES) in kG."""

    unit: str = "kG"
    element_attribute: str = "k1"

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        return (
            getattr(element, self.element_attribute)
            * element.length
            * get_magnetic_rigidity(energy)
        )

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        new_k1 = value / get_magnetic_rigidity(energy) / element.length
        setattr(element, self.element_attribute, new_k1)


class QuadrupoleBACTVariable(_ReadbackFromControlMixin, QuadrupoleBCTRLVariable):
    """Quadrupole readback magnetic strength (BACT) in kG."""


class SolenoidBCTRLVariable(CheetahWritableScalarVariable):
    """Solenoid control/desired field variable in kG-equivalent units."""

    unit: str = "kG"
    element_attribute: str = "k"

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        return getattr(element, self.element_attribute) * get_magnetic_rigidity(energy)

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        new_k = value / get_magnetic_rigidity(energy)
        setattr(element, self.element_attribute, new_k)


class SolenoidBACTVariable(_ReadbackFromControlMixin, SolenoidBCTRLVariable):
    """Solenoid readback field variable in kG-equivalent units."""


class SBendBCTRLVariable(CheetahWritableScalarVariable):
    """SBend control/desired momentum-field representation in GeV/c."""

    unit: str = "GeV/c"

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator, self.element_name)
        sbend = self._primary_element(element)

        if not all(hasattr(sbend, attr) for attr in ("g", "dg", "p0c")):
            raise ValueError(
                f"Element {self.element_name!r} does not expose sbend field attributes"
            )

        if sbend.g == 0:
            return 0
        p = sbend.p0c * (1 + sbend.dg / sbend.g)
        return p * 1e-9

    def _set(self, simulator, value):
        element, _ = self._resolve_element_and_energy(simulator, self.element_name)
        sbend = self._primary_element(element)

        if not all(hasattr(sbend, attr) for attr in ("g", "p0c")):
            raise ValueError(
                f"Element {self.element_name!r} does not expose sbend field attributes"
            )

        dp = (value * 1e9 - sbend.p0c) / sbend.p0c
        sbend.dg = dp * sbend.g


class SBendBACTVariable(_ReadbackFromControlMixin, SBendBCTRLVariable):
    """SBend readback momentum-field representation in GeV/c."""


class KickerBCTRLVariable(CheetahWritableScalarVariable):
    """Horizontal/vertical corrector control variable (BCTRL/BDES) in kG."""

    unit: str = "kG"
    element_attribute: str = "angle"

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        return getattr(element, self.element_attribute) * get_magnetic_rigidity(energy)

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        new_angle = value / get_magnetic_rigidity(energy)
        setattr(element, self.element_attribute, new_angle)


class KickerBACTVariable(_ReadbackFromControlMixin, KickerBCTRLVariable):
    """Horizontal/vertical corrector readback variable (BACT) in kG."""


class StatusVariable(TorchScalarVariable, _ReadOnlyActionMixin):
    """Read-only device status scalar value."""

    element_name: str

    def _get(self, simulator):
        return 1.0


class BminVariable(TorchScalarVariable, _ReadOnlyActionMixin):
    """Read-only lower operating limit variable for magnet-like devices."""

    element_name: str

    def _get(self, simulator):
        return -BCTRL_LIMIT


class BmaxVariable(TorchScalarVariable, _ReadOnlyActionMixin):
    """Read-only upper operating limit variable for magnet-like devices."""

    element_name: str

    def _get(self, simulator):
        return BCTRL_LIMIT


class ControlStateVariable(EnumVariable, _ReadOnlyActionMixin):
    """Read-only high-level control state enum for magnet-like devices."""

    element_name: str

    options: list[str] = ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"]
    default_value: str = "Ready"

    def _get(self, simulator):
        return "Ready"


class DummyEnumVariable(EnumVariable, _ReadOnlyActionMixin):
    """Read-only placeholder enum used for PVs that exist for interface parity."""

    element_name: str

    options: list[str] = ["0", "1"]
    default_value: str = "0"

    def _get(self, simulator):
        return "0"


class BPMXVariable(CheetahReadOnlyScalarVariable):
    """Read-only BPM X readout in millimeters."""

    unit: str = "mm"
    element_attribute: str = "reading"

    def _get(self, simulator):
        return super()._get(simulator)[0]


class BPMYVariable(CheetahReadOnlyScalarVariable):
    """Read-only BPM Y readout in millimeters."""

    unit: str = "mm"
    element_attribute: str = "reading"

    def _get(self, simulator):
        return super()._get(simulator)[1]


class BPMTMITDummyVariable(TorchScalarVariable, _ReadOnlyActionMixin):
    """Read-only placeholder BPM intensity variable for interface parity."""

    element_name: str

    unit: str = "arbitrary units"

    def _get(self, simulator):
        return 1.0


class CavityAREQVariable(CheetahWritableScalarVariable):
    """Writable cavity amplitude request variable in MV."""

    unit: str = "MV"

    def _get(self, simulator):
        return self._get_direct_attribute(simulator, "voltage")

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, "voltage", value)


class CavityAREQReadbackVariable(CheetahReadOnlyScalarVariable):
    """Read-only cavity amplitude readback variable in MV."""

    unit: str = "MV"

    def _get(self, simulator):
        return self._get_direct_attribute(simulator, "voltage")


class CavityPREQVariable(CheetahWritableScalarVariable):
    """Writable cavity phase request variable in degrees."""

    unit: str = "degrees"

    def _get(self, simulator):
        return self._get_direct_attribute(simulator, "phase")

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, "phase", value)


class CavityPREQReadbackVariable(CheetahReadOnlyScalarVariable):
    """Read-only cavity phase readback variable in degrees."""

    unit: str = "degrees"

    def _get(self, simulator):
        return self._get_direct_attribute(simulator, "phase")


class CavityMODECFGVariable(CheetahReadOnlyEnumVariable):
    """Read-only cavity mode configuration enum."""

    options: list[str] = ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"]
    default_value: str = "ACCEL_STDBY"

    def _get(self, simulator):
        return "ACCEL_STDBY"


class ScreenImageVariable(CheetahReadOnlyNDVariable):
    """Read-only screen image array variable."""

    element_attribute: str = "reading"

    def _get(self, simulator):
        return super()._get(simulator).T * 65535


class ScreenImageArraySizeVariable(CheetahReadOnlyScalarVariable):
    """Read-only scalar for screen image dimension metadata values."""

    element_attribute: str = "resolution"
    index: int

    def _get(self, simulator):
        return super()._get(simulator)[self.index]


class ScreenResolutionVariable(CheetahReadOnlyScalarVariable):
    """Read-only scalar for screen pixel resolution in micrometers."""

    element_attribute: str = "pixel_size"
    unit: str = "um"

    def _get(self, simulator):
        return super()._get(simulator)[0] * 1e6


class ScreenPneumaticVariable(CheetahWritableScalarVariable):
    """Writable scalar representing screen insertion/activation control."""

    element_attribute: str = "is_active"

    def _get(self, simulator):
        return 1.0 if bool(super()._get(simulator)) else 0.0

    def _set(self, simulator, value):
        super()._set(simulator, 1.0 if bool(value) else 0.0)


class ScreenCentroidVariable(TorchScalarVariable, _ReadOnlyActionMixin):
    """Read-only scalar for beam centroid multiplied by 1e3 (to convert to mm, mrad)."""

    element_name: str
    centroid_axis: str
    unit: str

    def _get(self, simulator):
        element = getattr(simulator.segment, self.element_name)
        if not isinstance(element, Screen):
            raise ValueError(
                f"Element {self.element_name!r} is not a Screen and cannot provide X readback"
            )
        return getattr(element.get_read_beam(), self.centroid_axis).mean().item() * 1e3


class ScreenXVariable(ScreenCentroidVariable):
    """Read-only scalar for beam x centroid."""

    centroid_axis: str = "x"
    unit: str = "mm"


class ScreenYVariable(ScreenCentroidVariable):
    """Read-only scalar for beam y centroid."""

    centroid_axis: str = "y"
    unit: str = "mm"
