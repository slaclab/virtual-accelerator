"""Action-variable implementations for Cheetah-backed virtual accelerator PVs.

This module provides the Cheetah action layer used by the virtual accelerator
package. Conversion and composite-device behavior live directly on action
classes.
"""

from typing import Any

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import EnumVariable
from lume_torch.variables import TorchNDVariable, TorchScalarVariable


BCTRL_LIMIT = 100.0


def get_magnetic_rigidity(energy):
    """Calculate magnetic rigidity ($B\rho$) in kG-m for energy in eV."""
    return 33.356 * energy / 1e9


def _normalize_element_name(name: str) -> str:
    """Return normalized element name used for split-element matching."""
    return name.split("#")[0].upper()


def _as_element_list(element: Any) -> list[Any]:
    """Return a uniform list representation for single or split elements."""
    if isinstance(element, list):
        return element
    return [element]


def _resolve_segment_element(segment, element_name: str):
    """Resolve a segment element by logical name.

    Returns a list when multiple split elements match the same logical name.
    """
    target = element_name.upper()
    exact_matches = [element for element in segment.elements if element.name.upper() == target]
    if exact_matches:
        return exact_matches[0]

    normalized_target = _normalize_element_name(element_name)
    normalized_matches = [
        element
        for element in segment.elements
        if _normalize_element_name(element.name) == normalized_target
    ]

    if not normalized_matches:
        raise ValueError(f"Element {element_name!r} not found in Cheetah segment")

    if len(normalized_matches) == 1:
        return normalized_matches[0]

    return normalized_matches


def _resolve_energy(simulator, element: Any):
    """Resolve beam energy for a single element or split-element list."""
    candidate_names = [subelement.name for subelement in _as_element_list(element)]

    for name in candidate_names:
        if name in simulator.energies:
            return simulator.energies[name]

    normalized_target = _normalize_element_name(candidate_names[0])
    for name, energy in simulator.energies.items():
        if _normalize_element_name(name) == normalized_target:
            return energy

    raise ValueError(
        f"No beam energy found for element {candidate_names[0]!r} in simulator energies"
    )


class _CheetahElementAccessMixin:
    """Shared element/energy resolution and direct attribute helpers."""

    element_name: str
    pv_attribute: str

    @staticmethod
    def _elements(element: Any) -> list[Any]:
        return _as_element_list(element)

    @staticmethod
    def _primary_element(element: Any) -> Any:
        return _as_element_list(element)[0]

    def _resolve_element_and_energy(self, simulator):
        """Resolve target element object(s) and matching beam energy."""
        element = _resolve_segment_element(simulator.segment, self.element_name)
        energy = _resolve_energy(simulator, element)
        return element, energy

    def _get_direct_attribute(self, simulator, attribute_name: str):
        """Read a direct element attribute from the first resolved element."""
        element, _ = self._resolve_element_and_energy(simulator)
        return getattr(self._primary_element(element), attribute_name)

    def _set_direct_attribute(self, simulator, attribute_name: str, value):
        """Set a direct element attribute on all resolved split elements."""
        element, _ = self._resolve_element_and_energy(simulator)
        for subelement in self._elements(element):
            setattr(subelement, attribute_name, value)


class CheetahScalarVariable(_CheetahElementAccessMixin, TorchScalarVariable):
    """Base scalar action variable for Cheetah-backed PVs."""

    element_name: str
    pv_attribute: str


class CheetahNDVariable(_CheetahElementAccessMixin, TorchNDVariable):
    """Base array-valued action variable for Cheetah-backed PVs."""

    element_name: str
    pv_attribute: str


class CheetahEnumVariable(_CheetahElementAccessMixin, EnumVariable):
    """Base enum action variable for Cheetah-backed PVs."""

    element_name: str
    pv_attribute: str


class _ReadOnlyFromImplementationMixin(ReadOnlyActionMixin):
    """Read-only behavior mixin that delegates `_get` to class implementations."""

    read_only: bool = True

    def _get(self, simulator):
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator, value) -> None:
        raise RuntimeError(f"{self.name} is read-only")


class _ReadbackFromControlMixin(_ReadOnlyFromImplementationMixin):
    """Read-only mixin for readback variables sharing control `_get` logic."""


class CheetahWritableScalarVariable(CheetahScalarVariable, WritableActionMixin):
    """Writable scalar action variable base."""

    read_only: bool = False


class CheetahReadOnlyScalarVariable(
    _ReadOnlyFromImplementationMixin, CheetahScalarVariable
):
    """Read-only scalar action variable base."""


class CheetahWritableNDVariable(CheetahNDVariable, WritableActionMixin):
    """Writable array action variable base."""

    read_only: bool = False


class CheetahReadOnlyNDVariable(_ReadOnlyFromImplementationMixin, CheetahNDVariable):
    """Read-only array action variable base."""


class CheetahWritableEnumVariable(CheetahEnumVariable, WritableActionMixin):
    """Writable enum action variable base."""

    read_only: bool = False


class CheetahReadOnlyEnumVariable(
    _ReadOnlyFromImplementationMixin, CheetahEnumVariable
):
    """Read-only enum action variable base."""


class QuadrupoleBCTRLVariable(CheetahWritableScalarVariable):
    """Quadrupole control/desired magnetic strength (BCTRL/BDES) in kG."""

    unit: str = "kG"

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator)
        elements = self._elements(element)
        total_length = sum(subelement.length for subelement in elements)
        return elements[0].k1 * total_length * get_magnetic_rigidity(energy)

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator)
        elements = self._elements(element)
        total_length = sum(subelement.length for subelement in elements)
        new_k1 = value / get_magnetic_rigidity(energy) / total_length
        for subelement in elements:
            subelement.k1 = new_k1


class QuadrupoleBACTVariable(_ReadbackFromControlMixin, QuadrupoleBCTRLVariable):
    """Quadrupole readback magnetic strength (BACT) in kG."""


class SolenoidBCTRLVariable(CheetahWritableScalarVariable):
    """Solenoid control/desired field variable in kG-equivalent units."""

    unit: str = "kG"

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator)
        return self._primary_element(element).k * get_magnetic_rigidity(energy)

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator)
        new_k = value / get_magnetic_rigidity(energy)
        for subelement in self._elements(element):
            subelement.k = new_k


class SolenoidBACTVariable(_ReadbackFromControlMixin, SolenoidBCTRLVariable):
    """Solenoid readback field variable in kG-equivalent units."""


class SBendBCTRLVariable(CheetahWritableScalarVariable):
    """SBend control/desired momentum-field representation in GeV/c."""

    unit: str = "GeV/c"

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator)
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
        element, _ = self._resolve_element_and_energy(simulator)
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

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator)
        return self._primary_element(element).angle * get_magnetic_rigidity(energy)

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator)
        new_angle = value / get_magnetic_rigidity(energy)
        for subelement in self._elements(element):
            subelement.angle = new_angle


class KickerBACTVariable(_ReadbackFromControlMixin, KickerBCTRLVariable):
    """Horizontal/vertical corrector readback variable (BACT) in kG."""


class StatusVariable(CheetahReadOnlyScalarVariable):
    """Read-only device status scalar value."""

    def _get(self, simulator):
        return 1.0


class BminVariable(CheetahReadOnlyScalarVariable):
    """Read-only lower operating limit variable for magnet-like devices."""

    def _get(self, simulator):
        return -BCTRL_LIMIT


class BmaxVariable(CheetahReadOnlyScalarVariable):
    """Read-only upper operating limit variable for magnet-like devices."""

    def _get(self, simulator):
        return BCTRL_LIMIT


class ControlStateVariable(CheetahReadOnlyEnumVariable):
    """Read-only high-level control state enum for magnet-like devices."""

    options: list[str] = ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"]
    default_value: str = "Ready"

    def _get(self, simulator):
        return "Ready"


class BPMXVariable(CheetahReadOnlyScalarVariable):
    """Read-only BPM X readout in millimeters."""

    unit: str = "mm"

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator)
        return self._primary_element(element).reading[0]


class BPMYVariable(CheetahReadOnlyScalarVariable):
    """Read-only BPM Y readout in millimeters."""

    unit: str = "mm"

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator)
        return self._primary_element(element).reading[1]


class BPMTMITDummyVariable(CheetahReadOnlyScalarVariable):
    """Read-only placeholder BPM intensity variable for interface parity."""

    unit: str = "arbitrary units"

    def _get(self, simulator):
        return 1.0


class KlystronENLDVariable(CheetahWritableScalarVariable):
    """Writable klystron amplitude-like variable in MeV."""

    unit: str = "MeV"


class KlystronPDESVariable(CheetahWritableScalarVariable):
    """Writable klystron phase setpoint in degrees."""

    unit: str = "degrees"


class KlystronPACTVariable(CheetahReadOnlyScalarVariable):
    """Read-only klystron phase readback in degrees."""

    unit: str = "degrees"


class KlystronStatVariable(CheetahReadOnlyEnumVariable):
    """Read-only klystron binary status enum."""

    options: list[str] = ["0", "1"]
    default_value: str = "0"

    def _get(self, simulator):
        return "0"


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
        if self.pv_attribute == "AMPL_W0CH0":
            return 0.0
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
        if self.pv_attribute == "PACT_AVGNT":
            return 0.0
        return self._get_direct_attribute(simulator, "phase")


class CavityMODECFGVariable(CheetahReadOnlyEnumVariable):
    """Read-only cavity mode configuration enum."""

    options: list[str] = ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"]
    default_value: str = "ACCEL_STDBY"

    def _get(self, simulator):
        return "ACCEL_STDBY"


class DummyEnumVariable(CheetahReadOnlyEnumVariable):
    """Read-only placeholder enum variable for unsupported controls."""

    options: list[str] = ["0", "1"]
    default_value: str = "0"

    def _get(self, simulator):
        if self.pv_attribute == "RF_ENABLE":
            return 1.0
        return 0.0


class ScreenImageVariable(CheetahReadOnlyNDVariable):
    """Read-only screen image array variable."""

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator)
        return self._primary_element(element).reading.T * 65535


class ScreenImageArraySizeVariable(CheetahReadOnlyScalarVariable):
    """Read-only scalar for screen image dimension metadata values."""

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator)
        screen = self._primary_element(element)
        if self.pv_attribute in {"Image:ArraySize1_RBV", "N_OF_ROW"}:
            return screen.resolution[0]
        return screen.resolution[1]


class ScreenResolutionVariable(CheetahReadOnlyScalarVariable):
    """Read-only scalar for screen pixel resolution in micrometers."""

    unit: str = "um"

    def _get(self, simulator):
        element, _ = self._resolve_element_and_energy(simulator)
        return self._primary_element(element).pixel_size[0] * 1e6


class ScreenPneumaticVariable(CheetahWritableScalarVariable):
    """Writable scalar representing screen insertion/activation control."""

    def _get(self, simulator):
        return 1.0 if bool(self._get_direct_attribute(simulator, "is_active")) else 0.0

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, "is_active", bool(value))
