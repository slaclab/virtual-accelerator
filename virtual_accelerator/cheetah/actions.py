"""Action-variable implementations for Cheetah-backed virtual accelerator PVs.

This module provides a Cheetah equivalent to the BMAD action layer. Each
variable class encapsulates get/set behavior against a Cheetah simulator while
preserving existing SLAC PV semantics implemented in the Cheetah utility
attribute access mappings.
"""

from typing import Any

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import EnumVariable, NDVariable, ScalarVariable

from virtual_accelerator.cheetah.utils import access_cheetah_attribute


def _normalize_element_name(name: str) -> str:
    """Return a normalized element name used for split-element matching.

    Cheetah split elements may include a ``#<index>`` suffix (for example
    ``Q1#1``). This helper strips the suffix and normalizes case for lookup.
    """
    return name.split("#")[0].upper()


def _resolve_segment_element(segment, element_name: str):
    """Resolve a Cheetah segment element reference from a logical element name.

    Resolution behavior:
    1. Prefer exact case-insensitive name match.
    2. Fallback to normalized split-element match.
    3. Return a list when multiple split elements represent one logical device.
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
    """Resolve beam energy for a single element or composite split-element list.

    The simulator stores energies keyed by element name. For split elements,
    this helper attempts exact key matches first, then normalized-name fallback.
    """
    if isinstance(element, list):
        candidate_names = [subelement.name for subelement in element]
    else:
        candidate_names = [element.name]

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


class _CheetahAttributeAccessMixin:
    """Shared accessor logic for action variables using PV attribute mappings.

    Subclasses provide ``element_name`` and ``pv_attribute`` fields. The mixin
    resolves both the target element object and corresponding beam energy, then
    delegates conversion logic to ``access_cheetah_attribute``.
    """

    element_name: str
    pv_attribute: str

    def _resolve_element_and_energy(self, simulator):
        """Resolve the element object (or split list) and matching beam energy."""
        element = _resolve_segment_element(simulator.segment, self.element_name)
        energy = _resolve_energy(simulator, element)
        return element, energy

    def _get(self, simulator):
        """Read a PV value from simulator state through the Cheetah mapping layer."""
        element, energy = self._resolve_element_and_energy(simulator)
        return access_cheetah_attribute(element, self.pv_attribute, energy)

    def _set(self, simulator, value):
        """Write a PV value to simulator state through the Cheetah mapping layer."""
        element, energy = self._resolve_element_and_energy(simulator)
        access_cheetah_attribute(element, self.pv_attribute, energy, set_value=value)


class CheetahScalarVariable(_CheetahAttributeAccessMixin, ScalarVariable):
    """Base scalar action variable for Cheetah-backed PVs."""

    element_name: str
    pv_attribute: str


class CheetahNDVariable(_CheetahAttributeAccessMixin, NDVariable):
    """Base array-valued action variable for Cheetah-backed PVs."""

    element_name: str
    pv_attribute: str


class CheetahEnumVariable(_CheetahAttributeAccessMixin, EnumVariable):
    """Base enum action variable for Cheetah-backed PVs."""

    element_name: str
    pv_attribute: str


class _ReadOnlyFromAccessMixin(ReadOnlyActionMixin):
    """Read-only action behavior for variables backed by attribute mappings."""

    read_only: bool = True

    def _get(self, simulator):
        """Read value using the next class in method resolution order."""
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator, value) -> None:
        """Reject writes consistently for read-only PVs."""
        raise RuntimeError(f"{self.name} is read-only")


class CheetahWritableScalarVariable(CheetahScalarVariable, WritableActionMixin):
    """Writable scalar action variable base."""

    read_only: bool = False


class CheetahReadOnlyScalarVariable(_ReadOnlyFromAccessMixin, CheetahScalarVariable):
    """Read-only scalar action variable base."""

    pass


class CheetahWritableNDVariable(CheetahNDVariable, WritableActionMixin):
    """Writable array action variable base."""

    read_only: bool = False


class CheetahReadOnlyNDVariable(_ReadOnlyFromAccessMixin, CheetahNDVariable):
    """Read-only array action variable base."""

    pass


class CheetahWritableEnumVariable(CheetahEnumVariable, WritableActionMixin):
    """Writable enum action variable base."""

    read_only: bool = False


class CheetahReadOnlyEnumVariable(_ReadOnlyFromAccessMixin, CheetahEnumVariable):
    """Read-only enum action variable base."""

    pass


class QuadrupoleBCTRLVariable(CheetahWritableScalarVariable):
    """Quadrupole control/desired magnetic strength (BCTRL/BDES) in kG."""

    unit: str = "kG"


class QuadrupoleBACTVariable(CheetahReadOnlyScalarVariable):
    """Quadrupole readback magnetic strength (BACT) in kG."""

    unit: str = "kG"


class SolenoidBCTRLVariable(CheetahWritableScalarVariable):
    """Solenoid control/desired field variable in kG-equivalent units."""

    unit: str = "kG"


class SolenoidBACTVariable(CheetahReadOnlyScalarVariable):
    """Solenoid readback field variable in kG-equivalent units."""

    unit: str = "kG"


class SBendBCTRLVariable(CheetahWritableScalarVariable):
    """SBend control/desired momentum-field representation in GeV/c."""

    unit: str = "GeV/c"


class SBendBACTVariable(CheetahReadOnlyScalarVariable):
    """SBend readback momentum-field representation in GeV/c."""

    unit: str = "GeV/c"


class KickerBCTRLVariable(CheetahWritableScalarVariable):
    """Horizontal/vertical corrector control variable (BCTRL/BDES) in kG."""

    unit: str = "kG"


class KickerBACTVariable(CheetahReadOnlyScalarVariable):
    """Horizontal/vertical corrector readback variable (BACT) in kG."""

    unit: str = "kG"


class StatusVariable(CheetahReadOnlyScalarVariable):
    """Generic read-only device status scalar variable."""

    pass


class BminVariable(CheetahReadOnlyScalarVariable):
    """Read-only lower operating limit variable for magnet-like devices."""

    pass


class BmaxVariable(CheetahReadOnlyScalarVariable):
    """Read-only upper operating limit variable for magnet-like devices."""

    pass


class ControlStateVariable(CheetahReadOnlyEnumVariable):
    """Read-only high-level control state enum for magnet-like devices."""

    options: list[str] = ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"]
    default_value: str = "Ready"


class BPMXVariable(CheetahReadOnlyScalarVariable):
    """Read-only BPM X readout in millimeters."""

    unit: str = "mm"


class BPMYVariable(CheetahReadOnlyScalarVariable):
    """Read-only BPM Y readout in millimeters."""

    unit: str = "mm"


class BPMTMITDummyVariable(CheetahReadOnlyScalarVariable):
    """Read-only placeholder BPM intensity variable used for interface parity."""

    unit: str = "arbitrary units"


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


class CavityAREQVariable(CheetahWritableScalarVariable):
    """Writable cavity amplitude request variable in MV."""

    unit: str = "MV"


class CavityAREQReadbackVariable(CheetahReadOnlyScalarVariable):
    """Read-only cavity amplitude readback variable in MV."""

    unit: str = "MV"


class CavityPREQVariable(CheetahWritableScalarVariable):
    """Writable cavity phase request variable in degrees."""

    unit: str = "degrees"


class CavityPREQReadbackVariable(CheetahReadOnlyScalarVariable):
    """Read-only cavity phase readback variable in degrees."""

    unit: str = "degrees"


class CavityMODECFGVariable(CheetahReadOnlyEnumVariable):
    """Read-only cavity mode configuration enum."""

    options: list[str] = ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"]
    default_value: str = "ACCEL_STDBY"


class DummyEnumVariable(CheetahReadOnlyEnumVariable):
    """Read-only placeholder enum variable used for unsupported controls."""

    options: list[str] = ["0", "1"]
    default_value: str = "0"


class ScreenImageVariable(CheetahReadOnlyNDVariable):
    """Read-only screen image array variable."""

    pass


class ScreenImageArraySizeVariable(CheetahReadOnlyScalarVariable):
    """Read-only scalar for screen image dimension metadata values."""

    pass


class ScreenResolutionVariable(CheetahReadOnlyScalarVariable):
    """Read-only scalar for screen pixel resolution in micrometers."""

    unit: str = "um"


class ScreenPneumaticVariable(CheetahWritableScalarVariable):
    """Writable scalar representing screen insertion/activation control."""

    pass
