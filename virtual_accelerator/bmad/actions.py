from typing import Any


from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable, EnumVariable
from lume_bmad.actions import ScaledEleScalarVariable
from pytao import Tao

import logging

logger = logging.getLogger(__name__)


class BmadScalarVariable(ScalarVariable):
    """all bmad variables should have the bmad element name associated with them"""

    element_name: str

    def _get_ele_attr(self, simulator: Tao) -> dict[str, Any]:
        return simulator.ele_gen_attribs(self.element_name)

    def _set_ele_attr(self, simulator: Tao, attribute_name: str, value: Any) -> None:
        simulator.cmd(f"set ele {self.element_name} {attribute_name} = {value}")


class BmadEnumVariable(EnumVariable):
    """Base class for Bmad variables that have a discrete set of options."""

    element_name: str


class _ReadbackFromControlMixin(ReadOnlyActionMixin):
    """Common readback behavior for variables that share control get logic."""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        # Skip ReadOnlyActionMixin's abstract _get and delegate to the next class.
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        raise RuntimeError(f"{self.name} is read-only")


class _QuadrupoleGradientVariable(BmadScalarVariable):
    """Shared quadrupole conversion helpers."""

    def _get_bctrl_value(self, simulator: Tao) -> Any:
        ele_attr = self._get_ele_attr(simulator)
        return -ele_attr["B1_GRADIENT"] * ele_attr["L"] * 10

    def _set_bctrl_value(self, simulator: Tao, value: Any) -> None:
        ele_attr = self._get_ele_attr(simulator)
        bmad_value = -value / (ele_attr["L"] * 10)
        self._set_ele_attr(simulator, "B1_GRADIENT", bmad_value)


class _SBendFieldVariable(BmadScalarVariable):
    """
    Shared sbend conversion helpers.

    SLAC PV units are in GeV/c, which needs to be mapped
    to the Bmad dg parameter:
    dg = g_0 * dp, where dp = (p - p_0) / p_0, and g_0 is the nominal g value for the sbend. The Bmad dg parameter is in units of 1/m, so we need to convert from GeV/c to 1/m.

    """

    def _get_bctrl_value(self, simulator: Tao) -> Any:
        ele_attr = self._get_ele_attr(simulator)
        g = ele_attr["G"]  # 1/m
        dg = ele_attr["DG"]
        p0c = ele_attr["P0C"]  # eV

        if g == 0:
            logger.warning(
                f"Element {self.element_name} has g=0, cannot compute BCTRL value. Returning 0."
            )
            return 0
        p = p0c * (1 + dg / g)
        return p * 1e-9

    def _set_bctrl_value(self, simulator: Tao, value: Any) -> None:
        ele_attr = self._get_ele_attr(simulator)
        p0c = ele_attr["P0C"]  # eV
        g = ele_attr["G"]  # 1/m
        dp = (value * 1e9 - p0c) / p0c
        dg = dp * g
        simulator.cmd(f"set ele {self.element_name} DG = {dg}")


class _ScaledElementAttributeVariable(BmadScalarVariable):
    """Shared linear scaling helpers for element attributes."""

    attribute_name: str
    bmad_to_external_scale: float

    def _get_scaled_value(self, simulator: Tao) -> Any:
        ele_attr = self._get_ele_attr(simulator)
        return ele_attr[self.attribute_name] * self.bmad_to_external_scale

    def _set_scaled_value(self, simulator: Tao, value: Any) -> None:
        bmad_value = value / self.bmad_to_external_scale
        self._set_ele_attr(simulator, self.attribute_name, bmad_value)


class QuadrupoleBCTRLVariable(_QuadrupoleGradientVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Quadrupoles"""

    read_only: bool = False
    unit: str = "kG"

    def _get(self, simulator: Tao) -> Any:
        return self._get_bctrl_value(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        self._set_bctrl_value(simulator, value)


class QuadrupoleBACTVariable(_ReadbackFromControlMixin, QuadrupoleBCTRLVariable):
    """Action that operates on the BACT property of Quadrupoles"""


class SolenoidBCTRLVariable(_ScaledElementAttributeVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Solenoids"""

    attribute_name: str = "BS_FIELD"
    bmad_to_external_scale: float = 10.0

    def _get(self, simulator: Tao) -> Any:
        return self._get_scaled_value(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        self._set_scaled_value(simulator, value)


class SolenoidBACTVariable(_ReadbackFromControlMixin, SolenoidBCTRLVariable):
    """Action that operates on the BACT property of Solenoids"""


class SBendBCTRLVariable(_SBendFieldVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of SBends"""

    read_only: bool = False
    unit: str = "GeV/c"

    def _get(self, simulator: Tao) -> Any:
        return self._get_bctrl_value(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        self._set_bctrl_value(simulator, value)


class SBendBACTVariable(_ReadbackFromControlMixin, SBendBCTRLVariable):
    """Action that operates on the BACT property of SBends"""


class KickerBCTRLVariable(_ScaledElementAttributeVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Kicker magnets"""

    attribute_name: str = "BL_KICK"
    bmad_to_external_scale: float = -10.0

    def _get(self, simulator: Tao) -> Any:
        return self._get_scaled_value(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        self._set_scaled_value(simulator, value)


class KickerBACTVariable(_ReadbackFromControlMixin, KickerBCTRLVariable):
    """Action that operates on the BACT property of Kicker magnets"""


class StatusVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the status of a device (e.g. STATCTRLSUB.T)"""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return 0  # TODO: add logic for status of device


class BminVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMIN/DRVL property of a device"""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return -100  # TODO: add logic for these limits


class BmaxVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMAX/DRVH property of a device"""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return 100  # TODO: add logic for these limits


class ControlStateVariable(BmadEnumVariable, ReadOnlyActionMixin):
    """Action that operates on the control state (e.g. CTRL) of a device"""

    read_only: bool = True
    options: list[str] = ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"]
    default_value: str = "Ready"

    def _get(self, simulator: Tao) -> Any:
        return "Ready"


class BPMXVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the X position of a BPM"""

    unit: str = "mm"
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele(self.element_name).orbit.x * 1e3  # convert from m to mm


class BPMYVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the Y position of a BPM"""

    unit: str = "mm"
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele(self.element_name).orbit.y * 1e3  # convert from m to mm


class BPMTMITDummyVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Dummy variable for BPM TMIT (total beam intensity) for testing purposes."""

    unit: str = "arbitrary units"
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        # Return a dummy value for TMIT
        return 1.0  # This can be adjusted as needed for testing


class KlystronENLDVariable(BmadScalarVariable, WritableActionMixin):
    """
    Action that operates on the amplitude of a klystron which acts on an overlay in Bmad
    """

    unit: str = "MeV"

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele(self.element_name).control_vars["ENLD_MEV"]

    def _set(self, simulator: Tao, value: Any) -> None:
        simulator.cmd(f"set ele {self.element_name} ENLD_MEV = {value}")


class KlystronPDESVariable(BmadScalarVariable, WritableActionMixin):
    """
    Action that operates on the phase of a klystron which acts on an overlay in Bmad
    Note: the element_name for this variable should be set for a Bmad overlay element

    """

    unit: str = "degrees"

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele(self.element_name).control_vars["PHASE_DEG"]

    def _set(self, simulator: Tao, value: Any) -> None:
        simulator.cmd(f"set ele {self.element_name} PHASE_DEG = {value}")


class KlystronPACTVariable(_ReadbackFromControlMixin, KlystronPDESVariable):
    """
    Action that operates on the actual phase of a klystron which acts on an overlay in Bmad
    Note: the element_name for this variable should be set for a Bmad overlay element

    """


class KlystronStatVariable(BmadEnumVariable, WritableActionMixin):
    """
    Action that operates on the status of a klystron which acts on an overlay in Bmad
    Note: the element_name for this variable should be set for a Bmad overlay element

    """

    read_only: bool = True
    options: list[str] = ["0", "1"]
    default_value: str = "0"

    _logic_mapping = {"0": True, "1": False}

    def _get(self, simulator: Tao) -> Any:
        inverse_logic_mapping = {v: k for k, v in self._logic_mapping.items()}
        return inverse_logic_mapping[
            simulator.ele(self.element_name).control_vars["IN_USE"]
        ]

    def _set(self, simulator: Tao, value: Any) -> None:
        simulator.cmd(
            f"set ele {self.element_name} IN_USE = {self._logic_mapping[value]}"
        )


class CavityAREQVariable(ScaledEleScalarVariable):
    """
    Action that operates on the amplitude property of a cavity

    """

    unit: str = "MV"
    scale_factor: float = 1e6
    property_name: str = "VOLTAGE"


class CavityAREQReadbackVariable(_ReadbackFromControlMixin, CavityAREQVariable):
    """Read-only variant of cavity amplitude request variable."""


class CavityPREQVariable(ScaledEleScalarVariable):
    """
    Action that operates on the phase property of a cavity

    """

    unit: str = "degrees"
    property_name: str = "PHI0"
    scale_factor: float = 1 / 360.0  # scale degrees to rad / 2pi


class CavityPREQReadbackVariable(_ReadbackFromControlMixin, CavityPREQVariable):
    """Read-only variant of cavity phase request variable."""


class DummyEnumVariable(BmadEnumVariable, WritableActionMixin):
    """
    Dummy variable for testing purposes. This variable does not correspond to any real element or property in the Bmad model.
    It is used to test the behavior of the model when a variable is requested that does not exist in the model.

    """

    options: list[str] = ["0", "1"]
    default_value: str = "0"

    _value: str = "0"

    def _get(self, simulator: Tao) -> Any:
        return self._value

    def _set(self, simulator: Tao, value: Any) -> None:
        self._value = value


class CavityMODECFGVariable(BmadEnumVariable, WritableActionMixin):
    """
    Action that operates on the mode configuration property of a cavity
    """

    options: list[str] = ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"]
    default_value: str = "ACCEL_STDBY"

    def _get(self, simulator: Tao) -> Any:
        return "ACCEL_STDBY" if simulator.ele(self.element_name).head.is_on else "STDBY"

    def _set(self, simulator: Tao, value: Any) -> None:
        if value == "ACCEL_STDBY":
            simulator.cmd(f"set ele {self.element_name} is_on = True")
        elif value == "STDBY":
            simulator.cmd(f"set ele {self.element_name} is_on = False")
        else:
            raise ValueError(f"Invalid value for CavityMODECFGVariable: {value}")
