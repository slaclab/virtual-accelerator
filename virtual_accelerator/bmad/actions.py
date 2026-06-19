from typing import Any

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable, EnumVariable
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


class _QuadrupoleGradientVariable(BmadScalarVariable):
    """Shared quadrupole conversion helpers."""

    def _get_bctrl_value(self, simulator: Tao) -> Any:
        ele_attr = self._get_ele_attr(simulator)
        return -ele_attr["B1_GRADIENT"] * ele_attr["L"] * 10

    def _set_bctrl_value(self, simulator: Tao, value: Any) -> None:
        ele_attr = self._get_ele_attr(simulator)
        bmad_value = -value / (ele_attr["L"] * 10)
        self._set_ele_attr(simulator, "B1_GRADIENT", bmad_value)


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


class QuadrupoleBACTVariable(_QuadrupoleGradientVariable, ReadOnlyActionMixin):
    """Action that operates on the BACT property of Quadrupoles"""

    read_only: bool = True
    unit: str = "kG"

    def _get(self, simulator: Tao) -> Any:
        return self._get_bctrl_value(simulator)


class SolenoidBCTRLVariable(_ScaledElementAttributeVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Solenoids"""

    attribute_name: str = "BS_FIELD"
    bmad_to_external_scale: float = 10.0

    def _get(self, simulator: Tao) -> Any:
        return self._get_scaled_value(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        self._set_scaled_value(simulator, value)


class SolenoidBACTVariable(_ScaledElementAttributeVariable, ReadOnlyActionMixin):
    """Action that operates on the BACT property of Solenoids"""

    attribute_name: str = "BS_FIELD"
    bmad_to_external_scale: float = 10.0
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return self._get_scaled_value(simulator)


class KickerBCTRLVariable(_ScaledElementAttributeVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Kicker magnets"""

    attribute_name: str = "BL_KICK"
    bmad_to_external_scale: float = -10.0

    def _get(self, simulator: Tao) -> Any:
        return self._get_scaled_value(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        self._set_scaled_value(simulator, value)


class KickerBACTVariable(_ScaledElementAttributeVariable, ReadOnlyActionMixin):
    """Action that operates on the BACT property of Kicker magnets"""

    attribute_name: str = "BL_KICK"
    bmad_to_external_scale: float = -10.0
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return self._get_scaled_value(simulator)


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

    read_only: bool = True
    unit: str = "mm"

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele(self.element_name).orbit.x * 1e3  # convert from m to mm


class BPMYVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the Y position of a BPM"""

    read_only: bool = True
    unit: str = "mm"

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele(self.element_name).orbit.y * 1e3  # convert from m to mm
