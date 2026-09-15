from typing import Any


from impact import Impact
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable, EnumVariable

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


class _QuadrupoleGradientVariable(ImpactScalarVariable):
    """Shared quadrupole conversion helpers."""

    def _get_bctrl_value(self, simulator: Impact) -> Any:
        ele_attr = self._get_ele_attr(simulator)
        return -ele_attr["b1_gradient"] * ele_attr["L_effective"] * 10

    def _set_bctrl_value(self, simulator: Impact, value: Any) -> None:
        ele_attr = self._get_ele_attr(simulator)
        self._set_ele_attr(
            simulator, "b1_gradient", -value / (ele_attr["L_effective"] * 10)
        )


class QuadrupoleBCTRLVariable(_QuadrupoleGradientVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Quadrupoles"""

    read_only: bool = False
    unit: str = "kG"

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
