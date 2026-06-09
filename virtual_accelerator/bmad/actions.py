from abc import ABC, abstractmethod
from typing import Any

from lume.actions import Action, WritableAction
from lume.variables import Variable
from pytao import Tao

class BmadAction(Action[Tao]):
    """Base class for actions that operate on a Tao object."""

class WritableBmadAction(WritableAction[Tao]):
    """Base class for writable actions that operate on a Tao object."""

def get_attributes_from_pv(tao, variable_name: str) -> tuple[str, str]:
    """Helper function to extract the Bmad element name and attribute from a control variable name."""
    # example
    mapping = {"QUAD:IN20:525:BCTRL": "QM02"}
    element_name = mapping[variable_name]
    return tao.ele_gen_attribs(element_name)


class BCTRLAction(WritableBmadAction, ABC):
    """Action that operates on the bctrl control variable for a magnet in the Bmad model."""

    name: str

    @abstractmethod
    def _map_get(self, attrs: dict):
        """Maps the attributes of a Bmad element to the value of the bctrl variable."""
        ...

    @abstractmethod
    def _map_set(self, attrs: dict, value: Any):
        """Maps the value of the bctrl variable to the simulator commands needed to set the value in the Bmad model."""
        ...

    def get(self, simulator: Tao, variable: Variable):
        return self._map_get(get_attributes_from_pv(simulator, variable.name))
    
    def set(self, simulator: Tao, variable: Variable, value: Any) -> None:
        simulator.cmd(self._map_set(get_attributes_from_pv(simulator, variable.name), value))


class BCTRLQuadrupoleAction(BCTRLAction):
    """Action that operates on the bctrl control variable for a quadrupole magnet in the Bmad model."""
    name: str = "bctrl_quadrupole"

    def _map_get(self, attrs: dict):
        return -attrs["B1_GRADIENT"] * attrs["L"] * 10
    
    def _map_set(self, attrs: dict, value: Any):
        k1 = -value / (attrs["L"] * 10)
        return f"set {attrs['name']} B1_GRADIENT = {k1}"
    
class BCTRLKickerAction(BCTRLAction):
    """Action that operates on the bctrl control variable for a kicker magnet in the Bmad model."""
    name: str = "bctrl_kicker"

    def _map_get(self, attrs: dict):
        return attrs["BL_KICK"]
    
    def _map_set(self, attrs: dict, value: Any):
        return f"set {attrs['name']} BL_KICK = {value}"

class TrackTypeAction(WritableBmadAction):
    """Action that sets the track type for a beam tracking simulation in the Bmad model."""

    name: str = "track_type"

    def set(self, simulator: Tao, variable: Variable, value: Any) -> None:
        simulator.cmd(f"set global track_type = {value}")

    def get(self, simulator: Tao, variable: Variable):
        return simulator.tao_global()["track_type"]


class StatAction(BmadAction):
    """Action that operates on a single statistic in the Bmad model."""
    name: str = "stat"
    statistic_name: str

    def get(self, simulator: Tao, variable: Variable):
        return simulator.lat_list("*", f"ele.{self.statistic_name}")

