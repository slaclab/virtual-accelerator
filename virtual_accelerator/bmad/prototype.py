from typing import Any
from lume.model import LUMEModel
from lume.variables import NDVariable, ScalarVariable, Variable
from pytao import Tao

from virtual_accelerator.bmad.actions import BCTRLQuadrupoleAction, StatAction


class PrototypeLUMEBmadModel(LUMEModel):
    def __init__(self, tao: Tao, variables: dict[str, Variable]):
        super().__init__()
        self.tao = tao
        self.variables = variables

        self._state = {}
        self.update_state()

    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Internal method to retrieve current values for specified variables.

        Parameters
        ----------
        names : list[str]
            List of variable names to retrieve

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their current values
        """
        out = {name: self._state[name] for name in names}
        print(f"Getting variables: {out}")
        return out
    
    def _set(self, values: dict[str, Any]) -> None:
        """
        Internal method to set values for specified variables and update the state.

        Parameters
        ----------
        values : dict[str, Any]
            Dictionary mapping variable names to the values to set
        """
        # disable lattice calculations in Bmad
        self.tao.cmd("set global lattice_calc_on = F")

        for name, value in values.items():
            self.variables[name].set(self.tao, value)

        # re-enable lattice calculations after setting all variables
        self.tao.cmd("set global lattice_calc_on = T")

        # update internal state after setting new values
        self.update_state()

    def update_state(self):
        for name, var in self.variables.items():
            self._state[name] = var.get(self.tao)

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self.variables
    
    def reset(self):
        # Implement any necessary reset logic for the Bmad model here
        pass


if __name__ == "__main__":
    import os
    root_path = os.environ["LCLS_LATTICE"]
    relative_init_path = "bmad/models/cu_hxr/tao.init"
    init_file = os.path.join(root_path, relative_init_path)

    start_element = "OTR2"
    end_element = "OTR4"
    tao = Tao(f"-init {init_file} -noplot -slice_lattice {start_element}:{end_element}")


    # set tracking to start_element
    tao.cmd(f"set beam track_start = {start_element}")

    n_elements = len(tao.lat_list("*", "ele.name"))
    variables = {
        "QUAD:IN20:525:BCTRL": ScalarVariable(
            name="QUAD:IN20:525:BCTRL",
            action=BCTRLQuadrupoleAction(),
        ),
        "s": NDVariable(name="s", action=StatAction(statistic_name="s"), shape=(n_elements,)),
        "a.beta": NDVariable(name="a.beta", action=StatAction(statistic_name="a.beta"), shape=(n_elements,)),
        "b.beta": NDVariable(name="b.beta", action=StatAction(statistic_name="b.beta"), shape=(n_elements,)),
    }

    model = PrototypeLUMEBmadModel(tao=tao, variables=variables)
    print(model.supported_variables)

    model.get(["s", "a.beta", "b.beta"])


