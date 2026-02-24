from abc import ABC, abstractmethod
from virtual_accelerator.cheetah.utils import access_cheetah_attribute

class CheetahTransformer(ABC):
    """
    Class that handles transformations between control system
    names and values and cheetah element names and attributes.

    Should be subclassed for each specific accelerator facility to handle
    any necessary unit conversions or special cases in the mapping
    between control variables and cheetah properties.

    """
    

    @abstractmethod
    def get_cheetah_property(self, simulator, control_variable_name):
        """
        Given a control variable name, return the corresponding cheetah value.

        Parameters
        ----------
        control_variable_name : str
            The name of the control variable (e.g. "QUAD:IN20:511:BCTRL")

        Returns
        -------
        Any
            The corresponding cheetah element property value
        """
        pass

    @abstractmethod
    def set_cheetah_property(self, simulator, control_variable_name, value):
        """
        Given a control variable name and value, set the corresponding cheetah property.

        Parameters
        ----------
        control_variable_name : str
            The name of the control variable (e.g. "QUAD:IN20:511:BCTRL")
        value : Any
            The value to set for the corresponding cheetah property
        """
        pass


class SLACCheetahTransformer(CheetahTransformer):
    """
    CheetahTransformer subclass for SLAC accelerator simulations.

    This class can be extended to include any necessary unit conversions or
    special handling for SLAC-specific control variables and their mapping
    to cheetah properties.

    Attributes
    ----------
    control_name_to_cheetah : dict[str, str]
        A dictionary mapping control variable names to cheetah elements 
        (e.g. {"QUAD:IN20:511 : "qe03"})
        #same something about how bctrl maps to k1, in utils.py
    """
    
    def __init__(self, control_name_to_cheetah: dict[str, str]):
        self._control_name_to_cheetah = control_name_to_cheetah

    @property
    def control_name_to_cheetah(self):
        return self._control_name_to_cheetah

    def get_cheetah_property(self, simulator, control_variable_name):
        """
        Get a property of a Cheetah element based on the control
        variable name and return its value in EPICS units.

        Parameters
        ----------
        simulator : CheetahSimulator
            The simulator instance containing the segment and elements.
        control_variable_name : str
            The name of the control variable (e.g. "QUAD:IN20:511:BCTRL")
        energy : float
            The beam energy in eV, used for unit conversions if necessary.
        """
        #get the last part after the last colon, which is the attribute name
        control_name, attribute = control_variable_name.rsplit(":", 1)
        element_name = self.control_name_to_cheetah.get(control_name) # mapping { "QUAD:IN20:511:BCTRL" : "QE03"}
        if element_name is None:
            raise ValueError(f"No mapping found for control variable '{control_variable_name}'")

        element = getattr(simulator.segment, element_name)
        beam_energy_at_element = simulator.energies[element_name] 
        # due to getting beam energy this calc is very slow, maybe some list format should
        # be passable for args. 
        return access_cheetah_attribute(element, attribute, beam_energy_at_element)

    def set_cheetah_property(self, simulator, control_variable_name, value):
        """
        Set a property of a Cheetah element based on the control variable
        name and value in EPICS units.

        Parameters
        ----------
        simulator : CheetahSimulator
            The simulator instance containing the segment and elements.
        control_variable_name : str
            The name of the control variable (e.g. "QUAD:IN20:511:BCTRL")
        value : Any
            The value to set for the corresponding cheetah property, in EPICS units.
        """

        control_name, attribute = control_variable_name.rsplit(":", 1) 
        element_name = self.control_name_to_cheetah.get(control_name)
        if element_name is None:
            raise ValueError(f"No mapping found for control variable '{control_variable_name}'")

        element = getattr(simulator.segment, element_name)
        beam_energy_at_element = simulator.energies[element_name]
        access_cheetah_attribute(element, attribute, beam_energy_at_element, set_value=value)


