from typing import Any
from lume_bmad.transformer import BmadTransformer
from pytao import Tao
from lume_bmad.utils import get_beam_info
from beamphysics.interfaces.bmad import write_bmad
from os import getcwd
import numpy as np


class SCBmadTransformer(BmadTransformer):
    """
    Attributes
    ----------
    control_name_to_bmad : dict[str, str]
        A dictionary mapping control variable names to Bmad elements
        (e.g. {"QUAD:IN20:511": "QE03"})
        #same something about how bctrl maps to k1, in utils.py

    """
        def __init__(
        self,
        control_name_to_bmad: dict[str, str],
        screen_attributes: dict[str, str] = None,
    ):
        """
        Initialize the CUBmadTransformer.

        Parameters
        ----------
        control_name_to_bmad : dict[str, str]
            A dictionary mapping control variable names to Bmad elements
            (e.g. {"QUAD:IN20:511": "QE03"})
        screen_attributes: dict[str, str]
            A dictionary of screen attributes needed to convert OTR histograms to images. Example:
            {
                "OTR2": {
                    "bins": 1024,
                    "resolution": 10, # um/pixel
                }
            }

        Notes
        -----
        - screens are assumed to be centered on the beam, if you want something better contact Bmad Developers

        """

        super().__init__(control_name_to_bmad=control_name_to_bmad)
        self.screen_attributes = screen_attributes

 def get_beam_elements(self):
    return {"input_element": "BEAM0", "output_element": "BEGUNDH"}

