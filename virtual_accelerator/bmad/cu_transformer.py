from typing import Any
from lume_bmad.transformer import BmadTransformer
from pytao import Tao
from lume_bmad.utils import get_beam_info, get_particle_group_at_element
from beamphysics.interfaces.bmad import write_bmad
from os import getcwd
import numpy as np

class CUBmadTransformer(BmadTransformer):
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

    def get_tao_property(self, tao: Tao, control_name: str):
        """
        Get a property of an element from Bmad via Tao and
        return its value in control system (EPICS) units.

        # TODO: implment other variable types as needed,
        # utilize future datamaps functionality from lcls-live or database

        Parameters
        ----------
        tao: Tao
            Instance of the Tao class.
        control_name: str
            Name of the control variable to retrieve. Example: "QUAD:IN20:511:BCTRL"

        Returns
        -------
        Any
            Value of the requested property.

        """

        # Map control name to element and attribute
        # get prefix
        if "Image" in control_name:
            element_name = self.control_name_to_bmad[
                ":".join(control_name.split(":")[:3])
            ]
            attr = ":".join(control_name.split(":")[3:])
        else:
            element_name = self.control_name_to_bmad[
                ":".join(control_name.split(":")[:-1])
            ]
            attr = ":".join(control_name.split(":")[-1:])

        device_type = control_name.split(":")[0]  # QUAD, KLYS, etc.
        ele_attr = tao.ele_gen_attribs(element_name)

        if device_type == "QUAD":
            if attr in ["BCTRL", "BACT", "BDES"]:
                # convert from Bmad units to EPICS units
                return -ele_attr["B1_GRADIENT"] * ele_attr["L"] * 10
        elif device_type == "SOLN":
            if attr in ["BCTRL", "BACT", "BDES"]:
                return ele_attr["BS_FIELD"] * 10  # TODO confirm this conversion
        elif device_type in ["KLYS", "ACCL"]:
            if attr in ["ENLD", "ADES"]:
                tao.ele_control_var(element_name)
                return tao.ele_control_var(element_name)["ENLD_MEV"]
            if attr in ["PHAS", "PDES"]:
                return tao.ele_control_var(element_name)["PHASE_DEG"]
            if attr == "BEAMCODE1_STAT":
                return tao.ele_control_var(element_name)["IN_USE"]
        elif device_type in ["XCOR", "YCOR", "EFC"]:
            return tao.ele_gen_attribs(element_name)["BL_KICK"]
        elif device_type == "OTRS":
            if attr == "Image:ArrayData":
                bins = self.screen_attributes[element_name]["bins"]
                resolution = (
                    self.screen_attributes[element_name]["resolution"] * 1e-6
                )  # um / pixel
                range = bins * resolution / 2

                if tao.tao_global()["track_type"] != "beam":
                    return np.zeros((bins[0], bins[1]))
                else:
                    beam = get_particle_group_at_element(tao, element_name)
                    H, _ = beam.histogramdd(
                        "x",
                        "y",
                        bins=bins,
                        range=np.stack(((-range[0], range[0]), (-range[1], range[1]))),
                    )
                    return H
            elif attr == "Image:ArraySize1_RBV":
                return self.screen_attributes[element_name]["bins"][0]
            elif attr == "Image:ArraySize0_RBV":
                return self.screen_attributes[element_name]["bins"][1]
            elif attr == "RESOLUTION":
                return self.screen_attributes[element_name]["resolution"]

        else:
            return ele_attr[attr]

    def get_tao_commands(self, tao: Tao, pvdata: dict[str, Any]) -> list[str]:
        """
        Get Tao commands to set a property of an element in Bmad via Tao. Handle
        mapping control names to element attributes and any necessary unit conversions as needed.

        Parameters
        ----------
        tao: Tao
            Instance of the Tao class.
        pvdata: dict[str, Any]
            Dictionary of control variable names and values to set

        Returns
        -------
        list[str]
            List of Tao commands to execute

        """
        klys_attr_to_bmad = {
            "ENLD": "ENLD_MEV",
            "PDES": "PHASE_DEG",
            "BEAMCODE1_STAT": "IN_USE",
            "BEAMCODE2_STAT": "IN_USE",
        }
        tao_cmds = []

        for pv in pvdata.keys():
            value = pvdata[pv]
            if "input_beam" == pv:
                # Save ParticleGroup to file and re-init beam
                beam_info = get_beam_info(tao)
                if beam_info["track_type"] == "beam":
                    file_name = getcwd() + "/model_bmad_beam_pg"
                    write_bmad(value, file_name, p0c=value["mean_p"])
                    tao.cmd(f"set beam_init position_file = {file_name}")
                else:
                    print("Single Particle tracking, no beam information saved")
                continue
            pv_name = ":".join(pv.split(":")[0:3])
            attr = pv.split(":")[3]
            element = self.control_name_to_bmad[pv_name]
            device_type = pv_name.split(":")[0]  # QUAD, KLYS, etc.
            if device_type == "OTRS":
                continue
            if device_type == "QUAD":
                if attr == "BCTRL" or attr == "BDES":
                    # convert from EPICS units to Bmad units
                    ele_attr = tao.ele_gen_attribs(element)
                    bmad_value = -value / (ele_attr["L"] * 10)
                    bmad_attr = "b1_gradient"
            elif device_type in ["XCOR", "YCOR"]:
                if attr == "BCTRL" or attr == "BDES":
                    bmad_value = -0.1 * value
                    bmad_attr = "bl_kick"
            elif device_type == "SOLN":
                if attr == "BCTRL" or attr == "BDES":
                    bmad_value = -0.1 * value
                    bmad_attr = "BS_FIELD"
            elif device_type == "KLYS":
                bmad_value = value
                bmad_attr = klys_attr_to_bmad[attr]
            else:
                print(f"Do not know about {device_type} {attr}")
            # print(f"set ele for {device_type} {attr} {element} {bmad_attr} = {bmad_value}")
            tao_cmd = f"set ele {element} {bmad_attr} = {bmad_value}"
            tao_cmds.append(tao_cmd)

        return tao_cmds

    def get_beam_elements(self):
        return {"input_element": "OTR2", "output_element": "BEGUNDH"}
