import yaml
from lume.variables import ScalarVariable, ParticleGroupVariable
from typing import Any
from lume_bmad.transformer import BmadTransformer
from pytao import Tao
from beamphysics.interfaces.bmad import (
    bmad_to_particlegroup_data,
    particlegroup_to_bmad,
    write_bmad)
from beamphysics import ParticleGroup

###############################################################
# Utility functions for importing control and output variables
################################################################


def import_control_variables(control_variable_file: str):
    """
    Import control variables from a YAML file and define them as ScalarVariables.
    Also get the mapping between device PV names and Bmad element names.

    TODO: move SLAC specific mapping and unit conversions to slac-tools

    Parameters
    ----------
    control_variable_file: str
        Path to the YAML file containing control variable definitions.

    Returns
    -------
    dict[str, ScalarVariable]
        Dictionary of pv variables to ScalarVariable instances.
    dict[str, str]
        Mapping between PV names and Bmad element names + attributes.
    """

    control_name_to_bmad = {}
    var_dict = {}

    with open(control_variable_file, "r") as file:
        control_variable = yaml.safe_load(file)

    # handle quadrupoles
    quads = control_variable.get("quad", [])
    for quad in quads:
        pv_name = quad["pvname"]
        device_name = ':'.join(pv_name.split(':')[0:3])

        # map pv to bmad element name and attribute
        control_name_to_bmad[device_name] = quad["bmad_name"]
        
        var_dict[pv_name] = ScalarVariable(
            name=pv_name,
            #value_range=(quad["min_value"], quad["max_value"]),
            unit="kG-m",
            read_only=False,
        )

    # handle correctors
    corrs = control_variable.get("correctors", [])
    for corr in corrs:
        pv_name = corr["pvname"]
        device_name = ':'.join(pv_name.split(':')[0:3])

        # map pv to bmad element name and attribute
        control_name_to_bmad[device_name] = corr["bmad_name"]
        
        var_dict[pv_name] = ScalarVariable(
            name=pv_name,
            #value_range=(quad["min_value"], quad["max_value"]),
            unit="kG-m",
            read_only=False,
        )

    # handle klystrons
    klystron_keys = [k for k in control_variable.keys() if k[0] == 'K']
    for station in klystron_keys:
        klys = control_variable.get(station)[0]
        # map pv to bmad element name and attribute
        pv_name = klys['ampl_des_pvname']
        device_name = ':'.join(pv_name.split(':')[0:3])
        control_name_to_bmad[device_name] = klys["name"]
        
        var_dict[pv_name] = ScalarVariable(
            name=pv_name,
            value_range=(-180, 180),
            unit="kG-m",
            read_only=False,
        )
        pv_name = klys['phase_des_pvname']
        if klys["name"] in ['K24_1', 'K24_2', 'K24_3']:
            accl_name = ':'.join(pv_name.split(':')[0:3])
            control_name_to_bmad[accl_name] = klys["name"]
        
        var_dict[pv_name] = ScalarVariable(
            name=pv_name,
            #value_range=(quad["min_value"], quad["max_value"]),
            unit="Deg_S",
            read_only=False,
        )
        if pv_name.split(':')[0] == 'KLYS':
            pv_name = klys['accelerate_pvname']        
            var_dict[pv_name] = ScalarVariable(
                name=pv_name,
                value_range=(0, 1),
                unit="",
                read_only=False,
            )
        
    return var_dict, control_name_to_bmad
