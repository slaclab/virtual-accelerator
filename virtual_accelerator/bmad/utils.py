import yaml
from lume.variables import ScalarVariable

TAO_OUTPUT_UNITS = {
    "ele.name": "",
    "ele.ix_ele": "",
    "ele.ix_branch": "",
    "ele.a.beta": "m",
    "ele.a.alpha": "",
    "ele.a.eta": "m",
    "ele.a.etap": "",
    "ele.a.gamma": "1/m",
    "ele.a.phi": "",
    "ele.b.beta": "m",
    "ele.b.alpha": "",
    "ele.b.eta": "m",
    "ele.b.etap": "",
    "ele.b.gamma": "1/m",
    "ele.b.phi": "",
    "ele.x.eta": "m",
    "ele.x.etap": "",
    "ele.y.eta": "m",
    "ele.y.etap": "",
    "ele.s": "m",
    "ele.l": "m",
    "ele.e_tot": "eV",
    "ele.p0c": "eV",
    "ele.mat6": "",
    "ele.vec0": "m",
}


def import_control_variables(control_variable_file: str):
    """
    Import control variables from a YAML file and define them as Variable instances.
    Also get the mapping between device PV names and Bmad element names.

    TODO: move SLAC specific mapping and unit conversions to slac-tools

    Parameters
    ----------
    control_variable_file: str
        Path to the YAML file containing control variable definitions.

    Returns
    -------
    dict[str, Variable]
        Dictionary of pv variables to Variable instances.
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

        # map pv to bmad element name and attribute
        control_name_to_bmad[pv_name] = " ".join(
            [quad["bmad_name"], quad["bmad_attribute"]]
        )
        var_dict[pv_name] = ScalarVariable(
            name=pv_name,
            value_range=(quad["min_value"], quad["max_value"]),
            unit="kG",
            read_only=False,
        )

    return var_dict, control_name_to_bmad


def import_output_variables(output_variable_file: str):
    """
    Import output variables from a YAML file and define them as Variable instances.
    Note that output variables are read-only.

    TODO: move SLAC specific mapping and unit conversions to slac-tools

    Parameters
    ----------
    output_variable_file: str
        Path to the YAML file containing output variable definitions.

    Returns
    -------
    dict[str, Variable]
        Dictionary of output variables mapped by their names.
    """

    out_dict = {}

    with open(output_variable_file, "r") as file:
        output_variables = yaml.safe_load(file)

    for ele in output_variables.keys():
        for attr in output_variables[ele].keys():
            name = attr.replace("ele", "").replace(".", "_")
            name = ele + name + "_"
            out_dict[name] = ScalarVariable(
                name=name,
                unit=TAO_OUTPUT_UNITS[attr],
                read_only=True,
            )

    return out_dict


###############################################################
# Utility classes / functions for Bmad/Tao interaction
###############################################################
