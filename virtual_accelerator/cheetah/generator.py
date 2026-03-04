from __future__ import annotations
import importlib
from typing import Any
import os
import yaml
from pathlib import Path
import pprint
from cheetah.accelerator import Segment

from lume.variables import Variable, ScalarVariable, NDVariable
from virtual_accelerator.cheetah.utils import get_devs_from_lattice
from copy import copy

LCLS_ELEMENTS = os.path.join(
    Path(__file__).parent.resolve(),
    "lcls_elements.csv",
)
SLAC_VARIABLE_CONFIG_FILE = os.path.join(
    Path(__file__).parent.resolve(),
    "slac_variable_config.yaml",
)

def resolve_slac_variables(element, control_name, dev_attr_mapping):
    """
    Instantiate variables for a single lattice element using a SLAC variable
    configuration mapping.

    This function converts a device attribute configuration (typically loaded
    from the SLAC YAML variable config) into instantiated LUME `Variable`
    objects.

    For each configured attribute:
      - A full PV name is constructed as "{control_name}:{attr}".
      - The configured `variable_class` is instantiated.
      - Additional keyword arguments from the configuration are passed through.

    Special handling is applied for NDVariables whose shape must be determined
    from the lattice element (e.g. screen image arrays).

    Parameters
    ----------
    element : Any
        Cheetah lattice element instance (e.g. Screen, BPM, Quadrupole).

    control_name : str
        Control-system PV prefix for this device
        (e.g. "QUAD:IN20:425").

    dev_attr_mapping : dict[str, dict[str, Any]]
        Mapping of PV attribute -> variable specification loaded from the
        SLAC variable configuration.

        Example structure::

            {
                "Image:ArrayData": {
                    "variable_class": NDVariable,
                    "read_only": True,
                    "unit": "pixel",
                    "shape": None
                },
                "PNEUMATIC": {
                    "variable_class": ScalarVariable,
                    "read_only": False
                }
            }

    Returns
    -------
    dict[str, Variable]
        Mapping of full PV name -> instantiated LUME Variable
        (ScalarVariable or NDVariable).
    """
    variables = {}

    for attr, var_config in dev_attr_mapping.items():
        var_config = copy(var_config)
        variable_name = ':'.join((control_name, attr))
        variable_class = var_config.pop('variable_class')

        is_nd = isinstance(variable_class, type) and issubclass(variable_class, NDVariable)
        if is_nd and attr == "Image:ArrayData":
            var_config["shape"] = tuple(element.resolution)    

        variables[variable_name] = variable_class(name=variable_name, **var_config)
            #elif isinstance(variable_class, ScalarVariable) and attr == 'Image:ArraySize0_RBV':
                # can pass because these don't need defaults

    return variables
            
def build_variables_and_mapping(
    devices: dict[str, str],
    variable_config: dict,
    lattice: Segment,
    resolve_variable: callable | None = None
) -> tuple[dict[str, Variable], dict[str, str]]:
    """
    Build LUME variables for all devices present in a lattice.

    This function iterates over a set of lattice devices and uses the
    SLAC variable configuration to instantiate the appropriate LUME
    Variable objects for each device.

    For each lattice element:
      - The element type is determined from the lattice object.
      - The corresponding variable configuration block is retrieved.
      - Variables are instantiated using the provided resolver function.

    A reverse mapping from control-system prefix to lattice element name
    is also generated.

    Parameters
    ----------
    devices : dict[str, str]
        Mapping of lattice element name -> control-system PV prefix.

        Example::

            {
                "qb03": "QUAD:IN20:731",
                "yc01": "YCOR:IN20:425"
            }

    variable_config : dict
        Device-type -> PV attribute -> variable specification mapping
        loaded from the SLAC variable YAML configuration.

    lattice : Segment
        Cheetah accelerator lattice containing the device objects.

    resolve_variable : callable, optional
        Function responsible for creating variables for a single device.
        Defaults to `resolve_slac_variables`.

    Returns
    -------
    tuple[dict[str, Variable], dict[str, str]]

        all_vars
            Mapping of full PV name -> instantiated Variable.

        mapping
            Mapping of control-system PV prefix -> lattice element name.
    """


    if resolve_variable is None:
        resolve_variable = resolve_slac_variables

    all_vars: dict[str, Variable] = {}
    mapping: dict[str,str] = {}
    for lattice_element_name, control_name in devices.items():
        try:
            element = getattr(lattice, lattice_element_name)
            element_type = type(element).__name__
        except AttributeError as e:
            raise KeyError(f"Element {lattice_element_name!r} not found in lattice") from e
        
        dev_attr_mapping = variable_config.get(element_type)
        if dev_attr_mapping is not None:
            variables = resolve_variable(
                element,
                control_name,
                dev_attr_mapping
            )

            all_vars.update(variables)

            mapping[control_name] = lattice_element_name
        else:
            continue

    return all_vars, mapping
    

def generate_variables_and_mapping(
    lattice: Segment,
    lcls_elements_path: str | None = None,
    variable_config: dict[str, dict[str, dict[str, Any]]] | None = None,
    config_file: str | None = None,
) -> tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]:
    """
    Generate LUME variables for a SLAC-style control system from a Cheetah lattice.

    This is the primary high-level entry point for building variables used by the
    virtual accelerator. The function performs the following steps:

      1. Resolves the variable configuration (either directly provided or
         loaded from a YAML configuration file).
      2. Determines which devices from the SLAC elements table exist in the
         provided lattice.
      3. Instantiates the appropriate Variable objects for each device.
      4. Returns both the variables and a reverse mapping to lattice elements.

    Parameters
    ----------
    lattice : Segment
        Cheetah accelerator segment containing the lattice elements.

    lcls_elements_path : str, optional
        Path to the CSV file describing SLAC lattice elements and their
        control-system identifiers. Defaults to the module-level
        `LCLS_ELEMENTS`.

    variable_config : dict, optional
        Pre-loaded SLAC variable configuration mapping.

        If provided, this configuration will be used directly instead of
        loading a YAML configuration file.

    config_file : str, optional
        Path to a YAML configuration file containing `SLAC_VARIABLE_CONFIG`.
        Used only when `variable_config` is not provided.

    Returns
    -------
    tuple[dict[str, Variable], dict[str, str]]

        all_vars
            Mapping of full PV name -> instantiated Variable
            (ScalarVariable or NDVariable).

        mapping
            Mapping of control-system PV prefix -> lattice element name.
    """
    if lcls_elements_path is None:
        lcls_elements_path = str(LCLS_ELEMENTS)

    variable_config_resolved = resolve_variable_config(
        variable_config=variable_config,
        config_file=config_file,
    )


    devices = get_devs_from_lattice(lcls_elements_path,lattice) #rename this function

    all_vars, mapping = build_variables_and_mapping(devices,variable_config_resolved,lattice)

    return all_vars, mapping

def load_config_from_file(path: str) -> tuple[dict[str, Any], list[str]]:
    """
    Load the SLAC variable configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to a YAML configuration file containing the key
        `SLAC_VARIABLE_CONFIG`.

    Returns
    -------
    dict[str, Any]
        Parsed SLAC variable configuration mapping.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    TypeError
        If `SLAC_VARIABLE_CONFIG` exists but is not a dictionary.

    Notes
    -----
    Expected YAML structure::

        SLAC_VARIABLE_CONFIG:
            DEVICE_TYPE:
                PV_ATTR:
                    variable_class: lume.variables.ScalarVariable
                    read_only: true
                    unit: mm
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    vc = cfg.get("SLAC_VARIABLE_CONFIG")

    if not isinstance(vc, dict):
        raise TypeError("Config key SLAC_VARIABLE_CONFIG must be a dict.")
    return vc



def resolve_variable_config(
    variable_config,
    config_file,
) -> dict:
    """
    Resolve and normalize the SLAC variable configuration.

    This function ensures a usable variable configuration mapping is available.
    If a configuration dictionary is not provided directly, it is loaded from
    a YAML configuration file.

    Additionally, string values for `variable_class` are resolved into actual
    Python classes using `import_from_dotted_path`.

    Parameters
    ----------
    variable_config : dict or None
        Pre-loaded configuration mapping. If provided, it is used directly.

    config_file : str or None
        YAML configuration file used when `variable_config` is not supplied.
        Defaults to `SLAC_VARIABLE_CONFIG_FILE`.

    Returns
    -------
    dict
        Normalized variable configuration where `variable_class` entries
        are Python classes instead of dotted strings.
    """

    if variable_config is None:
        if config_file is None:
            config_file = SLAC_VARIABLE_CONFIG_FILE

        variable_config = load_config_from_file(config_file)
    #print(variable_config)

    # resolve variable_class strings
    for device_type, attrs in variable_config.items():
        for attr, spec in attrs.items():
            vc = spec.get("variable_class")
            if isinstance(vc, str):
                spec["variable_class"] = import_from_dotted_path(vc)

    return variable_config

def split_control_and_observable(
    all_vars: dict[str, ScalarVariable],
) -> tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]:
    """
    Separate variables into control and observable groups.

    Parameters
    ----------
    all_vars : dict[str, ScalarVariable]
        Mapping of PV name to ScalarVariable.

    Returns
    -------
    tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]
        (control_vars, observable_vars)

    Notes
    -----
    Classification is based on the `read_only` attribute of
    each ScalarVariable:
        read_only == False → control variable
        read_only == True  → observable variable
    """
    control_vars = {k: v for k, v in all_vars.items() if not v.read_only}
    observable_vars = {k: v for k, v in all_vars.items() if v.read_only}
    return control_vars, observable_vars

def import_from_dotted_path(path: str):
    """
    Import and return an object from a dotted module path.

    This utility resolves strings of the form:

        "package.module.ClassName"

    into the corresponding Python object by importing the module
    and retrieving the named attribute.

    Parameters
    ----------
    path : str
        Fully-qualified dotted path to an importable attribute or class.

    Returns
    -------
    Any
        The imported attribute (typically a class or callable).

    Raises
    ------
    ValueError
        If `path` does not contain both a module and attribute
        component (i.e., missing a dot separator).

    ModuleNotFoundError
        If the module portion of the path cannot be imported.

    AttributeError
        If the requested attribute does not exist in the module.

    Examples
    --------
    >>> import_from_dotted_path(
    ...     "lume.variables.ScalarVariable"
    ... )
    <class 'lume.variables.ScalarVariable'>

    Notes
    -----
    This function is commonly used when loading configuration from
    YAML or JSON files where Python classes must be specified as
    strings and resolved at runtime.
    """
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Expected dotted path 'module.Class', got {path!r}"
        )

    module = importlib.import_module(module_path)
    return getattr(module, class_name)