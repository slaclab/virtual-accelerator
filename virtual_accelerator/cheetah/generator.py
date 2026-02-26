from __future__ import annotations
import importlib
from typing import Any
import os
import yaml
from pathlib import Path

from cheetah.accelerator import Segment
from lume.variables import ScalarVariable
from virtual_accelerator.cheetah.utils import get_devices_from_lattice

LCLS_ELEMENTS = os.path.join(
    Path(__file__).parent.resolve(),
    "lcls_elements.csv",
)
SLAC_VARIABLE_CONFIG_FILE = os.path.join(
    Path(__file__).parent.resolve(),
    "slac_variable_config.yaml",
)


def build_variables_and_mapping(
    devices: dict[str, dict[str,str]],
    variable_config: dict,
) -> dict[str, ScalarVariable]:
    """ 
    Instantiate ScalarVariables for a set of devices using a device-type mapping.

    Given a mapping of lattice/beamline devices to their control-system metadata
    (control PV prefix and device type), this function:

      - Looks up the configured PV attributes for each device type in
        `variable_config`
      - Builds full PV names as "{control_name}:{attr}"
      - Instantiates the configured `variable_class` with `name=<pv>` plus any
        remaining spec kwargs
      - Builds a reverse mapping from control_name -> device identifier

    Parameters
    ----------
    devices : dict[str, dict[str, str]]
        Mapping of device identifier -> device metadata. Each device metadata
        dict must contain exactly two entries whose values are, in iteration
        order, (control_name, dev_type). For example:
            {
                "QUAD_001": {"control_name": "QUAD:IN20:731", "type": "QUAD"},
                ...
            }
        Note: the current implementation uses `dev_config.values()`; if dict
        key order is not guaranteed by construction, prefer explicit keys.

    variable_config : dict
        Device-type -> PV-attribute -> spec mapping. Each spec must include:
            - "variable_class": callable to construct a ScalarVariable
        and may include additional keyword arguments passed to the constructor
        (e.g., unit, read_only, limits, metadata).

    Returns
    -------
    tuple[dict[str, ScalarVariable], dict[str, str]]
        (all_vars, mapping) where:
          - all_vars maps full PV name -> ScalarVariable instance
          - mapping maps control_name -> device identifier

    Raises
    ------
    KeyError
        If a spec is missing required keys (e.g., "variable_class").
    TypeError
        If "variable_class" is not callable or constructor kwargs are invalid.
    Exception
        Propagates any exception raised during variable instantiation.
    """


    all_vars: dict[str, ScalarVariable] = {}
    mapping: dict[str,str] = {}
    for device, dev_config in devices.items():
        control_name, dev_type = dev_config.values()
        dev_attr_mapping = variable_config.get(dev_type)
        if dev_attr_mapping is None:
            continue

        for attr in dev_attr_mapping:
            try:
                spec = dev_attr_mapping[attr]
                variable_class = spec["variable_class"]
                kwargs = {k: v for k, v in spec.items() if k != "variable_class"}
                variable_name = ':'.join((control_name,attr))
                all_vars[variable_name] = variable_class(name=variable_name, **kwargs)
                mapping[control_name] = device
            except Exception as exc:
                raise exc

    return all_vars, mapping

def generate_variables_and_mapping(
    lattice: Segment,
    lcls_elements_path: str | None = None,
    variable_config: dict[str, dict[str, dict[str, Any]]] | None = None,
    config_file: str | None = None,
) -> tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]:
    """
    Generate ScalarVariables for SLAC-style controls from a Cheetah lattice.

    This is a single-entry functional API that:

      1. Resolves the variable configuration (either provided directly or loaded
         from a YAML file).
      2. Filters/collects devices compatible with the provided lattice using the
         LCLS elements table.
      3. Instantiates variables for each device using the device-type -> PV-attr
         mapping.
      4. Returns the created variables and a reverse mapping from control PV
         prefix to lattice device identifier.

    Parameters
    ----------
    lattice : Segment
        Cheetah accelerator segment used to determine which devices are present.

    lcls_elements_path : str, optional
        Path to the LCLS elements CSV used to map/identify control names and
        device types. If not provided, defaults to the module-level LCLS_ELEMENTS.

    variable_config : dict, optional
        Device-type -> PV-attribute -> variable specification mapping. If not
        provided, the configuration is loaded from `config_file` (or the default
        SLAC_VARIABLE_CONFIG_FILE).

    config_file : str, optional
        YAML configuration file containing `SLAC_VARIABLE_CONFIG`. Used only when
        `variable_config` is not provided.

    Returns
    -------
    tuple[dict[str, ScalarVariable], dict[str, str]]
        (all_vars, mapping) where:
          - all_vars maps full PV name -> ScalarVariable instance
          - mapping maps control_name -> device identifier

    Raises
    ------
    FileNotFoundError
        If `config_file` is required but does not exist.
    TypeError
        If the loaded configuration has an invalid structure (e.g., wrong types).
    ValueError
        If configuration cannot be resolved or required information is missing.
    Exception
        Propagates exceptions raised during device filtering or variable creation.

    Notes
    -----
    This function is stateless and deterministic given the same inputs and file
    contents. Errors during variable instantiation are not suppressed.
    """
    if lcls_elements_path is None:
        lcls_elements_path = str(LCLS_ELEMENTS)


    variable_config_resolved = resolve_variable_config(
        variable_config=variable_config,
        config_file=config_file,
    )

    

    devices = get_devices_from_lattice(lcls_elements_path,lattice) #rename this function

    
    all_vars, mapping = build_variables_and_mapping(
        devices=devices,
        variable_config=variable_config_resolved, 
    )

    return all_vars, mapping

def load_config_from_file(path: str) -> tuple[dict[str, Any], list[str]]:
    """
    Load SLAC variable configuration and ignore flags from YAML.

    Parameters
    ----------
    path : str
        Path to a YAML configuration file containing:
            - SLAC_VARIABLE_CONFIG (dict)

    Returns
    -------
    dict[str, Any] - variable_config

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TypeError
        If required keys are missing or have incorrect types.

    YAML Format
    -----------
    SLAC_VARIABLE_CONFIG:
        DEVICE_TYPE:
            PV_ATTR:
                variable_class: SomeScalarVariableClass
                read_only: true
                ...
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
    Resolve the variable configuration and normalize variable_class entries.

    If `variable_config` is provided, it is used directly. Otherwise the
    configuration is loaded from `config_file` (or the default
    SLAC_VARIABLE_CONFIG_FILE).

    Additionally, this function resolves any string-valued "variable_class"
    entries (e.g., "pkg.module.ClassName") into the corresponding Python class
    object via `import_from_dotted_path`.

    Parameters
    ----------
    variable_config : dict or None
        Device-type -> PV-attribute -> spec mapping. If None, the configuration
        is loaded from `config_file`.

    config_file : str or None
        YAML configuration file used when `variable_config` is None. If None,
        defaults to SLAC_VARIABLE_CONFIG_FILE.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Resolved configuration with "variable_class" entries normalized to class
        objects (callables).

    Raises
    ------
    FileNotFoundError
        If loading from `config_file` and the file does not exist.
    TypeError
        If the loaded configuration structure is invalid.
    ValueError
        If a "variable_class" string is not a valid dotted path.
    ImportError
        If a module/class referenced by "variable_class" cannot be imported.
    """

    if variable_config is None:
        if config_file is None:
            config_file = SLAC_VARIABLE_CONFIG_FILE

        variable_config = load_config_from_file(config_file)

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