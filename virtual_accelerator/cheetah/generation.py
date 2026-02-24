from __future__ import annotations
import importlib
from typing import Any
import os
import yaml
from pathlib import Path

from cheetah.accelerator import Segment
from lume.variables import ScalarVariable
from virtual_accelerator.cheetah.utils import get_compatible_devices

LCLS_ELEMENTS = os.path.join(
    Path(__file__).parent.resolve(),
    "lcls_elements.csv",
)
SLAC_VARIABLE_CONFIG_FILE = os.path.join(
    Path(__file__).parent.resolve(),
    "slac_variable_config.yaml",
)


def build_variables_for_devices(
    *,
    devices: dict[str, dict[str,str]],
    variable_config: dict,
) -> dict[str, ScalarVariable]:

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

def generate_slac_variables(
    lattice: Segment,
    *,
    lcls_elements_path: str | None = None,
    variable_config: dict[str, dict[str, dict[str, Any]]] | None = None,
    config_file: str | None = None,
) -> tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]:
    """
    Generate LUME ScalarVariables from SLAC control YAML definitions.

    This is a single-entry functional API that:

        1. Loads and merges SLAC device YAML files
        2. Filters devices to those present in a Cheetah lattice
        3. Applies a device-type → PV-attribute mapping
        4. Instantiates ScalarVariable objects
        5. Splits them into control and observable variables

    Parameters
    ----------
    lattice : Segment
        Cheetah accelerator segment.
    slac_tools_dir : str, optional
        Base directory containing SLAC YAML files.
    variable_config : dict, optional
        Device-type → PV-attribute → variable specification mapping.
    config_file : str, optional
        YAML config file containing both:
            - SLAC_VARIABLE_CONFIG

    Returns
    -------
    tuple[dict[str, ScalarVariable], dict[str, ScalarVariable]]
        (control_variables, observable_variables),
        both keyed by full PV name.

    Raises
    ------
    ValueError
        If configuration cannot be resolved.
    TypeError
        If configuration structure is invalid.

    Design Notes
    ------------
    - Stateless: no internal storage.
    - Deterministic: same inputs → same outputs.
    - Safe-by-default: errors during variable creation
      are handled via callback instead of stopping execution.
    """
    if lcls_elements_path is None:
        lcls_elements_path = str(LCLS_ELEMENTS)


    variable_config_resolved = resolve_variable_config(
        variable_config=variable_config,
        config_file=config_file,
    )

    print(variable_config_resolved)

    devices = get_compatible_devices(lcls_elements_path,lattice) #rename this function

    
    all_vars, mapping = build_variables_for_devices(
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
    *,
    variable_config,
    config_file,
) -> dict:

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
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Expected dotted path 'module.Class', got {path!r}"
        )

    module = importlib.import_module(module_path)
    return getattr(module, class_name)