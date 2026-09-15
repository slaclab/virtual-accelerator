from impact import Impact
from distgen import Generator
import os
from pathlib import Path
import yaml
from dataclasses import dataclass

from impact.model.distgen.distgen_impact_model import LUMEDistgenImpactModel
from virtual_accelerator.impact.actions import ImpactGroupVariable
from virtual_accelerator.utils.variables import (
    get_element_attr_mapping,
    get_element_name_to_base_pv_mapping,
)
from virtual_accelerator.impact.variables import get_variables


@dataclass(frozen=True)
class ImpactModelSpec:
    lattice_env_var: str
    distgen_file: str
    n_particles: int
    profmon_config_filename: str
    stop_location: str | float = None
    impact_file: str = None
    impact_yaml_file: str = None
    numprocs: int = 1
    space_charge: bool = False


def get_impact_and_distgen(spec: ImpactModelSpec):
    """Get the Impact and Distgen objects based on the provided specification."""
    # get files
    lattice_root = os.environ[spec.lattice_env_var]
    distgen_file = os.path.join(lattice_root, spec.distgen_file)

    # either an impact file or an impact YAML file must be provided
    impact_file = (
        os.path.join(lattice_root, spec.impact_file) if spec.impact_file else None
    )
    impact_yaml_file = (
        os.path.join(lattice_root, spec.impact_yaml_file)
        if spec.impact_yaml_file
        else None
    )

    if impact_file is None and impact_yaml_file is None:
        raise ValueError(
            "Either an impact file or an impact YAML file must be provided"
        )

    # if an impact YAML file is provided, use it to create the Impact object
    if impact_yaml_file is not None:
        impact = Impact.from_yaml(impact_yaml_file)
    else:
        impact = Impact(impact_file)

    distgen = Generator(distgen_file)

    return impact, distgen


def get_actions_from_groups(impact: Impact, spec: ImpactModelSpec):
    """
    Get the action variables for the impact model based
    on the groups defined in the impact YAML file in the model spec.
    """

    lattice_root = os.environ[spec.lattice_env_var]
    with open(
        os.path.join(lattice_root, spec.impact_yaml_file), "r", encoding="utf-8"
    ) as f:
        impact_config_dict = yaml.safe_load(f)

    actions = []
    for group_name, group_info in impact_config_dict.get("group", {}).items():
        # only add the action if the group ele_names are present in the impact.ele attribute
        if not all(
            ele_name in impact.ele for ele_name in group_info.get("ele_names", [])
        ):
            continue

        action = ImpactGroupVariable(
            name=f"group:{group_name}",
            group_name=group_name,
            group_key=group_info["var_name"],
        )
        actions.append(action)
    return actions


def set_stop_location(impact: Impact, stop_location: str | float):
    """
    Set z stop location based on the beginning of the named element or a float value

    Parameters:
    -----------
    impact : Impact
        The impact model object.
    stop_location : str | float
        The stop location, either as the name of an element (str) or a float value representing the z position.

    Returns:
    --------
    None

    """
    if isinstance(stop_location, str):
        try:
            element = impact.ele[stop_location]
            stop_location_z = element["s"]
        except KeyError:
            raise ValueError(
                f"Element '{stop_location}' not found in the impact model."
            )
    else:
        stop_location_z = float(stop_location)

    impact.stop = stop_location_z

    # remove elements that are downstream of the stop location
    impact.ele = {k: v for k, v in impact.ele.items() if v["s"] <= impact.stop}
    impact.input["lattice"] = [
        elem for elem in impact.lattice if elem.get("s", float("inf")) <= impact.stop
    ]
    return impact


def build_impact_model(spec: ImpactModelSpec):
    """Build and return the impact model based on the provided specification."""
    impact, distgen = get_impact_and_distgen(spec)

    # set the parameters of the impact model
    impact.header["Np"] = spec.n_particles
    impact.numprocs = spec.numprocs
    impact.header["Bcurr"] = 1 if spec.space_charge else 0

    if spec.stop_location is not None:
        impact = set_stop_location(impact, spec.stop_location)

    impact.run()

    # set the parameters of the distgen model
    distgen["n_particle"] = spec.n_particles

    # create the LUMEDistgenImpactModel from the distgen and impact objects
    model = LUMEDistgenImpactModel.from_objects(distgen, impact)

    # register additional actions to lume model
    element_name_to_base_pv_mapping = get_element_name_to_base_pv_mapping(
        os.environ[spec.lattice_env_var]
    )

    # get the screen configuration dictionary from the profmon config file
    config_path = Path(__file__).parent / ".." / "utils" / spec.profmon_config_filename
    with config_path.open("r", encoding="utf-8") as f:
        screen_config_dict = yaml.safe_load(f)

    # get the action variables for the impact model based on the lattice elements
    action_variables = get_variables(
        impact,
        get_element_attr_mapping(),
        screen_config_dict,
        element_name_to_base_pv_mapping,
    )
    for var in action_variables:
        model.register_impact_action_variable(var)

    # add actions based on groups
    if spec.impact_yaml_file is not None:
        for action in get_actions_from_groups(impact, spec):
            model.register_impact_action_variable(action)

    return model
