import os
from dataclasses import dataclass
from pathlib import Path
import yaml

from virtual_accelerator.bmad.variables import get_all_element_types, get_variables
from virtual_accelerator.utils.optional_dependencies import import_optional
from virtual_accelerator.utils.variables import get_element_attr_mapping

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BmadModelSpec:
    feature: str
    lattice_env_var: str
    tao_init_relpath: str
    profmon_config_filename: str
    mapping_beampath: str | None = None
    database_relpath: str = "bmad/conversion/from_oracle/lcls_elements.csv"
    default_track_start: str | None = None
    default_beam_relpath: str | None = None


def _check_optional_modules(module_names: list[str], feature: str, extra: str) -> None:
    """Validate all optional modules for a feature in a single gate check."""
    for module_name in module_names:
        import_optional(module_name, feature=feature, extra=extra)


def build_bmad_model(
    spec: BmadModelSpec,
    start_element: str,
    end_element: str,
    track_beam: bool,
    custom_beam_path: str | None,
    custom_tao_commands: list[str] | None = None,
    custom_aliases: dict[str, str] | None = None,
):
    """Build a lattice-specific LUMEBmadModel from a shared implementation."""

    _check_optional_modules(
        [
            "pytao",
            "lume_bmad.model",
            "beamphysics.interfaces.bmad",
            "virtual_accelerator.bmad.variables",
        ],
        feature=spec.feature,
        extra="bmad",
    )

    from pytao import Tao
    from lume_bmad.model import LUMEBmadModel

    lattice_root = os.environ[spec.lattice_env_var]
    init_file = os.path.join(lattice_root, spec.tao_init_relpath)
    tao = Tao(f"-init {init_file} -noplot -slice_lattice {start_element}:{end_element}")

    # set tracking to start_element
    tao.cmd(f"set beam track_start = {start_element}")

    # apply any custom tao commands (e.g. for setting up custom aliases or other tao configuration needed for the model)
    if custom_tao_commands is not None:
        for cmd in custom_tao_commands:
            tao.cmd(cmd)

    # handle custom aliases if provided
    if custom_aliases is not None:
        for element, alias in custom_aliases.items():
            try:
                tao.cmd(f"set ele {element} alias = {alias}")
            except Exception as e:
                logger.warning(f"Failed to set custom alias for element {element}: {e}")

    # get screen configuration for the model based on the provided spec
    config_path = Path(__file__).parent / ".." / "utils" / spec.profmon_config_filename
    with config_path.open("r", encoding="utf-8") as f:
        screen_config_dict = yaml.safe_load(f)

    # get variables for all elements in the lattice based on the element attribute mapping and screen configuration for the model
    variables = get_variables(tao, get_element_attr_mapping(), screen_config_dict)

    # get list of screens that are present in the lattice
    element_types = get_all_element_types(tao)
    active_screens = tuple(
        element
        for element, element_type in element_types.items()
        if element_type == "Screen"
    )

    # create LUMEBmadModel with the Tao instance, variables, and active screens for beam dumping
    model = LUMEBmadModel(
        tao=tao,
        action_variables=variables,
        dump_locations=list(active_screens),
    )

    # if tracking is enabled, set up the beam in the model based on the provided custom beam path or default beam path in the spec
    if track_beam:
        if custom_beam_path is not None:
            beam_path = custom_beam_path
        elif (
            spec.default_track_start is not None
            and spec.default_beam_relpath is not None
            and start_element == spec.default_track_start
        ):
            beam_path = Path(__file__).parent / ".." / spec.default_beam_relpath
        else:
            raise ValueError(
                "Cannot have track_beam=True for start_element "
                f"!= {spec.default_track_start} without providing custom_beam_path"
            )

        model.tao.cmd(f"set beam_init position_file = {beam_path}")
        model.set({"track_type": "beam"})

    return model
