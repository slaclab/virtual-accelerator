import os
from dataclasses import dataclass
from pathlib import Path
import yaml

from virtual_accelerator.utils.optional_dependencies import import_optional
from virtual_accelerator.utils.variables import get_element_attr_mapping


@dataclass(frozen=True)
class BmadModelSpec:
    feature: str
    lattice_env_var: str
    tao_init_relpath: str
    screens: tuple[str, ...]
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
    from virtual_accelerator.bmad.variables import (
        get_variables,
        get_screen_variables,
    )

    lattice_root = os.environ[spec.lattice_env_var]
    init_file = os.path.join(lattice_root, spec.tao_init_relpath)
    tao = Tao(f"-init {init_file} -noplot -slice_lattice {start_element}:{end_element}")

    # set tracking to start_element
    tao.cmd(f"set beam track_start = {start_element}")

    if custom_tao_commands is not None:
        for cmd in custom_tao_commands:
            tao.cmd(cmd)

    variables = get_variables(tao, get_element_attr_mapping())

    # add screen variables based on configuration
    config_path = Path(__file__).parent / ".." / "utils" / spec.profmon_config_filename
    with config_path.open("r", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    lattice_elements = {
        element_name.split("#")[0] for element_name in tao.lat_list("*", "ele.name")
    }
    active_screens = tuple(
        screen_name for screen_name in spec.screens if screen_name in lattice_elements
    )
    for screen_name in active_screens:
        variables.extend(get_screen_variables(tao, screen_name, info))

    model = LUMEBmadModel(
        tao=tao,
        action_variables=variables,
        dump_locations=list(active_screens),
    )

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
