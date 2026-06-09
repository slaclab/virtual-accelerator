import os
from dataclasses import dataclass
from pathlib import Path

from virtual_accelerator.utils.optional_dependencies import import_optional
from virtual_accelerator.utils.variables import (
    get_epics_to_name_or_overlay_mapping,
    split_control_and_observable,
)


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
            "lume_bmad.transformer",
            "beamphysics.interfaces.bmad",
            "virtual_accelerator.bmad.cu_transformer",
            "virtual_accelerator.bmad.variables",
        ],
        feature=spec.feature,
        extra="bmad",
    )

    from pytao import Tao
    from lume_bmad.model import LUMEBmadModel
    from virtual_accelerator.bmad.cu_transformer import CUBmadTransformer
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

    database_path = os.path.join(lattice_root, spec.database_relpath)
    control_name_to_element_name = get_epics_to_name_or_overlay_mapping(
        database_path,
        beampath=spec.mapping_beampath,
    )
    element_name_to_control_name = {
        v: k for k, v in control_name_to_element_name.items()
    }
    variables = get_variables(tao, element_name_to_control_name)

    control_variables, observable_variables = split_control_and_observable(variables)

    config_path = Path(__file__).parent / ".." / "utils" / spec.profmon_config_filename
    control_variables, screen_attributes, used_screens = get_screen_variables(
        tao,
        control_variables,
        list(spec.screens),
        config_path,
    )

    transformer = CUBmadTransformer(
        control_name_to_bmad=control_name_to_element_name,
        screen_attributes=screen_attributes,
    )

    model = LUMEBmadModel(
        tao=tao,
        control_variables=control_variables,
        output_variables=observable_variables,
        transformer=transformer,
        dump_locations=used_screens,
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
        model.set({"track_type": 1})

    return model
