from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from lume.variables import EnumVariable, ScalarVariable
from lume_torch.variables import TorchNDVariable
from scipy import constants

from virtual_accelerator.surrogates.beam_output import (
    BeamOutputAugmentation,
    BeamOutputModel,
)

DEFAULT_INJECTOR_FIELD_MAPPING = {
    "sigma_x": "OTRS:IN20:571:XRMS",
    "sigma_y": "OTRS:IN20:571:YRMS",
    "sigma_z": "sigma_z",
    "norm_emit_x": "norm_emit_x",
    "norm_emit_y": "norm_emit_y",
}


class BCTRLFamilyAugmentation(BeamOutputAugmentation):
    """Add BCTRL-compatible aliases (BACT/BDES/BMIN/BMAX/...) for controls."""

    def __init__(
        self,
        source_suffix: str = ":BCTRL",
        min_value: float = -100.0,
        max_value: float = 100.0,
        status_value: str = "Ready",
    ) -> None:
        self.source_suffix = source_suffix
        self.min_value = min_value
        self.max_value = max_value
        self.status_value = status_value

    def _base_name(self, name: str, suffix: str) -> str:
        return name[: -len(suffix)]

    def augment_supported_variables(
        self, supported_variables: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        new_variables: dict[str, Any] = {}

        for name, variable in supported_variables.items():
            if not name.endswith(self.source_suffix):
                continue

            base_name = self._base_name(name, self.source_suffix)
            unit = getattr(variable, "unit", "")
            for suffix in [":BACT", ":BMIN", ":BMAX", ":BCTRL.DRVL", ":BCTRL.DRVH"]:
                full_name = base_name + suffix
                new_variables[full_name] = ScalarVariable(
                    unit=unit, name=full_name, read_only=True
                )

            full_name = base_name + ":BDES"
            new_variables[full_name] = ScalarVariable(
                unit=unit,
                name=full_name,
                read_only=False,
            )

            for suffix in [":STATCTRLSUB.T", ":CTRL"]:
                full_name = base_name + suffix
                new_variables[full_name] = EnumVariable(
                    name=full_name,
                    options=[self.status_value],
                    read_only=True,
                )

        return new_variables

    def map_set_values(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        mapped_values: dict[str, Any] = {}

        for name, value in values.items():
            if name.endswith(":BDES"):
                base_name = self._base_name(name, ":BDES")
                mapped_values[base_name + self.source_suffix] = value
            else:
                mapped_values[name] = value

        return mapped_values

    def resolve_get_value(
        self, name: str, cache: Mapping[str, Any]
    ) -> tuple[bool, Any]:
        if name.endswith(":BACT") or name.endswith(":BDES"):
            base_name = self._base_name(
                name, ":BACT" if name.endswith(":BACT") else ":BDES"
            )
            return True, cache[base_name + self.source_suffix]

        if name.endswith(":BMIN") or name.endswith(":BCTRL.DRVL"):
            return True, self.min_value

        if name.endswith(":BMAX") or name.endswith(":BCTRL.DRVH"):
            return True, self.max_value

        if name.endswith(":STATCTRLSUB.T") or name.endswith(":CTRL"):
            return True, self.status_value

        return False, None


@dataclass(frozen=True)
class InjectorCovarianceAugmentation(BeamOutputAugmentation):
    """Derive covariance_matrix from scalar injector outputs."""

    energy_eV: float | None = None
    field_mapping: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_INJECTOR_FIELD_MAPPING)
    )

    def augment_supported_variables(
        self, supported_variables: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if "covariance_matrix" in supported_variables:
            return {}

        return {
            "covariance_matrix": TorchNDVariable(
                name="covariance_matrix",
                unit="",
                shape=(6, 6),
            )
        }

    def update_outputs(self, outputs: dict[str, Any], model: BeamOutputModel) -> None:
        energy = model.p0c if self.energy_eV is None else self.energy_eV
        outputs["covariance_matrix"] = torch.as_tensor(
            compute_injector_covariance_matrix(
                outputs,
                energy,
                field_mapping=self.field_mapping,
            ),
            dtype=torch.float32,
        )


def _to_scalar(value: Any) -> float:
    data = np.asarray(value)
    if data.size != 1:
        raise ValueError(f"Expected scalar surrogate output, got shape {data.shape}")
    return float(data.reshape(-1)[0])


def compute_injector_covariance_matrix(
    state: Mapping[str, Any],
    energy_eV: float,
    field_mapping: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Compute a diagonal covariance matrix from scalar beam outputs."""

    mapping = dict(DEFAULT_INJECTOR_FIELD_MAPPING)
    if field_mapping is not None:
        mapping.update(field_mapping)

    sigma_x = _to_scalar(state[mapping["sigma_x"]]) * 1e-6
    sigma_y = _to_scalar(state[mapping["sigma_y"]]) * 1e-6
    sigma_z = _to_scalar(state[mapping["sigma_z"]]) * 1e-6

    relativistic_gamma = energy_eV / (
        constants.value("electron mass energy equivalent in MeV") * 1e6
    )
    emit_x = _to_scalar(state[mapping["norm_emit_x"]]) / relativistic_gamma
    emit_y = _to_scalar(state[mapping["norm_emit_y"]]) / relativistic_gamma

    covariance_matrix = np.zeros((6, 6), dtype=np.float32)
    covariance_matrix[0, 0] = sigma_x**2
    covariance_matrix[2, 2] = sigma_y**2
    covariance_matrix[1, 1] = emit_x**2 * energy_eV**2 / covariance_matrix[0, 0]
    covariance_matrix[3, 3] = emit_y**2 * energy_eV**2 / covariance_matrix[2, 2]
    covariance_matrix[4, 4] = sigma_z**2

    return covariance_matrix


def load_augmentations_from_yaml(
    config_path: str | Path,
) -> list[BeamOutputAugmentation]:
    """Load augmentation objects from YAML configuration."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return load_augmentations_from_config(config)


def load_augmentations_from_config(
    config: Mapping[str, Any],
) -> list[BeamOutputAugmentation]:
    """Build augmentation objects from a parsed configuration mapping."""

    if "augmentations" not in config:
        raise ValueError("Augmentation config must define an 'augmentations' list")

    entries = config["augmentations"]
    if not isinstance(entries, list):
        raise ValueError("'augmentations' must be a list")

    augmentations: list[BeamOutputAugmentation] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Each augmentation entry must be a mapping")

        augmentation_type = entry.get("type")
        if augmentation_type == "bctrl_family":
            augmentations.append(
                BCTRLFamilyAugmentation(
                    source_suffix=entry.get("source_suffix", ":BCTRL"),
                    min_value=entry.get("min_value", -100.0),
                    max_value=entry.get("max_value", 100.0),
                    status_value=entry.get("status_value", "Ready"),
                )
            )
            continue

        if augmentation_type == "injector_covariance":
            energy = entry.get("energy_eV")
            field_mapping = entry.get("field_mapping", {})
            if not isinstance(field_mapping, Mapping):
                raise ValueError("injector_covariance.field_mapping must be a mapping")

            augmentations.append(
                InjectorCovarianceAugmentation(
                    energy_eV=energy,
                    field_mapping=dict(field_mapping),
                )
            )
            continue

        raise ValueError(f"Unknown augmentation type: {augmentation_type!r}")

    return augmentations
