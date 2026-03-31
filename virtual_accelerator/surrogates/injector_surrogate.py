from typing import Any, Iterable, Mapping
import tempfile
import numpy as np
import yaml
from lume_torch.models.torch_model import TorchModel
from lume_torch.base import LUMETorchModel
from lume.model import LUMEModel
from lume.variables import ParticleGroupVariable
from cheetah.particles import ParticleBeam
from scipy import constants
import torch
from pathlib import Path

OTR2_BEAM_ENERGY = 135.0e6  # eV


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Return a NumPy view/copy from tensor-like input on CPU without gradients."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_python_scalar(value: Any, key: str) -> Any:
    """Convert one-element tensors to scalars while preserving non-tensors."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.numel() != 1:
        raise ValueError(
            f"Expected scalar tensor for cache key '{key}', got shape {tuple(value.shape)}"
        )
    return value.item()


def to_openpmd_particlegroup(beam) -> "openpmd.ParticleGroup":  # noqa: F821
    """
    Convert the `ParticleBeam` to an openPMD `ParticleGroup` object.

    NOTE: openPMD uses boolean particle status flags, i.e. alive or dead. Cheetah's
        survival probabilities are converted to status flags by thresholding at 0.5.

    NOTE: At the moment this method only supports non-vectorised particle
        distributions.

    :return: openPMD `ParticleGroup` object with the `ParticleBeam`'s particles.
    """
    try:
        import pmd_beamphysics as openpmd
    except ImportError:
        raise ImportError(
            """To use the openPMD beam export, openPMD-beamphysics must be
            installed."""
        )

    # For now only support non-vectorised particle distributions
    if len(beam.particles.shape) != 2:
        raise ValueError("Only non-vectorised particle distributions are supported.")

    px = beam.px * beam.p0c
    py = beam.py * beam.p0c
    p_total = (beam.energies.square() - beam.species.mass_eV.square()).sqrt()
    pz = (p_total.square() - px.square() - py.square()).sqrt()
    t = beam.tau / constants.speed_of_light
    # TODO: To be discussed
    status = beam.survival_probabilities > 0.5

    data = {
        "x": _tensor_to_numpy(beam.x),
        "y": _tensor_to_numpy(beam.y),
        "z": _tensor_to_numpy(beam.tau),
        "px": _tensor_to_numpy(px),
        "py": _tensor_to_numpy(py),
        "pz": _tensor_to_numpy(pz),
        "t": _tensor_to_numpy(t),
        "weight": _tensor_to_numpy(beam.particle_charges),  # need to make at least 1d
        "status": _tensor_to_numpy(status.int()),  # need int
        "species": beam.species.name,
    }
    particle_group = openpmd.ParticleGroup(data=data)

    return particle_group


def create_beam_distribution_from_state(state, n_particles) -> ParticleBeam:
    sigma_x = torch.tensor(state["OTRS:IN20:571:XRMS"] * 1e-6)
    sigma_y = torch.tensor(state["OTRS:IN20:571:YRMS"] * 1e-6)
    sigma_z = torch.tensor(state["sigma_z"] * 1e-6)
    normalized_emittance_x = torch.tensor(state["norm_emit_x"] * 1e-6)
    normalized_emittance_y = torch.tensor(state["norm_emit_y"] * 1e-6)
    energy = OTR2_BEAM_ENERGY
    relativistic_gamma = energy / (
        constants.value("electron mass energy equivalent in MeV") * 1e6
    )
    beam = ParticleBeam.from_twiss(
        num_particles=n_particles,
        beta_x=sigma_x**2 / (normalized_emittance_x / relativistic_gamma),
        beta_y=sigma_y**2 / (normalized_emittance_y / relativistic_gamma),
        emittance_x=normalized_emittance_x / relativistic_gamma,
        emittance_y=normalized_emittance_y / relativistic_gamma,
        sigma_tau=sigma_z,
        energy=torch.tensor(energy),
    )
    beam.particles = beam.particles.squeeze()
    return beam


class InjectorSurrogate(LUMEModel):
    """LUME wrapper around injector torch surrogate with openPMD beam output."""

    #: Relative path from the project root to the submodule config.
    _SUBMODULE_RELATIVE = Path(".submodules") / "repo" / "model_config.yaml"

    #: Config keys whose values are resource paths that need resolving.
    _RESOURCE_KEYS = ("model", "input_transformers", "output_transformers")

    @classmethod
    def _find_config(cls) -> Path:
        """Locate ``model_config.yaml`` regardless of install mode.

        Walks up the directory tree from this file looking for a
        ``.submodules`` directory so the config is found both when running
        from the source tree and when the package is installed (e.g. in CI).

        Raises
        ------
        FileNotFoundError
            If the config cannot be found in any ancestor directory.
        """
        search = Path(__file__).resolve().parent
        for directory in [search, *search.parents]:
            candidate = directory / cls._SUBMODULE_RELATIVE
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find {cls._SUBMODULE_RELATIVE} in any directory above "
            f"{Path(__file__).resolve().parent}. "
            "Ensure the submodule has been initialised with "
            "'git submodule update --init'."
        )

    def __init__(self, n_particles: int = 10000) -> None:
        """Initialize surrogate model and internal cache copy.

        Resource paths inside ``model_config.yaml`` are relative to the
        submodule directory.  A temporary config file with those paths
        rewritten to absolute paths is passed to ``TorchModel`` so that
        initialization succeeds regardless of the current working directory.
        """
        super().__init__()
        tm = self._load_torch_model()
        self.model = LUMETorchModel(tm)
        self.n_particles = n_particles
        self._cache: dict[str, Any] = {}
        self.reset()

    @classmethod
    def _resolve_resource_paths(cls, config: dict, base_dir: Path) -> dict:
        """Return a copy of *config* with resource paths made absolute."""
        resolved = dict(config)
        for key in cls._RESOURCE_KEYS:
            if key not in resolved:
                continue
            value = resolved[key]
            if isinstance(value, str):
                resolved[key] = str((base_dir / value).resolve())
            elif isinstance(value, list):
                resolved[key] = [str((base_dir / v).resolve()) for v in value]
        return resolved

    @classmethod
    def _load_torch_model(cls) -> TorchModel:
        """Load :class:`TorchModel` with all resource paths resolved.

        Writes a temporary config YAML whose resource paths are absolute so
        that ``TorchModel`` can locate them regardless of the working directory.
        The temporary file is removed after loading.
        """
        config_path = cls._find_config()
        base_dir = config_path.parent

        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        resolved_config = cls._resolve_resource_paths(config, base_dir)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(resolved_config, tmp)
            tmp_path = Path(tmp.name)

        try:
            return TorchModel(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        """Update model state and regenerate exported output beam."""
        self.model.set(dict(values))

        model_cache = getattr(self.model, "_cache", {})
        scalarized_cache = {k: _to_python_scalar(v, k) for k, v in model_cache.items()}

        beam = create_beam_distribution_from_state(scalarized_cache, self.n_particles)
        scalarized_cache["output_beam"] = to_openpmd_particlegroup(beam)
        self._cache = scalarized_cache

    @property
    def supported_variables(self) -> dict[str, Any]:
        """Return supported variables without mutating wrapped model metadata."""
        variables = dict(self.model.supported_variables)
        variables["output_beam"] = ParticleGroupVariable(
            name="output_beam", read_only=True
        )
        return variables

    def reset(self):
        self.model.reset()
        self._cache = {}
