from pathlib import Path
import os
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from lume.model import LUMEModel
from lume.variables import ParticleGroupVariable
from scipy import constants

from virtual_accelerator.utils.optional_dependencies import (
    import_optional,
    import_optional_symbol,
)

OTR2_BEAM_ENERGY = 135.0e6  # eV

torch = import_optional(
    "torch",
    feature="injector surrogate",
    extra="surrogate",
)
LUMETorchModel = import_optional_symbol(
    "lume_torch.base",
    "LUMETorchModel",
    feature="injector surrogate",
    extra="surrogate",
)
TorchModel = import_optional_symbol(
    "lume_torch.models.torch_model",
    "TorchModel",
    feature="injector surrogate",
    extra="surrogate",
)

ParticleBeam = import_optional_symbol(
    "cheetah.particles",
    "ParticleBeam",
    feature="injector surrogate",
    extra="surrogate",
)
beamphysics = import_optional(
    "beamphysics",
    feature="openPMD beam export for injector surrogate",
    extra="surrogate",
)


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


def to_openpmd_particlegroup(beam) -> Any:
    """
    Convert the `ParticleBeam` to an openPMD `ParticleGroup` object.

    NOTE: openPMD uses boolean particle status flags, i.e. alive or dead. Cheetah's
        survival probabilities are converted to status flags by thresholding at 0.5.

    NOTE: At the moment this method only supports non-vectorised particle
        distributions.

    :return: openPMD `ParticleGroup` object with the `ParticleBeam`'s particles.
    """
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
        "weight": _tensor_to_numpy(
            -beam.particle_charges
        ),  # need to make at least 1d and negate
        "status": _tensor_to_numpy(status.int()),  # need int
        "species": beam.species.name,
    }
    particle_group = beamphysics.ParticleGroup(data=data)

    return particle_group


def create_beam_distribution_from_state(state: Mapping[str, Any], n_particles: int):
    sigma_x = torch.tensor(state["OTRS:IN20:571:XRMS"] * 1e-6)
    sigma_y = torch.tensor(state["OTRS:IN20:571:YRMS"] * 1e-6)
    sigma_z = torch.tensor(state["sigma_z"] * 1e-6)
    normalized_emittance_x = torch.tensor(state["norm_emit_x"])
    normalized_emittance_y = torch.tensor(state["norm_emit_y"])
    energy = OTR2_BEAM_ENERGY
    relativistic_gamma = energy / (
        constants.value("electron mass energy equivalent in MeV") * 1e6
    )
    beam = ParticleBeam.from_twiss(
        num_particles=n_particles,
        beta_x=sigma_x**2 / (normalized_emittance_x / relativistic_gamma),
        beta_y=sigma_y**2 / (normalized_emittance_y / relativistic_gamma),
        alpha_x=torch.tensor(0.1333896),
        alpha_y=torch.tensor(0.1333896),
        emittance_x=normalized_emittance_x / relativistic_gamma,
        emittance_y=normalized_emittance_y / relativistic_gamma,
        sigma_tau=sigma_z,
        energy=torch.tensor(energy),
    )
    beam.particles = beam.particles.squeeze()
    return beam


class BeamOutputWrapper(LUMEModel):
    """
    LUME wrapper around a surrogate model that adds an openPMD beam
    output variable based on a model predicting the beam covariance matrix.

    The surrogate model is expected to support at least the following variables:
    covariance_matrix: TorchNDVariable
        6x6 covariance matrix of the beam distribution in openpmd ParticleBeam order / units.
        Note: The units for openPMD ParticleBeam are meters and eV/c.

    """

    def __init__(
        self, surrogate: LUMEModel, n_particles: int = 10000, p0c: float = 1e8
    ) -> None:
        """
         Initialize wrapper with surrogate model and internal cache copy.

         Parameters
         ----------
        surrogate: LUMEModel
            The surrogate model to wrap, which must support the required input variables.
        n_particles: int, optional
            The number of particles to generate in the output beam distribution (default: 10000).
        p0c: float, optional
            The reference momentum in eV/c to use for generating the output beam distribution (default: 1e8).

        """
        super().__init__()
        self.surrogate = surrogate
        self.n_particles = n_particles
        self.p0c = p0c
        self._cache: dict[str, Any] = {}
        self.set({})  # Initializing with defaults of NN model
        self.update_state()

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        """Update model state and regenerate exported output beam."""
        for name, value in values.items():
            self._cache[name] = value
        self.surrogate.set(dict(values))
        self.update_state()

    @property
    def supported_variables(self) -> dict[str, Any]:
        """Return supported variables without mutating wrapped model metadata."""
        variables = dict(self.surrogate.supported_variables)
        variables["output_beam"] = ParticleGroupVariable(
            name="output_beam", read_only=True
        )
        return variables

    def reset(self):
        self.surrogate.reset()
        self._cache = {}

    def update_state(self):
        """Update internal cache from surrogate model and regenerate output beam."""
        self._cache.update(
            self.surrogate.get(list(self.surrogate.supported_variables.keys()))
        )
        covariance_matrix = self._cache["covariance_matrix"]

        # sample beam distribution from covariance matrix and convert to openPMD ParticleGroup
        particles = torch.distributions.MultivariateNormal(
            loc=torch.zeros(6), covariance_matrix=covariance_matrix
        ).sample((self.n_particles,))

        data = {
            "x": _tensor_to_numpy(particles[:, 0]),
            "y": _tensor_to_numpy(particles[:, 2]),
            "z": _tensor_to_numpy(particles[:, 4]),
            "px": _tensor_to_numpy(particles[:, 1]),
            "py": _tensor_to_numpy(particles[:, 3]),
            "pz": _tensor_to_numpy(particles[:, 5]),
            "t": 0.0,
            "weight": _tensor_to_numpy(
                torch.ones(self.n_particles) 
            ),  # need to make at least 1d and negate
            "status": _tensor_to_numpy(torch.ones(self.n_particles, dtype=torch.int32)),  # need int
            "species": "electron",
        }
        particle_group = beamphysics.ParticleGroup(data=data)
        self._cache["output_beam"] = particle_group


class InjectorSurrogate(LUMEModel):
    """LUME wrapper around the lcls injector torch surrogate with openPMD beam output."""

    # Config path relative to the project root (used when running from source)
    _SOURCE_RELATIVE = (
        Path("subtrees") / "lcls_cu_injector_ml_model" / "model_config.yaml"
    )

    # Config keys whose values are resource paths that need resolving
    _RESOURCE_KEYS = ("model", "input_transformers", "output_transformers")

    @classmethod
    def _candidate_config_roots(cls) -> list[Path]:
        """Return candidate root directories used to locate model config."""
        roots: list[Path] = []

        # Module location (source checkout or installed package layout)
        roots.extend(Path(__file__).resolve().parents)

        # GitHub Actions checkout root
        workspace = os.environ.get("GITHUB_WORKSPACE")
        if workspace:
            roots.append(Path(workspace).resolve())

        # Current working directory and its ancestors
        cwd = Path.cwd().resolve()
        roots.append(cwd)
        roots.extend(cwd.parents)

        # Deduplicate while preserving order
        seen: set[Path] = set()
        unique_roots: list[Path] = []
        for root in roots:
            if root not in seen:
                seen.add(root)
                unique_roots.append(root)
        return unique_roots

    @classmethod
    def _find_config(cls) -> Path:
        """Locate ``model_config.yaml`` regardless of install mode."""
        for root in cls._candidate_config_roots():
            candidate = root / cls._SOURCE_RELATIVE
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            "Could not find model_config.yaml. Looked for "
            f"{cls._SOURCE_RELATIVE} from module/cwd/workspace roots. "
            "Ensure the subtree exists in the checkout, e.g. "
            "'git subtree add --prefix subtrees/lcls_cu_injector_ml_model <remote> <ref>'."
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
        self.set({})  # Initializing with defaults of NN model
        self.update_state()

    @classmethod
    def _resolve_resource_paths(cls, config: dict, base_dir: Path) -> dict:
        """Return a copy of config with resource paths made absolute."""
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
    def _load_torch_model(cls) -> Any:
        """Load :class:`TorchModel` with all resource paths resolved.

        Writes a temporary config YAML whose resource paths are absolute so
        that ``TorchModel`` can locate them regardless of the working directory.
        The temporary file is removed after loading.
        """
        config_path = cls._find_config()
        base_dir = config_path.parent

        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        resolved_config = cls._resolve_resource_paths(config, base_dir)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(resolved_config, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)

        try:
            return TorchModel(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        """Update model state and regenerate exported output beam."""
        for name, value in values.items():
            self._cache[name] = value
        self.model.set(dict(values))
        self.update_state()

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

    def update_state(self):
        self._cache.update(self.model.get(list(self.model.supported_variables.keys())))

        self._cache = {k: _to_python_scalar(v, k) for k, v in self._cache.items()}
        beam = create_beam_distribution_from_state(self._cache, self.n_particles)
        self._cache["output_beam"] = to_openpmd_particlegroup(beam)
