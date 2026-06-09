from typing import Any, Iterable, Mapping

import numpy as np
from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn
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
load_model = import_optional_symbol(
    "lcls_cu_inj_model",
    "load_model",
    feature="injector surrogate model package",
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


class InjectorSurrogate(LUMEModel, FinalParticlesMixIn):
    """LUME wrapper around the lcls injector torch surrogate with openPMD beam output."""

    def __init__(self, n_particles: int = 10000) -> None:
        """Initialize surrogate model and internal cache copy."""
        super().__init__()
        tm = load_model()
        self.model = LUMETorchModel(tm)
        self.n_particles = n_particles
        self._cache: dict[str, Any] = {}
        self.set({})  # Initializing with defaults of NN model
        self.update_state()

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        """Update model state and regenerate exported output beam."""
        self.model.set(dict(values))
        self.update_state()

    @property
    def supported_variables(self) -> dict[str, Any]:
        """Return supported variables without mutating wrapped model metadata."""
        return dict(self.model.supported_variables)

    def reset(self):
        self.model.reset()
        self._cache = {"output_beam": None}

    def update_state(self):
        self._cache.update(self.model.get(list(self.model.supported_variables.keys())))

        self._cache = {k: _to_python_scalar(v, k) for k, v in self._cache.items()}
        beam = create_beam_distribution_from_state(self._cache, self.n_particles)
        self._cache["output_beam"] = to_openpmd_particlegroup(beam)

    @property
    def final_particles(self):
        return self._cache["output_beam"]
