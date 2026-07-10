from typing import Any, Mapping

import numpy as np
import torch
import yaml
from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn
from lume_torch.base import LUMETorchModel
from lume_torch.models.torch_model import TorchModel
from scipy import constants
import beamphysics
from distgen import Generator


class BeamOutputModel(LUMEModel, FinalParticlesMixIn):
    """
    LUME wrapper around a surrogate model that adds an openPMD beam
    output variable based on a model predicting the beam covariance matrix.

    The surrogate model is expected to support at least the following variables:
    covariance_matrix: TorchNDVariable
        6x6 covariance matrix in the surrogate convention/order [x, px, y, py, t, pz].
        Units are [m, eV/c, m, eV/c, s, eV/c]. This wrapper converts the t-axis to z (meters)
        using z = -c * t before generating the output ParticleGroup.
    """

    def __init__(
        self,
        surrogate: TorchModel,
        n_particles: int = 10000,
        p0c: float = 1e8,
        t0: float = 0.0,
        z0: float = 0.0,
        total_charge: float = 1e-9,
    ) -> None:
        """
         Initialize wrapper with surrogate model and internal cache copy.

         Parameters
         ----------
        surrogate: TorchModel
            The surrogate model to wrap, which must support the required input variables.
        n_particles: int, optional
            The number of particles to generate in the output beam distribution (default: 10000).
        p0c: float, optional
            The reference momentum in eV/c to use for generating the output beam distribution (default: 1e8).
        t0: float, optional
            The reference time in seconds to use for generating the output beam distribution (default: 0.0).
        z0: float, optional
            The reference position in meters to use for generating the output beam distribution (default: 0.0).
        total_charge: float, optional
            The total charge in Coulombs to use for generating the output beam distribution (default: 1e-9 C).

        """
        super().__init__()
        self.surrogate = LUMETorchModel(surrogate)
        self.n_particles = n_particles
        self.p0c = p0c
        self.t0 = t0
        self.z0 = z0
        self.total_charge = total_charge
        self._cache: dict[str, Any] = {"output_beam": None}
        self.set({})  # Initializing with defaults of NN model
        self.update_state()

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        """Update model state and regenerate exported output beam."""
        # handle updates to input variables
        for name, value in values.items():
            self._cache[name] = value

        # update surrogate model with new input variables
        self.surrogate.set(dict(values))

        self.update_state()

    @property
    def supported_variables(self) -> dict[str, Any]:
        """Return supported variables without mutating wrapped model metadata."""
        return self.surrogate.supported_variables

    def reset(self):
        self.surrogate.reset()
        self._cache = {"output_beam": None}

    def update_state(self):
        """Update internal cache from surrogate model and regenerate output beam."""
        outputs = self.surrogate.get(list(self.surrogate.supported_variables.keys()))

        # change outputs to numpy if they are torch tensors -- except for the covariance matrix
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if key != "covariance_matrix":
                    outputs[key] = value.detach().cpu().numpy()

        self._cache.update(outputs)
        self._generate_output_beam()

    def _generate_output_beam(self):
        """Generate the output beam ParticleGroup from cached surrogate outputs and class attributes."""

        # get the covariance matrix from the cache
        # units and variable order: [x, px, y, py, z, pz]
        # units: [m, eV/c, m, eV/c, s, eV/c]
        covariance_matrix = torch.as_tensor(self._cache["covariance_matrix"]).squeeze()
        if covariance_matrix.shape != (6, 6):
            raise ValueError(
                "Expected covariance_matrix with shape (6, 6) or singleton-batched equivalent; "
                f"got shape {tuple(covariance_matrix.shape)}"
            )

        # convert covariance matrix time axis to z using speed of light units for the
        # surrogate and openPMD ParticleBeam convention
        scaled_covariance_matrix = covariance_matrix.clone()
        scaled_covariance_matrix[4, :] *= -constants.speed_of_light
        scaled_covariance_matrix[:, 4] *= -constants.speed_of_light

        # reorder covariance matrix from surrogate variable order to openPMD ParticleBeam order
        # surrogate variable order: [x, px, y, py, z, pz]
        mean = np.array(
            [0.0, 0.0, 0.0, 0.0, self.z0, self.p0c], dtype=np.float32
        )  # units: [m, eV/c, m, eV/c, m, eV/c]
        inputs = {
            "n_particle": self.n_particles,
            "species": "electron",
            "nd_gaussian_dist": {
                "method": "cholesky",
                "centroid": {
                    "x": str(mean[0]) + " m",
                    "px": str(mean[1]) + " eV/c",
                    "y": str(mean[2]) + " m",
                    "py": str(mean[3]) + " eV/c",
                    "z": str(mean[4]) + " m",
                    "pz": str(mean[5]) + " eV/c",
                },
                "cov_matrix": scaled_covariance_matrix.tolist(),
            },
            "start": {"tstart": str(self.t0) + " s", "type": "time"},
            "total_charge": str(self.total_charge) + " C",
        }
        # convert to yaml
        inputs_yaml = yaml.safe_dump(inputs, sort_keys=False)

        generator = Generator(inputs_yaml)

        particle_group = generator.run()
        self._cache["output_beam"] = particle_group

    @property
    def final_particles(self) -> beamphysics.ParticleGroup:
        """Return the final particle distribution as an openPMD ParticleGroup."""
        return self._cache["output_beam"]
