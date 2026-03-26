from lume_torch.models.torch_model import TorchModel
from lume_torch.base import LUMETorchModel
from lume.model import LUMEModel
from lume.variables import ParticleGroupVariable
from cheetah.particles import ParticleBeam
from scipy import constants
import os
import torch
from pathlib import Path

OTR2_BEAM_ENERGY = 135.0e6  # eV

def create_beam_distribution_from_state(state, n_particles) -> ParticleBeam:
    sigma_x = torch.tensor(state["OTRS:IN20:571:XRMS"]*1e-6)
    sigma_y = torch.tensor(state["OTRS:IN20:571:YRMS"]*1e-6)
    sigma_z = torch.tensor(state["sigma_z"]*1e-6)
    normalized_emittance_x = torch.tensor(state["norm_emit_x"]*1e-6)
    normalized_emittance_y = torch.tensor(state["norm_emit_y"]*1e-6)
    energy = OTR2_BEAM_ENERGY
    relativistic_gamma = energy / (constants.value("electron mass energy equivalent in MeV") * 1e6)
    beam = ParticleBeam.from_twiss(
        num_particles=n_particles,
        beta_x=sigma_x**2 / (normalized_emittance_x / relativistic_gamma),
        beta_y=sigma_y**2 / (normalized_emittance_y / relativistic_gamma),
        emittance_x = normalized_emittance_x / relativistic_gamma,
        emittance_y = normalized_emittance_y / relativistic_gamma,
        sigma_tau=sigma_z,
        energy = torch.tensor(energy),
    )
    beam.particles = beam.particles.squeeze()
    return beam

class InjectorSurrogate(LUMEModel):
    def __init__(self, n_particles=10000):
        super().__init__()
        config_path = os.path.join(
            Path(__file__).parent, 
            "../../.submodules/repo/model_config.yaml"
        )
        tm = TorchModel(config_path)
        self.model = LUMETorchModel(tm)
        self.n_particles = n_particles

        self._state = self.model._state
        self.update_state()

    def _get(self, names):
        return {name: self._state[name] for name in names}
    
    def _set(self, values):
        for name, value in values.items():
            self._state[name] = value

        self.model.set(values)

        self.update_state()

    @property
    def supported_variables(self):
        v = self.model.supported_variables
        v.update({"output_beam": ParticleGroupVariable(name="output_beam", read_only=True)})
        return v

    def reset(self):
        self.model.reset()

    def update_state(self):
        self._state.update(self.model.get(list(self.model.supported_variables.keys())))

        # replace torch tensors with floats
        for key, value in self._state.items():
            if isinstance(value, torch.Tensor):
                self._state[key] = value.item()

        # update a outgoing beam distribution
        beam = create_beam_distribution_from_state(self._state, self.n_particles)
        self._state["output_beam"] = beam.to_openpmd_particlegroup()

