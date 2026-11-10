from typing import Any, Mapping

from lume.model import LUMEModel
from lume.staged_model import InitialParticlesMixIn
from lume_torch.base import LUMETorchModel

from models.lcls_fel_genesis import load_model


_TWISS_INPUT_NAMES = frozenset(
    {"alphax", "alphay", "betax", "betay", "emitx", "emity", "beam_length"}
)


class GenesisFELSurrogate(LUMEModel, InitialParticlesMixIn):
    """
    LCLS FEL Genesis surrogate accepting a beam distribution.

    The underlying ``models.lcls_fel_genesis`` TorchModel expects Twiss parameters
    (``alphax``, ``alphay``, ``betax``, ``betay``, ``emitx``, ``emity``) and bunch
    length (``beam_length``) as scalar inputs. This wrapper extracts those from an
    openPMD ParticleGroup set via ``initial_particles``, leaving the physics
    inputs the beam cannot supply (``chirp``, ``taper``, ``ustart``) as writable
    variables.
    """

    def __init__(self) -> None:
        super().__init__()
        self.surrogate = LUMETorchModel(load_model())
        self._initial_particles = None
        self._cache: dict[str, Any] = {}
        self.set({})
        self.update_state()

    @property
    def initial_particles(self):
        return self._initial_particles

    @initial_particles.setter
    def initial_particles(self, particles) -> None:
        self._initial_particles = particles
        if particles is None:
            return

        twiss_x = particles.twiss("x")
        twiss_y = particles.twiss("y")
        self.surrogate.set(
            {
                "alphax": twiss_x["alpha_x"],
                "alphay": twiss_y["alpha_y"],
                "betax": twiss_x["beta_x"],
                "betay": twiss_y["beta_y"],
                "emitx": twiss_x["norm_emit_x"],
                "emity": twiss_y["norm_emit_y"],
                "beam_length": particles["sigma_z"],
            }
        )
        self.update_state()

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        for name in values:
            if name in _TWISS_INPUT_NAMES:
                raise ValueError(
                    f"{name!r} is derived from initial_particles and cannot be set "
                    "directly on GenesisFELSurrogate."
                )
        self.surrogate.set(dict(values))
        self.update_state()

    @property
    def supported_variables(self) -> dict[str, Any]:
        return {
            name: var
            for name, var in self.surrogate.supported_variables.items()
            if name not in _TWISS_INPUT_NAMES
        }

    def reset(self) -> None:
        self.surrogate.reset()
        self._initial_particles = None
        self._cache = {}

    def update_state(self) -> None:
        self._cache.update(
            self.surrogate.get(list(self.surrogate.supported_variables.keys()))
        )
