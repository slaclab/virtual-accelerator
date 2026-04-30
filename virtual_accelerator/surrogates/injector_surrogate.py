import os
from typing import Any, Iterable, Mapping

import numpy as np
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
mlflow = import_optional(
    "mlflow",
    feature="injector surrogate",
    extra="surrogate",
)

_DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow.american-science-cloud.org/"
_DEFAULT_MLFLOW_CONFIG = {
    "mlflow": {
        "username": "globus:smiskov@slac.stanford.edu",
        "api_key_env": "api_key",
    }
}
_UNSET = object()


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
            "status": _tensor_to_numpy(
                torch.ones(self.n_particles, dtype=torch.int32)
            ),  # need int
            "species": "electron",
        }
        particle_group = beamphysics.ParticleGroup(data=data)
        self._cache["output_beam"] = particle_group


class InjectorSurrogate(LUMEModel):
    """LUME wrapper around the lcls injector torch surrogate with openPMD beam output."""

    tracking_uri = _DEFAULT_MLFLOW_TRACKING_URI
    registered_model_name = "lcls-cu-inj-model"
    model_version: str | None = None
    mlflow_config: Mapping[str, Any] | None = _DEFAULT_MLFLOW_CONFIG

    def __init__(
        self,
        n_particles: int = 10000,
        *,
        tracking_uri: str | None = None,
        registered_model_name: str | None = None,
        model_version: str | None | object = _UNSET,
        mlflow_config: Mapping[str, Any] | None | object = _UNSET,
    ) -> None:
        """Initialize surrogate model and internal cache copy."""
        super().__init__()
        self._tracking_uri = self._resolve_tracking_uri(tracking_uri)
        self._registered_model_name = (
            registered_model_name or self.registered_model_name
        )
        self._model_version = (
            self.model_version if model_version is _UNSET else model_version
        )
        self._mlflow_config = (
            self.mlflow_config if mlflow_config is _UNSET else mlflow_config
        )
        self.model = self._load_model()
        self.n_particles = n_particles
        self._cache: dict[str, Any] = {}
        self.set({})  # Initializing with defaults of NN model
        self.update_state()

    @classmethod
    def _resolve_tracking_uri(cls, tracking_uri: str | None) -> str:
        if tracking_uri:
            return tracking_uri
        return (
            os.getenv("VIRTUAL_ACCELERATOR_MLFLOW_TRACKING_URI")
            or os.getenv("MLFLOW_TRACKING_URI")
            or cls.tracking_uri
        )

    @staticmethod
    def enable_amsc_x_api_key(config_dict: Mapping[str, Any] | None) -> None:
        """Patch MLflow requests to include the AmSC API key header."""
        import mlflow.utils.rest_utils as rest_utils

        mlflow_cfg = config_dict.get("mlflow") if config_dict is not None else None
        if not isinstance(mlflow_cfg, Mapping):
            raise KeyError(
                "Missing 'mlflow' configuration section required for AmSC MLflow authentication."
            )

        api_key_env = mlflow_cfg.get("api_key_env")
        if not api_key_env:
            raise KeyError(
                "Missing 'api_key_env' in 'mlflow' configuration. "
                "Please specify the name of the environment variable containing the AmSC API key."
            )

        api_key = os.getenv(api_key_env)
        if api_key is None:
            raise KeyError(
                f"The environment variable '{api_key_env}' specified in 'mlflow.api_key_env' "
                "is not set. Please export it with the AmSC MLflow API key."
            )

        if getattr(rest_utils, "_virtual_accelerator_amsc_patch", None) == api_key_env:
            return

        original_http_request = rest_utils.http_request

        def patched(host_creds, endpoint, method, *args, **kwargs):
            if "headers" in kwargs and kwargs["headers"] is not None:
                headers = dict(kwargs["headers"])
                headers["X-Api-Key"] = api_key
                kwargs["headers"] = headers
            else:
                headers = dict(kwargs.get("extra_headers") or {})
                headers["X-Api-Key"] = api_key
                kwargs["extra_headers"] = headers
            return original_http_request(host_creds, endpoint, method, *args, **kwargs)

        rest_utils.http_request = patched
        rest_utils._virtual_accelerator_amsc_patch = api_key_env

    @classmethod
    def _configure_mlflow(
        cls, tracking_uri: str, config_dict: Mapping[str, Any] | None
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow_cfg = config_dict.get("mlflow") if config_dict is not None else None
        if isinstance(mlflow_cfg, Mapping):
            username = mlflow_cfg.get("username")
            if username:
                os.environ.setdefault("MLFLOW_TRACKING_USERNAME", str(username))
            cls.enable_amsc_x_api_key(config_dict)

    @classmethod
    def _coerce_model(cls, loaded_model: Any) -> LUMEModel:
        if isinstance(loaded_model, LUMEModel):
            return loaded_model
        if isinstance(loaded_model, TorchModel):
            return LUMETorchModel(loaded_model)

        unwrap = getattr(loaded_model, "unwrap_python_model", None)
        if callable(unwrap):
            return cls._coerce_model(unwrap())

        inner_model = getattr(loaded_model, "model", None)
        if inner_model is not None and inner_model is not loaded_model:
            return cls._coerce_model(inner_model)

        raise TypeError(
            "Loaded MLflow model is not a supported LUME model type. "
            f"Got {type(loaded_model)!r}."
        )

    def _model_uri(self) -> str:
        version = self._model_version if self._model_version is not None else "latest"
        return f"models:/{self._registered_model_name}/{version}"

    def _load_model(self) -> LUMEModel:
        self._configure_mlflow(self._tracking_uri, self._mlflow_config)
        loaded_model = mlflow.pyfunc.load_model(self._model_uri())
        return self._coerce_model(loaded_model)

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
