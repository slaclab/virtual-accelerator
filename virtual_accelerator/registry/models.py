"""Registry of available virtual-accelerator models for LCLS and FACET-II."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelEntry:
    """
    Metadata describing one model and how to configure it.

    Parameters
    ----------
    name : str
        Registry key, also used to qualify per-stage kwargs.
    description : str
        One-line summary shown by ``models_available``.
    facility : str
        "lcls" or "facet2". Models of different facilities cannot be staged.
    engine : str
        "bmad", "impact", "surrogate" or "cheetah".
    builder : str
        Builder function as a ``"module:function"`` string rather than a callable,
        so that importing this module does not import pytao / torch / impact.
        Discovery must work with no optional dependencies installed.
    extras : tuple[str, ...]
        Pip extras the builder needs, e.g. ``("bmad",)``.
    params : dict[str, Any]
        Configurable parameter name to default value. Also the allow-list against
        which kwargs are validated.
    handoff_points : tuple[str, ...]
        Suggested start, end and handoff elements, in lattice order. A discovery
        aid rather than a restriction: any element in the underlying lattice may
        be used. Screens are enumerated exhaustively, so a screen-shaped name
        absent from this tuple is treated as a typo and rejected; anything else
        passes through to the engine. Positions refer to the entrance face of the
        element, the only reference plane Bmad and IMPACT express identically.
    start_param : str | None, optional
        Builder kwarg controlling the start element. Default is None, meaning the
        start is fixed and the model cannot be a downstream stage.
    end_param : str | None, optional
        Builder kwarg controlling the end element. Default is None, meaning the
        end is not configurable.
    default_start : str | None, optional
        Standard start element. Default is None.
    default_end : str | None, optional
        Standard end element, used to infer the handoff when this model is
        upstream in a chain. Default is None.
    shared_params : frozenset[str], optional
        Params that must hold the same value in every stage of a chain. The beam
        flows through the stages, so a particle count differing between them is
        physically meaningless. These are broadcast to every stage that declares
        them, and the per-stage ``"<model>.<param>"`` form is rejected for them --
        letting the values diverge would break the invariant rather than
        configure anything. Default is empty.
    """

    name: str
    description: str
    facility: str
    engine: str
    builder: str
    extras: tuple[str, ...]
    params: dict[str, Any]
    handoff_points: tuple[str, ...]
    start_param: str | None = None
    end_param: str | None = None
    default_start: str | None = None
    default_end: str | None = None
    shared_params: frozenset[str] = frozenset()

    @property
    def configurable_extent(self) -> bool:
        """Whether either end of the model's tracking range can be set."""
        return self.start_param is not None or self.end_param is not None


_ALL_CU_HXR_SCREENS = (
    "YAG02",
    "YAG03",
    "OTRH1",
    "OTRH2",
    "OTR1",
    "OTR2",
    "OTR3",
    "OTR4",
    "OTR11",
    "OTR12",
    "OTR21",
    "OTRDMP",
)


MODELS: dict[str, ModelEntry] = {
    "impact_cu_inj": ModelEntry(
        name="impact_cu_inj",
        description="IMPACT-T LCLS injector, cathode -> YAG03",
        facility="lcls",
        engine="impact",
        builder="virtual_accelerator.models.cu_hxr:get_cu_inj_impact_model",
        extras=("impact",),
        params={"n_particles": 100, "end_element": "YAG03"},
        # YAG01 and OTR3 exist in the deck but their lines are commented out;
        # OTR4 is past stop_1 at z=16.5.
        handoff_points=("YAG02", "YAG03"),
        end_param="end_element",
        default_end="YAG03",
        shared_params=frozenset({"n_particles"}),
    ),
    "bmad_cu_hxr": ModelEntry(
        name="bmad_cu_hxr",
        description="Bmad CU-HXR linac, injector handoff -> END",
        facility="lcls",
        engine="bmad",
        builder="virtual_accelerator.models.cu_hxr:get_cu_hxr_bmad_model",
        extras=("bmad",),
        params={
            "start_element": "OTR2",
            "end_element": "END",
            "track_beam": False,
            "custom_beam_path": None,
        },
        # Starts wherever the upstream injector hands off: YAG03 from
        # impact_cu_inj, OTR2 from surrogate_cu_inj.
        handoff_points=("CATHODE", *_ALL_CU_HXR_SCREENS, "END"),
        start_param="start_element",
        end_param="end_element",
        default_start="OTR2",
        default_end="END",
    ),
    "surrogate_cu_inj": ModelEntry(
        name="surrogate_cu_inj",
        description="NN LCLS injector surrogate, cathode -> OTR2",
        facility="lcls",
        engine="surrogate",
        builder=(
            "virtual_accelerator.models.cu_hxr:get_cu_hxr_injector_surrogate_model"
        ),
        extras=("surrogate",),
        params={"n_particles": 1000},
        handoff_points=("OTR2",),
        default_end="OTR2",
        shared_params=frozenset({"n_particles"}),
    ),
    "cheetah_cu_hxr": ModelEntry(
        name="cheetah_cu_hxr",
        description="Cheetah nc_hxr, cathode -> END",
        facility="lcls",
        engine="cheetah",
        builder="virtual_accelerator.models.cu_hxr:get_cu_hxr_cheetah_model",
        extras=("cheetah",),
        params={"n_particles": 1000},
        handoff_points=(),
        shared_params=frozenset({"n_particles"}),
    ),
    "impact_f2e_inj": ModelEntry(
        name="impact_f2e_inj",
        description="IMPACT-T FACET-II injector, cathode -> PR10241",
        facility="facet2",
        engine="impact",
        builder="virtual_accelerator.models.facet2:get_facet_impact_model",
        extras=("impact",),
        params={"n_particles": 100, "end_element": "PR10241"},
        handoff_points=("PR10241",),
        end_param="end_element",
        default_end="PR10241",
        shared_params=frozenset({"n_particles"}),
    ),
    "surrogate_f2e_inj": ModelEntry(
        name="surrogate_f2e_inj",
        description="NN FACET-II injector surrogate, cathode -> PR10241",
        facility="facet2",
        engine="surrogate",
        builder="virtual_accelerator.models.facet2:get_facet_injector_surrogate_model",
        extras=("surrogate",),
        params={"n_particles": 10000, "surrogate_inputs": "machine"},
        handoff_points=("PR10241",),
        default_end="PR10241",
        shared_params=frozenset({"n_particles"}),
    ),
    "bmad_f2_elec": ModelEntry(
        name="bmad_f2_elec",
        description="Bmad FACET-II e- linac, injector handoff -> END",
        facility="facet2",
        engine="bmad",
        builder="virtual_accelerator.models.facet2:get_facet_bmad_model",
        extras=("bmad",),
        params={
            "start_element": "L0AFEND",
            "end_element": "END",
            "track_beam": False,
            "custom_beam_path": None,
        },
        # Both FACET injectors end at PR10241, so that is the only shared handoff.
        # The screens downstream of it are listed so they can be used as end_ele.
        # CATHODEF, not CATHODE -- FACET's cathode element carries the "F" suffix.
        handoff_points=(
            "CATHODEF",
            "PR10241",
            "L0AFEND",
            "PR10465",
            "PR10471",
            "PR10571",
            "PR10711",
            "END",
        ),
        start_param="start_element",
        end_param="end_element",
        default_start="L0AFEND",
        default_end="END",
    ),
}
