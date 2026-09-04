"""Single entry point for building virtual-accelerator models by name.

from virtual_accelerator.registry import get_model, models_available

print(models_available)
model = get_model("bmad_cu_hxr", end_ele="OTR4", track_beam=True)
model = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03")
"""

import importlib
import logging
from typing import Any

from virtual_accelerator.registry.models import MODELS, ModelEntry

logger = logging.getLogger(__name__)

__all__ = [
    "get_model",
    "models_available",
    "list_models",
    "list_handoff_points",
    "common_handoff_points",
]


class _ModelCatalog(dict):
    """Mapping of model name -> description that prints as an aligned table."""

    def __repr__(self) -> str:
        if not self:
            return "(no models registered)"
        width = max(len(name) for name in self)
        return "\n".join(f"{name:<{width}}  {desc}" for name, desc in self.items())


models_available = _ModelCatalog(
    (name, entry.description) for name, entry in MODELS.items()
)


def list_models(facility: str | None = None, engine: str | None = None) -> list[str]:
    """
    Get the names of registered models, optionally filtered.

    Parameters
    ----------
    facility : str, optional
        Restrict to one facility, "lcls" or "facet2". Default is None, meaning all.
    engine : str, optional
        Restrict to one engine, e.g. "bmad". Default is None, meaning all.

    Returns
    -------
    list[str]
        Registry names, in registration order.
    """
    return [
        name
        for name, entry in MODELS.items()
        if (facility is None or entry.facility == facility)
        and (engine is None or entry.engine == engine)
    ]


def list_handoff_points(model_name: str) -> tuple[str, ...]:
    """
    Get the suggested start, end and handoff elements for one model.

    Parameters
    ----------
    model_name : str
        Registry name.

    Returns
    -------
    tuple[str, ...]
        Element names in lattice order. A discovery aid rather than an exhaustive
        list -- any element in the underlying lattice may be used.

    Raises
    ------
    KeyError
        If ``model_name`` is not registered.
    """
    return _entry(model_name).handoff_points


CATHODE = "CATHODE"


def common_handoff_points(*model_names: str) -> tuple[str, ...]:
    """
    Get the elements every named model can hand off at, in lattice order.

    Parameters
    ----------
    *model_names : str
        Two or more registry names.

    Returns
    -------
    tuple[str, ...]
        Shared handoff elements, ordered by the first model's lattice order.
        Empty if the models share none.

    Raises
    ------
    ValueError
        If fewer than two model names are given.
    KeyError
        If any name is not registered.

    Notes
    -----
    ``CATHODE`` is always excluded: it marks the front of the machine, so nothing
    can hand over to a stage beginning there.

    This is the set intersection, not the union. A union would admit planes only
    one stage can reach -- ``impact_cu_inj`` stops by z=16.5 m and so cannot reach
    ``OTR4`` at 17.80 m, but ``bmad_cu_hxr`` lists it, so a union would wrongly
    accept ``handoff_loc="OTR4"`` for that pair.
    """
    if len(model_names) < 2:
        raise ValueError("Need at least two models to find common handoff points.")

    entries = [_entry(name) for name in model_names]
    shared = set(entries[0].handoff_points)
    for entry in entries[1:]:
        shared &= set(entry.handoff_points)
    shared.discard(CATHODE)

    return tuple(name for name in entries[0].handoff_points if name in shared)


def _entry(name: str) -> ModelEntry:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(sorted(MODELS))}"
        ) from None


def _load_builder(entry: ModelEntry):
    module_path, _, func_name = entry.builder.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        extras = ", ".join(entry.extras) or "none"
        raise ImportError(
            f"Cannot import builder for {entry.name!r} ({entry.builder}). "
            f"Required extras: {extras}. Install with "
            f'`pip install "virtual-accelerator[{",".join(entry.extras)}]"`.'
        ) from exc
    return getattr(module, func_name)


def _normalize(name: str | None) -> str | None:
    """Canonicalise a user-supplied element name to upper case.

    Lattice element names are upper case everywhere. Tao is case-insensitive so a
    lower-case name would appear to work, but IMPACT's ``impact.ele[...]`` is a
    plain dict lookup, and the registry's own ``handoff_points`` lookups would
    silently miss.
    """
    return name if name is None else name.upper()


def _check_element(entry: ModelEntry, name: str, role: str) -> None:
    """Validate a start/end/handoff element.

    Any element in the underlying lattice is allowed, so this cannot be an
    exhaustive check -- quads, markers and drifts are all legitimate and there are
    thousands of them. Screens *are* enumerated exhaustively though, so a
    screen-shaped name missing from ``handoff_points`` is a typo worth catching
    early rather than letting it fail deep inside Tao.
    """
    if name in entry.handoff_points:
        return
    if name.startswith(("OTR", "YAG", "PR")):
        raise ValueError(
            f"{name!r} is not an available {role} screen for {entry.name!r}. "
            f"Suggested points: {', '.join(entry.handoff_points)}"
        )


def _route_kwargs(
    entries: list[ModelEntry], kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Distribute flat kwargs across stages using the registry's declared params.

    Routing is a table lookup, not signature introspection, so the error
    messages can name the candidate stages.
    """
    routed: list[dict[str, Any]] = [{} for _ in entries]
    by_name = {entry.name: i for i, entry in enumerate(entries)}

    # Builder spellings of the extent params. Flat use is rejected: which stage a
    # bare "end_element" means is ambiguous, and start_ele/end_ele already say it.
    extent_params = {
        param
        for entry in entries
        for param in (entry.start_param, entry.end_param)
        if param is not None
    }

    for key, value in kwargs.items():
        stage_name, sep, param = key.partition(".")
        if sep:
            if stage_name not in by_name:
                raise ValueError(
                    f"{key!r} targets stage {stage_name!r}, which is not in this model. "
                    f"Stages: {', '.join(by_name)}"
                )
            index = by_name[stage_name]
            # Accept the get_model spelling per stage, e.g. "bmad_cu_hxr.end_ele".
            param = {
                "start_ele": entries[index].start_param,
                "end_ele": entries[index].end_param,
            }.get(param, param)
            if param in entries[index].shared_params:
                raise ValueError(
                    f"{param!r} must be the same in every stage, so it cannot be set "
                    f"per stage. Pass {param}=... instead of {key!r}."
                )
            if param not in entries[index].params:
                raise ValueError(
                    f"{param!r} is not a parameter of {stage_name!r}. "
                    f"Accepted: {', '.join(sorted(entries[index].params))}"
                )
            routed[index][param] = value
            continue

        if key in extent_params:
            role = (
                "start_ele" if any(e.start_param == key for e in entries) else "end_ele"
            )
            raise ValueError(
                f"Do not pass {key!r} directly -- it is the builder's own name and does "
                f"not say which stage it applies to. Use {role}=... for the overall "
                f'extent, or "<model_name>.{key}=..." to target one stage.'
            )

        accepting = [i for i, entry in enumerate(entries) if key in entry.params]
        if not accepting:
            known = sorted({p for entry in entries for p in entry.params})
            raise ValueError(
                f"{key!r} is not a parameter of any stage. Accepted: {', '.join(known)}"
            )

        shared = any(key in entries[i].shared_params for i in accepting)
        if len(accepting) > 1 and not shared:
            names = ", ".join(entries[i].name for i in accepting)
            raise ValueError(
                f"{key!r} is ambiguous across stages ({names}). "
                f'Qualify it, e.g. "{entries[accepting[0]].name}.{key}=...".'
            )
        for i in accepting:
            routed[i][key] = value

    return routed


def _build(
    entry: ModelEntry,
    call_kwargs: dict[str, Any],
    start_ele: str | None,
    end_ele: str | None,
) -> Any:
    kwargs = dict(call_kwargs)

    if start_ele is not None:
        if entry.start_param is None:
            raise ValueError(
                f"{entry.name!r} has a fixed start and does not accept start_ele."
            )
        _check_element(entry, start_ele, "start")
        kwargs[entry.start_param] = start_ele

    if end_ele is not None:
        if entry.end_param is None:
            raise ValueError(
                f"{entry.name!r} has a fixed end and does not accept end_ele."
            )
        _check_element(entry, end_ele, "end")
        kwargs[entry.end_param] = end_ele

    return _load_builder(entry)(**kwargs)


def _resolve_handoffs(
    entries: list[ModelEntry], handoff_loc: str | list[str] | None
) -> list[str]:
    """Determine the handoff element between each consecutive pair of stages."""
    n_handoffs = len(entries) - 1

    if handoff_loc is None:
        handoffs = []
        for upstream in entries[:-1]:
            if upstream.default_end is None:
                raise ValueError(
                    f"handoff_loc is required: {upstream.name!r} has no default end "
                    "to infer it from."
                )
            handoffs.append(upstream.default_end)
        return handoffs

    handoffs = [handoff_loc] if isinstance(handoff_loc, str) else list(handoff_loc)
    if len(handoffs) != n_handoffs:
        raise ValueError(
            f"{len(entries)} stages need {n_handoffs} handoff location(s), "
            f"got {len(handoffs)}."
        )
    return handoffs


def _validate_pair(upstream: ModelEntry, downstream: ModelEntry, handoff: str) -> None:
    if upstream.facility != downstream.facility:
        raise ValueError(
            f"Cannot stage {upstream.name!r} ({upstream.facility}) onto "
            f"{downstream.name!r} ({downstream.facility}): different facilities."
        )

    if downstream.start_param is None:
        reason = (
            "IMPACT models can only start at the cathode"
            if downstream.engine == "impact"
            else f"{downstream.name!r} has a fixed start"
        )
        raise ValueError(f"{downstream.name!r} cannot be a downstream stage: {reason}.")

    if handoff == CATHODE:
        raise ValueError(
            f"{CATHODE!r} cannot be a handoff location: nothing is upstream of it."
        )

    shared = common_handoff_points(upstream.name, downstream.name)
    if handoff not in shared:
        raise ValueError(
            f"{handoff!r} is not a shared handoff point for {upstream.name!r} -> "
            f"{downstream.name!r}. Available: {', '.join(shared) or 'none'}"
        )


def _strip_overlapping_variables(upstream, downstream, upstream_name, downstream_name):
    """
    Remove variables the downstream stage shares with the upstream stage.

    Parameters
    ----------
    upstream : LUMEModel
        Stage that tracks the beam to the handoff plane and so owns its PVs.
    downstream : LUMEModel
        Stage the duplicates are removed from. Must support
        ``unregister_action_variable``.
    upstream_name : str
        Registry name of ``upstream``, used in error messages.
    downstream_name : str
        Registry name of ``downstream``, used in error messages.

    Returns
    -------
    list[str]
        Variable names removed from ``downstream``, empty if there was no overlap.

    Raises
    ------
    ValueError
        If any shared variable is writable.
    TypeError
        If ``downstream`` cannot unregister variables.

    Notes
    -----
    Both stages include the handoff element, so both publish its PVs and
    ``StagedModel`` would reject the pair as duplicates.

    A writable overlap means something different and worse: both stages would be
    driving the same magnet, so their extents overlap rather than meeting at a
    plane, and dropping it downstream would leave that stage tracking a stale
    value.
    """
    from lume.actions import WritableActionMixin

    downstream_vars = downstream.supported_variables
    overlap = sorted(set(upstream.supported_variables) & set(downstream_vars))
    if not overlap:
        return []

    writable = [
        name
        for name in overlap
        if isinstance(downstream_vars[name], WritableActionMixin)
    ]
    if writable:
        raise ValueError(
            f"{upstream_name!r} and {downstream_name!r} both control "
            f"{len(writable)} writable variable(s), so their extents overlap rather "
            f"than meeting at a plane: {', '.join(writable[:5])}"
            f"{' ...' if len(writable) > 5 else ''}. Check the handoff element."
        )

    if not hasattr(downstream, "unregister_action_variable"):
        raise TypeError(
            f"{downstream_name!r} shares {len(overlap)} variable(s) with "
            f"{upstream_name!r} but does not support unregister_action_variable, so "
            "the duplicates cannot be resolved."
        )

    for name in overlap:
        downstream.unregister_action_variable(name)
    logger.debug(
        "Removed %d variable(s) from %s already provided by %s",
        len(overlap),
        downstream_name,
        upstream_name,
    )
    return overlap


def get_model(
    spec: str | list[str],
    *,
    handoff_loc: str | list[str] | None = None,
    start_ele: str | None = None,
    end_ele: str | None = None,
    **kwargs: Any,
):
    """
    Build a model, or a staged chain of models, by registry name.

    Parameters
    ----------
    spec : str or list[str]
        A registry name, or an ordered list of names to stage together, upstream
        first.
    handoff_loc : str or list[str], optional
        Element where each consecutive pair hands the beam over. Must be a shared
        handoff point of both stages, see ``common_handoff_points``. A list is
        required for more than two stages. Default is None, meaning it is inferred
        from the upstream stage's standard end.
    start_ele : str, optional
        Element to start tracking from. For a staged model this applies to the
        first stage. Default is None, meaning the model's own default.
    end_ele : str, optional
        Element to stop tracking at. For a staged model this applies to the last
        stage; interior extents come from ``handoff_loc``. Default is None,
        meaning the model's own default.
    **kwargs
        Builder parameters. Params listed in a model's ``shared_params`` are sent
        to every stage declaring them and cannot be set per stage. Others are
        routed to the single stage declaring them, or qualified as
        ``"<model_name>.<param>"`` when more than one does.

    Returns
    -------
    LUMEModel
        A single model, or a ``StagedModel`` wrapping the chain.

    Raises
    ------
    KeyError
        If a name in ``spec`` is not registered.
    ValueError
        If the stages cannot be chained, the handoff is not shared by both, or a
        kwarg cannot be routed unambiguously.

    Notes
    -----
    For staged chains this handles two things that are easy to get wrong by hand.

    Duplicate variables at the handoff are removed automatically. Both stages
    include the handoff element, so both publish its PVs -- an IMPACT model stopped
    at YAG03 keeps the screen, since it prunes to ``s <= stop``, and so does a Bmad
    model sliced from YAG03. ``StagedModel`` would reject the pair as duplicates.
    The upstream stage owns them, because it is the stage that tracks the beam to
    that plane, so they are unregistered from the downstream stage before the chain
    is assembled.

    Beam tracking is forced on for every stage that supports it, since a non-final
    stage must produce ``final_particles`` and a non-first stage must accept
    ``initial_particles``.

    See ``docs/model_registry_usage.md`` for worked examples.
    """
    start_ele, end_ele = _normalize(start_ele), _normalize(end_ele)

    if isinstance(spec, str):
        entry = _entry(spec)
        (routed,) = _route_kwargs([entry], kwargs)
        return _build(entry, routed, start_ele, end_ele)

    names = list(spec)
    if len(names) < 2:
        raise ValueError("Staging requires at least two models.")

    entries = [_entry(name) for name in names]
    handoffs = [_normalize(h) for h in _resolve_handoffs(entries, handoff_loc)]
    routed = _route_kwargs(entries, kwargs)

    for upstream, downstream, handoff in zip(entries, entries[1:], handoffs):
        _validate_pair(upstream, downstream, handoff)

    stages = []
    for i, entry in enumerate(entries):
        stage_start = start_ele if i == 0 else handoffs[i - 1]
        stage_end = end_ele if i == len(entries) - 1 else handoffs[i]

        stage_kwargs = dict(routed[i])
        # Every stage needs tracking on: a non-final stage has to produce
        # final_particles, and a non-first stage has to accept initial_particles
        # (lume_bmad rejects those unless track_type is 'beam').
        if "track_beam" in entry.params:
            stage_kwargs["track_beam"] = True

        stages.append(
            _build(
                entry,
                stage_kwargs,
                stage_start if entry.start_param else None,
                stage_end if entry.end_param else None,
            )
        )

    # Both stages include the handoff element and so publish its PVs. Resolve the
    # duplicates before StagedModel validation, which would otherwise reject them.
    for i in range(1, len(stages)):
        _strip_overlapping_variables(
            stages[i - 1], stages[i], entries[i - 1].name, entries[i].name
        )

    from lume.staged_model import StagedModel

    return StagedModel(stages)
