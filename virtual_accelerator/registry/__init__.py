"""Single entry point for building virtual-accelerator models by name.

from virtual_accelerator.registry import get_model, list_models

print(list_models())
model = get_model("bmad_cu_hxr", end_ele="OTR4", track_beam=True)
model = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03")
model = get_model("high_fidelity_cu_hxr_s2e", n_particles=1000)
"""

import importlib
import logging
from typing import Any

from virtual_accelerator.registry.models import MODELS, ModelEntry

logger = logging.getLogger(__name__)

__all__ = [
    "get_model",
    "list_models",
    "list_handoff_points",
    "common_handoff_points",
]


# Display labels for the facility/simulator enums. Registry keys stay lower case,
# but the printed table uses the canonical spelling ("LCLS", "Facet2", "IMPACT").
_FACILITY_LABELS = {
    "lcls": "LCLS",
    "hxr": "HXR",
    "sxr": "SXR",
    "lcls2": "LCLS2",
    "facet2": "Facet2",
}
_SIMULATOR_LABELS = {
    "bmad": "Bmad",
    "impact": "IMPACT",
    "surrogate": "Surrogate",
    "cheetah": "Cheetah",
    "zfel": "ZFEL",
}


# Standard staged chains, each addressable by its alias in `get_model`. A row is
# (alias, upstream, downstream); the alias resolves to the stage pair so
# `get_model("high_fidelity_cu_hxr_s2e")` behaves like passing the pair as a list.
_STANDARD_CHAINS: tuple[tuple[str, str, str], ...] = (
    ("high_fidelity_cu_hxr_s2e", "impact_cu_inj", "bmad_cu_hxr"),
    ("fast_cu_hxr_s2e", "surrogate_cu_inj", "bmad_cu_hxr"),
    ("high_fidelity_facet2_s2e", "impact_f2e_inj", "bmad_f2_elec"),
    ("fast_facet2_s2e", "surrogate_f2e_inj", "bmad_f2_elec"),
)

_CHAIN_ALIASES: dict[str, tuple[str, str]] = {
    alias: (upstream, downstream) for alias, upstream, downstream in _STANDARD_CHAINS
}


class _ModelCatalog(dict):
    """Mapping of model name -> ModelEntry that prints as a bordered ASCII table."""

    _COLS = ("name", "facility", "simulator", "start", "end", "description")

    def __init__(self, entries, chains=()):
        super().__init__(entries)
        # Extra display-only rows for standard staged chains.
        self._chains = tuple(chains)

    def _row(self, entry: ModelEntry) -> tuple[str, ...]:
        return (
            entry.name,
            _FACILITY_LABELS.get(entry.facility, entry.facility),
            _SIMULATOR_LABELS.get(entry.simulator, entry.simulator),
            entry.default_start or "-",
            entry.default_end or "-",
            entry.description,
        )

    def _chain_row(
        self, alias: str, upstream: ModelEntry, downstream: ModelEntry
    ) -> tuple[str, ...]:
        simulator = "+".join(
            dict.fromkeys(
                _SIMULATOR_LABELS.get(e.simulator, e.simulator)
                for e in (upstream, downstream)
            )
        )
        # Handoff plane comes from the upstream stage's default end -- that is what
        # get_model will pick when handoff_loc is not passed, so it is the honest
        # default to advertise here.
        handoff = upstream.default_end or "?"
        return (
            alias,
            _FACILITY_LABELS.get(upstream.facility, upstream.facility),
            simulator,
            upstream.default_start or "-",
            downstream.default_end or "-",
            f"{upstream.name} -> {downstream.name} (handoff {handoff})",
        )

    def __repr__(self) -> str:
        if not self:
            return "(no models registered)"
        rows = [self._row(entry) for entry in self.values()]
        chain_rows = [
            self._chain_row(alias, self[up], self[down])
            for alias, up, down in self._chains
            if up in self and down in self
        ]
        all_rows = rows + chain_rows
        widths = [
            max(len(col), *(len(row[i]) for row in all_rows))
            for i, col in enumerate(self._COLS)
        ]
        sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        header = (
            "| " + " | ".join(f"{c:<{w}}" for c, w in zip(self._COLS, widths)) + " |"
        )
        body = [
            "| " + " | ".join(f"{v:<{w}}" for v, w in zip(row, widths)) + " |"
            for row in all_rows
        ]
        parts = [sep, header, sep, *body, sep]
        if chain_rows:
            # A second separator between single models and staged chains so the
            # table makes the split obvious rather than looking like more entries.
            split_at = 3 + len(rows)
            parts.insert(split_at, sep)
        return "\n".join(parts)


def list_models(
    facility: str | None = None, simulator: str | None = None
) -> "_ModelCatalog":
    """
    Get registered models, optionally filtered, as a printable table.

    Parameters
    ----------
    facility : str, optional
        Restrict to one facility, "lcls" or "facet2". Default is None, meaning all.
    simulator : str, optional
        Restrict to one simulator, e.g. "bmad". Default is None, meaning all.

    Returns
    -------
    _ModelCatalog
        Mapping of registry name to ``ModelEntry``, in registration order. Prints
        as an ASCII table with facility, simulator, start, end and description
        columns, plus a block of standard staged chains as discovery aids.
        Iterate for names.
    """
    kept = {
        name: entry
        for name, entry in MODELS.items()
        if (facility is None or entry.facility == facility)
        and (simulator is None or entry.simulator == simulator)
    }
    # Only include a chain when both stages survived the filter, so the block
    # tracks the visible rows rather than advertising unreachable staging.
    chains = tuple(
        (alias, up, down)
        for alias, up, down in _STANDARD_CHAINS
        if up in kept and down in kept
    )
    return _ModelCatalog(kept.items(), chains=chains)


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


# Facility-specific cathode element names. Cathodes mark the front of the
# machine, so nothing is upstream of them and they can never be handoff points,
# regardless of which facility's naming convention the model uses -- FACET's
# element carries an "F" suffix (CATHODEF) while LCLS uses the bare name.
CATHODES: frozenset[str] = frozenset({"CATHODE", "CATHODEF"})


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
    Cathode elements are always excluded (``CATHODE`` for LCLS, ``CATHODEF`` for
    FACET): they mark the front of the machine, so nothing can hand over to a
    stage beginning there.

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
    shared -= CATHODES

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


def _resolve_spec(spec: str | list[str]) -> tuple[list[str], str | None]:
    """Resolve ``spec`` into an ordered list of stage names.

    Parameters
    ----------
    spec : str or list[str]
        A registry name, a chain alias, or an ordered list of stage names.

    Returns
    -------
    names : list[str]
        Length 1 for a single model, >= 2 for a chain.
    alias : str or None
        The chain alias if ``spec`` was one, else None. Only used for error
        messages so the user sees the name they typed.

    Raises
    ------
    ValueError
        If a list spec is shorter than two entries, or has a duplicate name.
    """
    if isinstance(spec, str):
        if spec in _CHAIN_ALIASES:
            return list(_CHAIN_ALIASES[spec]), spec
        return [spec], None

    names = list(spec)
    if len(names) < 2:
        raise ValueError("Staging requires at least two models.")

    seen: set[str] = set()
    for name in names:
        # A duplicate name would make stage_kwargs keys ambiguous ("which of the
        # two bmad_cu_hxr stages is this dict for?"), and legitimate use cases
        # for repeating a stage are hard to construct -- the two instances would
        # be identically configured and produce identical outputs.
        if name in seen:
            raise ValueError(
                f"Duplicate stage {name!r}: each stage name may appear at most "
                "once in a chain so stage_kwargs keys stay unambiguous."
            )
        seen.add(name)

    return names, None


def _route_chain_kwargs(
    entries: list[ModelEntry],
    stage_kwargs: dict[str, dict[str, Any]] | None,
    broadcast_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Distribute chain kwargs across stages.

    Top-level (``broadcast_kwargs``) is for shared params only -- ``n_particles``
    is the canonical example. Anything else must go through ``stage_kwargs``,
    keyed by stage name, because the caller is the only one who can say which
    stage a stage-specific param applies to.

    Precedence (highest wins):
      1. ``stage_kwargs[stage][param]`` -- explicit per stage
      2. top-level ``param=`` -- broadcast to every stage declaring it
      3. builder default -- the registry does not send the key at all

    In practice (1) and (2) never touch the same param, because shared params
    are rejected inside ``stage_kwargs`` and stage-specific params are rejected
    at the top level. Every param has exactly one legal home; the precedence
    order is stated for the general case and to match user intuition when a
    future change makes the two homes overlap.
    """
    stage_kwargs = dict(stage_kwargs or {})
    by_name = {entry.name: i for i, entry in enumerate(entries)}

    # Superset of "shared" across every stage in the chain. A param is shared
    # iff at least one stage that declares it lists it as shared -- the flag
    # says "this cannot diverge between stages", so any stage marking it
    # constrains the chain as a whole.
    all_shared = {p for entry in entries for p in entry.shared_params}

    # (A) Top-level kwargs must be shared params.
    for key in broadcast_kwargs:
        if key in all_shared:
            continue
        accepting = [e.name for e in entries if key in e.params]
        if accepting:
            hint = (
                f' Try stage_kwargs={{"{accepting[0]}": {{"{key}": ...}}}}'
                f" to target one stage."
            )
        else:
            known = sorted({p for e in entries for p in e.params} | all_shared)
            hint = f" Accepted params: {', '.join(known)}."
        raise ValueError(
            f"{key!r} is stage-specific and cannot be passed at the top level "
            f"for a chain -- it does not say which stage it applies to.{hint}"
        )

    # (B) stage_kwargs shape and stage names.
    for stage_name, params in stage_kwargs.items():
        if stage_name not in by_name:
            raise ValueError(
                f"{stage_name!r} in stage_kwargs is not a stage of this model. "
                f"Stages: {', '.join(by_name)}."
            )
        if not isinstance(params, dict):
            raise TypeError(
                f"stage_kwargs[{stage_name!r}] must be a dict of param -> value, "
                f"got {type(params).__name__}."
            )

    # (C) Assemble per-stage kwargs.
    routed: list[dict[str, Any]] = []
    for entry in entries:
        stage: dict[str, Any] = {}

        # Broadcast layer: every shared param a stage declares receives the
        # top-level value. Non-declaring stages skip it silently -- e.g.
        # bmad_cu_hxr does not declare n_particles because Bmad tracks each
        # particle individually rather than sampling a count.
        for key, value in broadcast_kwargs.items():
            if key in entry.params:
                stage[key] = value

        # Per-stage layer: overrides win over broadcast for the same key.
        per_stage = stage_kwargs.get(entry.name, {})
        for key, value in per_stage.items():
            # Accept the get_model spelling per stage, e.g. "end_ele" for a bmad
            # stage's "end_element". The dict form makes the mapping natural --
            # the stage name is already explicit, only the param needs aliasing.
            builder_key = {
                "start_ele": entry.start_param,
                "end_ele": entry.end_param,
            }.get(key, key)

            if builder_key is None:
                raise ValueError(
                    f"{entry.name!r} has no configurable {key!r} -- its extent "
                    "is fixed."
                )
            if builder_key in all_shared:
                raise ValueError(
                    f"{builder_key!r} is a shared param and cannot be set per "
                    "stage (the beam flows through the stages, so divergent "
                    f"values are physically invalid). Pass {builder_key}=... "
                    "at the top level."
                )
            if builder_key not in entry.params:
                accepted = sorted(entry.params)
                raise ValueError(
                    f"{key!r} is not a parameter of {entry.name!r}. "
                    f"Accepted: {', '.join(accepted)}."
                )
            stage[builder_key] = value

        routed.append(stage)

    return routed


def _route_single_kwargs(entry: ModelEntry, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate kwargs for a single-model call.

    No routing to do -- there is one stage. This just rejects unknown params so
    a typo like ``n_particle`` surfaces here rather than as a TypeError from the
    builder several imports later.
    """
    for key in kwargs:
        if key not in entry.params:
            accepted = sorted(entry.params)
            raise ValueError(
                f"{key!r} is not a parameter of {entry.name!r}. "
                f"Accepted: {', '.join(accepted)}."
            )
    return dict(kwargs)


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
            if downstream.simulator == "impact"
            else f"{downstream.name!r} has a fixed start"
        )
        raise ValueError(f"{downstream.name!r} cannot be a downstream stage: {reason}.")

    if handoff in CATHODES:
        raise ValueError(
            f"{handoff!r} cannot be a handoff location: nothing is upstream of it."
        )

    shared = common_handoff_points(upstream.name, downstream.name)
    if handoff not in shared:
        raise ValueError(
            f"{handoff!r} is not a shared handoff point for {upstream.name!r} -> "
            f"{downstream.name!r}. Available: {', '.join(shared) or 'none'}"
        )


def _exclusive_end(entry: ModelEntry, handoff: str) -> str:
    """
    Get the end element for an upstream stage handing over at ``handoff``.

    Parameters
    ----------
    entry : ModelEntry
        The upstream stage.
    handoff : str
        Element where the beam is handed to the next stage.

    Returns
    -------
    str
        Element to pass as the upstream stage's end. For Bmad this is Tao's
        ``"<handoff>-1"`` offset form; other simulators end at ``handoff`` itself.

    Notes
    -----
    Tracking stops at the handoff plane without passing through the element, so
    the downstream stage owns it and the two stages meet rather than overlap.

    Bmad's ``-slice_lattice`` accepts ``"OTR4-1"`` to mean the element before
    OTR4, which also sidesteps naming the predecessor -- several of them are
    duplicated in the lattice (both OTR1 and OTR3 follow an element called DE05)
    and would otherwise need a ``##N`` index.

    For IMPACT the exclusion happens in ``set_stop_location``, which prunes
    elements at or beyond the stop plane, so the name needs no adjustment here.
    """
    return f"{handoff}-1" if entry.simulator == "bmad" else handoff


def _strip_overlapping_variables(upstream, downstream, upstream_name, downstream_name):
    """
    Remove variables the downstream stage shares with the upstream stage.

    Parameters
    ----------
    upstream : LUMEModel
        Stage that tracks the beam up to (but not through) the handoff plane.
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
    A safeguard, not the primary handoff mechanism. The upstream stage already
    ends before the handoff element (via Bmad's ``"<handoff>-1"`` slice or
    IMPACT's ``include_end_element=False``), so the downstream stage owns the
    handoff plane and there is usually no overlap. Anything that does turn out
    to be shared is unregistered from the downstream stage before
    ``StagedModel`` sees the pair, since it would otherwise reject the chain
    outright.

    A writable overlap means something different and worse: both stages would
    be driving the same magnet, so their extents overlap rather than meeting at
    a plane, and dropping it downstream would leave that stage tracking a stale
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
    stage_kwargs: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
):
    """
    Build a model, or a staged chain of models, by registry name.

    Parameters
    ----------
    spec : str or list[str]
        A registry name, a chain alias (e.g. ``"high_fidelity_cu_hxr_s2e"``), or
        an ordered list of registry names to stage together, upstream first.
        Duplicate names in a list are rejected -- each stage must appear at most
        once so ``stage_kwargs`` keys stay unambiguous.
    handoff_loc : str or list[str], optional
        Element where each consecutive pair hands the beam over. Must be a
        shared handoff point of both stages, see ``common_handoff_points``. A
        list is required for more than two stages. Default is None, meaning it
        is inferred from the upstream stage's standard end.
    start_ele : str, optional
        Element to start tracking from. For a staged model this applies to the
        first stage. Default is None, meaning the model's own default.
    end_ele : str, optional
        Element to stop tracking at. For a staged model this applies to the last
        stage; interior extents come from ``handoff_loc``. Default is None,
        meaning the model's own default.
    stage_kwargs : dict[str, dict], optional
        Per-stage parameter overrides for a chain, keyed by stage name. Values
        are dicts of ``{param: value}`` for that stage's builder. Not accepted
        for single-model calls -- pass params as keyword arguments there.
    **kwargs
        For a single model: any builder parameter. For a chain: shared
        parameters only (broadcast to every stage that declares them).
        Stage-specific parameters must go through ``stage_kwargs``.

    Returns
    -------
    LUMEModel
        A single model, or a ``StagedModel`` wrapping the chain.

    Raises
    ------
    KeyError
        If a name in ``spec`` is not registered.
    ValueError
        If the stages cannot be chained, the handoff is not shared by both, a
        stage_kwargs key does not name a stage, a top-level kwarg is
        stage-specific in a chain, or a stage_kwargs entry names a shared param.

    Notes
    -----
    Precedence for a chain: ``stage_kwargs[stage][param]`` beats a top-level
    broadcast, which beats the builder default. In practice the two homes never
    touch the same param, because shared params are rejected inside
    ``stage_kwargs`` and stage-specific params are rejected at the top level.

    For staged chains this handles two things that are easy to get wrong by
    hand. The upstream stage ends immediately before the handoff element, so
    the downstream stage owns its PVs; any unexpected duplicate read-only
    variables are removed before wrapping the stages. Beam tracking is forced
    on for every stage that supports it, since a non-final stage must produce
    ``final_particles`` and a non-first stage must accept
    ``initial_particles``.

    See ``docs/model_registry_usage.md`` for worked examples.
    """
    start_ele, end_ele = _normalize(start_ele), _normalize(end_ele)
    names, _alias = _resolve_spec(spec)

    if len(names) == 1:
        if stage_kwargs:
            raise ValueError(
                "stage_kwargs is only meaningful for a chain -- for a single "
                "model, pass parameters directly as keyword arguments."
            )
        entry = _entry(names[0])
        routed = _route_single_kwargs(entry, kwargs)
        return _build(entry, routed, start_ele, end_ele)

    entries = [_entry(name) for name in names]
    routed = _route_chain_kwargs(entries, stage_kwargs, kwargs)
    handoffs = [_normalize(h) for h in _resolve_handoffs(entries, handoff_loc)]

    for upstream, downstream, handoff in zip(entries, entries[1:], handoffs):
        _validate_pair(upstream, downstream, handoff)

    stages = []
    for i, entry in enumerate(entries):
        stage_start = start_ele if i == 0 else handoffs[i - 1]
        stage_end = (
            end_ele if i == len(entries) - 1 else _exclusive_end(entry, handoffs[i])
        )

        stage_kw = dict(routed[i])
        # Every stage needs tracking on: a non-final stage has to produce
        # final_particles, and a non-first stage has to accept initial_particles
        # (lume_bmad rejects those unless track_type is 'beam').
        if "track_beam" in entry.params:
            stage_kw["track_beam"] = True

        # An upstream stage stops at the handoff plane without keeping the
        # element, so the downstream stage owns it. Bmad does this via the "-1"
        # offset in _exclusive_end; IMPACT needs the flag because its prune is
        # inclusive.
        if i < len(entries) - 1 and "include_end_element" in entry.params:
            stage_kw["include_end_element"] = False

        stages.append(
            _build(
                entry,
                stage_kw,
                stage_start if entry.start_param else None,
                stage_end if entry.end_param else None,
            )
        )

    # Both stages include the handoff element and so publish its PVs. Resolve
    # the duplicates before StagedModel validation, which would otherwise reject
    # them.
    for i in range(1, len(stages)):
        _strip_overlapping_variables(
            stages[i - 1], stages[i], entries[i - 1].name, entries[i].name
        )

    from lume.staged_model import StagedModel

    return StagedModel(stages)
