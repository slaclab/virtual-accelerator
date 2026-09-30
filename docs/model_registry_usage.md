# Virtual Accelerator (VA) Registry — README

## Overview

The `virtual_accelerator.registry` module provides a unified interface for loading,
configuring, and chaining accelerator simulation models for the LCLS copper linac (CU)
and FACET-II. Models can be used standalone or chained together to simulate the full
beamline from cathode to end.

Building a model needs the matching lattice checkout on the environment:
`$LCLS_LATTICE` for the `*_cu_*` models, `$FACET2_LATTICE` for the `*_f2*` ones.

The `>>>` blocks in this document are executable doctests — running `pytest` runs
them, so the printed outputs stay in sync with the code. Heavy examples that would
launch a real simulator are marked `# doctest: +SKIP`.

### Installation

```bash
pip install git+https://github.com/slaclab/virtual-accelerator.git
```

### Quick Start

```python
>>> from virtual_accelerator.registry import (
...     get_model,
...     list_models,
...     list_handoff_points,
...     common_handoff_points,
... )
>>> from virtual_accelerator.registry.models import MODELS

```

### Available Models

Print all registered models as a table:

```python
>>> print(list_models())
+--------------------------+----------+----------------+----------+---------+-----------------------------------------------------+
| name                     | facility | simulator      | start    | end     | description                                         |
+--------------------------+----------+----------------+----------+---------+-----------------------------------------------------+
| impact_cu_inj            | LCLS     | IMPACT         | CATHODE  | YAG03   | LCLS CU injector                                    |
| bmad_cu_hxr              | LCLS     | Bmad           | OTR2     | END     | LCLS CU-HXR linac                                   |
| surrogate_cu_inj         | LCLS     | Surrogate      | CATHODE  | OTR2    | LCLS CU injector (NN surrogate)                     |
| cheetah_cu_hxr           | LCLS     | Cheetah        | CATHODE  | END     | LCLS CU-HXR full beamline                           |
| zfel_cu_hxr              | LCLS     | ZFEL           | -        | -       | ZFEL model for LCLS CU-HXR                          |
| impact_f2e_inj           | Facet2   | IMPACT         | CATHODEF | PR10241 | FACET-II injector                                   |
| surrogate_f2e_inj        | Facet2   | Surrogate      | CATHODEF | PR10241 | FACET-II injector (NN surrogate)                    |
| bmad_f2_elec             | Facet2   | Bmad           | CATHODEF | END     | FACET-II e- linac                                   |
+--------------------------+----------+----------------+----------+---------+-----------------------------------------------------+
| high_fidelity_cu_hxr_s2e | LCLS     | IMPACT+Bmad    | CATHODE  | END     | impact_cu_inj -> bmad_cu_hxr (handoff YAG03)        |
| fast_cu_hxr_s2e          | LCLS     | Surrogate+Bmad | CATHODE  | END     | surrogate_cu_inj -> bmad_cu_hxr (handoff OTR2)      |
| high_fidelity_facet2_s2e | Facet2   | IMPACT+Bmad    | CATHODEF | END     | impact_f2e_inj -> bmad_f2_elec (handoff PR10241)    |
| fast_facet2_s2e          | Facet2   | Surrogate+Bmad | CATHODEF | END     | surrogate_f2e_inj -> bmad_f2_elec (handoff PR10241) |
+--------------------------+----------+----------------+----------+---------+-----------------------------------------------------+

```

The chain rows below the separator are **aliases**: `get_model("high_fidelity_cu_hxr_s2e")`
builds the pair `impact_cu_inj -> bmad_cu_hxr` for you, using the handoff shown in
the description. Pass the pair as a list if you want to spell it out.

Filter by facility or simulator:

```python
>>> list(list_models(facility="facet2"))
['impact_f2e_inj', 'surrogate_f2e_inj', 'bmad_f2_elec']

>>> list(list_models(simulator="bmad"))
['bmad_cu_hxr', 'bmad_f2_elec']

```

### Handoff Points

Each model exposes a set of suggested handoff points — named locations where two
chained models exchange beam state. Cathodes and `END` are omitted below: they are
endpoints, not intermediate handoffs.

```python
>>> for m in ["impact_cu_inj", "bmad_cu_hxr", "surrogate_cu_inj", "cheetah_cu_hxr"]:
...     pts = tuple(p for p in list_handoff_points(m) if p not in {"CATHODE", "CATHODEF", "END"})
...     print(f"{m:<18} {pts}")
impact_cu_inj      ('YAG02', 'YAG03')
bmad_cu_hxr        ('YAG02', 'YAG03', 'OTRH1', 'OTRH2', 'OTR1', 'OTR2', 'OTR3', 'OTR4', 'OTR11', 'OTR12', 'OTR21', 'OTRDMP')
surrogate_cu_inj   ('OTR2',)
cheetah_cu_hxr     ('YAG02', 'YAG03', 'OTRH1', 'OTRH2', 'OTR1', 'OTR2', 'OTR3', 'OTR4', 'OTR11', 'OTR12', 'OTR21', 'OTRDMP')

```

Cathodes and `END` are still valid values for `start_ele` / `end_ele` on the Bmad
models, whose extents are configurable. The injector models always begin at the
cathode and cannot be told otherwise. FACET's injectors expose only `PR10241`, which
restricts every FACET chain to that one handoff plane.

### Shared Handoff Points

`common_handoff_points()` returns the locations two models can actually hand over at
— the intersection of their handoff points, with `CATHODE` excluded since nothing is
upstream of it.

```python
>>> common_handoff_points("impact_cu_inj", "bmad_cu_hxr")
('YAG02', 'YAG03')

>>> common_handoff_points("surrogate_cu_inj", "bmad_cu_hxr")
('OTR2',)

>>> common_handoff_points("impact_f2e_inj", "bmad_f2_elec")
('PR10241',)

```

### Loading a Single Model

Use `get_model()` with a model ID and an optional `end_ele` to stop tracking at a
specific screen.

```python
>>> get_model("bmad_cu_hxr", end_ele="TD11")            # doctest: +SKIP
<lume_bmad.model.LUMEBmadModel object at 0x150a9d370>

>>> get_model("impact_cu_inj", end_ele="YAG03")         # doctest: +SKIP
<impact.model.distgen.distgen_impact_model.LUMEDistgenImpactModel object at 0x1666b6450>

```

## Staged Models

Pass a list of two model IDs to `get_model()` to chain an injector model into a linac
model. The upstream model hands off beam particles to the downstream model at a shared
handoff point.

`surrogate_cu_inj` → `bmad_cu_hxr` — no `handoff_loc` needed, it is inferred from the
surrogate's fixed end (OTR2):

```python
>>> m = get_model(["surrogate_cu_inj", "bmad_cu_hxr"], end_ele="OTR4", n_particles=500)   # doctest: +SKIP
>>> m.set({"QUAD:IN20:525:BCTRL": -10.0})                                                 # doctest: +SKIP
>>> print(m.get("OTR4_beam")["norm_emit_y"])                                              # doctest: +SKIP
5.850087235892218e-07

```

`impact_cu_inj` → `bmad_cu_hxr` — hand off at YAG03:

```python
>>> model = get_model(                                                    # doctest: +SKIP
...     ["impact_cu_inj", "bmad_cu_hxr"],
...     handoff_loc="YAG03",
...     end_ele="TD11",
...     n_particles=1000,
... )
>>> model.set({"QUAD:IN20:525:BCTRL": -7.5})                              # doctest: +SKIP
>>> print(model.get("OTR4_beam")["norm_emit_y"])                          # doctest: +SKIP
2.3638227838794528e-07

```

### FACET-II

Both FACET chains hand off at PR10241, so `handoff_loc` can be left out:

```python
>>> m = get_model(["surrogate_f2e_inj", "bmad_f2_elec"], end_ele="PR10711", n_particles=2000)   # doctest: +SKIP
>>> m = get_model(["impact_f2e_inj", "bmad_f2_elec"], end_ele="PR10711", n_particles=200)       # doctest: +SKIP

```

### Handoff Validation

`handoff_loc` must be a shared point of both stages (`common_handoff_points`); the
cathode is never valid, and cross-facility staging is refused. Everything is checked
before any model is built, so a bad handoff never costs an IMPACT run.

Standard chains:

| Alias | Upstream | Downstream | Handoff |
|---|---|---|---|
| `high_fidelity_cu_hxr_s2e` | `impact_cu_inj` | `bmad_cu_hxr` | YAG03 |
| `fast_cu_hxr_s2e` | `surrogate_cu_inj` | `bmad_cu_hxr` | OTR2 (inferred) |
| `high_fidelity_facet2_s2e` | `impact_f2e_inj` | `bmad_f2_elec` | PR10241 (inferred) |
| `fast_facet2_s2e` | `surrogate_f2e_inj` | `bmad_f2_elec` | PR10241 (inferred) |

LCLS needs two handoff planes because its injector models end at different places
and neither can move: the NN surrogate predicts `OTRS:IN20:571` (OTR2) at 135 MeV
and cannot produce a beam at YAG03, which sits before L0B at 64 MeV.

### The Handoff Element Belongs To The Downstream Stage

Tracking stops *at* the handoff plane without carrying on through the element, so the
upstream stage ends just before it and the downstream stage owns it. `get_model()`
arranges this; nothing is required of the caller.

The two simulators express it differently:

| stage | how the exclusion is done |
|---|---|
| Bmad upstream | sliced to Tao's `"<handoff>-1"`, the element before the handoff |
| IMPACT upstream | `include_end_element=False`, so the element on the stop plane is pruned |

Neither changes where the beam stops. Every handoff point is zero-length, and IMPACT's
stop plane is already the element's entrance, so ending "before" the element and
ending "at" it are the same z. Only ownership of its PVs changes.

The result is that the handoff element appears in exactly one stage:

```python
>>> m = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", end_ele="OTR4")   # doctest: +SKIP
>>> imp, bmad = m.lume_model_instances                                                     # doctest: +SKIP
>>> "YAG03" in imp.impact_model.simulator.ele                                              # doctest: +SKIP
False
>>> set(imp.supported_variables) & set(bmad.supported_variables)                           # doctest: +SKIP
set()

```

Note this applies only to the handoff. A `start_ele` or `end_ele` you ask for yourself
stays inclusive, so `end_ele="OTR4"` still gives you `OTR4_beam` and the OTR4 image PVs.

#### If the extents overlap anyway

Because the stages meet at a plane rather than overlapping, there is normally nothing
to deduplicate. As a safeguard, any variables that do turn out to be shared are
unregistered from the downstream stage — `StagedModel` rejects duplicates outright,
and this keeps the failure from surfacing only after a full IMPACT run.

A *writable* overlap raises instead of being dropped. That means both stages drive the
same magnet, so their extents genuinely overlap rather than meeting at a plane, and
dropping it downstream would leave that stage tracking a stale value. Check the
handoff element if you see it.

### Targeting One Stage With kwargs

Parameters fall into two kinds, and which one it is decides where you pass it.

**Shared parameters must hold the same value in every stage.** `n_particles` is the
only one: the beam flows through the stages, so a particle count that differs between
them is physically meaningless. Pass it at the top level and it broadcasts to every
stage that declares one.

```python
>>> m = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=1000)   # doctest: +SKIP

```

Setting a shared parameter per stage is refused — divergence would break the
invariant rather than configure anything.

**Stage-specific parameters go through `stage_kwargs`**, a dict keyed by stage name
whose values are that stage's parameter overrides. `track_beam` and
`custom_beam_path` are declared by `bmad_cu_hxr`, so they belong under its entry:

```python
>>> m = get_model(                                                        # doctest: +SKIP
...     ["surrogate_cu_inj", "bmad_cu_hxr"],
...     stage_kwargs={"bmad_cu_hxr": {"custom_beam_path": "beam.h5"}},
... )

```

Passing `custom_beam_path=` at the top level of a chain call is refused, because it
does not say which stage it applies to. The error names the valid `stage_kwargs`
form.

Start and end elements are role-based: use `start_ele` / `end_ele` for the overall
extent — the first and last stage respectively:

```python
>>> m = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", end_ele="TD11")   # doctest: +SKIP

```

Or set them per stage inside `stage_kwargs` — `end_ele` there means the same thing
as the builder's `end_element`:

```python
>>> m = get_model(                                                        # doctest: +SKIP
...     ["impact_cu_inj", "bmad_cu_hxr"],
...     handoff_loc="YAG03",
...     stage_kwargs={"bmad_cu_hxr": {"end_ele": "TD11"}},
... )

```

**Chain aliases** are the same shape — pass an alias instead of a list, and
`stage_kwargs` keys are the stage names the alias resolves to (see `list_models()`):

```python
>>> m = get_model(                                                        # doctest: +SKIP
...     "high_fidelity_cu_hxr_s2e",
...     n_particles=1000,
...     stage_kwargs={"bmad_cu_hxr": {"custom_beam_path": "beam.h5"}},
... )

```

Precedence when both homes touch the same key (rare): the `stage_kwargs` value
wins over the top-level broadcast, which wins over the builder default. In
normal use the two homes are disjoint — shared params are rejected inside
`stage_kwargs` and stage-specific params are rejected at the top level, so
every parameter has exactly one legal home.

To see what a model accepts:

```python
>>> MODELS["bmad_cu_hxr"].params
{'start_element': 'OTR2', 'end_element': 'END', 'track_beam': False, 'custom_beam_path': None}

>>> MODELS["impact_cu_inj"].shared_params
frozenset({'n_particles'})

```

## API Reference

```
get_model(spec, *, handoff_loc=None, start_ele=None, end_ele=None,
          stage_kwargs=None, **kwargs)
```

| Parameter | Type | Description |
|---|---|---|
| `spec` | str or list[str] | Model ID, chain alias (`high_fidelity_cu_hxr_s2e` etc.), or `[upstream, downstream]` to chain. Duplicate names in a list are rejected |
| `handoff_loc` | str | Where the stages exchange beam. Inferred from the upstream model's standard end when omitted. Must be in `common_handoff_points()` |
| `start_ele` | str | Element to start tracking from (first stage) |
| `end_ele` | str | Element to stop tracking at (last stage) |
| `stage_kwargs` | dict[str, dict] | Per-stage overrides for a chain, keyed by stage name. Values are `{param: value}` dicts. Not accepted for single-model calls |
| `**kwargs` | any | Single-model call: any builder parameter. Chain call: shared parameters only (broadcast to every stage declaring them) |

For staged chains `get_model()` also removes variables that both stages publish at the
handoff, and forces beam tracking on for every stage that supports it.

```
list_models(facility: str | None = None, simulator: str | None = None)
```

Printable table of registered models with facility, simulator, start, end and
description columns, plus a block of standard staged chains as discovery aids.
Optionally filtered by facility (`"lcls"` / `"facet2"`) or simulator (`"bmad"`,
`"impact"`, `"surrogate"`, `"cheetah"`); chain rows are only shown when both stages
survive the filter.

```
list_handoff_points(model_id: str) -> tuple[str, ...]
```

Returns the suggested handoff point names for a given model, in lattice order. A
discovery aid, not a restriction.

```
common_handoff_points(*model_ids: str) -> tuple[str, ...]
```

Returns the handoff points shared by all named models, in lattice order, excluding
`CATHODE`. Use it to see where two models can legally hand over.
