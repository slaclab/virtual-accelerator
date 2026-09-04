# Virtual Accelerator (VA) Registry — README
## Overview
The virtual_accelerator.registry module provides a unified interface for loading, configuring, and chaining accelerator simulation models for the LCLS copper linac (CU) and FACET-II. Models can be used standalone or chained together to simulate the full beamline from cathode to end.

Building a model needs the matching lattice checkout on the environment:
`$LCLS_LATTICE` for the `*_cu_*` models, `$FACET2_LATTICE` for the `*_f2*` ones.

### Installation
```bash
pip install git+https://github.com/slaclab/virtual-accelerator.git
```

### Quick Start
```python
from virtual_accelerator.registry import (
    get_model,
    models_available,
    list_models,
    list_handoff_points,
    common_handoff_points,
)
from virtual_accelerator.registry.models import MODELS
```

### Available Models
Print all registered models and their descriptions:

```python
>>> print(models_available)
impact_cu_inj      IMPACT-T LCLS injector, cathode -> YAG03
bmad_cu_hxr        Bmad CU-HXR linac, injector handoff -> END
surrogate_cu_inj   NN LCLS injector surrogate, cathode -> OTR2
cheetah_cu_hxr     Cheetah nc_hxr, cathode -> END
impact_f2e_inj     IMPACT-T FACET-II injector, cathode -> PR10241
surrogate_f2e_inj  NN FACET-II injector surrogate, cathode -> PR10241
bmad_f2_elec       Bmad FACET-II e- linac, injector handoff -> END
```

Filter by facility or engine:

```python
>>> list_models(facility="facet2")
['impact_f2e_inj', 'surrogate_f2e_inj', 'bmad_f2_elec']

>>> list_models(engine="bmad")
['bmad_cu_hxr', 'bmad_f2_elec']
```

### Handoff Points
Each model exposes a set of suggested handoff points — named locations where beam tracking can start or stop, and where chained models exchange beam state.

```python
>>> for m in ["impact_cu_inj", "bmad_cu_hxr", "surrogate_cu_inj", "cheetah_cu_hxr"]:
...     print(m, list_handoff_points(m))

impact_cu_inj      ('YAG02', 'YAG03')
bmad_cu_hxr        ('CATHODE', 'YAG02', 'YAG03', 'OTRH1', 'OTRH2', 'OTR1', 'OTR2', 'OTR3', 'OTR4', 'OTR11', 'OTR12', 'OTR21', 'OTRDMP', 'END')
surrogate_cu_inj   ('OTR2',)
cheetah_cu_hxr     ()
impact_f2e_inj     ('PR10241',)
surrogate_f2e_inj  ('PR10241',)
bmad_f2_elec       ('CATHODEF', 'PR10241', 'L0AFEND', 'PR10465', 'PR10471', 'PR10571', 'PR10711', 'END')
```

The cathode appears only on the Bmad models, whose start is configurable — you can slice
from the front of the machine with `start_ele="CATHODE"` (LCLS) or `start_ele="CATHODEF"`
(FACET; the element carries an `F` suffix in that lattice). The injector models always
begin at the cathode and cannot be told otherwise, so listing it there would advertise
something you cannot pass.

FACET's injectors list only `PR10241`, which is what restricts every FACET chain to that
one handoff plane. `bmad_f2_elec` still lists the downstream screens so they remain usable
as `end_ele`.

### Shared Handoff Points
`common_handoff_points()` returns the locations two models can actually hand over at
— the intersection of their handoff points, with `CATHODE` excluded since nothing is
upstream of it.

```python
>>> common_handoff_points("impact_cu_inj", "bmad_cu_hxr")
('YAG02', 'YAG03')

>>> common_handoff_points("surrogate_cu_inj", "bmad_cu_hxr")
('OTR2',)

>>> common_handoff_points("cheetah_cu_hxr", "bmad_cu_hxr")
()

>>> common_handoff_points("impact_f2e_inj", "bmad_f2_elec")
('PR10241',)
```

### Loading a Single Model
Use get_model() with a model ID and an optional end_ele to stop tracking at a specific screen.

```python
>>> get_model("bmad_cu_hxr", end_ele="TD11")
<lume_bmad.model.LUMEBmadModel object at 0x150a9d370>

>>> get_model("impact_cu_inj", end_ele="YAG03")
<impact.model.distgen.distgen_impact_model.LUMEDistgenImpactModel object at 0x1666b6450>
```

### Error: Unknown Model Name
Model IDs must be exact. Partial names are not supported:

```python
>>> get_model("bmad_cu_hx", end_ele="TD11")
KeyError: "Unknown model 'bmad_cu_hx'. Available: bmad_cu_hxr, cheetah_cu_hxr, impact_cu_inj, surrogate_cu_inj"
```

### Error: Invalid End Element
end_ele must be one of the model's listed handoff points:

```python
>>> get_model("impact_cu_inj", end_ele="otr99")
ValueError: 'OTR99' is not an available end screen for 'impact_cu_inj'.
Suggested points: YAG02, YAG03
```

## Staged Models
Pass a list of two model IDs to get_model() to chain an injector model into a linac model. The upstream model hands off beam particles to the downstream model at a shared handoff point.

surrogate_cu_inj → bmad_cu_hxr — no `handoff_loc` needed, it is inferred from the
surrogate's fixed end (OTR2):
```python
>>> m = get_model(["surrogate_cu_inj", "bmad_cu_hxr"], end_ele="OTR4", n_particles=500)

>>> m.set({"QUAD:IN20:525:BCTRL": -10.0})

>>> print(m.get("OTR4_beam")["norm_emit_y"])
5.850087235892218e-07

>>> print(m.get("OTRS:IN20:711:Image:ArrayData").shape)
(1040, 1392)

>>> print([n.split("#")[0] for n in m.lume_model_instances[1].get("name")][:3])
['BEGINNING', 'OTR2', 'DE06D']
```

impact_cu_inj → bmad_cu_hxr — hand off at YAG03:

```python
>>> model = get_model(
...     ["impact_cu_inj", "bmad_cu_hxr"],
...     handoff_loc="YAG03",
...     end_ele="TD11",
...     n_particles=1000,
... )

>>> model.set({"QUAD:IN20:525:BCTRL": -7.5})

>>> print(model.get("OTR4_beam")["norm_emit_y"])
2.3638227838794528e-07
```

### FACET-II
Both FACET chains hand off at PR10241, so `handoff_loc` can be left out:

```python
>>> m = get_model(["surrogate_f2e_inj", "bmad_f2_elec"], end_ele="PR10711", n_particles=2000)
>>> m = get_model(["impact_f2e_inj", "bmad_f2_elec"], end_ele="PR10711", n_particles=200)
```

Standalone:

```python
>>> get_model("bmad_f2_elec", end_ele="PR10711", track_beam=True)
<lume_bmad.model.LUMEBmadModel object at 0x...>
```

Anything other than PR10241 is refused, as is mixing facilities:

```python
>>> get_model(["impact_f2e_inj", "bmad_f2_elec"], handoff_loc="PR10571")
ValueError: 'PR10571' is not a shared handoff point for 'impact_f2e_inj' -> 'bmad_f2_elec'.
Available: PR10241

>>> get_model(["impact_cu_inj", "bmad_f2_elec"], handoff_loc="YAG03")
ValueError: Cannot stage 'impact_cu_inj' (lcls) onto 'bmad_f2_elec' (facet2): different facilities.
```

### Handoff Validation
`handoff_loc` must be a point both stages share. Anything else is rejected before any
model is built, so you do not pay for an IMPACT run to find out:

```python
>>> get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="OTR4")
ValueError: 'OTR4' is not a shared handoff point for 'impact_cu_inj' -> 'bmad_cu_hxr'.
Available: YAG02, YAG03

>>> get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="CATHODE")
ValueError: 'CATHODE' cannot be a handoff location: nothing is upstream of it.
```

Standard chains:

| Upstream | Downstream | Handoff |
|---|---|---|
| `impact_cu_inj` | `bmad_cu_hxr` | YAG03 |
| `surrogate_cu_inj` | `bmad_cu_hxr` | OTR2 (inferred) |

LCLS needs two handoff planes because its injector models end at different places and
neither can move. The NN surrogate predicts `OTRS:IN20:571` (OTR2) at 135 MeV and
cannot produce a beam at YAG03, which sits before L0B at 64 MeV.

### Overlapping Variables Are Handled For You
Both stages include the handoff element, so both publish its PVs — an IMPACT model
stopped at `YAG03` keeps the screen (it prunes to `s <= stop`) and so does a Bmad model
sliced from `YAG03`. `StagedModel` would reject the pair as duplicates.

`get_model()` resolves this automatically: the upstream stage owns those PVs, because it
is the stage that tracks the beam to that plane, so they are unregistered from the
downstream stage before the chain is assembled. Nothing is required of the caller.

The removal is surgical — only genuine collisions go. At YAG03 the IMPACT stage publishes
four PVs; the Bmad stage publishes those four plus `:X` and `:Y` centroid readbacks that
IMPACT does not provide. So the four move to IMPACT and the two Bmad-only ones stay:

```python
>>> m = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", end_ele="TD11")
>>> imp, bmad = m.lume_model_instances

>>> sorted(v for v in imp.supported_variables if "IN20:351" in v)
['YAGS:IN20:351:Image:ArrayData', 'YAGS:IN20:351:Image:ArraySize0_RBV',
 'YAGS:IN20:351:Image:ArraySize1_RBV', 'YAGS:IN20:351:RESOLUTION']

>>> sorted(v for v in bmad.supported_variables if "IN20:351" in v)
['YAGS:IN20:351:X', 'YAGS:IN20:351:Y']
```

A *writable* overlap raises instead of being dropped. That means both stages drive the
same magnet — their extents overlap rather than meeting at a plane — and dropping it
downstream would leave that stage tracking a stale value.

### Targeting One Stage With kwargs
Parameters fall into two kinds, and which one it is decides how you pass it.

**Shared parameters must hold the same value in every stage.** `n_particles` is the only
one: the beam flows through the stages, so a particle count that differs between them is
physically meaningless. Pass it flat and it reaches every stage that declares one.

```python
>>> m = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=1000)
```

Setting a shared parameter per stage is refused — divergence would break the invariant
rather than configure anything:

```python
>>> get_model([...], **{"impact_cu_inj.n_particles": 200})
ValueError: 'n_particles' must be the same in every stage, so it cannot be set per stage.
Pass n_particles=... instead of 'impact_cu_inj.n_particles'.
```

**Everything else means something different to each stage**, so it is either unambiguous
or you say which stage. `track_beam` and `custom_beam_path` are only declared by
`bmad_cu_hxr`, so they route there on their own:

```python
>>> m = get_model(["surrogate_cu_inj", "bmad_cu_hxr"], custom_beam_path="beam.h5")
```

Start and end elements need naming, because both stages have them. Use `start_ele` /
`end_ele` for the overall extent — first and last stage respectively:

```python
>>> m = get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", end_ele="TD11")
```

Or prefix with the model ID to set one stage:

```python
>>> m = get_model(
...     ["impact_cu_inj", "bmad_cu_hxr"],
...     handoff_loc="YAG03",
...     **{"bmad_cu_hxr.end_ele": "TD11"},
... )
```

The dotted form accepts either spelling — the registry's (`end_ele`, `start_ele`) or the
underlying builder's (`end_element`, `start_element`):

```python
>>> "bmad_cu_hxr.end_ele"      # same as
>>> "bmad_cu_hxr.end_element"
```

Passing a builder spelling *flat* is refused, since it does not say which stage it means:

```python
>>> get_model(["impact_cu_inj", "bmad_cu_hxr"], end_element="TD11")
ValueError: Do not pass 'end_element' directly -- it is the builder's own name and does
not say which stage it applies to. Use end_ele=... for the overall extent, or
"<model_name>.end_element=..." to target one stage.
```

Unknown parameters are rejected outright, listing what is accepted:

```python
>>> get_model("bmad_cu_hxr", n_particle=5)
ValueError: 'n_particle' is not a parameter of any stage.
Accepted: custom_beam_path, end_element, start_element, track_beam
```

To see what a model accepts:

```python
>>> MODELS["bmad_cu_hxr"].params
{'start_element': 'OTR2', 'end_element': 'END', 'track_beam': False, 'custom_beam_path': None}

>>> MODELS["impact_cu_inj"].shared_params
frozenset({'n_particles'})
```

## API Reference
```get_model(spec, *, handoff_loc=None, start_ele=None, end_ele=None, **kwargs)```

| Parameter | Type | Description |
|---|---|---|
| `spec` | str or list[str] | Model ID, or `[upstream, downstream]` to chain |
| `handoff_loc` | str | Where the stages exchange beam. Inferred from the upstream model's standard end when omitted. Must be in `common_handoff_points()` |
| `start_ele` | str | Element to start tracking from (first stage) |
| `end_ele` | str | Element to stop tracking at (last stage) |
| `**kwargs` | any | Builder parameters. Prefix with `"<model_id>."` to target one stage |

For staged chains `get_model()` also removes variables that both stages publish at the
handoff, and forces beam tracking on for every stage that supports it.

```models_available```
Printable summary of all registered models and their descriptions.

```list_handoff_points(model_id: str) -> tuple[str, ...]```

Returns the suggested handoff point names for a given model, in lattice order. A
discovery aid, not a restriction.

```common_handoff_points(*model_ids: str) -> tuple[str, ...]```

Returns the handoff points shared by all named models, in lattice order, excluding
`CATHODE`. Use it to see where two models can legally hand over.
