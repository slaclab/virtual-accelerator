This repository includes SLAC-specific python code to be utilized with creating and running virtual accelerators of SLAC beamlines via the LUME framework (see https://github.com/slaclab/lume-base, https://github.com/lume-science).

## Installation
Clone this repo into a local location, enter it and run
```
pip install .
```
the `-e` flag can be added if editing the files.

Install backend-specific extras depending on which simulation types you need:

```
pip install .[bmad]
pip install .[cheetah]
pip install .[impact]
pip install .[pva]
pip install .[surrogate]
pip install .[all]
```

### Optional Dependency Keys by Model

| Model / Factory Function | Optional dependency key(s) | Notes |
| --- | --- | --- |
| `get_cu_hxr_bmad_model` | `bmad` | Requires BMAD/PyTAO backend. |
| `get_cu_hxr_cheetah_model` | `cheetah` | Requires Cheetah backend. |
| `get_sc_diag0_cheetah_model` | `cheetah` | Requires Cheetah backend. |
| `InjectorSurrogate` | `surrogate` | Uses torch surrogate + cheetah particles. |
| `FacetInjectorSurrogate` | `surrogate` | Uses FACET-II covariance surrogate + beam sampling wrapper. |
| `get_cu_hxr_staged_model` | `surrogate`, `bmad` | Stages `InjectorSurrogate` + CU HXR BMAD model. |
| `get_facet_staged_model` | `surrogate`, FACET BMAD implementation | Stages `FacetInjectorSurrogate` + FACET BMAD model once `virtual_accelerator.models.facet.get_facet_bmad_model` is implemented. |
| `virtual_accelerator.models.runners` CLI | `pva` (+ model backend key) | Runner requires `pva`; selected model backend must also be installed. |

The package now lazily imports backend-specific dependencies. If you call a model
whose optional dependency is not installed, you will get an actionable error with
the matching extra to install.

`InjectorSurrogate` also depends on the installable Cu injector model package:

```
pip install git+https://github.com/slaclab/lcls_cu_injector_ml_model.git@packaging
```

`FacetInjectorSurrogate` depends on the installable FACET-II injector model package:

```
pip install git+https://github.com/slaclab/facet2_inj_ml_model.git
```

This repository now includes a `get_facet_staged_model` scaffold. It still requires
`virtual_accelerator.models.facet.get_facet_bmad_model(**kwargs)` to be implemented
with a FACET BMAD model exposing `input_beam` and `output_beam` as
`ParticleGroupVariable` values before the staged model can run end-to-end.

Creating the model instances requires the `$LCLS_LATTICE` environment variable to be set to a location containing the
contents of the lcls-lattice repo https://github.com/slaclab/lcls-lattice.

