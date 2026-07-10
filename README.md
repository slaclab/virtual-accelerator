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
| `get_facet_bmad_model` | `bmad` | FACET-II BMAD model; requires `FACET2_LATTICE`. |
| `get_cu_hxr_injector_surrogate_model` | `surrogate` | Uses torch surrogate + cheetah particles. |
| `get_facet_staged_model` | `surrogate`, `bmad` | FACET-II staged model (injector surrogate + FACET-II BMAD). |
| `get_cu_hxr_staged_model` | `surrogate`, `bmad` | Stages `InjectorSurrogate` + CU HXR BMAD model. |
| `virtual_accelerator.models.runners` CLI | `pva` (+ model backend key) | Runner requires `pva`; selected model backend must also be installed. |

The package now lazily imports backend-specific dependencies. If you call a model
whose optional dependency is not installed, you will get an actionable error with
the matching extra to install.

Creating model instances requires the `$LCLS_LATTICE` environment variable for LCLS-based models and
`$FACET2_LATTICE` for FACET-II models; each should point to a location containing the
contents of the lcls-lattice repo https://github.com/slaclab/lcls-lattice or the facet2-lattice
repo https://github.com/slaclab/facet2-lattice.


#### Note
The Cu Injector model is present in subtrees/lcls_cu_injector_ml_model.
To pull latest changes from the Cu Inj repo

```
pip install git+https://github.com/slaclab/lcls_cu_injector_ml_model.git
```

Creating the model instances requires the `$LCLS_LATTICE` environment variable to be set to a location containing the
contents of the lcls-lattice repo https://github.com/slaclab/lcls-lattice.
