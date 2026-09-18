This repository includes SLAC-specific python code to be utilized with creating and running virtual accelerators of SLAC beamlines via the LUME framework (see https://github.com/slaclab/lume-base, https://github.com/lume-science).

## Installation

First, clone this repo to a local location and enter the directory.
Also install Conda if you don't already have it. (we recommended using Conda from [Miniforge](https://conda-forge.org/download/))

Then create a new conda environment using mamba, containing bmad and pytao:
```
mamba create -n va-env -c conda-forge python=3.12 bmad pytao
```
(you can also use an existing environment, although it could lead to dependency conflicts)

Now activate the newly created environment:
```
conda activate va-env
```

and then install the remaining required packages with pip by running:
```
pip install .
```
the `-e` flag can be added if you plan to edit the virtual accelerator code.

Lastly, install backend-specific extras depending on which simulation types you need:
```
pip install .[bmad]
pip install .[cheetah]
pip install .[impact]
pip install .[pva]
pip install .[surrogate]
pip install .[all]
```

Supported models (see `docs/model_registry_usage.md` for the full API):

| Model | Facility | Simulator | Start | End | Extras |
| --- | --- | --- | --- | --- | --- |
| `impact_cu_inj` | LCLS | IMPACT | CATHODE | YAG03 | `impact` |
| `bmad_cu_hxr` | LCLS | Bmad | OTR2 | END | `bmad` |
| `surrogate_cu_inj` | LCLS | Surrogate | CATHODE | OTR2 | `surrogate` |
| `cheetah_cu_hxr` | LCLS | Cheetah | CATHODE | END | `cheetah` |
| `zfel_cu_hxr` | LCLS | ZFEL | — | — | `zfel` |
| `impact_f2e_inj` | Facet2 | IMPACT | CATHODEF | PR10241 | `impact` |
| `surrogate_f2e_inj` | Facet2 | Surrogate | CATHODEF | PR10241 | `surrogate` |
| `bmad_f2_elec` | Facet2 | Bmad | CATHODEF | END | `bmad` |

Standard staged chains (build with `get_model([upstream, downstream], ...)`):

| Alias | Upstream | Downstream | Handoff |
| --- | --- | --- | --- |
| `high_fidelity_cu_hxr_s2e` | `impact_cu_inj` | `bmad_cu_hxr` | YAG03 |
| `fast_cu_hxr_s2e` | `surrogate_cu_inj` | `bmad_cu_hxr` | OTR2 |
| `high_fidelity_facet2_s2e` | `impact_f2e_inj` | `bmad_f2_elec` | PR10241 |
| `fast_facet2_s2e` | `surrogate_f2e_inj` | `bmad_f2_elec` | PR10241 |

The `Runner` CLI additionally needs the `pva` extra.

The package now lazily imports backend-specific dependencies. If you call a model
whose optional dependency is not installed, you will get an actionable error with
the matching extra to install.

Creating model instances also requires the `$LCLS_LATTICE` environment variable for LCLS-based models and
`$FACET2_LATTICE` for FACET-II models; each should point to a location containing the
contents of the lcls-lattice repo https://github.com/slaclab/lcls-lattice or the facet2-lattice
repo https://github.com/slaclab/facet2-lattice.

## Running the models

You can use the runner script to start the model. The script allows you to specify the model backend,
number of particles, and end element to run with.

For example:
```
python virtual_accelerator/models/runners.py cu_hxr_bmad --end-element OTR4
```

For more info, run:
```
python virtual_accelerator/models/runners.py -h
```

#### Note
The Cu Injector model is present in subtrees/lcls_cu_injector_ml_model.
To pull latest changes from the Cu Inj repo

```
pip install git+https://github.com/slaclab/lcls_cu_injector_ml_model.git
```
