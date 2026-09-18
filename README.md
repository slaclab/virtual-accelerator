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

Optional Dependency Keys by Model:
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

Creating model instances also requires the `$LCLS_LATTICE` environment variable for LCLS-based models and
`$FACET2_LATTICE` for FACET-II models; each should point to a location containing the
contents of the lcls-lattice repo https://github.com/slaclab/lcls-lattice or the facet2-lattice
repo https://github.com/slaclab/facet2-lattice.

## Cached beam distributions

Reference beam distributions at handoff planes (e.g. FACET PR10241, L0AFEND) are stored
under `virtual_accelerator/beams/` via **Git LFS**. Before cloning or pulling this repo,
install Git LFS once per machine:

```
brew install git-lfs      # or: conda install -c conda-forge git-lfs
git lfs install
```

If you have already cloned without LFS, run `git lfs pull` to fetch the beam blobs.

### Layout

Beams are grouped by scenario (one subdirectory per date-tagged run) and named by
handoff plane and particle count. Each `.h5` ships with a `.h5.meta.json` sidecar
of the same basename — the doubled extension makes it obvious the JSON is metadata
*about* the neighboring HDF5 file (and pairs trivially: `path + ".meta.json"`):

```
virtual_accelerator/beams/
    2024-10-22_facet2_oneBunch/
        L0AFEND_100000.h5     # + L0AFEND_100000.h5.meta.json
        PR10241_100000.h5     # + PR10241_100000.h5.meta.json
        PR10241_10000.h5      # + PR10241_10000.h5.meta.json    (small version for smoke tests)
```

### Adding a new beam

1. Place the `.h5` in an appropriate subdirectory of `virtual_accelerator/beams/` (create
   a new date-tagged subdirectory if the scenario is new). The `*.h5` LFS filter in
   `.gitattributes` handles the tracking automatically.
2. Write a `<name>.h5.meta.json` sidecar next to it. Required fields:

   ```json
   {
     "plane": "PR10241",
     "s_m": 0.942,
     "ref_energy_eV": 6.099e6,
     "generator": "impact",
     "n_particles": 100000,
     "charge_C": 1.6e-9,
     "species": "electron",
     "date_generated": "YYYY-MM-DD",
     "mode": "nominal_one_bunch",
     "source": "where this beam came from (upstream repo path, notebook, run command, ...)",
     "notes": ""
   }
   ```

   The `plane`, `s_m`, `ref_energy_eV`, and `n_particles` fields can be filled from
   `pmd_beamphysics.ParticleGroup(path).avg("z")`, `.avg("energy")`, `.n_particle`.

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
