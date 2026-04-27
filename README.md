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

The package now lazily imports backend-specific dependencies. If you call a model
whose optional dependency is not installed, you will get an actionable error with
the matching extra to install.

Creating the model instances requires the `$LCLS_LATTICE` environment variable to be set to a location containing the
contents of the lcls-lattice repo https://github.com/slaclab/lcls-lattice.


#### Note
The Cu Injector model is present in subtrees/lcls_cu_injector_ml_model.
To pull latest changes from the Cu Inj repo

```
git subtree pull --prefix subtrees/lcls_cu_injector_ml_model https://github.com/slaclab/lcls_cu_injector_ml_model.git main --squash
```
