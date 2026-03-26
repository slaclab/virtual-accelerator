# LCLS Cu Injector NN Model

This repository contains the model files corresponding to the LCLS Cu injector NN surrogate model, and example notebooks illustrating how to load and use the model. Using [LUME-model](https://github.com/slaclab/lume-model) is recommended.

## Model Description

The model was trained on IMPACT-T simulation data by Auralee Edelen to predict beam properties at OTR2 using injector PVs for LCLS. As the model was trained with normalized data, input and output transformations have to be applied to use it on simulation data. Another layer of transformations is required for using it with EPICS data. See provided examples for more information.

The model YAML provided in this repo handles the full PV to sim to model transformations.

<br/>
<img src="docs/transformers.png" alt="drawing" width="1000"/>
<br/><br/>

## Dependencies

```shell
lume-model
```

## Usage

From the main repository directory, call

```python
from lume_model.models import TorchModel

# load model from yaml
model = TorchModel("model_config.yaml")

# evaluate the model at a given point
print(model.evaluate({"QUAD:IN20:425:BACT": -1}))

# get model input variables
print(model.input_variables)

# get model output variables
print(model.output_variables)
```

NOTE: when not specified, input variables are set to their default values as defined in model_config.yaml

## Examples

* [Load and print model information](docs/examples/info.ipynb)
* [Load as LUME-model for use with EPICS data](docs/examples/lume_model_epics.ipynb)

## Default Input Variables

The default value for `QE01:b1_gradient` in the [simulation variable specification](https://github.com/slaclab/lcls_cu_injector_ml_model/blob/old-deployment/model/sim_variables.yml)
has been noticed to lie outside the given value range (likewise for `QUAD:IN20:425:BACT` in the [PV variable specification](https://github.com/slaclab/lcls_cu_injector_ml_model/blob/old-deployment/model/pv_variables.yml)).
Thus, a new value was determined by minimizing the model prediction of the transverse beam size within the valid range
(documented in [this notebook](https://github.com/slaclab/lcls_cu_injector_ml_model/blob/old-deployment/correct_inconsistent_default_value.ipynb)).

## Notes about Working with EPICS PV Values

### Unmeasured Input PVs

Some of the input features used as features of the model are not available in EPICS. These include:

#### `distgen:t_dist:length:value`

This is the **pulse length** within the simulation. There has been some discussion about creating a PV to record the pulse length but for now, a reference value of 1.8550514181818183 (PV units) or 3.06083484 (sim units) is used by default.

#### `L0B_scale:voltage`

As demonstrated by the train input min and max values in [model.json](https://github.com/slaclab/lcls_cu_injector_ml_model/blob/old-deployment/info/model.json), this value was treated as a constant when training the surrogate model. However in reality, its PV value `ACCL:IN20:400:L0B_ADES` shows a distribution of values. If it was to be used in the model for predictions, the error would increase dramatically and therefore any measured values from EPICS are overwritten by the value seen during training, scaled to PV units.

#### `distgen:total_charge:value`

As above, the **charge value** was constant in the training dataset but its PV value `FBCK:BCI0:1:CHRG_S` shows a distribution of values. Measured values from EPICS are overwritten by the value seen during training, scaled to PV units.

#### `distgen:r_dist:sigma_xy:value`

The value for the **beam size** (r_dist) is not measured directly in EPICS but we do measure the XRMS and YRMS value of the beam. We use these PVs (`CAMR:IN20:186:XRMS` and `CAMR:IN20:186:YRMS`) to calculate a value for the beam size using the formula:

```python
r_dist = np.sqrt(data["CAMR:IN20:186:XRMS"].values ** 2 + data["CAMR:IN20:186:YRMS"].values ** 2)
```

We call this computed PV `CAMR:IN20:186:R_DIST`. Therefore, when pulling data from the archive, this step needs to be completed in any data processing.
