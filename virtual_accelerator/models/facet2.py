import tempfile

from copy import copy

import logging
from typing import Any

from virtual_accelerator.impact.actions import _ReadbackFromControlMixin
from virtual_accelerator.utils.variables import get_element_attr_mapping

logger = logging.getLogger(__name__)

IMPACT_GROUP_PV_MAPPING = {
    "group:L0AF_phase": {"pv": "KLYS:IN10:81:PDES", "element": "L0AF_entrance"},
    "group:L0BF_phase": {"pv": "KLYS:IN10:41:PDES", "element": "L0BF_entrance"},
    "group:L0AF_scale": {
        "pv": "KLYS:IN10:81:ADES",
        "scale": 1e6,
        "element": "L0AF_entrance",
    },
    "group:L0BF_scale": {
        "pv": "KLYS:IN10:41:ADES",
        "scale": 1e6,
        "element": "L0BF_entrance",
    },
    "group:GUNF_phase": {"pv": "KLYS:IN10:31:PDES", "element": "GUNF"},
    "group:GUNF_scale": {"pv": "KLYS:IN10:31:ADES", "scale": 1e6, "element": "GUNF"},
}


def add_facet_custom_bmad_variables(model) -> None:
    """
    Add custom variables to the FACET-II model.

    Parameters
    ----------
    model : LUMEBmadModel
        The FACET-II model to which custom variables will be added.
    """
    from virtual_accelerator.bmad.actions import CavityPREQReadbackVariable
    from lume_bmad.actions import EleScalarVariable

    # Add the L0B RF phase feedback variable to the model if L0BF#1 is included in the model
    if "L0BF#1" in model.get("name"):
        logger.debug("Adding L0B RF phase feedback variable to the model.")
        model.register_action_variable(
            EleScalarVariable(
                name="KLYS:LI10:41:SFB_PDES",
                element_name="L0BF",
                property_name="PHI0",
                unit="degrees",
            )
        )

    if "TCY10490#1" in model.get("name"):
        logger.debug("Adding TCY10490 phase readback variable to the model.")
        model.register_action_variable(
            CavityPREQReadbackVariable(
                name="KLYS:LI10:51:PACT05",
                element_name="TCY10490",
            )
        )

def add_facet_custom_impact_variables(model) -> None:
    """
    Add custom Impact variables to the FACET-II model.

    This includes:
    - SolenoidBCTRLVariable which connects BCTRL variables to the solenoid field scale in the Impact simulator via: solenoid_field_scale = BCTRL / 1.6
    - SolenoidBACTVariable which provides a readback of the solenoid field scale in the Impact simulator.

    Parameters
    ----------
    model : ImpactModel
        The FACET-II model to which custom Impact variables will be added.
    """
    from virtual_accelerator.impact.actions import ImpactScalarVariable, WritableActionMixin
    from virtual_accelerator.impact.actions import BminVariable, BmaxVariable, StatusVariable, _ReadbackFromControlMixin, ControlStateVariable
    from impact import Impact

    class SolenoidBCTRLVariable(ImpactScalarVariable, WritableActionMixin):
        """Action that operates on the BCTRL/BDES property of Solenoids"""
        read_only: bool = False
        unit: str = "kG-m"

        def _get(self, simulator: Impact) -> Any:
            ele_attr = self._get_ele_attr(simulator)
            return ele_attr["solenoid_field_scale"] * 1.6  # emperically known for FACET-II

        def _set(self, simulator: Impact, value: Any) -> None:
            ele_attr = self._get_ele_attr(simulator)
            self._set_ele_attr(
                simulator, "solenoid_field_scale", value / 1.6  # emperically known for FACET-II
            )

    class SolenoidBACTVariable(_ReadbackFromControlMixin, SolenoidBCTRLVariable):
        """Action that operates on the BACT property of Solenoids"""


    base_pv = "SOLN:IN10:121"
    element_name = "SOL10111"
    mapping = get_element_attr_mapping()["Solenoid"]
    
    # register variables based on mapping -- convert string to class type defined above
    for suffix, var_class in mapping.items():
        model.register_impact_action_variable(
            locals().get(var_class)(
                name=f"{base_pv}:{suffix}",
                element_name=element_name,
            )
        )


def get_facet_bmad_model(
    start_element="L0AFEND", end_element="END", track_beam=False, custom_beam_path=None
):
    """
    Get the LUMEBmadModel for the FACET-II lattice from `start_element` to `end_element`.

    Parameters
    -------------
    start_element: str, optional
        The starting element for the model. Default is "L0AFEND".
    end_element: str, optional
        The ending element for the model. Default is "END".
    track_beam: bool, optional
        Whether to enable beam tracking in the model. Default is False.
    custom_beam_path: str, optional
        Path to custom beam file for tracking. If None, will use default design beam. Default is None.


    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the FACET-II lattice.

    Notes
    -----
    - The model is built using the BmadModelSpec for FACET-II
    - To match real PVs custom aliases are added for the PROF and KLYS elements.
    - Custom tao commands are added to the model to disable certain effects and set parameters including:
        - absolute_time_tracking=true
        - lr_wakes_on=false
        - sr_wakes_on=false
        - n_rf_steps=1000 for lcavity elements
        - is_on=false for fixer elements
    """
    from virtual_accelerator.bmad.factory import BmadModelSpec, build_bmad_model

    custom_aliases = {
        "PR10241": "PROF:IN10:241",
        "PR10571": "PROF:IN10:571",
        "PR10711": "PROF:IN10:711",
        "TCY10490": "KLYS:LI10:51",
    }

    spec = BmadModelSpec(
        feature="FACET-II Bmad model",
        lattice_env_var="FACET2_LATTICE",
        tao_init_relpath="bmad/models/f2_elec/tao.init",
        mapping_beampath=None,
        profmon_config_filename="facet2_profmon_info.yaml",
        default_beam_relpath="../beams/2024-10-22_oneBunch.h5",
        default_track_start="L0AFEND",
    )
    model = build_bmad_model(
        spec=spec,
        start_element=start_element,
        end_element=end_element,
        track_beam=track_beam,
        custom_beam_path=custom_beam_path,
        custom_aliases=custom_aliases,
        custom_tao_commands=[
            "set bmad_com absolute_time_tracking=true",
            "set bmad_com lr_wakes_on=false",
            "set bmad_com sr_wakes_on=false",
            "set ele lcavity::* n_rf_steps=1000",
            "set ele fixer::* is_on=false",
        ],
    )

    add_facet_custom_bmad_variables(model)

    return model


def get_facet_staged_model(n_particles=10000, surrogate_inputs="machine", **kwargs):
    """
    Get the StagedModel for the FACET-II lattice from PR10241 to END, with an injector surrogate model.

    Parameters
    -------------
    n_particles: int, optional
        Number of particles to simulate in the surrogate model. Default is 10000.
    surrogate_inputs: str, optional
        Input for the surrogate model either "machine" or "sim". Default is "machine".
    **kwargs:
        Keyword arguments to be passed to the bmad LUMEModel instance as needed.

    Returns
    -------
    StagedModel
        Instance of the StagedModel for the FACET-II lattice.
    """
    from facet2_inj_ml_model import load_model
    from virtual_accelerator.surrogates.beam_output import BeamOutputModel
    from lume.staged_model import StagedModel

    injector_surrogate = BeamOutputModel(
        load_model(surrogate_inputs),
        n_particles=n_particles,
        t0=3.15391398e-09,
        p0c=6.3e06,
        z0=0.9420843,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".h5")
    fname = tmp.name
    tmp.close()
    injector_surrogate.final_particles.write(fname)

    facet_bmad_model = get_facet_bmad_model(
        start_element="PR10241", track_beam=True, custom_beam_path=fname, **kwargs
    )

    staged_model = StagedModel([injector_surrogate, facet_bmad_model])

    return staged_model


def get_facet_impact_model(n_particles: int = 100, end_element="PR10571"):
    from virtual_accelerator.impact.factory import (
        ImpactModelSpec,
        build_impact_model,
        get_actions_from_groups,
    )

    spec = ImpactModelSpec(
        lattice_env_var="FACET2_LATTICE",
        distgen_file="distgen/models/f2e_inj/v0/distgen.yaml",
        impact_yaml_file="impact/models/f2e_inj/v0/ImpactT.yaml",
        profmon_config_filename="facet2_profmon_info.yaml",
        n_particles=n_particles,
        numprocs=1,
        space_charge=False,
        stop_location=end_element,
    )
    model = build_impact_model(spec)

    # register custom action variables for solenoids based on the element attribute mapping
    add_facet_custom_impact_variables(model)

    # register custom actions for linac L0A and L0B sections
    group_actions = get_actions_from_groups(model.impact_model.simulator, spec)

    for action in group_actions:
        old_name = copy(action.name)
        action.name = IMPACT_GROUP_PV_MAPPING[old_name]["pv"]
        action.scale = IMPACT_GROUP_PV_MAPPING[old_name].get("scale", 1.0)

        if (
            IMPACT_GROUP_PV_MAPPING[old_name]["element"]
            in model.impact_model.simulator.ele
        ):
            model.register_impact_action_variable(action)

    return model
