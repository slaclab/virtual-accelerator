def get_cu_hxr_rmat(start_element: str, end_element: str):
    """
    Dedicated model configuration optimized for fast CU HXR R-matrix calculations
    """
    from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model
    from virtual_accelerator.bmad.actions import RMatrixAction

    model = get_cu_hxr_bmad_model(start_element, end_element, track_beam=False)

    # unregister any read only action variables
    # -- need to keep non read-only ones for model consistency
    for action_var in list(model.supported_variables.keys()):
        if model.supported_variables[action_var].read_only:
            model.unregister_action_variable(action_var)

    # register R-matrix specific action variable
    model.register_action_variable(
        RMatrixAction(
            name=f"rmat:{start_element}_{end_element}",
            start_element=start_element,
            end_element=end_element,
        )
    )

    # turn off radiation calculations
    model.tao.cmd("set bmad_com radiation_fluctuations_on = F")

    return model
