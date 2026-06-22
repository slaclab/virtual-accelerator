from virtual_accelerator.bmad.factory import BmadModelSpec, build_bmad_model


def get_cu_hxr_bmad_model(
    start_element="OTR2", end_element="END", track_beam=False, custom_beam_path=None
):
    """
    Get the LUMEBmadModel for the CU_HXR lattice from OTR2 to END.

    Parameters
    -------------
    start_element: str, optional
        The starting element for the model. Default is "OTR2".
    end_element: str, optional
        The ending element for the model. Default is "END".
    track_beam: bool, optional
        Whether to enable beam tracking in the model. Default is False.
    custom_beam_path: str, optional
        Path to custom beam file for tracking. If None, will use default design beam. Default is None.


    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the CU_HXR lattice.
    """

    spec = BmadModelSpec(
        feature="CU HXR Bmad model",
        lattice_env_var="LCLS_LATTICE",
        tao_init_relpath="bmad/models/cu_hxr/tao.init",
        profmon_config_filename="cu_hxr_profmon_info.yaml",
        default_beam_relpath="bmad/bmad_set_beam2000_pg",
        default_track_start="OTR2",
    )
    return build_bmad_model(
        spec=spec,
        start_element=start_element,
        end_element=end_element,
        track_beam=track_beam,
        custom_beam_path=custom_beam_path,
    )
