from virtual_accelerator.bmad.factory import BmadModelSpec, build_bmad_model


def get_facet_bmad_model(
    start_element="PROF241", end_element="END", track_beam=False, custom_beam_path=None
):
    """
    Get the LUMEBmadModel for the FACET-II lattice from PROF241 to END.

    Parameters
    -------------
    start_element: str, optional
        The starting element for the model. Default is "PROF241".
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
    """

    spec = BmadModelSpec(
        feature="FACET-II Bmad model",
        lattice_env_var="FACET2_LATTICE",
        tao_init_relpath="bmad/models/f2_elec/tao.init",
        mapping_beampath="F2_ELEC",
        screens=("PR10571", "PR10711"),
        profmon_config_filename="facet2_profmon_info.yaml",
        default_track_start="PROF241",
    )
    return build_bmad_model(
        spec=spec,
        start_element=start_element,
        end_element=end_element,
        track_beam=track_beam,
        custom_beam_path=custom_beam_path,
    )
