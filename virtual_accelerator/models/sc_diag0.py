import os

from virtual_accelerator.utils.optional_dependencies import import_optional
from virtual_accelerator.utils.variables import (
    get_epics_to_name_mapping,
    split_control_and_observable,
)


def _check_optional_modules(module_names: list[str], feature: str, extra: str) -> None:
    """Validate all optional modules for a feature in a single gate check."""
    for module_name in module_names:
        import_optional(module_name, feature=feature, extra=extra)


def get_sc_diag0_cheetah_model():
    """
    Get the LUMECheetahModel for the SC_DIAG0 lattice from BEAM0 to OTRDG02.

    Returns
    -------
    LUMECheetahModel
        Instance of the LUMECheetahModel for the SC_DIAG0 lattice.
    """

    _check_optional_modules(
        [
            "lume_cheetah",
            "cheetah.accelerator",
            "cheetah.particles",
            "virtual_accelerator.cheetah.transformer",
            "virtual_accelerator.cheetah.variables",
            "torch",
        ],
        feature="SC DIAG0 Cheetah model",
        extra="cheetah",
    )

    from lume_cheetah import LUMECheetahModel, CheetahSimulator
    from cheetah.accelerator import Segment
    from cheetah.particles import ParticleBeam
    from virtual_accelerator.cheetah.transformer import SLACCheetahTransformer
    from virtual_accelerator.cheetah.variables import get_variables_from_segment
    import torch

    incoming_beam = ParticleBeam.from_twiss(
        beta_x=torch.tensor(9.34),
        alpha_x=torch.tensor(-1.6946),
        emittance_x=torch.tensor(1e-7),
        beta_y=torch.tensor(9.34),
        alpha_y=torch.tensor(-1.6946),
        emittance_y=torch.tensor(1e-7),
        energy=torch.tensor(90e6),
    )
    incoming_beam.particle_charges = torch.tensor(1.0)

    # Create lattice from file
    segment = get_diag0_beamline()

    # Ensure screen elements can support vectorization
    for element in segment.elements:
        element_type = type(element).__name__
        if element_type == "Screen":
            element.method = "kde"

    # Define the simulator using lattice and particle beam
    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=incoming_beam,
    )

    # get control system device to cheetah mapping
    control_name_to_element_name = {
        k: v.lower() for k, v in get_epics_to_name_mapping().items()
    }

    # Create transformer that handles maps get/set calls
    transformer = SLACCheetahTransformer(
        control_name_to_cheetah=control_name_to_element_name
    )

    # Get supported control system variables
    # for the model
    variables = get_variables_from_segment(segment)
    # Define the controllable and observable variables
    control_variables, observable_variables = split_control_and_observable(variables)

    # Create model
    model = LUMECheetahModel(
        simulator=simulator,
        transformer=transformer,
        control_variables=control_variables,
        observable_variables=observable_variables,
    )

    return model
