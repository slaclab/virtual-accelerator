import argparse

from virtual_accelerator.utils.optional_dependencies import import_optional_symbol

import logging


def main():
    parser = argparse.ArgumentParser(
        description="Run the CU HXR model with BMAD or CHEETAH backend"
    )
    choices = ["cu_hxr_bmad", "cu_hxr_cheetah", "facet_bmad"]
    parser.add_argument(
        "model",
        choices=choices,
        help="Model backend to run (cu_hxr_bmad, cu_hxr_cheetah, or facet_bmad)",
    )
    parser.add_argument(
        "--end-element",
        default="END",
        help="End lattice element for BMAD models (default: END)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.getLogger("pytao").setLevel(logging.WARNING)

    Runner = import_optional_symbol(
        "lume_pva.runner",
        "Runner",
        feature="virtual accelerator runner CLI",
        extra="pva",
    )

    from virtual_accelerator.models.cu_hxr import (
        get_cu_hxr_bmad_model,
        get_cu_hxr_cheetah_model,
    )
    from virtual_accelerator.models.facet2 import get_facet_bmad_model

    # Get the appropriate model based on user input
    if args.model == "cu_hxr_bmad":
        model = get_cu_hxr_bmad_model(end_element=args.end_element)
    elif args.model == "cu_hxr_cheetah":  # cu_hxr_cheetah
        if args.end_element != "END":
            parser.error(
                "--end-element is only supported for BMAD models "
                "(cu_hxr_bmad, facet_bmad)."
            )
        model = get_cu_hxr_cheetah_model()
    elif args.model == "facet_bmad":
        model = get_facet_bmad_model(end_element=args.end_element, track_beam=True)
    else:
        raise ValueError(f"Invalid model choice. Please choose one of {choices}.")

    # Run the model
    runner = Runner(model)
    runner.run()


if __name__ == "__main__":
    main()
